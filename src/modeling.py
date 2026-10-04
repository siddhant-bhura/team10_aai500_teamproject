"""
modeling.py

Owner: Umesh
Purpose: Model development, model comparison, validation, final
integration, and technical consistency review.

Dataset: UCI "Default of Credit Card Clients". Reads the cleaned file from
data_cleaning.py and follows the feature recommendations in
reports/eda_findings.md:

  - Start from repayment behaviour (MONTHS_DELINQUENT, PAY_1, PAID_SHARE)
  - Add an inactive-account flag and banded age
  - Prefer recent-vs-distant delinquency over PAY_1 - PAY_6 trends
  - Hold demographics and the full bill block until they earn a place
  - Evaluate with PR-AUC / precision / recall, not accuracy alone
  - Stratify every split

Outputs:
    reports/model_findings.md
    reports/figures/model_*.png

Run:
    python src/modeling.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_PATH = BASE_DIR / "data" / "processed" / "credit_card_default_cleaned.csv"
FIGURE_DIR = BASE_DIR / "reports" / "figures"
FINDINGS_PATH = BASE_DIR / "reports" / "model_findings.md"

TARGET = "default payment next month"
PAY_COLUMNS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

# Primary selection metric under class imbalance (~22% positives).
PRIMARY_METRIC = "average_precision"

logger = logging.getLogger(__name__)
_findings: list[str] = []

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def record(line: str = "") -> None:
    """Print a line and keep it for the written findings file."""
    print(line)
    _findings.append(line)


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name)
    plt.close(fig)


def load_cleaned_data(path: Path = CLEANED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {path}. Run src/data_cleaning.py first."
        )
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the EDA-recommended feature set.

    None of these invent new information; they restate repayment behaviour in
    forms that survive the PAY-scale inconsistency documented in the EDA
    (status 1 is effectively PAY_1-only, so differencing months is unsafe).
    """
    out = df.copy()

    out["UTILIZATION"] = out["BILL_AMT1"] / out["LIMIT_BAL"].replace(0, np.nan)
    out["UTILIZATION"] = out["UTILIZATION"].fillna(0.0).clip(-1.0, 3.0)

    # Safe under the scale anomaly: counts any late month (status >= 1).
    out["MONTHS_DELINQUENT"] = (out[PAY_COLUMNS] >= 1).sum(axis=1)
    out["MAX_DELINQUENCY"] = out[PAY_COLUMNS].max(axis=1)

    # Share of the outstanding bill actually repaid (amounts, not status codes).
    bill_ref = out["BILL_AMT2"]
    out["PAID_SHARE"] = np.where(bill_ref > 0, out["PAY_AMT1"] / bill_ref, 0.0)
    out["PAID_SHARE"] = out["PAID_SHARE"].clip(0.0, 2.0)

    # Dormant accounts default above the base rate in the EDA.
    out["INACTIVE"] = (
        (out[BILL_COLUMNS] == 0).all(axis=1) | (out[PAY_AMT_COLUMNS] == 0).all(axis=1)
    ).astype(int)

    # Recency grouping: recent trouble outweighs old trouble.
    out["RECENT_LATE"] = (out[["PAY_1", "PAY_2"]] >= 1).sum(axis=1)
    out["DISTANT_LATE"] = (out[["PAY_5", "PAY_6"]] >= 1).sum(axis=1)

    # Age is U-shaped; band it rather than feeding a linear term.
    out["AGE_BAND"] = pd.cut(
        out["AGE"],
        bins=[20, 25, 30, 35, 40, 45, 50, 60, 80],
        labels=["20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-60", "60-80"],
        include_lowest=True,
    ).astype(str)
    out.loc[out["AGE_BAND"] == "nan", "AGE_BAND"] = "unknown"

    return out


# Feature sets deliberately escalate complexity so the comparison answers the
# EDA question: does anything beat the repayment-only baseline?
FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "baseline_eda": {
        "numeric": [
            "MONTHS_DELINQUENT",
            "PAY_1",
            "PAID_SHARE",
            "INACTIVE",
            "RECENT_LATE",
            "DISTANT_LATE",
        ],
        "categorical": ["AGE_BAND"],
    },
    "extended": {
        "numeric": [
            "MONTHS_DELINQUENT",
            "PAY_1",
            "PAY_2",
            "PAID_SHARE",
            "INACTIVE",
            "RECENT_LATE",
            "DISTANT_LATE",
            "MAX_DELINQUENCY",
            "LIMIT_BAL",
            "UTILIZATION",
            "PAY_AMT1",
        ],
        "categorical": ["AGE_BAND"],
    },
    # Full behavioural / financial block — deliberately omits SEX / EDUCATION /
    # MARRIAGE (weak signal in the EDA, protected-attribute exposure in lending).
    "kitchen_sink": {
        "numeric": (
            ["LIMIT_BAL", "AGE", "UTILIZATION", "MONTHS_DELINQUENT", "MAX_DELINQUENCY",
             "PAID_SHARE", "INACTIVE", "RECENT_LATE", "DISTANT_LATE"]
            + PAY_COLUMNS
            + BILL_COLUMNS
            + PAY_AMT_COLUMNS
        ),
        "categorical": [],
    },
}


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models() -> dict[str, object]:
    """Three estimators spanning linear, bagging, and boosting families."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=250,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(model_name: str, estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    # Trees do not need scaling; keep a no-op numeric path for a uniform API.
    if model_name == "logistic_regression":
        preprocessor = build_preprocessor(numeric, categorical)
    else:
        transformers = []
        if numeric:
            transformers.append(("num", "passthrough", numeric))
        if categorical:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical,
                )
            )
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def evaluate_holdout(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick the probability cutoff that maximises F1 on the training fold.

    Default 0.5 is rarely ideal under class imbalance with class_weight
    balancing; this keeps the operating point explicit rather than implicit.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def cross_val_scores(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )
    summary = {}
    for key in scoring:
        values = results[f"test_{key}"]
        summary[f"cv_{key}_mean"] = float(values.mean())
        summary[f"cv_{key}_std"] = float(values.std())
    return summary


def run_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train every (feature set x model) combination and return a leaderboard."""
    y = df[TARGET].astype(int)
    X_all = df.drop(columns=[TARGET, "ID"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    record("## 1. Setup")
    record()
    record(
        f"- Stratified hold-out: {1 - TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test "
        f"(random_state={RANDOM_STATE})."
    )
    record(
        f"- Train defaults: {int(y_train.mean() * 10000) / 100:.2f}% · "
        f"Test defaults: {int(y_test.mean() * 10000) / 100:.2f}%."
    )
    record(
        f"- Selection metric: **{PRIMARY_METRIC}** (PR-AUC), with ROC-AUC, "
        "precision, recall and F1 reported alongside."
    )
    record(
        "- All estimators use class_weight='balanced' (or balanced_subsample "
        "for the forest) so the minority class is not ignored."
    )
    record()

    record("## 2. Feature sets")
    record()
    for name, cols in FEATURE_SETS.items():
        n_num = len(cols["numeric"])
        n_cat = len(cols["categorical"])
        record(
            f"- **{name}**: {n_num} numeric"
            + (f", {n_cat} categorical ({', '.join(cols['categorical'])})" if n_cat else "")
            + f" — {', '.join(cols['numeric'][:6])}"
            + ("..." if n_num > 6 else "")
        )
    record()
    record(
        "- `baseline_eda` is the EDA-recommended starting point. `extended` "
        "adds limit/utilization/payment amount. `kitchen_sink` adds the full "
        "bill/payment panel (still excluding SEX/EDUCATION/MARRIAGE) as a "
        "complexity check against collinear financial columns."
    )
    record()

    models = build_models()
    rows = []
    artifacts: dict = {
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "fitted": {},
    }

    record("## 3. Cross-validated comparison")
    record()

    for feature_name, cols in FEATURE_SETS.items():
        numeric = cols["numeric"]
        categorical = cols["categorical"]
        feature_cols = numeric + categorical
        X_tr = X_train[feature_cols]
        X_te = X_test[feature_cols]

        for model_name, estimator in models.items():
            pipe = make_pipeline(model_name, estimator, numeric, categorical)
            cv_summary = cross_val_scores(pipe, X_tr, y_train)
            pipe.fit(X_tr, y_train)

            train_prob = pipe.predict_proba(X_tr)[:, 1]
            threshold = choose_threshold(y_train.to_numpy(), train_prob)
            test_prob = pipe.predict_proba(X_te)[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            holdout = evaluate_holdout(y_test.to_numpy(), test_prob, test_pred)

            key = f"{feature_name}__{model_name}"
            artifacts["fitted"][key] = {
                "pipeline": pipe,
                "threshold": threshold,
                "feature_set": feature_name,
                "model": model_name,
                "feature_cols": feature_cols,
                "numeric": numeric,
                "categorical": categorical,
                "y_prob": test_prob,
                "y_pred": test_pred,
            }

            rows.append(
                {
                    "feature_set": feature_name,
                    "model": model_name,
                    "threshold": round(threshold, 3),
                    **{k: round(v, 4) for k, v in cv_summary.items()},
                    **{f"test_{k}": round(v, 4) for k, v in holdout.items()},
                }
            )

    leaderboard = pd.DataFrame(rows).sort_values(
        ["cv_average_precision_mean", "test_pr_auc"],
        ascending=False,
    ).reset_index(drop=True)

    record("Cross-validated leaderboard (sorted by CV PR-AUC):")
    record("```")
    display_cols = [
        "feature_set",
        "model",
        "cv_average_precision_mean",
        "cv_average_precision_std",
        "cv_roc_auc_mean",
        "cv_f1_mean",
        "cv_recall_mean",
        "cv_precision_mean",
    ]
    record(leaderboard[display_cols].to_string(index=False))
    record("```")
    record()

    record("Hold-out test metrics (threshold tuned for F1 on the training set):")
    record("```")
    test_cols = [
        "feature_set",
        "model",
        "threshold",
        "test_pr_auc",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    record(leaderboard[test_cols].to_string(index=False))
    record("```")
    record()

    best_row = leaderboard.iloc[0]
    best_key = f"{best_row['feature_set']}__{best_row['model']}"
    artifacts["best_key"] = best_key
    artifacts["leaderboard"] = leaderboard

    record(
        f"**Selected model:** `{best_row['model']}` on feature set "
        f"`{best_row['feature_set']}` "
        f"(CV PR-AUC = {best_row['cv_average_precision_mean']:.4f} ± "
        f"{best_row['cv_average_precision_std']:.4f}; "
        f"test PR-AUC = {best_row['test_pr_auc']:.4f})."
    )
    record()
    return leaderboard, artifacts


def analyze_best_model(artifacts: dict) -> None:
    best_key = artifacts["best_key"]
    best = artifacts["fitted"][best_key]
    y_test = artifacts["y_test"].to_numpy()
    y_prob = best["y_prob"]
    y_pred = best["y_pred"]
    threshold = best["threshold"]

    record("## 4. Selected-model analysis")
    record()
    record(
        f"- Pipeline: **{best['model']}** · features: **{best['feature_set']}** · "
        f"decision threshold: **{threshold:.3f}** (F1-maximising on train)."
    )
    record()

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No default", "Default"],
        digits=3,
    )
    record("Classification report on the hold-out test set:")
    record("```")
    record(report.rstrip())
    record("```")
    record()

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    record("Confusion matrix (rows = actual, columns = predicted):")
    record("```")
    record(f"                 Pred No-default   Pred Default")
    record(f"Actual No-default      {tn:8d}        {fp:8d}")
    record(f"Actual Default         {fn:8d}        {tp:8d}")
    record("```")
    record()
    record(
        f"- Of {tp + fn:,} actual defaulters, the model catches {tp:,} "
        f"({tp / (tp + fn):.1%} recall) and misses {fn:,}."
    )
    record(
        f"- Of {tp + fp:,} predicted defaulters, {tp:,} are correct "
        f"({tp / (tp + fp) if (tp + fp) else 0:.1%} precision)."
    )
    record(
        "- For a collections triage use-case, missing a defaulter (FN) is "
        "usually costlier than a false alarm (FP), so the F1-tuned threshold "
        "is a compromise rather than a business-cost optimum. A bank would "
        "re-tune against its own loss/contact costs."
    )
    record()

    # Majority-class sanity check.
    majority_acc = float((y_test == 0).mean())
    model_acc = float((y_pred == y_test).mean())
    record(
        f"- Majority-class accuracy floor: {majority_acc:.1%}. Selected model "
        f"accuracy: {model_acc:.1%}. Accuracy alone would have preferred the "
        "majority classifier; PR-AUC / recall are the relevant numbers."
    )
    record()

    _plot_curves(y_test, y_prob, best)
    _plot_confusion(cm, best)
    _plot_leaderboard(artifacts["leaderboard"])
    _interpret_model(best, artifacts)


def _plot_curves(y_test: np.ndarray, y_prob: np.ndarray, best: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, color="#C44E52",
                 label=f"ROC-AUC = {roc_auc_score(y_test, y_prob):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(fontsize=8)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(
        recall,
        precision,
        color="#4C72B0",
        label=f"PR-AUC = {average_precision_score(y_test, y_prob):.3f}",
    )
    axes[1].axhline(y_test.mean(), color="#999999", linestyle="--", linewidth=1,
                    label=f"base rate = {y_test.mean():.3f}")
    axes[1].axvline(best["threshold"], color="#C44E52", linestyle=":", linewidth=1,
                    label=f"train F1 threshold ≈ {best['threshold']:.2f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–recall curve")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"Selected model: {best['model']} / {best['feature_set']}",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, "model_01_roc_pr_curves.png")


def _plot_confusion(cm: np.ndarray, best: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No default", "Default"],
        yticklabels=["No default", "Default"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix · {best['model']}")
    fig.tight_layout()
    save_figure(fig, "model_02_confusion_matrix.png")


def _plot_leaderboard(leaderboard: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = leaderboard.copy()
    plot_df["label"] = plot_df["feature_set"] + " / " + plot_df["model"]
    plot_df = plot_df.sort_values("cv_average_precision_mean")
    colours = {
        "baseline_eda": "#4C72B0",
        "extended": "#55A868",
        "kitchen_sink": "#C44E52",
    }
    ax.barh(
        plot_df["label"],
        plot_df["cv_average_precision_mean"],
        xerr=plot_df["cv_average_precision_std"],
        color=[colours[s] for s in plot_df["feature_set"]],
    )
    ax.set_xlabel("CV PR-AUC (mean ± std)")
    ax.set_title("Model comparison by feature set")
    fig.tight_layout()
    save_figure(fig, "model_03_leaderboard.png")


def _interpret_model(best: dict, artifacts: dict) -> None:
    record("## 5. Interpretation")
    record()
    pipe: Pipeline = best["pipeline"]
    model = pipe.named_steps["model"]
    feature_names = list(pipe.named_steps["preprocess"].get_feature_names_out())

    if hasattr(model, "coef_"):
        coefs = pd.Series(model.coef_.ravel(), index=feature_names)
        ranked = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
        record("Logistic regression coefficients (standardised features; top 12 by |coef|):")
        record("```")
        record(ranked.head(12).round(3).to_string())
        record("```")
        record()
        record(
            "- Positive coefficients raise default probability. Expect "
            "MONTHS_DELINQUENT / PAY_1 / RECENT_LATE on the positive side and "
            "PAID_SHARE on the negative side if the model agrees with the EDA."
        )
    elif hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        ranked = importances.sort_values(ascending=False)
        record("Random forest impurity-based feature importances (top 12):")
        record("```")
        record(ranked.head(12).round(4).to_string())
        record("```")
    else:
        # HistGradientBoosting: permutation importance on a sample for speed.
        from sklearn.inspection import permutation_importance

        X_te = artifacts["X_test"][best["feature_cols"]]
        y_te = artifacts["y_test"]
        result = permutation_importance(
            pipe,
            X_te,
            y_te,
            n_repeats=8,
            scoring="average_precision",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        ranked = pd.Series(result.importances_mean, index=best["feature_cols"])
        ranked = ranked.sort_values(ascending=False)
        record(
            "Permutation importance on the test set (drop in PR-AUC when shuffled):"
        )
        record("```")
        record(ranked.round(4).to_string())
        record("```")

    record()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    top = ranked.head(12).sort_values()
    ax.barh(top.index.astype(str), top.to_numpy(), color="#4C72B0")
    ax.set_xlabel("Importance / coefficient")
    ax.set_title(f"Driver features · {best['model']}")
    fig.tight_layout()
    save_figure(fig, "model_04_feature_importance.png")


def compare_against_baseline(leaderboard: pd.DataFrame) -> None:
    record("## 6. Did extra features earn their place?")
    record()

    def best_pr(feature_set: str) -> float:
        subset = leaderboard.loc[leaderboard["feature_set"] == feature_set]
        return float(subset["cv_average_precision_mean"].max())

    base = best_pr("baseline_eda")
    extended = best_pr("extended")
    kitchen = best_pr("kitchen_sink")
    record(
        f"- Best CV PR-AUC by feature set — baseline_eda: {base:.4f}, "
        f"extended: {extended:.4f}, kitchen_sink: {kitchen:.4f}."
    )
    record(
        f"- Extended vs baseline lift: {extended - base:+.4f}. "
        f"Kitchen-sink vs baseline lift: {kitchen - base:+.4f}."
    )
    if extended - base < 0.01 and kitchen - base < 0.01:
        record(
            "- The richer feature sets do not materially beat the EDA "
            "baseline. That supports keeping the simpler repayment-focused "
            "specification as the reported model."
        )
    elif kitchen > extended > base:
        record(
            "- Additional features improve PR-AUC. Report the best set, but "
            "note that the bill block is highly collinear (EDA VIFs > 10), so "
            "coefficients from a linear model on this set would be unstable "
            "even though tree ensembles handle the redundancy better."
        )
    else:
        record(
            "- Gains are modest. Prefer the simplest feature set within ~0.01 "
            "PR-AUC of the leader for interpretability and deployment risk."
        )
    record()


def write_limitations_and_recommendations(artifacts: dict) -> None:
    best = artifacts["fitted"][artifacts["best_key"]]
    record("## 7. Limitations")
    record()
    record(
        "1. **Preprocessing leakage from cleaning.** IQR winsorization in "
        "`data_cleaning.py` was fit on all 30,000 rows. The model split "
        "happens after that, so test rows have already influenced the "
        "capped bounds. A production pipeline should fit caps inside the "
        "training fold only."
    )
    record(
        "2. **Threshold is not cost-aware.** The cutoff maximises F1 on the "
        "training set. A lender should replace that with expected loss = "
        "FN_cost × miss_rate + FP_cost × false_alarm_rate."
    )
    record(
        "3. **Protected attributes.** SEX, EDUCATION and MARRIAGE are excluded "
        f"from every feature set, including the selected `{best['feature_set']}` "
        "model. The EDA found them statistically significant but practically "
        "negligible, and SEX in particular is a protected attribute in "
        "consumer lending."
    )
    record(
        "4. **PAY-scale inconsistency.** Features that count any late month "
        "or use amount-based PAID_SHARE are safe; a PAY_1 − PAY_6 trend was "
        "deliberately excluded per the EDA."
    )
    record(
        "5. **External validity.** The data is a single Taiwanese issuer in "
        "2005. Performance here is not a claim about current US / EU books."
    )
    record()

    record("## 8. Recommendations")
    record()
    record(
        f"1. **Ship the selected pipeline** (`{best['model']}` + "
        f"`{best['feature_set']}`) as the project model: it is the "
        "PR-AUC leader under stratified CV and aligns with the EDA story."
    )
    record(
        "2. **Operationalise a risk score, not only a hard label.** Rank "
        "accounts by predicted probability and apply policy cutoffs per "
        "segment (limit reduction, outreach, reserve)."
    )
    record(
        "3. **Refit preprocessing inside the train split** before any "
        "deployment claim, and monitor calibration on recent cohorts."
    )
    record(
        "4. **Keep demographics out of the decisioning model** unless a "
        "documented fairness review shows they improve outcomes without "
        "disparate impact — the EDA says they buy almost nothing."
    )
    record()


def write_summary(leaderboard: pd.DataFrame, artifacts: dict) -> None:
    best_row = leaderboard.iloc[0]
    record("## 9. Summary")
    record()
    record(
        f"- Final model: **{best_row['model']}** with **{best_row['feature_set']}** "
        f"features."
    )
    record(
        f"- Cross-validated PR-AUC: **{best_row['cv_average_precision_mean']:.4f}** "
        f"(± {best_row['cv_average_precision_std']:.4f}); "
        f"ROC-AUC: **{best_row['cv_roc_auc_mean']:.4f}**."
    )
    record(
        f"- Hold-out PR-AUC: **{best_row['test_pr_auc']:.4f}**, "
        f"precision **{best_row['test_precision']:.4f}**, "
        f"recall **{best_row['test_recall']:.4f}**, "
        f"F1 **{best_row['test_f1']:.4f}** "
        f"(threshold {best_row['threshold']:.3f})."
    )
    record(
        "- This beats a base-rate classifier on ranking quality (PR-AUC above "
        f"the ~{artifacts['y_test'].mean():.3f} positive rate) and recovers a "
        "meaningful share of defaulters that accuracy-only selection would miss."
    )
    record()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    raw_cleaned = load_cleaned_data()
    df = engineer_features(raw_cleaned)

    record("# Model Selection and Model Analysis")
    record()
    record(
        "UCI Default of Credit Card Clients — modeling stage. Built on the "
        "cleaned file from `data_cleaning.py` and the feature guidance in "
        "`eda_findings.md`."
    )
    record()
    record("Author: Umesh")
    record()

    leaderboard, artifacts = run_comparison(df)
    analyze_best_model(artifacts)
    compare_against_baseline(leaderboard)
    write_limitations_and_recommendations(artifacts)
    write_summary(leaderboard, artifacts)

    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_PATH.write_text("\n".join(_findings) + "\n", encoding="utf-8")
    print(f"\nFindings written to {FINDINGS_PATH.relative_to(BASE_DIR)}")
    print(f"Figures written to {FIGURE_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
