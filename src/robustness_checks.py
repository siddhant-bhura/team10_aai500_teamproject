"""Robustness checks on the selected model (read-only).

Purpose: measure an open limitation listed in the report, and add a
group-level error audit.

  1. Decision threshold picked from out-of-fold predictions (instead of the
     training-set predictions the model has already seen).
  2. False-positive and false-negative rates by sex and education.

Financial outliers are no longer winsorized in cleaning, so there is no
"fit caps on train only" variant to compare.

The selected model is histogram gradient boosting on the kitchen_sink inputs,
as chosen in modeling.py. This script reuses the cleaning and modeling code
and writes nothing to data/ or reports/. It only prints.

AI-tool disclosure: written with the help of Claude Code (Anthropic); the code
was run and its output checked. See README.md and the report (Section 1).

Run from the repository root, after the other scripts:
    python src/robustness_checks.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_cleaning as dc  # noqa: E402
import modeling as mdl  # noqa: E402

warnings.filterwarnings("ignore")

MODEL_NAME = "hist_gradient_boosting"
FEATURES = mdl.FEATURE_SETS["kitchen_sink"]["numeric"]
SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {
    1: "Graduate",
    2: "University",
    3: "High school",
    4: "Other",
}


def split(df):
    """Same stratified 80/20 split that modeling.py uses."""
    y = df[mdl.TARGET].astype(int)
    X = df.drop(columns=[mdl.TARGET, "ID"], errors="ignore")
    return train_test_split(
        X,
        y,
        test_size=mdl.TEST_SIZE,
        stratify=y,
        random_state=mdl.RANDOM_STATE,
    )


def new_pipeline():
    estimator = mdl.build_models()[MODEL_NAME]
    return mdl.make_pipeline(MODEL_NAME, estimator, FEATURES, [])


def run_variant(df, out_of_fold: bool, label: str) -> dict:
    """Fit, pick a threshold, score the hold-out set, print one line."""
    X_train, X_test, y_train, y_test = split(df)
    pipe = new_pipeline().fit(X_train[FEATURES], y_train)
    if out_of_fold:
        folds = StratifiedKFold(
            n_splits=mdl.N_FOLDS, shuffle=True, random_state=mdl.RANDOM_STATE
        )
        train_prob = cross_val_predict(
            new_pipeline(),
            X_train[FEATURES],
            y_train,
            cv=folds,
            method="predict_proba",
        )[:, 1]
    else:
        train_prob = pipe.predict_proba(X_train[FEATURES])[:, 1]
    threshold = mdl.choose_threshold(y_train.to_numpy(), train_prob)
    test_prob = pipe.predict_proba(X_test[FEATURES])[:, 1]
    pred = (test_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(
        f"{label}\n"
        f"  threshold={threshold:.3f}  "
        f"PR-AUC={average_precision_score(y_test, test_prob):.4f}  "
        f"ROC-AUC={roc_auc_score(y_test, test_prob):.4f}\n"
        f"  caught={tp}  missed={fn}  false_alarms={fp}  "
        f"precision={tp / (tp + fp):.3f}  recall={tp / (tp + fn):.3f}  "
        f"accuracy={(tp + tn) / len(y_test):.3f}"
    )
    return {"X_test": X_test, "y_test": y_test, "pred": pred}


def group_audit(result: dict) -> None:
    """Error rates by sex and education on the hold-out set."""
    X_test, y_test, pred = result["X_test"], result["y_test"], result["pred"]
    print(
        f"{'Group':<25}{'n':>6}{'default':>9}{'flagged':>9}"
        f"{'FPR':>7}{'FNR':>7}{'precision':>11}{'recall':>8}"
    )
    for column, labels in (
        ("SEX", SEX_LABELS),
        ("EDUCATION", EDUCATION_LABELS),
    ):
        codes = X_test[column].astype(int)
        for code, name in labels.items():
            mask = (codes == code).to_numpy()
            y = y_test.to_numpy()[mask]
            p = pred[mask]
            tp = int(((y == 1) & (p == 1)).sum())
            fn = int(((y == 1) & (p == 0)).sum())
            fp = int(((y == 0) & (p == 1)).sum())
            tn = int(((y == 0) & (p == 0)).sum())
            precision = tp / (tp + fp) if tp + fp else float("nan")
            print(
                f"{column + ' / ' + name:<25}{mask.sum():>6}"
                f"{y.mean():>9.3f}{p.mean():>9.3f}"
                f"{fp / (fp + tn):>7.3f}{fn / (fn + tp):>7.3f}"
                f"{precision:>11.3f}{tp / (tp + fn):>8.3f}"
            )


def main() -> None:
    report = dc.clean_dataframe()
    as_reported = mdl.engineer_features(report["df_cleaned"])

    print("== Threshold checks (hold-out set, 6,000 rows; outliers preserved) ==")
    base = run_variant(
        as_reported, False,
        "A. As in the report: threshold from train predictions",
    )
    run_variant(
        as_reported, True,
        "B. Threshold from out-of-fold predictions",
    )

    print("\n== Error rates by group, variant A (hold-out set) ==")
    print("FPR = share of non-defaulters flagged; FNR = share of "
          "defaulters missed.")
    group_audit(base)


if __name__ == "__main__":
    main()
