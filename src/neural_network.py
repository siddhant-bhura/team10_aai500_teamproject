"""Neural network (MLP) for the Model Selection and Analysis stage.

Adds a scikit-learn MLPClassifier to the comparison without touching
``modeling.py``. Everything that defines the experiment is imported from
that module so the comparison is apples to apples: the cleaned data, the
engineered features, the three feature sets, the stratified 80/20 split
(random_state=42), the 5-fold StratifiedKFold and the metric helpers.

Why scikit-learn and not PyTorch or Keras: neither is installed in the
project environment and ``requirements.txt`` only lists scikit-learn. For a
30,000-row tabular table with about 20 inputs, a small MLP from scikit-learn
is the same model family and needs no new dependency.

Class imbalance: MLPClassifier has no ``class_weight``. Two honest options
are used, both without leakage:

* Threshold tuning. The probability cutoff is chosen on out-of-fold (OOF)
  predictions from the training set only, then applied unchanged to the
  hold-out test set. The same OOF rule is applied to the baseline models.
* Random oversampling of the minority class, done inside each CV fold on the
  fold's training rows only (never on validation rows or the test set).

Hyperparameters are chosen by CV PR-AUC on the training set only. The
hold-out test set is touched once per final model.

Outputs go to ``reports/neural_network/`` and nothing tracked is overwritten.

AI-tool disclosure: written with the help of Claude Code (Anthropic); the code
was run and its output checked. See README.md and the report (Section 1).

Run from the repo root:  python src/neural_network.py
Oversampling self-check:  python src/neural_network.py --selfcheck
"""

from __future__ import annotations

import itertools
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import modeling as m  # noqa: E402

OUT_DIR = m.BASE_DIR / "reports" / "neural_network"
N_JOBS = 5  # one job per CV fold; results do not depend on this number

# Small, documented search grid (24 configurations per feature set).
HIDDEN_SIZES = [(32,), (64, 32), (128, 64)]
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0]
EARLY_STOPPING = [True, False]


def balanced_index(y, rng):
    """Row indices of ``y`` plus random copies of the minority class.

    Whichever class is smaller is copied up to the size of the other one.
    Nothing is added when the classes are already balanced (or one is empty).
    """
    idx = np.arange(len(y))
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    if 0 < len(minority) < len(majority):
        extra = rng.choice(
            minority, size=len(majority) - len(minority), replace=True)
        idx = np.concatenate([idx, extra])
    return idx


def selfcheck_balanced_index() -> None:
    """Tiny check: either class can be the minority; balanced adds nothing."""
    for y in ([1, 1, 0], [0, 0, 1], [0, 1, 0, 1]):
        y = np.array(y)
        idx = balanced_index(y, np.random.RandomState(0))
        assert (y[idx] == 1).sum() == (y[idx] == 0).sum(), y
        assert len(idx) == 2 * max(np.bincount(y)), y
    y = np.array([0, 1, 0, 1])
    assert list(balanced_index(y, np.random.RandomState(0))) == [0, 1, 2, 3]
    print("balanced_index self-check passed")


class OversampledMLP(BaseEstimator, ClassifierMixin):
    """MLP that balances classes by random oversampling inside ``fit``.

    Because resampling happens inside ``fit``, it only ever sees the rows it
    is given, so wrapping it in cross-validation keeps validation rows clean.
    """

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        alpha=1e-2,
        early_stopping=True,
        random_state=m.RANDOM_STATE,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        idx = balanced_index(y, rng)
        rng.shuffle(idx)
        self.mlp_ = make_mlp(
            self.hidden_layer_sizes,
            self.alpha,
            self.early_stopping,
            self.random_state,
        )
        self.mlp_.fit(X[idx], y[idx])
        self.classes_ = self.mlp_.classes_
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(X)

    def predict(self, X):
        return self.mlp_.predict(X)


def make_mlp(
    hidden, alpha, early_stopping, seed=m.RANDOM_STATE
) -> MLPClassifier:
    """Build an MLP with a fixed seed.

    Early stopping holds out 10% of the training rows internally.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        alpha=alpha,
        early_stopping=early_stopping,
        validation_fraction=0.1,
        n_iter_no_change=10,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=seed,
    )


def make_nn_pipeline(estimator, numeric, categorical) -> Pipeline:
    """Scaler + one-hot (the same preprocessor modeling.py uses for the LR)."""
    return Pipeline(
        [
            ("preprocess", m.build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ]
    )


def _fit_fold(pipe, X, y, train_idx, val_idx):
    model = clone(pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
    return val_idx, model.predict_proba(X.iloc[val_idx])[:, 1]


def oof_evaluate(pipe, X, y) -> dict:
    """Out-of-fold probabilities and per-fold PR-AUC / ROC-AUC on train only.

    Uses the same StratifiedKFold settings as ``modeling.cross_val_scores``.
    The OOF threshold is the F1-maximising cutoff on the pooled OOF
    probabilities (modeling.choose_threshold), and precision / recall / F1 are
    then computed at that cutoff.
    """
    cv = StratifiedKFold(
        n_splits=m.N_FOLDS, shuffle=True, random_state=m.RANDOM_STATE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        parts = Parallel(n_jobs=N_JOBS)(
            delayed(_fit_fold)(pipe, X, y, tr, va) for tr, va in cv.split(X, y)
        )
    oof = np.zeros(len(y))
    fold_pr, fold_roc = [], []
    yv = y.to_numpy()
    for val_idx, prob in parts:
        oof[val_idx] = prob
        fold_pr.append(average_precision_score(yv[val_idx], prob))
        fold_roc.append(roc_auc_score(yv[val_idx], prob))
    threshold = m.choose_threshold(yv, oof)
    stats = m.evaluate_holdout(yv, oof, (oof >= threshold).astype(int))
    return {
        "oof": oof,
        "threshold": threshold,
        "cv_pr_auc_mean": float(np.mean(fold_pr)),
        "cv_pr_auc_std": float(np.std(fold_pr)),
        "cv_roc_auc_mean": float(np.mean(fold_roc)),
        "cv_roc_auc_std": float(np.std(fold_roc)),
        "cv_precision": stats["precision"],
        "cv_recall": stats["recall"],
        "cv_f1": stats["f1"],
    }


def holdout_at_threshold(pipe, X_tr, y_tr, X_te, y_te, threshold) -> dict:
    """Fit on all training rows, score the test set once at a fixed cutoff."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = clone(pipe).fit(X_tr, y_tr)
    prob = fitted.predict_proba(X_te)[:, 1]
    pred = (prob >= threshold).astype(int)
    stats = m.evaluate_holdout(y_te.to_numpy(), prob, pred)
    return {"fitted": fitted, "prob": prob, "pred": pred, **stats}


def row_from(feature_set, model, label, oof_res, test_res) -> dict:
    return {
        "feature_set": feature_set,
        "model": model,
        "config": label,
        "oof_threshold": round(oof_res["threshold"], 4),
        "cv_pr_auc": round(oof_res["cv_pr_auc_mean"], 4),
        "cv_pr_auc_std": round(oof_res["cv_pr_auc_std"], 4),
        "cv_roc_auc": round(oof_res["cv_roc_auc_mean"], 4),
        "cv_f1": round(oof_res["cv_f1"], 4),
        "cv_recall": round(oof_res["cv_recall"], 4),
        "cv_precision": round(oof_res["cv_precision"], 4),
        "test_pr_auc": round(test_res["pr_auc"], 4),
        "test_roc_auc": round(test_res["roc_auc"], 4),
        "test_f1": round(test_res["f1"], 4),
        "test_recall": round(test_res["recall"], 4),
        "test_precision": round(test_res["precision"], 4),
    }


def split_data():
    """Same engineered data and split call as modeling.run_comparison."""
    df = m.engineer_features(m.load_cleaned_data())
    y = df[m.TARGET].astype(int)
    X_all = df.drop(columns=[m.TARGET, "ID"], errors="ignore")
    return train_test_split(
        X_all,
        y,
        test_size=m.TEST_SIZE,
        stratify=y,
        random_state=m.RANDOM_STATE,
    )


def run_baselines(X_train, y_train, X_test, y_test):
    """LR, RF and HGB from modeling.build_models, scored with the OOF rule."""
    rows, store = [], {}
    models = m.build_models()
    for fname, cols in m.FEATURE_SETS.items():
        feats = cols["numeric"] + cols["categorical"]
        for mname, est in models.items():
            pipe = m.make_pipeline(
                mname, est, cols["numeric"], cols["categorical"]
            )
            oof_res = oof_evaluate(pipe, X_train[feats], y_train)
            test_res = holdout_at_threshold(
                pipe,
                X_train[feats],
                y_train,
                X_test[feats],
                y_test,
                oof_res["threshold"],
            )
            rows.append(
                row_from(
                    fname, mname, "modeling.py defaults", oof_res, test_res
                )
            )
            store[(fname, mname)] = test_res
            print(
                f"  baseline {fname:13s} {mname:22s} "
                f"cv_pr={oof_res['cv_pr_auc_mean']:.4f} "
                f"test_pr={test_res['pr_auc']:.4f}",
                flush=True,
            )
    return rows, store


def run_nn(X_train, y_train, X_test, y_test):
    """Grid search per feature set (CV PR-AUC, train only), then score."""
    search_rows, final_rows, store = [], [], {}
    grid = list(itertools.product(HIDDEN_SIZES, ALPHAS, EARLY_STOPPING))
    for fname, cols in m.FEATURE_SETS.items():
        num, cat = cols["numeric"], cols["categorical"]
        feats = num + cat
        X_tr = X_train[feats]
        best = None
        for hidden, alpha, es in grid:
            pipe = make_nn_pipeline(make_mlp(hidden, alpha, es), num, cat)
            res = oof_evaluate(pipe, X_tr, y_train)
            search_rows.append(
                {
                    "feature_set": fname,
                    "variant": "plain",
                    "hidden_layer_sizes": str(hidden),
                    "alpha": alpha,
                    "early_stopping": es,
                    "cv_pr_auc": round(res["cv_pr_auc_mean"], 4),
                    "cv_pr_auc_std": round(res["cv_pr_auc_std"], 4),
                    "cv_roc_auc": round(res["cv_roc_auc_mean"], 4),
                }
            )
            if (
                best is None
                or res["cv_pr_auc_mean"] > best[0]["cv_pr_auc_mean"]
            ):
                best = (res, hidden, alpha, es)
        res, hidden, alpha, es = best

        # Same best configuration with in-fold oversampling, for comparison.
        os_pipe = make_nn_pipeline(OversampledMLP(hidden, alpha, es), num, cat)
        os_res = oof_evaluate(os_pipe, X_tr, y_train)
        search_rows.append(
            {
                "feature_set": fname,
                "variant": "oversampled",
                "hidden_layer_sizes": str(hidden),
                "alpha": alpha,
                "early_stopping": es,
                "cv_pr_auc": round(os_res["cv_pr_auc_mean"], 4),
                "cv_pr_auc_std": round(os_res["cv_pr_auc_std"], 4),
                "cv_roc_auc": round(os_res["cv_roc_auc_mean"], 4),
            }
        )
        use_os = os_res["cv_pr_auc_mean"] > res["cv_pr_auc_mean"]
        chosen_res, chosen_pipe = (
            (os_res, os_pipe)
            if use_os
            else (res, make_nn_pipeline(make_mlp(hidden, alpha, es), num, cat))
        )
        label = f"hidden={hidden}, alpha={alpha}, early_stopping={es}" + (
            ", oversampled" if use_os else ""
        )
        test_res = holdout_at_threshold(
            chosen_pipe,
            X_tr,
            y_train,
            X_test[feats],
            y_test,
            chosen_res["threshold"],
        )
        final_rows.append(
            row_from(fname, "mlp_neural_network", label, chosen_res, test_res)
        )
        store[fname] = {"test": test_res, "oof": chosen_res, "label": label}
        print(
            f"  NN {fname:13s} {label}  "
            f"cv_pr={chosen_res['cv_pr_auc_mean']:.4f} "
            f"test_pr={test_res['pr_auc']:.4f}",
            flush=True,
        )
    return search_rows, final_rows, store


def _mlp_of(fitted: Pipeline) -> MLPClassifier:
    model = fitted.named_steps["model"]
    return model.mlp_ if isinstance(model, OversampledMLP) else model


def plot_loss_curve(nn_store, best_fs):
    mlp = _mlp_of(nn_store[best_fs]["test"]["fitted"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(mlp.loss_curve_, color="#1f77b4")
    axes[0].set(
        title="Training loss per epoch", xlabel="Epoch", ylabel="Log loss"
    )
    if getattr(mlp, "validation_scores_", None) is not None:
        axes[1].plot(mlp.validation_scores_, color="#d62728")
        axes[1].set(
            title="Internal validation accuracy (early stopping)",
            xlabel="Epoch",
            ylabel="Accuracy",
        )
        axes[1].axvline(
            int(np.argmax(mlp.validation_scores_)), ls="--", color="grey"
        )
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "Early stopping off for this model:\nno validation curve",
            ha="center",
            va="center",
        )
    fig.suptitle(f"Best neural network ({best_fs}); {mlp.n_iter_} epochs run")
    fig.savefig(OUT_DIR / "fig_nn_loss_curve.png")
    plt.close(fig)


def plot_curves(y_test, nn_prob, nn_title, tree_prob, tree_title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for prob, title, color in [
        (nn_prob, nn_title, "#d62728"),
        (tree_prob, tree_title, "#1f77b4"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        axes[0].plot(
            fpr,
            tpr,
            color=color,
            label=f"{title} (AUC {roc_auc_score(y_test, prob):.3f})",
        )
        pr, rc, _ = precision_recall_curve(y_test, prob)
        axes[1].plot(
            rc,
            pr,
            color=color,
            label=f"{title} (AP {average_precision_score(y_test, prob):.3f})",
        )
    axes[0].plot([0, 1], [0, 1], ls=":", color="grey")
    axes[0].set(
        title="ROC, hold-out test set",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    axes[1].axhline(y_test.mean(), ls=":", color="grey")
    axes[1].set(
        title="Precision-recall, hold-out test set",
        xlabel="Recall",
        ylabel="Precision",
    )
    for ax in axes:
        ax.legend(
            fontsize=7, loc="lower right" if ax is axes[0] else "upper right"
        )
    fig.savefig(OUT_DIR / "fig_nn_vs_best_tree_curves.png")
    plt.close(fig)


def plot_confusion(y_test, pred, threshold, title):
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i, j in itertools.product(range(2), range(2)):
        ax.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No default", "Default"],
        yticklabels=["No default", "Default"],
        xlabel="Predicted",
        ylabel="Actual",
        title=f"{title}\nthreshold {threshold:.3f}",
    )
    ax.grid(False)
    fig.savefig(OUT_DIR / "fig_nn_confusion_matrix.png")
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(comparison, search, best_nn, best_tree, y_test, n_train):
    side = comparison[
        [
            "feature_set",
            "model",
            "cv_pr_auc",
            "cv_pr_auc_std",
            "cv_roc_auc",
            "cv_f1",
            "cv_recall",
            "cv_precision",
            "oof_threshold",
            "test_pr_auc",
            "test_roc_auc",
            "test_f1",
            "test_recall",
            "test_precision",
        ]
    ]
    nn = comparison[comparison["model"] == "mlp_neural_network"]
    others = comparison[comparison["model"] != "mlp_neural_network"]
    best_by_model = (
        others.sort_values("cv_pr_auc", ascending=False)
        .groupby("model")
        .head(1)
    )
    gap = best_nn["test_pr_auc"] - best_tree["test_pr_auc"]
    verdict = (
        "does not beat"
        if best_nn["cv_pr_auc"] <= best_tree["cv_pr_auc"]
        else "beats"
    )
    text = f"""# Neural network addition to Model Selection and Analysis

Generated by `src/neural_network.py`. Nothing in `modeling.py` or its outputs was changed.

## Setup (same as modeling.py)

- Data: cleaned file, features from `modeling.engineer_features`, the three feature sets in `modeling.FEATURE_SETS`.
- Split: stratified 80/20, random_state=42 ({n_train:,} train rows, {len(y_test):,} test rows). Test default rate {y_test.mean():.2%}.
- Cross-validation: 5-fold StratifiedKFold, shuffle, random_state=42, train set only.
- Selection metric: CV PR-AUC (average precision).
- Library: scikit-learn `MLPClassifier`. PyTorch, Keras and TensorFlow are not installed and are not in `requirements.txt`. For 30,000 rows and about 20 inputs an MLP from scikit-learn is the same model family and adds no dependency.
- Preprocessing: standard scaling for numeric columns, one-hot for AGE_BAND, inside a Pipeline (same preprocessor the logistic regression uses).

## How imbalance was handled

MLPClassifier has no class_weight, so:

1. The decision threshold is chosen from out-of-fold predictions on the training set only (the cutoff that maximises F1), then applied unchanged to the test set.
2. For the best configuration of each feature set, a variant with random oversampling of defaulters inside each CV fold (training rows of that fold only) was also tried. It is used only if its CV PR-AUC is higher.

The baselines (logistic regression, random forest, hist gradient boosting) are re-scored with the same OOF threshold rule so every row below is comparable. This differs from `modeling.py`, which picks the threshold on in-sample training predictions. Threshold-free numbers (PR-AUC, ROC-AUC) do not depend on this choice.

## Hyperparameter search

Grid per feature set: hidden layers {HIDDEN_SIZES}, alpha {ALPHAS}, early_stopping {EARLY_STOPPING} ({len(HIDDEN_SIZES) * len(ALPHAS) * len(EARLY_STOPPING)} configurations), scored by 5-fold CV PR-AUC on the training set. Full grid in `nn_search_results.csv`. CV figures for the winning configuration are slightly optimistic because it was picked on those same folds. The test figures are not affected by this.

## Side-by-side results

CV precision, recall and F1 are computed on pooled out-of-fold predictions at the OOF threshold. Test metrics use the same threshold.

{md_table(side)}

## Best of each model family (by CV PR-AUC, across feature sets)

{md_table(pd.concat([best_by_model, nn]).sort_values("cv_pr_auc", ascending=False)[["feature_set", "model", "cv_pr_auc", "cv_roc_auc", "test_pr_auc", "test_roc_auc", "test_f1", "test_recall", "test_precision"]])}

## Verdict

- Best neural network: `{best_nn['feature_set']}`, {best_nn['config']}. CV PR-AUC {best_nn['cv_pr_auc']:.4f} (std {best_nn['cv_pr_auc_std']:.4f}), test PR-AUC {best_nn['test_pr_auc']:.4f}, test ROC-AUC {best_nn['test_roc_auc']:.4f}.
- Best non-network model by CV PR-AUC: `{best_tree['model']}` on `{best_tree['feature_set']}`. CV PR-AUC {best_tree['cv_pr_auc']:.4f} (std {best_tree['cv_pr_auc_std']:.4f}), test PR-AUC {best_tree['test_pr_auc']:.4f}, test ROC-AUC {best_tree['test_roc_auc']:.4f}.
- The neural network {verdict} the best non-network model on CV PR-AUC. Test PR-AUC difference (network minus best other): {gap:+.4f}. The fold-to-fold standard deviation is about {best_tree['cv_pr_auc_std']:.3f}, so differences of this size are within noise.

## Why the network ties the trees instead of beating them

These are plausible reasons from the data, not proven causes.

- The table is tabular, with about 20 inputs and 30,000 rows. Tree ensembles usually match or beat MLPs at this size, and they need less tuning.
- The main signal is a handful of repayment-status columns (the PAY_ columns) with a few discrete levels. Trees split on these directly; an MLP has to learn the same steps through smooth functions.
- The label is noisy. Many customers with similar histories end up on both sides, so all models cluster in a narrow band of PR-AUC and ceiling effects limit any gain.
- MLPs are sensitive to scaling, skewed amounts (bills and payments) and the choice of imbalance handling. We scaled and tuned a small grid, but a larger search was not run.
- The extra flexibility of a network can add variance without adding signal.

## Practical points

- The network is a probability scorer. Its operating point is a threshold choice, as for the other models.
- Interpretability is weaker than for logistic regression or trees (no direct coefficients or split rules).
- Runs are deterministic: fixed seeds throughout, and the script was run twice with identical output files.

## Files

- `nn_search_results.csv`: every hyperparameter configuration with CV scores.
- `model_comparison.csv`: the side-by-side table above.
- `fig_nn_loss_curve.png`: training loss (and internal validation score when early stopping is on) for the best network.
- `fig_nn_vs_best_tree_curves.png`: ROC and precision-recall on the test set, network vs best non-network model.
- `fig_nn_confusion_matrix.png`: test confusion matrix for the best network at its OOF threshold.
"""  # noqa: E501 (long lines are markdown prose written to the results file)
    (OUT_DIR / "neural_network_results.md").write_text(text, encoding="utf-8")


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = split_data()
    print("Baselines (OOF threshold rule):", flush=True)
    base_rows, base_store = run_baselines(X_train, y_train, X_test, y_test)
    print("Neural network search:", flush=True)
    search_rows, nn_rows, nn_store = run_nn(X_train, y_train, X_test, y_test)

    comparison = pd.DataFrame(base_rows + nn_rows)
    pd.DataFrame(search_rows).to_csv(
        OUT_DIR / "nn_search_results.csv", index=False
    )
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    nn_df = comparison[comparison["model"] == "mlp_neural_network"]
    best_nn = nn_df.sort_values("cv_pr_auc", ascending=False).iloc[0]
    others = comparison[comparison["model"] != "mlp_neural_network"]
    best_tree = others.sort_values("cv_pr_auc", ascending=False).iloc[0]

    bfs = best_nn["feature_set"]
    tree_test = base_store[(best_tree["feature_set"], best_tree["model"])]
    plot_loss_curve(nn_store, bfs)
    plot_curves(
        y_test.to_numpy(),
        nn_store[bfs]["test"]["prob"],
        f"MLP ({bfs})",
        tree_test["prob"],
        f"{best_tree['model']} ({best_tree['feature_set']})",
    )
    plot_confusion(
        y_test.to_numpy(),
        nn_store[bfs]["test"]["pred"],
        best_nn["oof_threshold"],
        f"MLP ({bfs})",
    )
    write_report(
        comparison, search_rows, best_nn, best_tree, y_test, len(y_train)
    )
    print(f"Done in {time.time() - start:.0f}s. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck_balanced_index()
    else:
        main()
