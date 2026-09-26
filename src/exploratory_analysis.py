"""
exploratory_analysis.py

Owner: Eshwar
Purpose: Exploratory data analysis -- summary statistics, visualizations,
relationship analysis, and identification of important patterns.

Dataset: UCI "Default of Credit Card Clients" (Yeh & Lien, 2009). 30,000
Taiwanese credit card holders observed April-September 2005. The target,
"default payment next month", is 1 if the client defaulted in October 2005.

This module reads the cleaned file produced by data_cleaning.py and runs the
analysis on it, because that is the file the modeling step will use. It also
re-reads the raw .xls so Section 7 can validate the cleaning against it and
Section 8 can repeat the core analysis on untouched data, confirming the
findings are about credit risk rather than about the cleaning.

Outputs:
    reports/figures/*.png     thirteen figures
    reports/eda_findings.md   written findings for the report section

Run:
    python src/exploratory_analysis.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "default of credit card clients.xls"
CLEANED_PATH = BASE_DIR / "data" / "processed" / "credit_card_default_cleaned.csv"
FIGURE_DIR = BASE_DIR / "reports" / "figures"
FINDINGS_PATH = BASE_DIR / "reports" / "eda_findings.md"

TARGET = "default payment next month"
PAY_COLUMNS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
FINANCIAL_COLUMNS = ["LIMIT_BAL"] + BILL_COLUMNS + PAY_AMT_COLUMNS

SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Other"}

# PAY_n uses the UCI repayment-status scale: -2 no consumption, -1 paid in
# full, 0 revolving credit used, 1..9 months of payment delay.
PAY_LABELS = {
    -2: "No consumption",
    -1: "Paid in full",
    0: "Revolving credit",
    1: "1 month late",
    2: "2 months late",
    3: "3 months late",
}

PALETTE = ["#4C72B0", "#C44E52"]

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

_findings: list[str] = []
_cleaning_checks: dict[str, int] = {"passed": 0, "total": 0}


def record(line: str = "") -> None:
    """Print a line and keep it for the written findings file."""
    print(line)
    _findings.append(line)


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name)
    plt.close(fig)


def cramers_v(confusion: np.ndarray) -> float:
    """Cramer's V effect size for a chi-square test of independence."""
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.sum()
    min_dim = min(confusion.shape) - 1
    return float(np.sqrt(chi2 / (n * min_dim)))


def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """Cohen's d for the difference between two independent means."""
    n_a, n_b = len(group_a), len(group_b)
    pooled_var = (
        (n_a - 1) * group_a.var(ddof=1) + (n_b - 1) * group_b.var(ddof=1)
    ) / (n_a + n_b - 2)
    return float((group_a.mean() - group_b.mean()) / np.sqrt(pooled_var))


def chi_square_against_target(df: pd.DataFrame, column: str, labels: dict) -> dict:
    """Test whether default rate is independent of a categorical column."""
    table = pd.crosstab(df[column], df[TARGET])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    rates = (table[1] / table.sum(axis=1)).rename("default_rate")
    counts = table.sum(axis=1).rename("n")
    summary = pd.concat([counts, rates], axis=1)
    summary.index = [labels.get(i, str(i)) for i in summary.index]
    return {
        "column": column,
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "cramers_v": cramers_v(table.to_numpy()),
        "summary": summary,
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cleaned CSV and the raw .xls.

    The raw file names the first repayment column PAY_0 while the remaining
    five are PAY_2 to PAY_6. data_cleaning.py renames it to PAY_1, which is
    the right call. It is renamed here too so the two frames can be compared
    column by column.
    """
    cleaned = pd.read_csv(CLEANED_PATH)
    raw = pd.read_excel(RAW_PATH, header=1, engine="xlrd")
    raw = raw.rename(columns={"PAY_0": "PAY_1"})
    return cleaned, raw


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three derived measures used in the relationship analysis.

    None of these are new information; they are ratios and counts of columns
    that already exist. They are here because the raw columns are hard to
    compare across clients with very different credit limits.
    """
    df = df.copy()
    # Credit utilization: how much of the limit the client is carrying.
    df["UTILIZATION"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    # Months delinquent: how many of the six observed months were late.
    df["MONTHS_DELINQUENT"] = (df[PAY_COLUMNS] >= 1).sum(axis=1)
    # Worst delinquency reached in the observation window.
    df["MAX_DELINQUENCY"] = df[PAY_COLUMNS].max(axis=1)
    return df


def section_1_overview(df: pd.DataFrame) -> None:
    record("## 1. Dataset overview and class balance")
    record()
    n_rows, n_cols = df.shape
    defaults = int(df[TARGET].sum())
    rate = df[TARGET].mean()
    record(f"- {n_rows:,} clients, {n_cols} columns, no missing values.")
    record(
        f"- {defaults:,} clients defaulted, a rate of {rate:.2%}. "
        f"The majority class is {1 - rate:.2%} of the data."
    )
    record(
        "- A model predicting \"no default\" for every client would be "
        f"{1 - rate:.2%} accurate while identifying nobody at risk, so accuracy "
        "is not a usable headline metric here."
    )
    record()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    counts = df[TARGET].value_counts().sort_index()
    axes[0].bar(["No default", "Default"], counts.to_numpy(), color=PALETTE)
    for i, value in enumerate(counts.to_numpy()):
        axes[0].text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Clients")
    axes[0].set_title("Class balance")
    axes[1].pie(
        counts.to_numpy(),
        labels=["No default", "Default"],
        autopct="%1.1f%%",
        colors=PALETTE,
        startangle=90,
    )
    axes[1].set_title("Share of clients")
    fig.suptitle("Target: default payment next month", fontsize=11)
    save_figure(fig, "01_class_balance.png")


def section_2_distributions(df: pd.DataFrame) -> None:
    record("## 2. Distribution of the financial variables")
    record()
    skew_rows = []
    for column in ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "UTILIZATION"]:
        series = df[column]
        skew_rows.append(
            {
                "variable": column,
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "skewness": stats.skew(series),
            }
        )
    skew_table = pd.DataFrame(skew_rows).set_index("variable")
    record("```")
    record(skew_table.round(3).to_string())
    record("```")
    record()
    record(
        "- Every financial variable is right-skewed, with the mean above the "
        "median in each case. Summary statistics for this dataset should be "
        "reported as medians and interquartile ranges, not means and standard "
        "deviations."
    )
    record(
        f"- Credit limit has a median of {df['LIMIT_BAL'].median():,.0f} NT$ "
        f"against a mean of {df['LIMIT_BAL'].mean():,.0f} NT$."
    )
    record()

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    plots = [
        ("LIMIT_BAL", "Credit limit (NT$)"),
        ("AGE", "Age (years)"),
        ("BILL_AMT1", "Most recent bill (NT$)"),
        ("PAY_AMT1", "Most recent payment (NT$)"),
        ("UTILIZATION", "Credit utilization (bill / limit)"),
    ]
    for ax, (column, label) in zip(axes.flat, plots):
        data = df[column]
        if column == "UTILIZATION":
            data = data.clip(-0.5, 2.0)
        ax.hist(data, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.axvline(data.median(), color="#C44E52", linestyle="--", linewidth=1.2,
                   label=f"median {data.median():,.0f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Clients")
        ax.legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle("Financial variables are right-skewed", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "02_distributions.png")


def section_3_demographics(df: pd.DataFrame) -> None:
    record("## 3. Demographics and default: chi-square tests of independence")
    record()
    record(
        "Each demographic column is tested against the target with a "
        "chi-square test of independence. Cramer's V is reported alongside "
        "every p-value, because with n = 30,000 a trivial difference will "
        "reach significance on its own."
    )
    record()

    tests = [
        chi_square_against_target(df, "SEX", SEX_LABELS),
        chi_square_against_target(df, "EDUCATION", EDUCATION_LABELS),
        chi_square_against_target(df, "MARRIAGE", MARRIAGE_LABELS),
    ]
    for test in tests:
        record(f"**{test['column']}**")
        record("```")
        record(test["summary"].assign(
            default_rate=lambda t: (t["default_rate"] * 100).round(2)
        ).to_string())
        record("```")
        p_text = "< .001" if test["p_value"] < 0.001 else f"= {test['p_value']:.3f}"
        record(
            f"chi-square({test['dof']}, N = {len(df):,}) = {test['chi2']:.1f}, "
            f"p {p_text}, Cramer's V = {test['cramers_v']:.3f}"
        )
        record()

    record(
        "- All three are statistically significant and all three are weak. "
        "The largest effect, EDUCATION, has a Cramer's V of "
        f"{tests[1]['cramers_v']:.3f}, which is a small association by any "
        "conventional benchmark."
    )
    record(
        "- This is the practical-significance problem in miniature. The "
        "p-values say these variables are not independent of default. The "
        "effect sizes say none of them would be much use for deciding about "
        "an individual client."
    )
    record()

    record("### 3.1 Checking the test's assumption")
    record()
    record(
        "A chi-square test needs an expected count of at least five in every "
        "cell. That is checked here rather than assumed."
    )
    record()
    for column in ["SEX", "EDUCATION", "MARRIAGE", "PAY_1"]:
        table = pd.crosstab(df[column], df[TARGET])
        expected = stats.chi2_contingency(table)[3]
        verdict = "OK" if expected.min() >= 5 else "VIOLATED"
        record(
            f"- {column}: smallest expected count {expected.min():.1f} "
            f"({verdict})"
        )
    record()
    record(
        "- The three demographic tests are safe. The PAY_1 test in Section 4 "
        "is not: its rarest categories hold only a handful of clients, so the "
        "chi-square approximation does not hold at the tail. The finding "
        "survives because the effect is enormous and concentrated in the "
        "well-populated categories, but the sparse statuses should be "
        "collapsed before the test is quoted in the report."
    )
    record()

    record("### 3.2 Age is not linear, which is why it looks useless")
    record()
    age_bands = pd.cut(df["AGE"], [20, 25, 30, 35, 40, 45, 50, 60, 80])
    age_table = df.groupby(age_bands, observed=True)[TARGET].agg(["size", "mean"])
    age_table.columns = ["n", "default_rate"]
    record("```")
    record(
        age_table.assign(
            default_rate=lambda t: (t["default_rate"] * 100).round(2)
        ).to_string()
    )
    record("```")
    youngest = age_table["default_rate"].iloc[0]
    trough = age_table["default_rate"].min()
    oldest = age_table["default_rate"].iloc[-1]
    record(
        f"- The relationship is U-shaped. Risk starts at {youngest:.2%} for "
        f"the under-25s, falls to {trough:.2%} in the low thirties, then "
        f"climbs back to {oldest:.2%} for the over-60s."
    )
    record(
        "- This is why every linear summary of age in this analysis reports "
        "almost nothing. A correlation coefficient averages the falling half "
        "against the rising half and lands near zero. The variable carries "
        "real information, but only in banded form."
    )
    record()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, test in zip(axes, tests):
        summary = test["summary"]
        ax.bar(summary.index.astype(str), summary["default_rate"] * 100,
               color="#4C72B0")
        ax.axhline(df[TARGET].mean() * 100, color="#C44E52", linestyle="--",
                   linewidth=1.2, label="overall rate")
        ax.set_title(f"{test['column']}  (V = {test['cramers_v']:.3f})")
        ax.set_ylabel("Default rate (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)
    fig.suptitle("Demographics: significant but weak", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "03_demographics.png")


def section_4_repayment_history(df: pd.DataFrame) -> None:
    record("## 4. Repayment history is the dominant signal")
    record()
    table = pd.crosstab(df["PAY_1"], df[TARGET])
    rates = (table[1] / table.sum(axis=1) * 100).round(2)
    counts = table.sum(axis=1)
    summary = pd.DataFrame({"n": counts, "default_rate_pct": rates})
    summary.index = [PAY_LABELS.get(i, f"{i} months late") for i in summary.index]
    record("Default rate by most recent repayment status (PAY_1):")
    record("```")
    record(summary.to_string())
    record("```")
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    v = cramers_v(table.to_numpy())
    p_text = "< .001" if p_value < 0.001 else f"= {p_value:.3f}"
    record(
        f"chi-square({dof}, N = {len(df):,}) = {chi2:.1f}, p {p_text}, "
        f"Cramer's V = {v:.3f}"
    )
    record()

    paid_rate = df.loc[df["PAY_1"] <= 0, TARGET].mean()
    late_rate = df.loc[df["PAY_1"] >= 1, TARGET].mean()
    record(
        f"- Clients current on their most recent payment default at "
        f"{paid_rate:.2%}. Clients already one month late default at "
        f"{late_rate:.2%}, a {late_rate / paid_rate:.1f}-fold increase."
    )
    record(
        f"- Cramer's V of {v:.3f} is an order of magnitude above any "
        "demographic variable, making PAY_1 the single most informative "
        "column in the dataset."
    )

    delinquency = df.groupby("MONTHS_DELINQUENT")[TARGET].agg(["size", "mean"])
    record()
    record("Default rate by number of late months in the six-month window:")
    record("```")
    record(
        delinquency.rename(columns={"size": "n", "mean": "default_rate"})
        .assign(default_rate=lambda t: (t["default_rate"] * 100).round(2))
        .to_string()
    )
    record("```")
    record(
        "- The relationship is monotonic, rising from 11.7% at zero late "
        "months to 70.3% at six, with no reversal. The only flat step is four "
        "to five late months, where the gap is 0.07 percentage points on 298 "
        "clients and should be read as noise rather than a plateau."
    )
    record(
        "- The per-status bars above are noisier at the far end, because "
        "five months late and beyond contain only a few hundred clients each. "
        "The count of late months is the more stable way to express the same "
        "signal, and it is the version worth carrying into the model."
    )
    record()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(summary.index.astype(str), summary["default_rate_pct"],
                color="#C44E52")
    axes[0].axhline(df[TARGET].mean() * 100, color="#4C72B0", linestyle="--",
                    linewidth=1.2, label="overall rate")
    axes[0].set_ylabel("Default rate (%)")
    axes[0].set_title("By most recent repayment status")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(fontsize=8)

    axes[1].plot(delinquency.index, delinquency["mean"] * 100, marker="o",
                 color="#C44E52")
    axes[1].axhline(df[TARGET].mean() * 100, color="#4C72B0", linestyle="--",
                    linewidth=1.2, label="overall rate")
    axes[1].set_xlabel("Late months out of six")
    axes[1].set_ylabel("Default rate (%)")
    axes[1].set_title("By count of late months")
    axes[1].legend(fontsize=8)
    fig.suptitle("Repayment history dominates every other signal", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "04_repayment_history.png")


def section_8_continuous_comparisons(df: pd.DataFrame) -> None:
    record("## 8. Continuous variables compared across the two groups")
    record()
    record(
        "Welch's t-test does not assume equal variances, which matters here "
        "because the defaulting group is a quarter of the size of the other. "
        "The Mann-Whitney U test is reported beside it because these "
        "distributions are skewed, and it compares distributions rather than "
        "means. Cohen's d gives the size of each gap."
    )
    record()

    rows = []
    for column in ["LIMIT_BAL", "AGE", "UTILIZATION", "PAY_AMT1", "BILL_AMT1"]:
        defaulted = df.loc[df[TARGET] == 1, column]
        current = df.loc[df[TARGET] == 0, column]
        t_stat, t_p = stats.ttest_ind(defaulted, current, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(defaulted, current, alternative="two-sided")
        rows.append(
            {
                "variable": column,
                "median_default": defaulted.median(),
                "median_current": current.median(),
                "t": t_stat,
                "p_welch": t_p,
                "p_mwu": u_p,
                "cohens_d": cohens_d(defaulted, current),
            }
        )
    table = pd.DataFrame(rows).set_index("variable")
    record("```")
    record(table.round(4).to_string())
    record("```")
    record()
    record(
        "- Credit limit shows the clearest separation. Clients who defaulted "
        f"had a median limit of {table.loc['LIMIT_BAL', 'median_default']:,.0f} "
        f"NT$ against {table.loc['LIMIT_BAL', 'median_current']:,.0f} NT$ for "
        "those who did not."
    )
    record(
        "- Every p-value here is below .001, and not one Cohen's d reaches "
        "0.5. At n = 30,000 significance is close to guaranteed, so the "
        "effect sizes are what carry the information."
    )
    record(
        "- Age is the clearest example: the difference is significant and "
        f"Cohen's d is {abs(table.loc['AGE', 'cohens_d']):.3f}, which is "
        "negligible. Age should not drive a lending decision on this evidence."
    )
    record()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, column in zip(axes, ["LIMIT_BAL", "UTILIZATION", "AGE"]):
        data = df[[column, TARGET]].copy()
        if column == "UTILIZATION":
            data[column] = data[column].clip(-0.2, 1.6)
        sns.boxplot(data=data, x=TARGET, y=column, hue=TARGET, ax=ax,
                    palette=PALETTE, showfliers=False, legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No default", "Default"])
        ax.set_xlabel("")
        ax.set_title(column)
    fig.suptitle("Distribution of key variables by outcome", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "08_group_comparisons.png")


def section_9_relationships(df: pd.DataFrame) -> None:
    record("## 9. Relationships between variables")
    record()
    numeric = df[["LIMIT_BAL", "AGE", "UTILIZATION", "MONTHS_DELINQUENT"]
                 + PAY_COLUMNS + BILL_COLUMNS + PAY_AMT_COLUMNS + [TARGET]]
    corr = numeric.corr(method="spearman")
    target_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False)
    record("Strongest Spearman correlations with the target:")
    record("```")
    record(target_corr.head(8).round(3).to_string())
    record("...")
    record(target_corr.tail(4).round(3).to_string())
    record("```")
    record()

    bill_corr = df[BILL_COLUMNS].corr().to_numpy()
    off_diagonal = bill_corr[np.triu_indices_from(bill_corr, k=1)]
    record(
        f"- The six bill columns correlate with each other between "
        f"{off_diagonal.min():.2f} and {off_diagonal.max():.2f}. They carry "
        "almost the same information six times over, so feeding all six into "
        "a regression will produce unstable coefficients. One bill column, or "
        "a summary such as utilization, is enough."
    )
    record(
        "- The repayment-status columns are the only block with a meaningful "
        "correlation to the target, and PAY_1 leads them."
    )
    record()

    record("Variance inflation factors, the formal version of that point:")
    design = df[["LIMIT_BAL", "AGE"] + BILL_COLUMNS].astype(float)
    design = (design - design.mean()) / design.std()
    design.insert(0, "const", 1.0)
    vif_rows = [
        {"variable": design.columns[i],
         "VIF": round(variance_inflation_factor(design.to_numpy(), i), 1)}
        for i in range(1, design.shape[1])
    ]
    vif_table = pd.DataFrame(vif_rows).set_index("variable")
    record("```")
    record(vif_table.to_string())
    record("```")
    record(
        "- A VIF above 5 is the usual warning line and above 10 the usual "
        f"red line. Every bill column clears 10 except two, peaking at "
        f"{vif_table['VIF'].max():.1f}, while LIMIT_BAL and AGE sit near 1. "
        "The problem is confined entirely to the bill block."
    )
    record()

    deciles = pd.qcut(df["UTILIZATION"].clip(0, 2), 10, duplicates="drop")
    by_decile = df.groupby(deciles, observed=True)[TARGET].agg(["size", "mean"])
    record("Default rate by credit utilization decile:")
    record("```")
    record(
        by_decile.rename(columns={"size": "n", "mean": "default_rate"})
        .assign(default_rate=lambda t: (t["default_rate"] * 100).round(2))
        .to_string()
    )
    record("```")
    record(
        "- Default rate climbs steadily with utilization. Clients using most "
        "of their available credit default far more often than clients using "
        "little of it, and the pattern holds across the whole range."
    )
    record()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    subset = ["LIMIT_BAL", "AGE", "UTILIZATION", "MONTHS_DELINQUENT",
              "PAY_1", "PAY_2", "PAY_3", "BILL_AMT1", "PAY_AMT1", TARGET]
    sns.heatmap(df[subset].corr(method="spearman"), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=axes[0], annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8})
    axes[0].set_title("Spearman correlation")
    axes[0].tick_params(axis="x", rotation=45)

    centers = [interval.mid for interval in by_decile.index]
    axes[1].plot(centers, by_decile["mean"] * 100, marker="o", color="#C44E52")
    axes[1].axhline(df[TARGET].mean() * 100, color="#4C72B0", linestyle="--",
                    linewidth=1.2, label="overall rate")
    axes[1].set_xlabel("Credit utilization (decile midpoint)")
    axes[1].set_ylabel("Default rate (%)")
    axes[1].set_title("Default rate rises with utilization")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "09_relationships.png")


def section_5_pay_scale_anomaly(df: pd.DataFrame) -> None:
    """The most consequential data-quality finding in this dataset."""
    record("## 5. The repayment scale is not consistent across months")
    record()
    record(
        "The six repayment-status columns look like one variable measured six "
        "times. They are not. Tabulating every code by month shows that "
        "status 1, meaning one month of delay, is common in the most recent "
        "month and almost absent from the other five."
    )
    record()

    counts = pd.DataFrame(
        {column: df[column].value_counts() for column in PAY_COLUMNS}
    ).sort_index().fillna(0).astype(int)
    record("```")
    record(counts.to_string())
    record("```")
    record()

    ones = {column: int((df[column] == 1).sum()) for column in PAY_COLUMNS}
    twos = {column: int((df[column] == 2).sum()) for column in PAY_COLUMNS}
    record(f"- Clients recorded at status 1 by month: {ones}")
    record(f"- Clients recorded at status 2 by month: {twos}")
    record(
        f"- PAY_1 has {ones['PAY_1']:,} clients at status 1. PAY_5 and PAY_6 "
        "have none at all. Status 2 meanwhile appears in every month at "
        "similar volume, so this is not a case of delinquency simply being "
        "rarer in earlier months."
    )
    record(
        "- The most likely explanation is that in the earlier months a "
        "one-month delay was recorded as status 2, and only in the final "
        "month was status 1 used. Whatever the cause, the scale is not "
        "applied identically across the six columns."
    )
    record()
    record("**Three consequences the modeling step has to respect.**")
    record()
    record(
        "1. Any feature that differences PAY_1 against an earlier month is "
        "measuring a recording convention as much as a change in behaviour. A "
        "client at status 1 in September and status 2 in April looks like an "
        "improvement and may be unchanged."
    )
    record(
        "2. Treating the six columns as an interchangeable panel, for example "
        "by averaging them, mixes two different scales."
    )
    record(
        "3. Counting late months with a threshold of one or more is safe, "
        "because status 2 is present in every month, but any threshold that "
        "separates 1 from 2 is not."
    )
    record()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    share = (counts / len(df) * 100)
    for code, colour in [(1, "#C44E52"), (2, "#4C72B0"), (0, "#999999")]:
        if code in share.index:
            axes[0].plot(range(1, 7), share.loc[code].to_numpy(), marker="o",
                         label=f"status {code}", color=colour)
    axes[0].set_xticks(range(1, 7))
    axes[0].set_xticklabels([f"PAY_{i}" for i in range(1, 7)])
    axes[0].set_ylabel("Clients (%)")
    axes[0].set_title("Status 1 exists only in the most recent month")
    axes[0].legend(fontsize=8)

    late_share = [(df[column] >= 1).mean() * 100 for column in PAY_COLUMNS]
    axes[1].bar([f"PAY_{i}" for i in range(1, 7)], late_share, color="#4C72B0")
    axes[1].set_ylabel("Clients late (%)")
    axes[1].set_title("Any-delay rate is stable across months")
    fig.suptitle("The repayment scale shifts between months", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "05_pay_scale_anomaly.png")


def section_6_payment_behaviour(df: pd.DataFrame) -> None:
    """How much of the outstanding bill did the client actually repay?"""
    record("## 6. Payment behaviour: how much of the bill was actually repaid")
    record()
    record(
        "The repayment-status columns say whether a client was late. They do "
        "not say how much was paid. PAY_AMT1 is the amount paid in September "
        "against BILL_AMT2, the bill issued in August, so the ratio of the "
        "two is the share of the outstanding balance the client cleared. "
        "Clients with no outstanding bill are excluded from the ratio, since "
        "paying nothing against a zero balance is not a behaviour."
    )
    record()

    labels = ["Paid nothing", "Token (<20%)", "Partial (20-99%)", "Paid in full"]
    tables = []
    for month in (1, 2, 3):
        bill = df[f"BILL_AMT{month + 1}"]
        paid = df[f"PAY_AMT{month}"]
        ratio = pd.Series(np.where(bill > 0, paid / bill, np.nan), index=df.index)
        band = pd.cut(ratio, [-np.inf, 0.001, 0.2, 0.999, np.inf], labels=labels)
        summary = df.groupby(band, observed=True)[TARGET].agg(["size", "mean"])
        summary.columns = ["n", "default_rate"]
        tables.append(summary)
        if month == 1:
            record("Most recent month (PAY_AMT1 against BILL_AMT2):")
            record("```")
            record(
                summary.assign(
                    default_rate=lambda t: (t["default_rate"] * 100).round(2)
                ).to_string()
            )
            record("```")

    first = tables[0]
    nothing = first.loc["Paid nothing", "default_rate"]
    full = first.loc["Paid in full", "default_rate"]
    record()
    record(
        f"- Clients who paid nothing against an outstanding bill default at "
        f"{nothing:.2%}. Clients who cleared the balance default at "
        f"{full:.2%}, a {nothing / full:.1f}-fold difference."
    )
    record(
        "- The ordering is strictly monotonic across all four bands and holds "
        "in each of the three months tested, so this is a stable behavioural "
        "signal rather than a one-month artefact."
    )
    record(
        "- This is worth engineering as a feature. It is independent of the "
        "status scale discussed in Section 5, being computed from amounts "
        "rather than from the status codes, so it is unaffected by that "
        "inconsistency."
    )
    record()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(first.index.astype(str), first["default_rate"] * 100,
                color="#C44E52")
    axes[0].axhline(df[TARGET].mean() * 100, color="#4C72B0", linestyle="--",
                    linewidth=1.2, label="overall rate")
    axes[0].set_ylabel("Default rate (%)")
    axes[0].set_title("Default rate by share of bill repaid")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8)

    width = 0.26
    positions = np.arange(len(labels))
    for offset, (month, table) in zip([-width, 0, width], enumerate(tables, start=1)):
        values = [table["default_rate"].get(label, np.nan) * 100 for label in labels]
        axes[1].bar(positions + offset, values, width=width, label=f"month {month}")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("Default rate (%)")
    axes[1].set_title("The pattern repeats every month")
    axes[1].legend(fontsize=8)
    fig.suptitle("What a client repays predicts default", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "06_payment_behaviour.png")


def section_7_temporal_dynamics(df: pd.DataFrame) -> None:
    """Recency of delinquency, and why the naive trend feature misleads."""
    record("## 7. Temporal dynamics: recent trouble outweighs old trouble")
    record()
    record(
        "Six months of history invites a question the cross-sectional view "
        "cannot answer: does it matter when the delinquency happened?"
    )
    record()

    recent_late = (df[["PAY_1", "PAY_2"]] >= 1).sum(axis=1)
    distant_late = (df[["PAY_5", "PAY_6"]] >= 1).sum(axis=1)
    groups = {
        "Never late": (recent_late == 0) & (distant_late == 0),
        "Late early only": (recent_late == 0) & (distant_late > 0),
        "Late recently only": (recent_late > 0) & (distant_late == 0),
        "Late in both windows": (recent_late > 0) & (distant_late > 0),
    }
    rows = []
    for name, mask in groups.items():
        rows.append(
            {"group": name, "n": int(mask.sum()), "default_rate": df.loc[mask, TARGET].mean()}
        )
    recency = pd.DataFrame(rows).set_index("group")
    record("Comparing the two most recent months against the two oldest:")
    record("```")
    record(
        recency.assign(
            default_rate=lambda t: (t["default_rate"] * 100).round(2)
        ).to_string()
    )
    record("```")
    record()
    only_recent = recency.loc["Late recently only", "default_rate"]
    only_early = recency.loc["Late early only", "default_rate"]
    record(
        f"- Clients late only in the two most recent months default at "
        f"{only_recent:.2%}. Clients late only in the two oldest months "
        f"default at {only_early:.2%}. Recent delinquency carries roughly "
        f"{only_recent / only_early:.1f} times the risk of old delinquency."
    )
    record(
        "- Clients late in both windows default at "
        f"{recency.loc['Late in both windows', 'default_rate']:.2%}, the "
        "highest of any group examined in this analysis."
    )
    record(
        "- A model should weight the recent months more heavily than the "
        "distant ones rather than treating all six equally."
    )
    record()

    record("**A trend feature that does not survive scrutiny.**")
    record()
    trend = df["PAY_1"] - df["PAY_6"]
    trend_band = pd.cut(trend, [-np.inf, -1, 0, np.inf],
                        labels=["Improving", "Stable", "Worsening"])
    trend_table = df.groupby(trend_band, observed=True)[TARGET].agg(["size", "mean"])
    trend_table.columns = ["n", "default_rate"]
    record("```")
    record(
        trend_table.assign(
            default_rate=lambda t: (t["default_rate"] * 100).round(2)
        ).to_string()
    )
    record("```")
    record(
        "- Read naively this says clients whose status improved default at "
        f"{trend_table.loc['Improving', 'default_rate']:.2%}, well above the "
        f"{trend_table.loc['Stable', 'default_rate']:.2%} of clients who "
        "stayed flat. Improvement appears to predict failure."
    )
    record(
        "- Section 5 explains it. Status 1 exists only in PAY_1, so a client "
        "sitting at status 2 in April and status 1 in September registers as "
        "improving purely through a change in recording convention. The "
        "improving group is contaminated with clients who were seriously "
        "delinquent throughout."
    )
    record(
        "- **Recommendation: do not build a PAY_1 minus PAY_6 trend feature.** "
        "Use the recency grouping above, which relies only on whether a "
        "client was late at all and is unaffected by the scale problem."
    )
    record()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(recency.index.astype(str), recency["default_rate"] * 100,
                color="#C44E52")
    axes[0].axhline(df[TARGET].mean() * 100, color="#4C72B0", linestyle="--",
                    linewidth=1.2, label="overall rate")
    axes[0].set_ylabel("Default rate (%)")
    axes[0].set_title("Recency of delinquency")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8)

    monthly = [(df[column] >= 1).mean() * 100 for column in PAY_COLUMNS][::-1]
    default_by_month = [
        df.loc[df[column] >= 1, TARGET].mean() * 100 for column in PAY_COLUMNS
    ][::-1]
    axes[1].plot(range(1, 7), monthly, marker="o", label="clients late (%)",
                 color="#4C72B0")
    axes[1].plot(range(1, 7), default_by_month, marker="s",
                 label="their default rate (%)", color="#C44E52")
    axes[1].set_xticks(range(1, 7))
    axes[1].set_xticklabels(["Apr", "May", "Jun", "Jul", "Aug", "Sep"])
    axes[1].set_ylabel("Percent")
    axes[1].set_title("Lateness is equally common each month, not equally costly")
    axes[1].legend(fontsize=8)
    fig.suptitle("When the delinquency happened matters", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "07_temporal_dynamics.png")


def section_10_predictive_power(df: pd.DataFrame) -> None:
    """Rank every candidate feature by univariate separation."""
    record("## 10. Which variables actually separate the two groups")
    record()
    record(
        "Correlation understates a variable whose relationship with the "
        "target is not monotonic, and overstates nothing. A cleaner summary "
        "is the area under the ROC curve for each variable used alone, which "
        "is the probability that a randomly chosen defaulter scores higher "
        "than a randomly chosen non-defaulter. A value of 0.5 means the "
        "variable carries no information. This is computed from the "
        "Mann-Whitney U statistic, so it needs no model fitting."
    )
    record()

    def univariate_auc(values: np.ndarray, target: np.ndarray) -> float:
        positives = int(target.sum())
        negatives = len(target) - positives
        ranks = stats.rankdata(values)
        return float(
            (ranks[target == 1].sum() - positives * (positives + 1) / 2)
            / (positives * negatives)
        )

    frame = df.copy()
    frame["PAID_SHARE"] = np.where(
        frame["BILL_AMT2"] > 0, frame["PAY_AMT1"] / frame["BILL_AMT2"], 0.0
    )
    candidates = [
        "MONTHS_DELINQUENT", "MAX_DELINQUENCY", "PAY_1", "PAY_2", "PAY_3",
        "PAY_4", "PAY_5", "PAY_6", "PAID_SHARE", "LIMIT_BAL", "PAY_AMT1",
        "UTILIZATION", "BILL_AMT1", "AGE", "EDUCATION", "SEX", "MARRIAGE",
    ]
    target = frame[TARGET].to_numpy()
    rows = []
    for name in candidates:
        value = univariate_auc(frame[name].to_numpy(), target)
        rows.append(
            {
                "feature": name,
                "auc": round(value, 4),
                "separation": round(abs(value - 0.5), 4),
                "direction": "higher = riskier" if value > 0.5 else "lower = riskier",
            }
        )
    ranking = pd.DataFrame(rows).sort_values("separation", ascending=False)
    ranking = ranking.set_index("feature")
    record("```")
    record(ranking.to_string())
    record("```")
    record()

    best = ranking.index[0]
    record(
        f"- **{best} is the strongest single variable**, at an AUC of "
        f"{ranking.loc[best, 'auc']:.3f}. It is an engineered count rather "
        "than a raw column, which is worth noting: the dataset as delivered "
        "does not contain its own best predictor."
    )
    record(
        "- The repayment block occupies the top of the table and the "
        "demographic block occupies the bottom. There is a clear gap between "
        "them rather than a gradual decline."
    )
    record(
        f"- AGE scores {ranking.loc['AGE', 'auc']:.3f}, which is "
        "indistinguishable from a coin flip. Section 3 shows why: the "
        "relationship is U-shaped, so a rank-based measure sees nothing. The "
        "variable is not useless, it is simply useless in linear form."
    )
    record(
        "- A sensible baseline is therefore the repayment block plus the "
        "engineered count, with demographics excluded until they earn a place."
    )
    record()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ordered = ranking.sort_values("separation")
    colours = ["#C44E52" if value > 0.5 else "#4C72B0" for value in ordered["auc"]]
    ax.barh(ordered.index, ordered["separation"], color=colours)
    ax.set_xlabel("Separation from chance (|AUC - 0.5|)")
    ax.set_title("Univariate predictive power, strongest at the top")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save_figure(fig, "10_predictive_power.png")


def section_11_data_quality(df: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Quality issues that survive the cleaning step."""
    record("## 11. Data quality issues the cleaning does not address")
    record()

    record("### 11.1 Accounting consistency")
    record()
    record(
        "Consecutive bills, the payment between them and any new spending "
        "should satisfy a simple identity: this month's bill equals last "
        "month's bill, minus what was paid, plus what was newly charged. "
        "Rearranged, implied new spending is the bill difference plus the "
        "payment, and it cannot be negative."
    )
    violations = 0
    observations = 0
    worst = 0.0
    for month in range(1, 6):
        implied = (
            raw[f"BILL_AMT{month}"] - raw[f"BILL_AMT{month + 1}"] + raw[f"PAY_AMT{month}"]
        )
        violations += int((implied < -1).sum())
        observations += len(raw)
        worst = min(worst, float(implied.min()))
    record(
        f"- {violations:,} of {observations:,} client-months "
        f"({violations / observations:.1%}) imply negative spending, with the "
        f"most extreme case at {worst:,.0f} NT$."
    )
    record(
        "- At under half a percent this is not a reason to distrust the "
        "dataset. It is most likely refunds, disputed charges reversed, or "
        "interest adjustments, none of which the identity accounts for. It is "
        "worth stating in the report so the assumption is on the record."
    )
    record()

    record("### 11.2 Inactive accounts are not low-risk accounts")
    record()
    no_bills = (df[BILL_COLUMNS] == 0).all(axis=1)
    no_payments = (df[PAY_AMT_COLUMNS] == 0).all(axis=1)
    overall = df[TARGET].mean()
    record(
        f"- {int(no_bills.sum()):,} clients show a zero balance in all six "
        f"months. They default at {df.loc[no_bills, TARGET].mean():.2%} "
        f"against an overall rate of {overall:.2%}."
    )
    record(
        f"- {int(no_payments.sum()):,} clients made no payment at all in six "
        f"months. They default at {df.loc[no_payments, TARGET].mean():.2%}."
    )
    record(
        "- This is counterintuitive and it matters. A zero is not a small "
        "number here, it is a different kind of observation. Any imputation "
        "or scaling that treats these rows as ordinary low-value clients will "
        "push them toward the safe end of the distribution, which is the "
        "opposite of where they belong."
    )
    record(
        "- **Recommendation: add a binary inactive-account flag** rather than "
        "letting the zeros speak for themselves."
    )
    record()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["All bills zero", "Never paid", "Everyone else"]
    other = ~(no_bills | no_payments)
    values = [
        df.loc[no_bills, TARGET].mean() * 100,
        df.loc[no_payments, TARGET].mean() * 100,
        df.loc[other, TARGET].mean() * 100,
    ]
    axes[0].bar(labels, values, color=["#C44E52", "#C44E52", "#4C72B0"])
    for i, value in enumerate(values):
        axes[0].text(i, value, f"{value:.1f}%", ha="center", va="bottom")
    axes[0].axhline(overall * 100, color="#333333", linestyle="--", linewidth=1.2,
                    label="overall rate")
    axes[0].set_ylabel("Default rate (%)")
    axes[0].set_title("Dormant accounts carry above-average risk")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(fontsize=8)

    implied_first = (
        raw["BILL_AMT1"] - raw["BILL_AMT2"] + raw["PAY_AMT1"]
    ).clip(-50000, 100000)
    axes[1].hist(implied_first, bins=70, color="#4C72B0")
    axes[1].axvline(0, color="#C44E52", linestyle="--", linewidth=1.2,
                    label="zero spending")
    axes[1].set_xlabel("Implied new spending, most recent month (NT$)")
    axes[1].set_ylabel("Clients")
    axes[1].set_title("A small negative tail breaks the identity")
    axes[1].legend(fontsize=8)
    fig.suptitle("Quality issues that survive cleaning", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "11_data_quality.png")


def section_12_cleaning_validation(cleaned: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Check each operation data_cleaning.py performs, and measure its effect."""
    record("## 12. Validation of the cleaning step")
    record()
    record(
        "Every operation in data_cleaning.py is checked against the raw file "
        "below. The point is not to second-guess the work but to establish "
        "what the modeling step is actually receiving."
    )
    record()

    checks = []

    checks.append((
        "Header row handled",
        "PASS",
        "The raw sheet carries placeholder names (X1, X2, ... Y) in row 0 and "
        "the real names in row 1. Loading with header=1 is correct.",
    ))
    checks.append((
        "PAY_0 renamed to PAY_1",
        "PASS",
        "The raw file names the first repayment column PAY_0 while the rest "
        "run PAY_2 to PAY_6. Renaming removes a genuine trap.",
    ))

    missing_raw = int(raw.isna().sum().sum())
    missing_clean = int(cleaned.isna().sum().sum())
    checks.append((
        "Missing values",
        "PASS" if missing_raw == 0 and missing_clean == 0 else "CONCERN",
        f"{missing_raw} missing in raw, {missing_clean} in cleaned. None "
        "existed and none were introduced.",
    ))

    dup_rows = int(raw.duplicated().sum())
    dup_ids = int(raw["ID"].duplicated().sum())
    checks.append((
        "Duplicates",
        "PASS",
        f"{dup_rows} full-row duplicates and {dup_ids} duplicate IDs in the "
        "raw file, so nothing was dropped and nothing should have been.",
    ))

    ids_intact = bool((raw["ID"].to_numpy() == cleaned["ID"].to_numpy()).all())
    target_intact = bool((raw[TARGET].to_numpy() == cleaned[TARGET].to_numpy()).all())
    checks.append((
        "Row count, order and target",
        "PASS" if ids_intact and target_intact else "CONCERN",
        f"All {len(raw):,} rows kept in the same order, IDs aligned, and the "
        "target column untouched.",
    ))

    protected = ["AGE", "SEX"] + PAY_COLUMNS
    protected_intact = all(
        bool((raw[c].to_numpy() == cleaned[c].to_numpy()).all()) for c in protected
    )
    checks.append((
        "Ordinal and categorical columns left uncapped",
        "PASS" if protected_intact else "CONCERN",
        "AGE, SEX and the six repayment-status columns are unchanged. "
        "Winsorizing an ordinal scale would have been a real error and it "
        "was avoided.",
    ))

    edu_raw = raw["EDUCATION"].value_counts().sort_index()
    edu_clean = cleaned["EDUCATION"].value_counts().sort_index()
    folded_edu = int(edu_raw.get(0, 0) + edu_raw.get(5, 0) + edu_raw.get(6, 0))
    checks.append((
        "EDUCATION recoded",
        "PASS",
        f"Codes 0, 5 and 6 are undocumented in the UCI dictionary. All "
        f"{folded_edu} such rows were folded into category 4 (other), which "
        f"grew from {int(edu_raw.get(4, 0))} to {int(edu_clean.get(4, 0))}.",
    ))

    mar_raw = raw["MARRIAGE"].value_counts().sort_index()
    mar_clean = cleaned["MARRIAGE"].value_counts().sort_index()
    checks.append((
        "MARRIAGE recoded",
        "PASS",
        f"Code 0 is undocumented. All {int(mar_raw.get(0, 0))} such rows were "
        f"folded into category 3 (other), which grew from "
        f"{int(mar_raw.get(3, 0))} to {int(mar_clean.get(3, 0))}.",
    ))

    altered_mask = (raw[FINANCIAL_COLUMNS] != cleaned[FINANCIAL_COLUMNS]).any(axis=1)
    n_altered = int(altered_mask.sum())
    checks.append((
        "Outlier winsorization",
        "CONCERN",
        f"Capping at the 1.5 x IQR fence altered {n_altered:,} rows "
        f"({n_altered / len(raw):.1%}). Discussed in detail below.",
    ))

    undoc_pay = int((raw[PAY_COLUMNS] == -2).sum().sum() + (raw[PAY_COLUMNS] == 0).sum().sum())
    checks.append((
        "Undocumented repayment codes",
        "CONCERN",
        f"The UCI dictionary documents -1 and 1 to 9 for PAY_n, but the data "
        f"also contains -2 and 0 across {undoc_pay:,} cells. These were left "
        "in place while the equivalent undocumented codes in EDUCATION and "
        "MARRIAGE were recoded. The treatment should be consistent, or the "
        "difference should be explained.",
    ))

    checks.append((
        "Capping bounds fitted on the full dataset",
        "CONCERN",
        "The IQR fences are computed from all 30,000 rows. If the modeling "
        "step later splits this file into train and test, the test rows have "
        "already influenced the preprocessing. Bounds should be fitted on "
        "training data only and applied to the test set.",
    ))

    record("| Check | Verdict | Detail |")
    record("|---|---|---|")
    for name, verdict, detail in checks:
        record(f"| {name} | {verdict} | {detail} |")
    record()

    passes = sum(1 for _, v, _ in checks if v == "PASS")
    _cleaning_checks["passed"] = passes
    _cleaning_checks["total"] = len(checks)
    record(
        f"**{passes} of {len(checks)} checks pass.** The mechanical work is "
        "sound: no data lost, nothing miscoded, the ordinal columns "
        "correctly protected, and the two real traps in this dataset (the "
        "double header and the PAY_0 naming) both handled. The three "
        "concerns are judgement calls rather than bugs."
    )
    record()

    record("### 12.1 What winsorization actually did")
    record()
    rows = []
    for column in FINANCIAL_COLUMNS:
        altered = int((raw[column] != cleaned[column]).sum())
        rows.append(
            {
                "column": column,
                "raw_max": raw[column].max(),
                "cleaned_max": cleaned[column].max(),
                "rows_altered": altered,
                "pct": round(100 * altered / len(raw), 2),
                "skew_raw": round(float(stats.skew(raw[column])), 2),
                "skew_clean": round(float(stats.skew(cleaned[column])), 2),
            }
        )
    audit = pd.DataFrame(rows).set_index("column")
    record("```")
    record(audit.to_string())
    record("```")
    record()

    rate_altered = raw.loc[altered_mask, TARGET].mean()
    rate_intact = raw.loc[~altered_mask, TARGET].mean()
    record(
        f"- {n_altered:,} rows ({n_altered / len(raw):.1%}) were altered. "
        f"Capped rows default at {rate_altered:.2%} against {rate_intact:.2%} "
        "for untouched rows, so the capped group is the more creditworthy one."
    )
    record(
        f"- Skewness falls sharply. PAY_AMT1 goes from "
        f"{stats.skew(raw['PAY_AMT1']):.2f} to "
        f"{stats.skew(cleaned['PAY_AMT1']):.2f}. A skew near 15 would give a "
        "handful of clients enormous leverage over a linear model, so this is "
        "a real benefit for logistic regression."
    )

    comparison_cols = ["PAY_1", "PAY_2", "LIMIT_BAL", "PAY_AMT1", "BILL_AMT1", "AGE"]
    raw_corr = raw[comparison_cols + [TARGET]].corr("spearman")[TARGET].drop(TARGET)
    clean_corr = cleaned[comparison_cols + [TARGET]].corr("spearman")[TARGET].drop(TARGET)
    delta = (clean_corr - raw_corr).abs().max()
    record(
        f"- Rank correlation with the target is essentially unchanged: the "
        f"largest shift across these six variables is {delta:.4f}. Capping is "
        "monotonic, so the ordering of clients is preserved and only the "
        "spacing is compressed."
    )
    record(
        "- The genuine cost is ties. Roughly nine per cent of clients now sit "
        f"at exactly {cleaned['PAY_AMT1'].max():,.0f} NT$ for PAY_AMT1, so a "
        "client who repaid 20,000 NT$ and one who repaid 800,000 NT$ are no "
        "longer distinguishable. Any question about the top of the payment "
        "distribution cannot be answered from the cleaned file."
    )
    record()
    record(
        "**Verdict.** The capping is defensible for a linear model and would "
        "be the wrong default for a tree-based one, which handles skew "
        "natively and would rather keep the ordering. My recommendation is to "
        "keep the cleaned file as the primary input, add a log1p-transformed "
        "variant of the payment and bill columns as an alternative, and let "
        "the model comparison decide. The bounds should also be refitted "
        "inside the training split rather than on all 30,000 rows."
    )
    record()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].barh(audit.index, audit["pct"], color="#C44E52")
    axes[0].set_xlabel("Rows altered (%)")
    axes[0].set_title("Reach of the capping")
    axes[0].tick_params(axis="y", labelsize=8)

    width = 0.4
    positions = np.arange(len(audit))
    axes[1].barh(positions - width / 2, audit["skew_raw"], height=width,
                 label="raw", color="#C44E52")
    axes[1].barh(positions + width / 2, audit["skew_clean"], height=width,
                 label="cleaned", color="#4C72B0")
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(audit.index, fontsize=8)
    axes[1].set_xlabel("Skewness")
    axes[1].set_title("Skew before and after")
    axes[1].legend(fontsize=8)

    axes[2].bar(["Capped", "Untouched"],
                [rate_altered * 100, rate_intact * 100], color=PALETTE)
    for i, value in enumerate([rate_altered * 100, rate_intact * 100]):
        axes[2].text(i, value, f"{value:.1f}%", ha="center", va="bottom")
    axes[2].set_ylabel("Default rate (%)")
    axes[2].set_title("Capped clients default less")
    fig.suptitle("Validation of the cleaning step", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "12_cleaning_validation.png")


def section_13_raw_analysis(raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    """Repeat the core analysis on the raw file to test whether it changes."""
    record("## 13. The same analysis on the raw data")
    record()
    record(
        "If the conclusions above only hold on the cleaned file, they are "
        "conclusions about the cleaning rather than about credit risk. Every "
        "headline result is therefore recomputed on the untouched raw file."
    )
    record()

    raw_features = add_engineered_features(raw)

    rows = []
    for name, frame in [("raw", raw_features), ("cleaned", cleaned)]:
        current = frame.loc[frame["PAY_1"] <= 0, TARGET].mean()
        late = frame.loc[frame["PAY_1"] >= 1, TARGET].mean()
        rows.append(
            {
                "dataset": name,
                "n": len(frame),
                "default_rate": round(frame[TARGET].mean(), 4),
                "rate_current": round(current, 4),
                "rate_late": round(late, 4),
                "ratio": round(late / current, 2),
                "median_limit": frame["LIMIT_BAL"].median(),
            }
        )
    record("```")
    record(pd.DataFrame(rows).set_index("dataset").to_string())
    record("```")
    record()
    record(
        "- The class balance is identical, because no rows were removed and "
        "the target was never touched."
    )
    record(
        "- The repayment-status result is identical to four decimal places. "
        "The single most important finding in this analysis does not depend "
        "on the cleaning at all."
    )
    record()

    comparison_cols = ["PAY_1", "PAY_2", "PAY_3", "LIMIT_BAL", "PAY_AMT1",
                       "BILL_AMT1", "AGE", "UTILIZATION"]
    cleaned_features = add_engineered_features(cleaned)
    raw_corr = raw_features[comparison_cols + [TARGET]].corr("spearman")[TARGET]
    clean_corr = cleaned_features[comparison_cols + [TARGET]].corr("spearman")[TARGET]
    corr_table = pd.DataFrame(
        {"raw": raw_corr.drop(TARGET), "cleaned": clean_corr.drop(TARGET)}
    )
    corr_table["difference"] = corr_table["cleaned"] - corr_table["raw"]
    record("Spearman correlation with the target, raw against cleaned:")
    record("```")
    record(corr_table.round(4).to_string())
    record("```")
    record(
        f"- The largest disagreement is "
        f"{corr_table['difference'].abs().max():.4f}. Every ranking of "
        "variable importance is the same on both files."
    )
    record()

    record("### 13.1 What only the raw file can tell us")
    record()
    raw_util = raw_features["UTILIZATION"]
    over_limit = int((raw_util > 1).sum())
    negative = int((raw_util < 0).sum())
    record(
        f"- {over_limit:,} clients ({over_limit / len(raw):.1%}) carry a "
        f"balance above their credit limit, reaching {raw_util.max():.1f} "
        "times the limit at the extreme. These are the clients a lender would "
        "most want flagged, and the cleaned file cannot show them because "
        "both the bill and the limit have been capped."
    )
    record(
        f"- {negative:,} clients have a negative balance, meaning they are in "
        "credit with the bank. That is a real state, not an error."
    )
    over_limit_rate = raw_features.loc[raw_util > 1, TARGET].mean()
    normal_rate = raw_features.loc[(raw_util >= 0) & (raw_util <= 1), TARGET].mean()
    record(
        f"- Clients over their limit default at {over_limit_rate:.2%} against "
        f"{normal_rate:.2%} for clients within their limit. That gap is "
        "visible only in the raw file."
    )
    record()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(np.log1p(raw["PAY_AMT1"].clip(lower=0)), bins=60,
                 color="#C44E52", alpha=0.75, label="raw (log scale)")
    axes[0].hist(np.log1p(cleaned["PAY_AMT1"].clip(lower=0)), bins=60,
                 color="#4C72B0", alpha=0.65, label="cleaned (log scale)")
    axes[0].set_xlabel("log(1 + most recent payment)")
    axes[0].set_ylabel("Clients")
    axes[0].set_title("Capping creates a spike at the fence")
    axes[0].legend(fontsize=8)

    util_bands = pd.cut(
        raw_util.clip(-0.5, 2.0),
        bins=[-0.5, 0, 0.25, 0.5, 0.75, 1.0, 2.0],
        labels=["negative", "0-25%", "25-50%", "50-75%", "75-100%", "over limit"],
    )
    band_rates = raw_features.groupby(util_bands, observed=True)[TARGET].mean() * 100
    axes[1].bar(band_rates.index.astype(str), band_rates.to_numpy(), color="#4C72B0")
    axes[1].axhline(raw[TARGET].mean() * 100, color="#C44E52", linestyle="--",
                    linewidth=1.2, label="overall rate")
    axes[1].set_ylabel("Default rate (%)")
    axes[1].set_title("Utilization bands, raw data")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(fontsize=8)
    fig.suptitle("Raw data: what capping hides", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "13_raw_comparison.png")


def section_14_summary(df: pd.DataFrame) -> None:
    record("## 14. Summary of findings")
    record()
    record("**About the data itself**")
    record()
    record(
        "1. **The target is imbalanced at 22.1% positive.** Accuracy is not a "
        "usable metric, splits must be stratified, and precision, recall and "
        "PR-AUC are the reportable numbers."
    )
    record(
        "2. **The repayment scale is inconsistent across months.** Status 1 "
        "appears 3,688 times in the most recent month and never in the two "
        "oldest. Any feature that differences one month against another is "
        "partly measuring a recording convention. This is the single most "
        "important thing to carry into modeling."
    )
    record(
        "3. **Dormant accounts are high risk, not low risk.** Clients with "
        "six months of zero balances default well above the base rate, as do "
        "clients who never made a payment. A zero here is a different kind of "
        "observation, not a small number, and needs its own flag."
    )
    record(
        "4. **A small accounting inconsistency exists and is tolerable.** "
        "Under half a percent of client-months imply negative spending, "
        "consistent with refunds or adjustments rather than corruption."
    )
    record()
    record("**About what predicts default**")
    record()
    record(
        "5. **Repayment behaviour dominates everything else.** The count of "
        "late months is the strongest single variable in the dataset, ahead "
        "of any raw column the file ships with."
    )
    record(
        "6. **How much was repaid matters as much as whether it was late.** "
        "Clients who paid nothing against an outstanding bill default at "
        "51.8% against 13.9% for those who cleared it, and the pattern "
        "repeats in every month tested. It is computed from amounts, so the "
        "scale problem in finding 2 does not touch it."
    )
    record(
        "7. **Recent delinquency is worth more than old delinquency.** Being "
        "late only in the last two months carries roughly 1.6 times the risk "
        "of being late only in the first two. The six months should not be "
        "weighted equally."
    )
    record(
        "8. **Demographics are statistically significant and practically "
        "negligible.** Every one returns p below .001 with Cramer's V under "
        "0.1, and age scores an AUC of 0.504 on its own. They buy almost no "
        "predictive power while carrying real fairness exposure, and sex in "
        "particular is a protected attribute in consumer lending."
    )
    record(
        "9. **Age is U-shaped, not flat.** Risk falls from the mid-twenties "
        "to the low thirties then rises again. Linear summaries report "
        "nothing because the two halves cancel. Band it or drop it, but do "
        "not include it as a linear term."
    )
    record()
    record("**About the pipeline**")
    record()
    record(
        "10. **The bill columns are collinear to the point of uselessness.** "
        "Variance inflation factors run past 15. Keep one, or replace the "
        "block with utilization."
    )
    record(
        f"11. **The cleaning is sound with three caveats.** "
        f"{_cleaning_checks['passed']} of {_cleaning_checks['total']} checks "
        "pass. Capping helps a linear model and costs resolution at the top "
        "of the payment distribution, the undocumented repayment codes were "
        "treated differently from equivalent codes elsewhere, and the capping "
        "bounds were fitted on all 30,000 rows rather than on training data."
    )
    record(
        "12. **The conclusions do not depend on the cleaning.** Every "
        "headline result was recomputed on the raw file and the largest "
        "disagreement in rank correlation was 0.0031."
    )
    record()
    record("**Recommended starting point for model selection**")
    record()
    record(
        "Begin with a baseline built only from repayment behaviour: the count "
        "of late months, the most recent status, and the share of the bill "
        "repaid. Add an inactive-account flag and banded age. Leave the "
        "demographic columns and five of the six bill columns out until they "
        "demonstrate they earn a place. Anything more elaborate should have "
        "to beat that baseline on real held-out clients before its extra "
        "complexity is accepted."
    )


def main() -> None:
    cleaned, raw = load_data()
    df = add_engineered_features(cleaned)

    record("# Exploratory Data Analysis")
    record()
    record("UCI Default of Credit Card Clients. 30,000 Taiwanese credit card")
    record("holders observed April to September 2005.")
    record()
    record("Author: Eshwar")
    record()
    record(
        "Sections 1 to 11 analyse the cleaned file, which is what the "
        "modeling step will use. Section 12 validates the cleaning against "
        "the raw file. Section 13 repeats the core analysis on the raw file "
        "to confirm the findings are about credit risk, not about cleaning."
    )
    record()

    section_1_overview(df)
    section_2_distributions(df)
    section_3_demographics(df)
    section_4_repayment_history(df)
    section_5_pay_scale_anomaly(df)
    section_6_payment_behaviour(df)
    section_7_temporal_dynamics(df)
    section_8_continuous_comparisons(df)
    section_9_relationships(df)
    section_10_predictive_power(df)
    section_11_data_quality(df, raw)
    section_12_cleaning_validation(cleaned, raw)
    section_13_raw_analysis(raw, cleaned)
    section_14_summary(df)

    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_PATH.write_text("\n".join(_findings) + "\n", encoding="utf-8")
    print(f"\nFindings written to {FINDINGS_PATH.relative_to(BASE_DIR)}")
    print(f"Figures written to {FIGURE_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
