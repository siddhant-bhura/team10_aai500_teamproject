"""
data_cleaning.py

Owner: Siddhant
AI-tool disclosure: revised with the help of Claude Code (Anthropic); the code
was run and its output checked. See README.md and the report (Section 1).
Purpose: Data cleaning and preparation for the team's chosen dataset
(UCI "Default of Credit Card Clients") -- missing values, duplicates,
outliers, data types, and feature preparation.

Source file quirk: the raw .xls has a placeholder header row (X1, X2, ...,
Y) in row 0, with the real column names (LIMIT_BAL, SEX, EDUCATION, ...,
"default payment next month") in row 1. Data starts at row 2, so this
module loads with header=1.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "default of credit card clients.xls"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "credit_card_default_cleaned.csv"

FINANCIAL_COLUMNS = (
    ["LIMIT_BAL"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)

# Documented value ranges per the UCI data dictionary. Anything outside
# these sets is an undocumented/miscoded value observed in the real data.
EDUCATION_VALID = {1, 2, 3, 4}
EDUCATION_OTHER_CODE = 4
MARRIAGE_VALID = {1, 2, 3}
MARRIAGE_OTHER_CODE = 3

# Standard Tukey's-fences multiplier for IQR-based outlier bounds (the same
# cutoff a boxplot uses), not a value tuned for this dataset.
IQR_MULTIPLIER = 1.5


def load_raw_data(
    path: Path = RAW_PATH,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the raw workbook and coerce every column to numeric.

    Returns the dataframe plus a per-column count of values that were
    coerced to missing because they weren't numeric -- a typo, "grad_school"
    in a numeric column, a literal "inf" -- distinct from cells that were
    already blank, so a coercion-heavy column is visible instead of quietly
    blending into the missing-value counts.

    Caveat: this can't distinguish a blank cell from one of pandas' own
    recognized missing-value tokens (a source cell literally containing
    "N/A", "NULL", "NaN", etc.) -- pandas.read_excel treats those as blank
    at parse time, before this function ever sees them as text. That's the
    correct call, though: those tokens are already an intentional way of
    writing "missing", not a data-quality problem, so counting them as
    "already blank" rather than "coerced" is accurate, not a bug."""
    df = pd.read_excel(path, header=1, engine="xlrd")
    coerced_counts: dict[str, int] = {}
    # Every column in this dataset is documented as numeric (ID plus 23
    # int-coded features and the target). Coerce the whole frame here, once,
    # so any stray non-numeric cell surfaces as a tracked missing value at
    # the single point all data enters the pipeline, instead of crashing
    # several steps downstream in value_counts/quantile/rounding code that
    # assumes clean numeric input.
    for col in df.columns:
        was_null = df[col].isnull()
        # pd.to_numeric leaves a column that's entirely True/False as bool
        # dtype (it only converts mixed/object columns), and a bool column
        # crashes quantile() later. Cast it to int first so it's coerced like
        # everything else.
        series = df[col].astype(int) if df[col].dtype == bool else df[col]
        coerced = pd.to_numeric(series, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        newly_null = int((coerced.isnull() & ~was_null).sum())
        if newly_null:
            coerced_counts[col] = newly_null
        df[col] = coerced
    return df, coerced_counts


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()


def check_duplicates(
    df: pd.DataFrame, id_col: str = "ID"
) -> tuple[pd.DataFrame, int, int]:
    """Check full-row and ID duplicates; drop exact (full-row) duplicates
    if found."""
    full_dup_count = int(df.duplicated().sum())
    id_dup_count = (
        int(df[id_col].duplicated().sum()) if id_col in df.columns else 0
    )
    if full_dup_count > 0:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df, full_dup_count, id_dup_count


def cap_outliers_iqr(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Winsorize (clip, not drop) each column at Q1-IQR_MULTIPLIER*IQR /
    Q3+IQR_MULTIPLIER*IQR. Bounds are rounded to whole numbers since these
    are all whole-currency (NT dollar) columns. Assumes finite numeric
    input; load_raw_data (the pipeline's only entry point) guarantees this
    by coercing the whole raw frame up front."""
    df = df.copy()
    cap_counts: dict[str, int] = {}
    for col in columns:
        valid = df[col].dropna()
        if valid.empty:
            cap_counts[col] = 0
            continue
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower = round(q1 - IQR_MULTIPLIER * iqr)
        upper = round(q3 + IQR_MULTIPLIER * iqr)
        outside = (df[col] < lower) | (df[col] > upper)
        n_capped = int((df[col].notna() & outside).sum())
        cap_counts[col] = n_capped
        df[col] = df[col].clip(lower=lower, upper=upper).astype(df[col].dtype)
    return df, cap_counts


def recode_categoricals(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recode undocumented EDUCATION/MARRIAGE codes into the documented
    'other' bucket."""
    df = df.copy()
    recode_counts = {}

    education_mask = df["EDUCATION"].notna() & ~df["EDUCATION"].isin(
        EDUCATION_VALID
    )
    recode_counts["EDUCATION"] = int(education_mask.sum())
    df.loc[education_mask, "EDUCATION"] = EDUCATION_OTHER_CODE

    marriage_mask = df["MARRIAGE"].notna() & ~df["MARRIAGE"].isin(
        MARRIAGE_VALID
    )
    recode_counts["MARRIAGE"] = int(marriage_mask.sum())
    df.loc[marriage_mask, "MARRIAGE"] = MARRIAGE_OTHER_CODE

    return df, recode_counts


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rename PAY_0 -> PAY_1 for naming consistency, and cast categorical-coded
    integer columns to pandas 'category' dtype."""
    df = df.copy()
    if "PAY_0" in df.columns:
        df = df.rename(columns={"PAY_0": "PAY_1"})
    for col in ("SEX", "EDUCATION", "MARRIAGE"):
        df[col] = df[col].astype("category")
    return df


def clean_dataframe(raw_path: Path = RAW_PATH) -> dict:
    """Run the full cleaning pipeline and return a report dict alongside the
    original and cleaned dataframes. Pure data processing -- no printing, no
    file writes; see generate_cleaning_report() and save_cleaned_data()."""
    df_original, coerced_counts = load_raw_data(raw_path)
    original_shape = df_original.shape
    original_dtypes = df_original.dtypes.copy()

    missing_before = check_missing_values(df_original)

    df, full_dup_count, id_dup_count = check_duplicates(df_original)

    education_before = df["EDUCATION"].value_counts(dropna=False).sort_index()
    marriage_before = df["MARRIAGE"].value_counts(dropna=False).sort_index()

    # Snapshot the post-dedup, pre-cap population: this is what
    # cap_outliers_iqr itself computes its IQR bounds from. Self-checks must
    # recompute bounds from this same population, not df_original -- if a
    # future extract contains duplicate rows, dedup can shift a column's IQR,
    # and comparing against the raw (pre-dedup) distribution would produce
    # the wrong bounds.
    df_pre_cap = df.copy()

    df, cap_counts = cap_outliers_iqr(df, FINANCIAL_COLUMNS)
    df, recode_counts = recode_categoricals(df)

    education_after = df["EDUCATION"].value_counts(dropna=False).sort_index()
    marriage_after = df["MARRIAGE"].value_counts(dropna=False).sort_index()

    df = prepare_features(df)

    missing_after = check_missing_values(df)

    report = {
        "df_original": df_original,
        "df_pre_cap": df_pre_cap,
        "df_cleaned": df,
        "coerced_counts": coerced_counts,
        "original_shape": original_shape,
        "final_shape": df.shape,
        "original_dtypes": original_dtypes,
        "final_dtypes": df.dtypes,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "full_dup_found": full_dup_count,
        # drop_duplicates removes all found dups
        "full_dup_removed": full_dup_count,
        "id_dup_found": id_dup_count,
        "cap_counts": cap_counts,
        "recode_counts": recode_counts,
        "education_before": education_before,
        "education_after": education_after,
        "marriage_before": marriage_before,
        "marriage_after": marriage_after,
    }
    return report


def generate_cleaning_report(report: dict) -> str:
    """Build the human-readable cleaning summary as a single string. Pure
    formatting -- no I/O; the caller decides how to emit it (logger, print,
    a file), and this stays trivially testable (just check the return
    value)."""
    lines = []
    lines.append("=" * 70)
    lines.append("DATA CLEANING SUMMARY")
    lines.append("=" * 70)

    lines.append(f"\nOriginal shape: {report['original_shape']}")
    lines.append(f"Final shape:    {report['final_shape']}")

    lines.append("\n--- Non-numeric values coerced to missing (by column) ---")
    if report["coerced_counts"]:
        for col, n in report["coerced_counts"].items():
            lines.append(
                f"  {col}: {n} value(s) were non-numeric/non-finite, "
                "not just blank"
            )
    else:
        lines.append(
            "  None -- every value in the raw file was already numeric."
        )

    lines.append("\n--- Missing values (before cleaning) ---")
    lines.append(report["missing_before"].to_string())
    lines.append("\n--- Missing values (after cleaning) ---")
    lines.append(report["missing_after"].to_string())

    lines.append("\n--- Duplicates ---")
    lines.append(f"Full-row duplicates found:  {report['full_dup_found']}")
    lines.append(f"Full-row duplicates removed: {report['full_dup_removed']}")
    lines.append(f"ID-column duplicates found: {report['id_dup_found']}")

    lines.append(
        "\n--- Outlier capping (IQR method, per financial column) ---"
    )
    for col, n in report["cap_counts"].items():
        lines.append(f"  {col}: {n} value(s) capped")
    total_capped = sum(report["cap_counts"].values())
    lines.append(f"  Total values capped: {total_capped}")

    lines.append("\n--- EDUCATION value_counts (before recoding) ---")
    lines.append(report["education_before"].to_string())
    lines.append("\n--- EDUCATION value_counts (after recoding) ---")
    lines.append(report["education_after"].to_string())
    lines.append(
        f"EDUCATION values recoded to 'other' ({EDUCATION_OTHER_CODE}): "
        f"{report['recode_counts']['EDUCATION']}"
    )

    lines.append("\n--- MARRIAGE value_counts (before recoding) ---")
    lines.append(report["marriage_before"].to_string())
    lines.append("\n--- MARRIAGE value_counts (after recoding) ---")
    lines.append(report["marriage_after"].to_string())
    lines.append(
        f"MARRIAGE values recoded to 'other' ({MARRIAGE_OTHER_CODE}): "
        f"{report['recode_counts']['MARRIAGE']}"
    )

    lines.append("\n--- Dtype changes ---")
    lines.append("Before:")
    lines.append(report["original_dtypes"].to_string())
    lines.append(
        "\nAfter (PAY_0 renamed to PAY_1; "
        "SEX/EDUCATION/MARRIAGE -> category):"
    )
    lines.append(report["final_dtypes"].to_string())

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def save_cleaned_data(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """Write the cleaned dataframe to disk, creating the target directory
    if needed. Separated from clean_dataframe() so cleaning can be tested
    without touching the filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_self_checks(report: dict) -> None:
    cleaned_df = report["df_cleaned"]
    original_shape = report["original_shape"]

    # This dataset has zero missing values end to end (verified above), and the
    # pipeline has no imputation policy -- it preserves genuine missingness
    # rather than masking it (see recode_categoricals' .notna() guards). So
    # this check is a dataset-specific invariant, not a general guarantee: if
    # a future raw extract ever contains a real missing value, this should
    # fail loudly here rather than silently continue with unresolved missing
    # data. Plain `raise ValueError` rather than `assert`, since assertions
    # are silently skipped if Python ever runs with the -O optimization flag,
    # which would defeat the point of a data-validation check.
    if cleaned_df.isnull().sum().sum() != 0:
        raise ValueError("Cleaned data contains missing values")

    if "ID" in cleaned_df.columns:
        if not cleaned_df["ID"].is_unique:
            raise ValueError("ID column is not unique after cleaning")
    elif cleaned_df.duplicated().sum() != 0:
        raise ValueError("Cleaned data contains duplicate rows")

    if cleaned_df.shape[0] == 0:
        raise ValueError("Cleaned data has no rows")
    if cleaned_df.shape[0] > original_shape[0]:
        raise ValueError("Cleaned data has more rows than original")

    education_values = set(
        cleaned_df["EDUCATION"].dropna().astype(int).unique()
    )
    if not education_values <= EDUCATION_VALID:
        raise ValueError(
            f"Unexpected EDUCATION codes: {education_values - EDUCATION_VALID}"
        )

    marriage_values = set(cleaned_df["MARRIAGE"].dropna().astype(int).unique())
    if not marriage_values <= MARRIAGE_VALID:
        raise ValueError(
            f"Unexpected MARRIAGE codes: {marriage_values - MARRIAGE_VALID}"
        )

    for col in ("SEX", "EDUCATION", "MARRIAGE"):
        if str(cleaned_df[col].dtype) != "category":
            raise ValueError(f"{col} was not cast to category")

    if "PAY_0" in cleaned_df.columns:
        raise ValueError("PAY_0 was not renamed")
    if "PAY_1" not in cleaned_df.columns:
        raise ValueError("PAY_1 missing after rename")

    # Independently recompute IQR bounds from the post-dedup, pre-cap data (the
    # same population cap_outliers_iqr itself used) and confirm the cleaned
    # columns actually respect them -- proves capping happened, rather than
    # trusting cap_counts alone. Must use df_pre_cap, not df_original: if
    # duplicates existed and were removed, the raw (pre-dedup) distribution's
    # IQR can differ from the one actually used for capping.
    df_pre_cap = report["df_pre_cap"]
    for col in FINANCIAL_COLUMNS:
        valid = df_pre_cap[col].dropna()
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        lower = round(q1 - IQR_MULTIPLIER * iqr)
        upper = round(q3 + IQR_MULTIPLIER * iqr)
        if not cleaned_df[col].dropna().between(lower, upper).all():
            raise ValueError(
                f"{col} has values outside its IQR cap bounds "
                f"[{lower}, {upper}] after cleaning"
            )

    logger.info("All self-checks passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = clean_dataframe()
    logger.info(generate_cleaning_report(report))

    save_cleaned_data(report["df_cleaned"], OUTPUT_PATH)
    logger.info(f"\nCleaned data written to: {OUTPUT_PATH}")

    run_self_checks(report)
