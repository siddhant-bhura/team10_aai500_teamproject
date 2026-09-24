"""
data_cleaning.py

Owner: Siddhant
Purpose: Data cleaning and preparation for the team's chosen dataset
(UCI "Default of Credit Card Clients") -- missing values, duplicates,
outliers, data types, and feature preparation.

Source file quirk: the raw .xls has a placeholder header row (X1, X2, ...,
Y) in row 0, with the real column names (LIMIT_BAL, SEX, EDUCATION, ...,
"default payment next month") in row 1. Data starts at row 2, so this
module loads with header=1.
"""

from pathlib import Path

import numpy as np
import pandas as pd

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


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, header=1, engine="xlrd")
    # Every column in this dataset is documented as numeric (ID plus 23
    # int-coded features and the target). Coerce the whole frame here, once,
    # so any stray non-numeric cell (a hand-edited "N/A", a typo, a literal
    # "inf") surfaces as a tracked missing value at the single point all data
    # enters the pipeline, instead of crashing several steps downstream in
    # value_counts/quantile/rounding code that assumes clean numeric input.
    for col in df.columns:
        # pd.to_numeric leaves a column that's entirely True/False as bool
        # dtype (it only converts mixed/object columns), and a bool column
        # crashes quantile() later. Cast it to int first so it's coerced like
        # everything else.
        series = df[col].astype(int) if df[col].dtype == bool else df[col]
        df[col] = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()


def check_duplicates(df: pd.DataFrame, id_col: str = "ID") -> tuple[pd.DataFrame, int, int]:
    """Check full-row and ID duplicates; drop exact (full-row) duplicates if found."""
    full_dup_count = int(df.duplicated().sum())
    id_dup_count = int(df[id_col].duplicated().sum()) if id_col in df.columns else 0
    if full_dup_count > 0:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df, full_dup_count, id_dup_count


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Winsorize each column at Q1-1.5*IQR / Q3+1.5*IQR. Bounds are rounded to
    whole numbers since these are all whole-currency (NT dollar) columns.
    Assumes finite numeric input; load_raw_data (the pipeline's only entry
    point) guarantees this by coercing the whole raw frame up front."""
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
        lower = round(q1 - 1.5 * iqr)
        upper = round(q3 + 1.5 * iqr)
        n_capped = int((df[col].notna() & ((df[col] < lower) | (df[col] > upper))).sum())
        cap_counts[col] = n_capped
        df[col] = df[col].clip(lower=lower, upper=upper).astype(df[col].dtype)
    return df, cap_counts


def recode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recode undocumented EDUCATION/MARRIAGE codes into the documented 'other' bucket."""
    df = df.copy()
    recode_counts = {}

    education_mask = df["EDUCATION"].notna() & ~df["EDUCATION"].isin(EDUCATION_VALID)
    recode_counts["EDUCATION"] = int(education_mask.sum())
    df.loc[education_mask, "EDUCATION"] = EDUCATION_OTHER_CODE

    marriage_mask = df["MARRIAGE"].notna() & ~df["MARRIAGE"].isin(MARRIAGE_VALID)
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


def clean_data(raw_path: Path = RAW_PATH) -> dict:
    """Run the full cleaning pipeline and return a report dict alongside the
    original and cleaned dataframes."""
    df_original = load_raw_data(raw_path)
    original_shape = df_original.shape
    original_dtypes = df_original.dtypes.copy()

    missing_before = check_missing_values(df_original)

    df, full_dup_count, id_dup_count = check_duplicates(df_original)

    education_before = df["EDUCATION"].value_counts(dropna=False).sort_index()
    marriage_before = df["MARRIAGE"].value_counts(dropna=False).sort_index()

    # Snapshot the post-dedup, pre-cap population: this is what cap_outliers_iqr
    # itself computes its IQR bounds from. Self-checks must recompute bounds from
    # this same population, not df_original -- if a future extract contains
    # duplicate rows, dedup can shift a column's IQR, and comparing against the
    # raw (pre-dedup) distribution would produce the wrong bounds.
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
        "original_shape": original_shape,
        "final_shape": df.shape,
        "original_dtypes": original_dtypes,
        "final_dtypes": df.dtypes,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "full_dup_found": full_dup_count,
        "full_dup_removed": full_dup_count,  # drop_duplicates removes all found dups
        "id_dup_found": id_dup_count,
        "cap_counts": cap_counts,
        "recode_counts": recode_counts,
        "education_before": education_before,
        "education_after": education_after,
        "marriage_before": marriage_before,
        "marriage_after": marriage_after,
    }
    return report


def print_summary(report: dict) -> None:
    print("=" * 70)
    print("DATA CLEANING SUMMARY")
    print("=" * 70)

    print(f"\nOriginal shape: {report['original_shape']}")
    print(f"Final shape:    {report['final_shape']}")

    print("\n--- Missing values (before cleaning) ---")
    print(report["missing_before"].to_string())
    print("\n--- Missing values (after cleaning) ---")
    print(report["missing_after"].to_string())

    print("\n--- Duplicates ---")
    print(f"Full-row duplicates found:  {report['full_dup_found']}")
    print(f"Full-row duplicates removed: {report['full_dup_removed']}")
    print(f"ID-column duplicates found: {report['id_dup_found']}")

    print("\n--- Outlier capping (IQR method, per financial column) ---")
    for col, n in report["cap_counts"].items():
        print(f"  {col}: {n} value(s) capped")
    print(f"  Total values capped: {sum(report['cap_counts'].values())}")

    print("\n--- EDUCATION value_counts (before recoding) ---")
    print(report["education_before"].to_string())
    print("\n--- EDUCATION value_counts (after recoding) ---")
    print(report["education_after"].to_string())
    print(f"EDUCATION values recoded to 'other' ({EDUCATION_OTHER_CODE}): {report['recode_counts']['EDUCATION']}")

    print("\n--- MARRIAGE value_counts (before recoding) ---")
    print(report["marriage_before"].to_string())
    print("\n--- MARRIAGE value_counts (after recoding) ---")
    print(report["marriage_after"].to_string())
    print(f"MARRIAGE values recoded to 'other' ({MARRIAGE_OTHER_CODE}): {report['recode_counts']['MARRIAGE']}")

    print("\n--- Dtype changes ---")
    print("Before:")
    print(report["original_dtypes"].to_string())
    print("\nAfter (PAY_0 renamed to PAY_1; SEX/EDUCATION/MARRIAGE -> category):")
    print(report["final_dtypes"].to_string())

    print("\n" + "=" * 70)


def run_self_checks(report: dict) -> None:
    cleaned_df = report["df_cleaned"]
    original_shape = report["original_shape"]

    # This dataset has zero missing values end to end (verified above), and the
    # pipeline has no imputation policy -- it preserves genuine missingness
    # rather than masking it (see recode_categoricals' .notna() guards). So this
    # assertion is a dataset-specific invariant, not a general guarantee: if a
    # future raw extract ever contains a real missing value, this should fail
    # loudly here rather than silently continue with unresolved missing data.
    assert cleaned_df.isnull().sum().sum() == 0, "Cleaned data contains missing values"

    if "ID" in cleaned_df.columns:
        assert cleaned_df["ID"].is_unique, "ID column is not unique after cleaning"
    else:
        assert cleaned_df.duplicated().sum() == 0, "Cleaned data contains duplicate rows"

    assert cleaned_df.shape[0] > 0, "Cleaned data has no rows"
    assert cleaned_df.shape[0] <= original_shape[0], "Cleaned data has more rows than original"

    education_values = set(cleaned_df["EDUCATION"].dropna().astype(int).unique())
    assert education_values <= EDUCATION_VALID, f"Unexpected EDUCATION codes: {education_values - EDUCATION_VALID}"

    marriage_values = set(cleaned_df["MARRIAGE"].dropna().astype(int).unique())
    assert marriage_values <= MARRIAGE_VALID, f"Unexpected MARRIAGE codes: {marriage_values - MARRIAGE_VALID}"

    assert str(cleaned_df["SEX"].dtype) == "category", "SEX was not cast to category"
    assert str(cleaned_df["EDUCATION"].dtype) == "category", "EDUCATION was not cast to category"
    assert str(cleaned_df["MARRIAGE"].dtype) == "category", "MARRIAGE was not cast to category"

    assert "PAY_0" not in cleaned_df.columns, "PAY_0 was not renamed"
    assert "PAY_1" in cleaned_df.columns, "PAY_1 missing after rename"

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
        lower, upper = round(q1 - 1.5 * iqr), round(q3 + 1.5 * iqr)
        assert cleaned_df[col].dropna().between(lower, upper).all(), (
            f"{col} has values outside its IQR cap bounds [{lower}, {upper}] after cleaning"
        )

    print("\nAll self-checks passed.")


if __name__ == "__main__":
    report = clean_data()
    print_summary(report)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report["df_cleaned"].to_csv(OUTPUT_PATH, index=False)
    print(f"\nCleaned data written to: {OUTPUT_PATH}")

    run_self_checks(report)
