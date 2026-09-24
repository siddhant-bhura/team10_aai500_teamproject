# AAI-500 Final Team Project Report

**Team:** Prasad Pidaparty · Siddhant Bhura · Eshwar Reddy M · Umesh Sharma
**Dataset:** UCI "Default of Credit Card Clients" (30,000 records, 24 features + target)

---

## 1. Introduction

> **DRAFT — written by Siddhant to unblock the report, needs Prasad's review and his own voice/citations before this is final. This is Prasad's assigned section (dataset research, business problem definition).**

### Business Problem

Credit card issuers lose money when customers default on their monthly payments. Being able to flag customers who are likely to default *before* the missed payment happens gives a bank room to act early — lowering a credit limit, offering a payment plan, or increasing loss reserves for that account — rather than only reacting after the fact. This project treats that as a binary classification problem: given a customer's demographic profile, credit limit, and the last six months of billing and repayment activity, predict whether they will default on their payment next month.

### Dataset

The team selected the UCI **"Default of Credit Card Clients"** dataset, sourced from a Taiwanese bank and hosted on the UCI Machine Learning Repository. It was chosen from six candidate datasets the team evaluated (ranging from ~4,400 to ~542,000 rows) based on group discussion and a team vote, balancing a dataset large and rich enough to support a full analysis pipeline against one still practical to work with on personal laptops.

The dataset contains 30,000 customer records with 23 features plus the target variable:

- **Demographics:** sex, education level, marital status, age
- **Credit line:** credit limit (`LIMIT_BAL`)
- **Repayment history:** monthly repayment status for the 6 most recent months (`PAY_1`–`PAY_6`)
- **Billing history:** monthly bill statement amount for the same 6 months (`BILL_AMT1`–`BILL_AMT6`)
- **Payment history:** monthly amount actually paid for the same 6 months (`PAY_AMT1`–`PAY_AMT6`)
- **Target:** whether the customer defaulted on their payment the following month (binary)

The target is moderately imbalanced — about 22% of customers in the dataset defaulted, versus 78% who did not (roughly a 1:3.5 ratio). This is relevant to how the model section (Section 4) evaluates performance, since accuracy alone can be misleading on an imbalanced target.

### Expected Business Value

A working model gives the bank a ranked risk score per customer rather than a blanket policy, so credit and collections teams can prioritize outreach and risk-mitigation actions toward the customers most likely to actually default, instead of spreading effort evenly across the whole customer base.

---

## 2. Data Cleaning / Preparation

*Owner: Siddhant Bhura*

### Loading the Data

The raw file (`data/raw/default of credit card clients.xls`) is a legacy Excel binary (`.xls`, not `.xlsx`) with a formatting quirk: the first row contains placeholder column labels (`X1`, `X2`, ..., `Y`), and the real column names (`LIMIT_BAL`, `SEX`, `EDUCATION`, ..., `default payment next month`) are on the second row. The loader reads the sheet with the header on that second row so the real column names are used throughout.

### Missing Values

Every one of the 25 columns (ID, 23 features, target) was checked for missing values directly (`.isnull().sum()`), rather than assumed clean. **Zero missing values were found**, both before and after cleaning.

### Duplicate Records

Two duplicate checks were run: full-row duplicates, and duplicates on the `ID` column. **Neither found any duplicates** — all 30,000 `ID` values are unique, and no row is an exact duplicate of another. Separately, 35 rows were found to have *identical feature values* under *different* `ID`s — these are different customers who happen to share the same profile (credit limit, demographics, payment history), not the same record entered twice, and were correctly left in the dataset rather than dropped.

### Outlier Handling

The credit-limit and monthly billing/payment columns (13 columns total) are heavily right-skewed, which is typical for financial amounts — most customers cluster at moderate values with a long tail of high-limit or high-balance accounts. Rather than drop these rows and lose data, outliers were **capped (winsorized)** using the standard IQR method: any value below `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR` for a given column is pulled in to that boundary. In total, **32,425 individual values were capped across the 13 financial columns** (out of 390,000 cells in those columns, about 8.3%) — no rows were removed.

| Column group | Values capped |
|---|---|
| `LIMIT_BAL` | 167 |
| `BILL_AMT1`–`BILL_AMT6` | 2,400 / 2,395 / 2,469 / 2,622 / 2,725 / 2,693 |
| `PAY_AMT1`–`PAY_AMT6` | 2,745 / 2,714 / 2,598 / 2,994 / 2,945 / 2,958 |

### Categorical Recoding

The dataset's own documentation defines `EDUCATION` as 1–4 (graduate school / university / high school / **others**) and `MARRIAGE` as 1–3 (married / single / **others**). The raw data contains codes outside those documented ranges — `EDUCATION` values of 0, 5, and 6 (345 rows total), and `MARRIAGE` value of 0 (54 rows) — which are undocumented in the original data dictionary. These were folded into the dataset's own existing "others" bucket (`EDUCATION = 4`, `MARRIAGE = 3`) rather than treated as a new category, since that already exists in the schema and is the standard treatment used for this well-known dataset. This is a judgment call, not something UCI's documentation mandates outright, but it is the most defensible reading of an "undocumented code" on a field that already has an "others" option.

### Feature Preparation

- `PAY_0` was renamed to `PAY_1` for naming consistency with `PAY_2`–`PAY_6` (the raw column numbering inexplicably starts at 0 for the most recent month).
- `SEX`, `EDUCATION`, and `MARRIAGE` were cast to categorical dtype, since they are coded categories rather than continuous numbers.

### Output

The cleaned dataset (30,000 rows retained, same as the original — no rows were dropped) is written to `data/processed/credit_card_default_cleaned.csv` for use by the exploratory analysis and modeling stages.

### Verification

The cleaning script includes automated checks that confirm, on every run: no missing values remain, `ID` stays unique, `EDUCATION`/`MARRIAGE` only contain documented codes, and every financial column's values fall within its own capped bounds — so a future change to the raw data or the cleaning logic that silently breaks one of these guarantees would be caught immediately rather than discovered downstream.

---

## 3. Exploratory Data Analysis

*Owner: Eshwar Reddy M — not yet written.*

## 4. Model Selection and Model Analysis

*Owner: Umesh Sharma — not yet written.*

## 5. Conclusion and Recommendations

*Shared, primarily Prasad — not yet written.*

## Appendix

*Code appendix — not yet assembled.*
