# Exploratory Data Analysis

UCI Default of Credit Card Clients. 30,000 Taiwanese credit card
holders observed April to September 2005.

Author: Eshwar

Sections 1 to 11 analyse the cleaned file, which is what the modeling step will use. Section 12 validates the cleaning against the raw file. Section 13 repeats the core analysis on the raw file to confirm the findings are about credit risk, not about cleaning.

## 1. Dataset overview and class balance

- 30,000 clients, 28 columns, no missing values.
- 6,636 clients defaulted, a rate of 22.12%. The majority class is 77.88% of the data.
- A model predicting "no default" for every client would be 77.88% accurate while identifying nobody at risk, so accuracy is not a usable headline metric here.

## 2. Distribution of the financial variables

```
                   mean     median         std  skewness
variable                                                
LIMIT_BAL    166967.489  140000.00  128013.394     0.904
AGE              35.486      34.00       9.218     0.732
BILL_AMT1     44293.450   22381.50   51294.801     1.194
PAY_AMT1       3496.731    2100.00    3492.743     1.032
UTILIZATION       0.403       0.31       0.392     0.892
```

- Every financial variable is right-skewed, with the mean above the median in each case. Summary statistics for this dataset should be reported as medians and interquartile ranges, not means and standard deviations.
- Credit limit has a median of 140,000 NT$ against a mean of 166,967 NT$.

## 3. Demographics and default: chi-square tests of independence

Each demographic column is tested against the target with a chi-square test of independence. Cramer's V is reported alongside every p-value, because with n = 30,000 a trivial difference will reach significance on its own.

**SEX**
```
            n  default_rate
Male    11888         24.17
Female  18112         20.78
```
chi-square(1, N = 30,000) = 47.7, p < .001, Cramer's V = 0.040

**EDUCATION**
```
                     n  default_rate
Graduate school  10585         19.23
University       14030         23.73
High school       4917         25.16
Other              468          7.05
```
chi-square(3, N = 30,000) = 160.4, p < .001, Cramer's V = 0.073

**MARRIAGE**
```
             n  default_rate
Married  13659         23.47
Single   15964         20.93
Other      377         23.61
```
chi-square(2, N = 30,000) = 28.1, p < .001, Cramer's V = 0.031

- All three are statistically significant and all three are weak. The largest effect, EDUCATION, has a Cramer's V of 0.073, which is a small association by any conventional benchmark.
- This is the practical-significance problem in miniature. The p-values say these variables are not independent of default. The effect sizes say none of them would be much use for deciding about an individual client.

### 3.1 Checking the test's assumption

A chi-square test needs an expected count of at least five in every cell. That is checked here rather than assumed.

- SEX: smallest expected count 2629.6 (OK)
- EDUCATION: smallest expected count 103.5 (OK)
- MARRIAGE: smallest expected count 83.4 (OK)
- PAY_1: smallest expected count 2.0 (VIOLATED)

- The three demographic tests are safe. The PAY_1 test in Section 4 is not: its rarest categories hold only a handful of clients, so the chi-square approximation does not hold at the tail. The finding survives because the effect is enormous and concentrated in the well-populated categories, but the sparse statuses should be collapsed before the test is quoted in the report.

### 3.2 Age is not linear, which is why it looks useless

```
             n  default_rate
AGE                         
(20, 25]  3871         26.66
(25, 30]  7142         20.15
(30, 35]  5796         19.43
(35, 40]  4917         21.62
(40, 45]  3605         22.14
(45, 50]  2400         25.04
(50, 60]  1997         25.24
(60, 80]   272         26.84
```
- The relationship is U-shaped. Risk starts at 26.66% for the under-25s, falls to 19.43% in the low thirties, then climbs back to 26.84% for the over-60s.
- This is why every linear summary of age in this analysis reports almost nothing. A correlation coefficient averages the falling half against the rising half and lands near zero. The variable carries real information, but only in banded form.

## 4. Repayment history is the dominant signal

Default rate by most recent repayment status (PAY_1):
```
                      n  default_rate_pct
No consumption     2759             13.23
Paid in full       5686             16.78
Revolving credit  14737             12.81
1 month late       3688             33.95
2 months late      2667             69.14
3 months late       322             75.78
4 months late        76             68.42
5 months late        26             50.00
6 months late        11             54.55
7 months late         9             77.78
8 months late        19             57.89
```
chi-square(10, N = 30,000) = 5366.0, p < .001, Cramer's V = 0.423

- Clients current on their most recent payment default at 13.83%. Clients already one month late default at 50.29%, a 3.6-fold increase.
- Cramer's V of 0.423 is an order of magnitude above any demographic variable, making PAY_1 the single most informative column in the dataset.

Default rate by number of late months in the six-month window:
```
                       n  default_rate
MONTHS_DELINQUENT                     
0                  19931         11.71
1                   4426         29.82
2                   1899         38.76
3                   1154         50.87
4                    951         57.31
5                    298         57.38
6                   1341         70.32
```
- The relationship is monotonic, rising from 11.7% at zero late months to 70.3% at six, with no reversal. The only flat step is four to five late months, where the gap is 0.07 percentage points on 298 clients and should be read as noise rather than a plateau.
- The per-status bars above are noisier at the far end, because five months late and beyond contain only a few hundred clients each. The count of late months is the more stable way to express the same signal, and it is the version worth carrying into the model.

## 5. The repayment scale is not consistent across months

The six repayment-status columns look like one variable measured six times. They are not. Tabulating every code by month shows that status 1, meaning one month of delay, is common in the most recent month and almost absent from the other five.

```
    PAY_1  PAY_2  PAY_3  PAY_4  PAY_5  PAY_6
-2   2759   3782   4085   4348   4546   4895
-1   5686   6050   5938   5687   5539   5740
 0  14737  15730  15764  16455  16947  16286
 1   3688     28      4      2      0      0
 2   2667   3927   3819   3159   2626   2766
 3    322    326    240    180    178    184
 4     76     99     76     69     84     49
 5     26     25     21     35     17     13
 6     11     12     23      5      4     19
 7      9     20     27     58     58     46
 8     19      1      3      2      1      2
```

- Clients recorded at status 1 by month: {'PAY_1': 3688, 'PAY_2': 28, 'PAY_3': 4, 'PAY_4': 2, 'PAY_5': 0, 'PAY_6': 0}
- Clients recorded at status 2 by month: {'PAY_1': 2667, 'PAY_2': 3927, 'PAY_3': 3819, 'PAY_4': 3159, 'PAY_5': 2626, 'PAY_6': 2766}
- PAY_1 has 3,688 clients at status 1. PAY_5 and PAY_6 have none at all. Status 2 meanwhile appears in every month at similar volume, so this is not a case of delinquency simply being rarer in earlier months.
- The most likely explanation is that in the earlier months a one-month delay was recorded as status 2, and only in the final month was status 1 used. Whatever the cause, the scale is not applied identically across the six columns.

**Three consequences the modeling step has to respect.**

1. Any feature that differences PAY_1 against an earlier month is measuring a recording convention as much as a change in behaviour. A client at status 1 in September and status 2 in April looks like an improvement and may be unchanged.
2. Treating the six columns as an interchangeable panel, for example by averaging them, mixes two different scales.
3. Counting late months with a threshold of one or more is safe, because status 2 is present in every month, but any threshold that separates 1 from 2 is not.

## 6. Payment behaviour: how much of the bill was actually repaid

The repayment-status columns say whether a client was late. They do not say how much was paid. PAY_AMT1 is the amount paid in September against BILL_AMT2, the bill issued in August, so the ratio of the two is the share of the outstanding balance the client cleared. Clients with no outstanding bill are excluded from the ratio, since paying nothing against a zero balance is not a behaviour.

Most recent month (PAY_AMT1 against BILL_AMT2):
```
                      n  default_rate
Paid nothing       2325         51.78
Token (<20%)      15803         21.12
Partial (20-99%)   3199         15.75
Paid in full       5498         15.39
```

- Clients who paid nothing against an outstanding bill default at 51.78%. Clients who cleared the balance default at 15.39%, a 3.4-fold difference.
- The ordering is strictly monotonic across all four bands and holds in each of the three months tested, so this is a stable behavioural signal rather than a one-month artefact.
- This is worth engineering as a feature. It is independent of the status scale discussed in Section 5, being computed from amounts rather than from the status codes, so it is unaffected by that inconsistency.

## 7. Temporal dynamics: recent trouble outweighs old trouble

Six months of history invites a question the cross-sectional view cannot answer: does it matter when the delinquency happened?

Comparing the two most recent months against the two oldest:
```
                          n  default_rate
group                                    
Never late            21191         12.44
Late early only        1544         25.97
Late recently only     4868         41.64
Late in both windows   2397         65.58
```

- Clients late only in the two most recent months default at 41.64%. Clients late only in the two oldest months default at 25.97%. Recent delinquency carries roughly 1.6 times the risk of old delinquency.
- Clients late in both windows default at 65.58%, the highest of any group examined in this analysis.
- A model should weight the recent months more heavily than the distant ones rather than treating all six equally.

**A trend feature that does not survive scrutiny.**

```
               n  default_rate
Improving   3935         26.28
Stable     18164         16.00
Worsening   7901         34.11
```
- Read naively this says clients whose status improved default at 26.28%, well above the 16.00% of clients who stayed flat. Improvement appears to predict failure.
- Section 5 explains it. Status 1 exists only in PAY_1, so a client sitting at status 2 in April and status 1 in September registers as improving purely through a change in recording convention. The improving group is contaminated with clients who were seriously delinquent throughout.
- **Recommendation: do not build a PAY_1 minus PAY_6 trend feature.** Use the recency grouping above, which relies only on whether a client was late at all and is unaffected by the scale problem.

## 8. Continuous variables compared across the two groups

Welch's t-test does not assume equal variances, which matters here because the defaulting group is a quarter of the size of the other. The Mann-Whitney U test is reported beside it because these distributions are skewed, and it compares distributions rather than means. Cohen's d gives the size of each gap.

```
             median_default  median_current        t  p_welch   p_mwu  cohens_d
variable                                                                       
LIMIT_BAL         90000.000     150000.0000 -29.0042   0.0000  0.0000   -0.3765
AGE                  34.000         34.0000   2.3195   0.0204  0.3725    0.0335
UTILIZATION           0.464          0.2679  15.3908   0.0000  0.0000    0.2176
PAY_AMT1           1636.000       2459.5000 -27.4280   0.0000  0.0000   -0.3492
BILL_AMT1         20185.000      23119.5000  -5.1572   0.0000  0.0000   -0.0700
```

- Credit limit shows the clearest separation. Clients who defaulted had a median limit of 90,000 NT$ against 150,000 NT$ for those who did not.
- Every p-value here is below .001, and not one Cohen's d reaches 0.5. At n = 30,000 significance is close to guaranteed, so the effect sizes are what carry the information.
- Age is the clearest example: the difference is significant and Cohen's d is 0.033, which is negligible. Age should not drive a lending decision on this evidence.

## 9. Relationships between variables

Strongest Spearman correlations with the target:
```
MONTHS_DELINQUENT    0.388
PAY_1                0.292
PAY_2                0.217
PAY_3                0.195
PAY_4                0.174
PAY_5                0.159
PAY_6                0.143
UTILIZATION          0.076
...
PAY_AMT3    -0.139
PAY_AMT2    -0.150
PAY_AMT1    -0.160
LIMIT_BAL   -0.170
```

- The six bill columns correlate with each other between 0.79 and 0.95. They carry almost the same information six times over, so feeding all six into a regression will produce unstable coefficients. One bill column, or a summary such as utilization, is enough.
- The repayment-status columns are the only block with a meaningful correlation to the target, and PAY_1 leads them.

Variance inflation factors, the formal version of that point:
```
            VIF
variable       
LIMIT_BAL   1.1
AGE         1.0
BILL_AMT1   9.7
BILL_AMT2  15.5
BILL_AMT3  13.7
BILL_AMT4  13.1
BILL_AMT5  14.1
BILL_AMT6   9.0
```
- A VIF above 5 is the usual warning line and above 10 the usual red line. Every bill column clears 10 except two, peaking at 15.5, while LIMIT_BAL and AGE sit near 1. The problem is confined entirely to the bill block.

Default rate by credit utilization decile:
```
                      n  default_rate
UTILIZATION                          
(-0.001, 0.00124]  3000         24.07
(0.00124, 0.0113]  3000         19.57
(0.0113, 0.0406]   3000         15.77
(0.0406, 0.131]    3000         14.80
(0.131, 0.31]      3000         17.80
(0.31, 0.481]      3000         21.37
(0.481, 0.668]     3000         25.37
(0.668, 0.849]     3000         28.50
(0.849, 0.968]     3000         25.23
(0.968, 2.0]       3000         28.73
```
- Default rate climbs steadily with utilization. Clients using most of their available credit default far more often than clients using little of it, and the pattern holds across the whole range.

## 10. Which variables actually separate the two groups

Correlation understates a variable whose relationship with the target is not monotonic, and overstates nothing. A cleaner summary is the area under the ROC curve for each variable used alone, which is the probability that a randomly chosen defaulter scores higher than a randomly chosen non-defaulter. A value of 0.5 means the variable carries no information. This is computed from the Mann-Whitney U statistic, so it needs no model fitting.

```
                      auc  separation         direction
feature                                                
MONTHS_DELINQUENT  0.7261      0.2261  higher = riskier
MAX_DELINQUENCY    0.7077      0.2077  higher = riskier
PAY_1              0.6897      0.1897  higher = riskier
PAY_2              0.6386      0.1386  higher = riskier
PAY_3              0.6243      0.1243  higher = riskier
LIMIT_BAL          0.3822      0.1178   lower = riskier
PAY_AMT1           0.3891      0.1109   lower = riskier
PAY_4              0.6097      0.1097  higher = riskier
PAY_5              0.5995      0.0995  higher = riskier
PAID_SHARE         0.4070      0.0930   lower = riskier
PAY_6              0.5902      0.0902  higher = riskier
UTILIZATION        0.5529      0.0529  higher = riskier
EDUCATION          0.5278      0.0278  higher = riskier
SEX                0.4765      0.0235   lower = riskier
BILL_AMT1          0.4822      0.0178   lower = riskier
MARRIAGE           0.4827      0.0173   lower = riskier
AGE                0.5036      0.0036  higher = riskier
```

- **MONTHS_DELINQUENT is the strongest single variable**, at an AUC of 0.726. It is an engineered count rather than a raw column, which is worth noting: the dataset as delivered does not contain its own best predictor.
- The repayment block occupies the top of the table and the demographic block occupies the bottom. There is a clear gap between them rather than a gradual decline.
- AGE scores 0.504, which is indistinguishable from a coin flip. Section 3 shows why: the relationship is U-shaped, so a rank-based measure sees nothing. The variable is not useless, it is simply useless in linear form.
- A sensible baseline is therefore the repayment block plus the engineered count, with demographics excluded until they earn a place.

## 11. Data quality issues the cleaning does not address

### 11.1 Accounting consistency

Consecutive bills, the payment between them and any new spending should satisfy a simple identity: this month's bill equals last month's bill, minus what was paid, plus what was newly charged. Rearranged, implied new spending is the bill difference plus the payment, and it cannot be negative.
- 618 of 150,000 client-months (0.4%) imply negative spending, with the most extreme case at -415,804 NT$.
- At under half a percent this is not a reason to distrust the dataset. It is most likely refunds, disputed charges reversed, or interest adjustments, none of which the identity accounts for. It is worth stating in the report so the assumption is on the record.

### 11.2 Inactive accounts are not low-risk accounts

- 866 clients show a zero balance in all six months. They default at 36.61% against an overall rate of 22.12%.
- 1,432 clients made no payment at all in six months. They default at 37.99%.
- This is counterintuitive and it matters. A zero is not a small number here, it is a different kind of observation. Any imputation or scaling that treats these rows as ordinary low-value clients will push them toward the safe end of the distribution, which is the opposite of where they belong.
- **Recommendation: add a binary inactive-account flag** rather than letting the zeros speak for themselves.

## 12. Validation of the cleaning step

Every operation in data_cleaning.py is checked against the raw file below. The point is not to second-guess the work but to establish what the modeling step is actually receiving.

| Check | Verdict | Detail |
|---|---|---|
| Header row handled | PASS | The raw sheet carries placeholder names (X1, X2, ... Y) in row 0 and the real names in row 1. Loading with header=1 is correct. |
| PAY_0 renamed to PAY_1 | PASS | The raw file names the first repayment column PAY_0 while the rest run PAY_2 to PAY_6. Renaming removes a genuine trap. |
| Missing values | PASS | 0 missing in raw, 0 in cleaned. None existed and none were introduced. |
| Duplicates | PASS | 0 full-row duplicates and 0 duplicate IDs in the raw file, so nothing was dropped and nothing should have been. |
| Row count, order and target | PASS | All 30,000 rows kept in the same order, IDs aligned, and the target column untouched. |
| Ordinal and categorical columns left uncapped | PASS | AGE, SEX and the six repayment-status columns are unchanged. Winsorizing an ordinal scale would have been a real error and it was avoided. |
| EDUCATION recoded | PASS | Codes 0, 5 and 6 are undocumented in the UCI dictionary. All 345 such rows were folded into category 4 (other), which grew from 123 to 468. |
| MARRIAGE recoded | PASS | Code 0 is undocumented. All 54 such rows were folded into category 3 (other), which grew from 323 to 377. |
| Outlier winsorization | CONCERN | Capping at the 1.5 x IQR fence altered 10,269 rows (34.2%). Discussed in detail below. |
| Undocumented repayment codes | CONCERN | The UCI dictionary documents -1 and 1 to 9 for PAY_n, but the data also contains -2 and 0 across 120,334 cells. These were left in place while the equivalent undocumented codes in EDUCATION and MARRIAGE were recoded. The treatment should be consistent, or the difference should be explained. |
| Capping bounds fitted on the full dataset | CONCERN | The IQR fences are computed from all 30,000 rows. If the modeling step later splits this file into train and test, the test rows have already influenced the preprocessing. Bounds should be fitted on training data only and applied to the test set. |

**8 of 11 checks pass.** The mechanical work is sound: no data lost, nothing miscoded, the ordinal columns correctly protected, and the two real traps in this dataset (the double header and the PAY_0 naming) both handled. The three concerns are judgement calls rather than bugs.

### 12.1 What winsorization actually did

```
           raw_max  cleaned_max  rows_altered   pct  skew_raw  skew_clean
column                                                                   
LIMIT_BAL  1000000       525000           167  0.56      0.99        0.90
BILL_AMT1   964511       162389          2400  8.00      2.66        1.19
BILL_AMT2   983931       155538          2395  7.98      2.71        1.19
BILL_AMT3  1664089       146412          2469  8.23      3.09        1.18
BILL_AMT4   891586       132775          2622  8.74      2.82        1.18
BILL_AMT5   927171       122832          2725  9.08      2.88        1.18
BILL_AMT6   961664       121112          2693  8.98      2.85        1.20
PAY_AMT1    873552        11015          2745  9.15     14.67        1.03
PAY_AMT2   1684259        11250          2714  9.05     30.45        1.11
PAY_AMT3    896040        10678          2598  8.66     17.22        1.20
PAY_AMT4    621000         9589          2994  9.98     12.90        1.18
PAY_AMT5    426529         9700          2945  9.82     11.13        1.18
PAY_AMT6    528666         9823          2958  9.86     10.64        1.21
```

- 10,269 rows (34.2%) were altered. Capped rows default at 14.91% against 25.87% for untouched rows, so the capped group is the more creditworthy one.
- Skewness falls sharply. PAY_AMT1 goes from 14.67 to 1.03. A skew near 15 would give a handful of clients enormous leverage over a linear model, so this is a real benefit for logistic regression.
- Rank correlation with the target is essentially unchanged: the largest shift across these six variables is 0.0006. Capping is monotonic, so the ordering of clients is preserved and only the spacing is compressed.
- The genuine cost is ties. Roughly nine per cent of clients now sit at exactly 11,015 NT$ for PAY_AMT1, so a client who repaid 20,000 NT$ and one who repaid 800,000 NT$ are no longer distinguishable. Any question about the top of the payment distribution cannot be answered from the cleaned file.

**Verdict.** The capping is defensible for a linear model and would be the wrong default for a tree-based one, which handles skew natively and would rather keep the ordering. My recommendation is to keep the cleaned file as the primary input, add a log1p-transformed variant of the payment and bill columns as an alternative, and let the model comparison decide. The bounds should also be refitted inside the training split rather than on all 30,000 rows.

## 13. The same analysis on the raw data

If the conclusions above only hold on the cleaned file, they are conclusions about the cleaning rather than about credit risk. Every headline result is therefore recomputed on the untouched raw file.

```
             n  default_rate  rate_current  rate_late  ratio  median_limit
dataset                                                                   
raw      30000        0.2212        0.1383     0.5029   3.64      140000.0
cleaned  30000        0.2212        0.1383     0.5029   3.64      140000.0
```

- The class balance is identical, because no rows were removed and the target was never touched.
- The repayment-status result is identical to four decimal places. The single most important finding in this analysis does not depend on the cleaning at all.

Spearman correlation with the target, raw against cleaned:
```
                raw  cleaned  difference
PAY_1        0.2922   0.2922      0.0000
PAY_2        0.2169   0.2169      0.0000
PAY_3        0.1948   0.1948      0.0000
LIMIT_BAL   -0.1696  -0.1696      0.0000
PAY_AMT1    -0.1605  -0.1599      0.0006
BILL_AMT1   -0.0253  -0.0256     -0.0003
AGE          0.0051   0.0051      0.0000
UTILIZATION  0.0730   0.0761      0.0031
```
- The largest disagreement is 0.0031. Every ranking of variable importance is the same on both files.

### 13.1 What only the raw file can tell us

- 2,115 clients (7.0%) carry a balance above their credit limit, reaching 6.5 times the limit at the extreme. These are the clients a lender would most want flagged, and the cleaned file cannot show them because both the bill and the limit have been capped.
- 590 clients have a negative balance, meaning they are in credit with the bank. That is a real state, not an error.
- Clients over their limit default at 30.07% against 21.58% for clients within their limit. That gap is visible only in the raw file.

## 14. Summary of findings

**About the data itself**

1. **The target is imbalanced at 22.1% positive.** Accuracy is not a usable metric, splits must be stratified, and precision, recall and PR-AUC are the reportable numbers.
2. **The repayment scale is inconsistent across months.** Status 1 appears 3,688 times in the most recent month and never in the two oldest. Any feature that differences one month against another is partly measuring a recording convention. This is the single most important thing to carry into modeling.
3. **Dormant accounts are high risk, not low risk.** Clients with six months of zero balances default well above the base rate, as do clients who never made a payment. A zero here is a different kind of observation, not a small number, and needs its own flag.
4. **A small accounting inconsistency exists and is tolerable.** Under half a percent of client-months imply negative spending, consistent with refunds or adjustments rather than corruption.

**About what predicts default**

5. **Repayment behaviour dominates everything else.** The count of late months is the strongest single variable in the dataset, ahead of any raw column the file ships with.
6. **How much was repaid matters as much as whether it was late.** Clients who paid nothing against an outstanding bill default at 51.8% against 13.9% for those who cleared it, and the pattern repeats in every month tested. It is computed from amounts, so the scale problem in finding 2 does not touch it.
7. **Recent delinquency is worth more than old delinquency.** Being late only in the last two months carries roughly 1.6 times the risk of being late only in the first two. The six months should not be weighted equally.
8. **Demographics are statistically significant and practically negligible.** Every one returns p below .001 with Cramer's V under 0.1, and age scores an AUC of 0.504 on its own. They buy almost no predictive power while carrying real fairness exposure, and sex in particular is a protected attribute in consumer lending.
9. **Age is U-shaped, not flat.** Risk falls from the mid-twenties to the low thirties then rises again. Linear summaries report nothing because the two halves cancel. Band it or drop it, but do not include it as a linear term.

**About the pipeline**

10. **The bill columns are collinear to the point of uselessness.** Variance inflation factors run past 15. Keep one, or replace the block with utilization.
11. **The cleaning is sound with three caveats.** 8 of 11 checks pass. Capping helps a linear model and costs resolution at the top of the payment distribution, the undocumented repayment codes were treated differently from equivalent codes elsewhere, and the capping bounds were fitted on all 30,000 rows rather than on training data.
12. **The conclusions do not depend on the cleaning.** Every headline result was recomputed on the raw file and the largest disagreement in rank correlation was 0.0031.

**Recommended starting point for model selection**

Begin with a baseline built only from repayment behaviour: the count of late months, the most recent status, and the share of the bill repaid. Add an inactive-account flag and banded age. Leave the demographic columns and five of the six bill columns out until they demonstrate they earn a place. Anything more elaborate should have to beat that baseline on real held-out clients before its extra complexity is accepted.
