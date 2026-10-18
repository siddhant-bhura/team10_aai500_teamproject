# Model Selection and Model Analysis

UCI Default of Credit Card Clients — modeling stage. Built on the cleaned file from `data_cleaning.py` and the feature guidance in `eda_findings.md`.

Author: Umesh

## 1. Setup

- Stratified hold-out: 80% train / 20% test (random_state=42).
- Train defaults: 22.12% · Test defaults: 22.11%.
- Selection metric: **average_precision** (PR-AUC), with ROC-AUC, precision, recall and F1 reported alongside.
- All estimators use class_weight='balanced' (or balanced_subsample for the forest) so the minority class is not ignored.

## 2. Feature sets

- **baseline_eda**: 6 numeric, 1 categorical (AGE_BAND) — MONTHS_DELINQUENT, PAY_1, PAID_SHARE, INACTIVE, RECENT_LATE, DISTANT_LATE
- **extended**: 11 numeric, 1 categorical (AGE_BAND) — MONTHS_DELINQUENT, PAY_1, PAY_2, PAID_SHARE, INACTIVE, RECENT_LATE...
- **kitchen_sink**: 27 numeric — LIMIT_BAL, AGE, UTILIZATION, MONTHS_DELINQUENT, MAX_DELINQUENCY, PAID_SHARE...

- `baseline_eda` is the EDA-recommended starting point. `extended` adds limit/utilization/payment amount. `kitchen_sink` adds the full bill/payment panel (still excluding SEX/EDUCATION/MARRIAGE) as a complexity check against collinear financial columns.

## 3. Cross-validated comparison

Cross-validated leaderboard (sorted by CV PR-AUC):
```
 feature_set                  model  cv_average_precision_mean  cv_average_precision_std  cv_roc_auc_mean  cv_f1_mean  cv_recall_mean  cv_precision_mean
kitchen_sink          random_forest                     0.5583                    0.0080           0.7844      0.5445          0.5994             0.4989
kitchen_sink hist_gradient_boosting                     0.5577                    0.0090           0.7810      0.5385          0.6269             0.4720
    extended          random_forest                     0.5551                    0.0083           0.7815      0.5391          0.6105             0.4827
    extended hist_gradient_boosting                     0.5531                    0.0080           0.7794      0.5362          0.6188             0.4732
baseline_eda hist_gradient_boosting                     0.5412                    0.0085           0.7580      0.5337          0.6084             0.4754
baseline_eda          random_forest                     0.5395                    0.0104           0.7579      0.5290          0.6076             0.4685
baseline_eda    logistic_regression                     0.5101                    0.0121           0.7443      0.5244          0.5786             0.4795
kitchen_sink    logistic_regression                     0.5094                    0.0143           0.7638      0.5243          0.6144             0.4572
    extended    logistic_regression                     0.5090                    0.0138           0.7615      0.5244          0.6114             0.4591
```

Hold-out test metrics (threshold tuned for F1 on the training set):
```
 feature_set                  model  threshold  test_pr_auc  test_roc_auc  test_precision  test_recall  test_f1
kitchen_sink          random_forest      0.430       0.5554        0.7748          0.4363       0.6473   0.5212
kitchen_sink hist_gradient_boosting      0.541       0.5501        0.7774          0.5039       0.5780   0.5384
    extended          random_forest      0.522       0.5546        0.7718          0.4909       0.5720   0.5284
    extended hist_gradient_boosting      0.559       0.5539        0.7708          0.5182       0.5584   0.5375
baseline_eda hist_gradient_boosting      0.532       0.5343        0.7547          0.4997       0.5674   0.5314
baseline_eda          random_forest      0.518       0.5415        0.7532          0.4901       0.5787   0.5308
baseline_eda    logistic_regression      0.565       0.5081        0.7311          0.5196       0.5283   0.5239
kitchen_sink    logistic_regression      0.542       0.4994        0.7489          0.4766       0.5592   0.5146
    extended    logistic_regression      0.565       0.4980        0.7447          0.4876       0.5479   0.5160
```

**Selected model:** `random_forest` on feature set `kitchen_sink` (CV PR-AUC = 0.5583 ± 0.0080; test PR-AUC = 0.5554).

## 4. Selected-model analysis

- Pipeline: **random_forest** · features: **kitchen_sink** · decision threshold: **0.430** (F1-maximising on train).

Classification report on the hold-out test set:
```
              precision    recall  f1-score   support

  No default      0.884     0.762     0.819      4673
     Default      0.436     0.647     0.521      1327

    accuracy                          0.737      6000
   macro avg      0.660     0.705     0.670      6000
weighted avg      0.785     0.737     0.753      6000
```

Confusion matrix (rows = actual, columns = predicted):
```
                 Pred No-default   Pred Default
Actual No-default          3563            1110
Actual Default              468             859
```

- Of 1,327 actual defaulters, the model catches 859 (64.7% recall) and misses 468.
- Of 1,969 predicted defaulters, 859 are correct (43.6% precision).
- For a collections triage use-case, missing a defaulter (FN) is usually costlier than a false alarm (FP), so the F1-tuned threshold is a compromise rather than a business-cost optimum. A bank would re-tune against its own loss/contact costs.

- Majority-class accuracy floor: 77.9%. Selected model accuracy: 73.7%. Accuracy alone would have preferred the majority classifier; PR-AUC / recall are the relevant numbers.

## 5. Interpretation

Random forest impurity-based feature importances (top 12):
```
num__MAX_DELINQUENCY      0.1122
num__PAY_1                0.1100
num__RECENT_LATE          0.1038
num__MONTHS_DELINQUENT    0.1018
num__PAY_2                0.0453
num__UTILIZATION          0.0424
num__LIMIT_BAL            0.0351
num__BILL_AMT1            0.0338
num__PAY_AMT2             0.0313
num__PAY_AMT1             0.0303
num__PAY_AMT3             0.0299
num__PAY_3                0.0285
```

## 6. Did extra features earn their place?

- Best CV PR-AUC by feature set — baseline_eda: 0.5412, extended: 0.5551, kitchen_sink: 0.5583.
- Extended vs baseline lift: +0.0139. Kitchen-sink vs baseline lift: +0.0171.
- Additional features improve PR-AUC. Report the best set, but note that the bill block is highly collinear (EDA VIFs > 10), so coefficients from a linear model on this set would be unstable even though tree ensembles handle the redundancy better.

## 7. Limitations

1. **Skewed financial tails are kept as-is.** Cleaning no longer winsorizes credit limits, bills or payments. Tree models handle that skew naturally; a linear model may still want log1p or robust scaling fitted inside the training fold only.
2. **Threshold is not cost-aware.** The cutoff maximises F1 on the training set. A lender should replace that with expected loss = FN_cost × miss_rate + FP_cost × false_alarm_rate.
3. **Protected attributes.** SEX, EDUCATION and MARRIAGE are excluded from every feature set, including the selected `kitchen_sink` model. The EDA found them statistically significant but practically negligible, and SEX in particular is a protected attribute in consumer lending.
4. **PAY-scale inconsistency.** Features that count any late month or use amount-based PAID_SHARE are safe; a PAY_1 − PAY_6 trend was deliberately excluded per the EDA.
5. **External validity.** The data is a single Taiwanese issuer in 2005. Performance here is not a claim about current US / EU books.

## 8. Recommendations

1. **Ship the selected pipeline** (`random_forest` + `kitchen_sink`) as the project model: it is the PR-AUC leader under stratified CV and aligns with the EDA story.
2. **Operationalise a risk score, not only a hard label.** Rank accounts by predicted probability and apply policy cutoffs per segment (limit reduction, outreach, reserve).
3. **Refit preprocessing inside the train split** before any deployment claim, and monitor calibration on recent cohorts.
4. **Keep demographics out of the decisioning model** unless a documented fairness review shows they improve outcomes without disparate impact — the EDA says they buy almost nothing.

## 9. Summary

- Final model: **random_forest** with **kitchen_sink** features.
- Cross-validated PR-AUC: **0.5583** (± 0.0080); ROC-AUC: **0.7844**.
- Hold-out PR-AUC: **0.5554**, precision **0.4363**, recall **0.6473**, F1 **0.5212** (threshold 0.430).
- This beats a base-rate classifier on ranking quality (PR-AUC above the ~0.221 positive rate) and recovers a meaningful share of defaulters that accuracy-only selection would miss.

