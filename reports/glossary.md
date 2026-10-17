# Glossary of Test Scores and Models

Reference for the AAI-500 credit-card default project (UCI Default of Credit Card Clients). Numbers for the **selected model** come from [`model_findings.md`](model_findings.md). EDA score examples come from [`eda_findings.md`](eda_findings.md).

**Selected pipeline:** Random Forest + `kitchen_sink` features  
**Default rate (positive class):** ~22.1%  
**Negative class:** customers who did *not* default (~77.9%)

Think of the modeling task as a screening tool for the bank: each customer gets a risk score (probability of default). Metrics tell you how good that screening is; models are different recipes for producing the score.

---

## Quick reference

| Name | Type | Role in this project | Selected / typical value |
|---|---|---|---|
| PR-AUC (Average Precision) | Metric | **Primary** model selection | 0.5583 CV / 0.5554 test |
| ROC-AUC | Metric | Ranking quality (secondary) | 0.7844 CV / 0.7748 test |
| Precision | Metric | Of predicted defaulters, how many are real | 0.4363 (test) |
| Recall | Metric | Of actual defaulters, how many are caught | 0.6473 (test) |
| F1-score | Metric | Threshold tuning on train; reported on test | 0.5212 (test) |
| Accuracy | Metric | **Not** used for selection | 0.737 model vs 0.779 majority floor |
| Decision threshold | Protocol | Probability cutoff (F1-max on train) | 0.430 |
| Random Forest | Model | **Final chosen model** | `kitchen_sink` features |
| Hist Gradient Boosting | Model | Close runner-up | CV PR-AUC 0.5577 |
| Logistic Regression | Model | Interpretable linear baseline | CV PR-AUC ~0.51 |
| `baseline_eda` | Feature set | EDA-recommended starting point | Best CV PR-AUC 0.5412 |
| `extended` | Feature set | Adds limit / utilization / payments | Best CV PR-AUC 0.5551 |
| `kitchen_sink` | Feature set | Full financial panel (no demographics) | Best CV PR-AUC 0.5583 |
| Cramer's V | EDA score | Effect size for categorical association | PAY_1 = 0.423; demographics &lt; 0.1 |
| Cohen's d | EDA score | Effect size for group mean gaps | All &lt; 0.5 (LIMIT_BAL ≈ −0.38) |
| VIF | EDA score | Multicollinearity among bills | Bill columns peak ~15.5 |
| Univariate AUC | EDA score | Single-feature separation | MONTHS_DELINQUENT = 0.726 |

---

## 0. One picture that unlocks most metrics

Every hard “yes/no” prediction falls into one of four buckets:

```
                        Predicted NO default     Predicted DEFAULT
Actually NO default          TN (correct)           FP (false alarm)
Actually DEFAULTED           FN (missed risk)       TP (caught risk)
```

- **Precision** looks only at the *predicted DEFAULT* column: how trustworthy are the alarms?
- **Recall** looks only at the *actually DEFAULTED* row: how many risky customers did we catch?
- **Accuracy** looks at the whole table: how often are we correct overall?
- **F1** blends precision and recall into one number.

ROC-AUC and PR-AUC do **not** need a single cutoff. They evaluate the ranking of risk scores across many possible cutoffs.

---

## 1. Classification metrics (modeling test scores)

### Accuracy

**Plain English:** “Out of 100 predictions, how many were right?”

**Formula in words:** (correct predictions) ÷ (all predictions) = (TP + TN) ÷ N.

**Intuition:** Accuracy treats every customer the same. If most customers never default, you can look accurate by never raising an alarm.

**Worked intuition for this dataset:**
- About 78 out of 100 customers do not default.
- A lazy rule “predict no default for everyone” is ~**77.9% accurate**.
- That rule’s recall is **0%** — it never finds a defaulter.
- So a higher accuracy is *not* automatically a better credit-risk model.

**How to read a value:** Compare against the majority-class floor (~77.9%). If a model’s accuracy is near that floor but recall is high, it may still be useful. If accuracy is high only because it predicts “no default” almost always, it is not useful.

**This project:** Selected model hold-out accuracy = **73.7%** (below the 77.9% floor) while catching **64.7%** of defaulters. We deliberately do **not** pick models by accuracy.

---

### Precision

**Plain English:** “When the model says ‘this customer will default,’ how often is it right?”

**Formula in words:** TP ÷ (TP + FP) = correct default flags ÷ all default flags.

**Intuition:** Precision is about **false alarms**. Low precision means collections/risk teams waste effort on customers who were fine.

**Everyday analogy:** A spam filter with low precision dumps lots of real email into spam. A credit model with low precision flags lots of good customers as risky.

**Trade-off:** You can raise precision by being stricter (higher threshold) — fewer flags, but you miss more true defaulters (recall falls).

**How to read a value:**
- 1.0 = every flagged customer really defaults (rare in practice).
- 0.5 = half of flagged customers are true defaulters.
- Near the base rate (~0.22) with a random-ish rule is weak.

**This project (selected model, test):** **0.4363** — of 1,969 predicted defaulters, 859 were correct (~44% of alarms were real).

---

### Recall (sensitivity / true positive rate)

**Plain English:** “Of all customers who really defaulted, what fraction did we catch?”

**Formula in words:** TP ÷ (TP + FN) = caught defaulters ÷ all actual defaulters.

**Intuition:** Recall is about **missed risk**. Low recall means many defaulters slip through unlabeled.

**Everyday analogy:** Airport security with low recall lets threats through. For a bank, low recall means loss events you never triaged early.

**Trade-off:** You can raise recall by being looser (lower threshold) — more alarms, more false positives (precision falls).

**How to read a value:**
- 1.0 = every defaulter was flagged.
- 0.65 = about two-thirds of defaulters were flagged (selected model).
- 0.0 = the model never predicts default.

**Business note:** In many lending/collections settings, a false negative (miss) costs more than a false positive (unnecessary contact). That is why recall is reported carefully even when precision is moderate.

**This project (selected model, test):** **0.6473** — of 1,327 actual defaulters, 859 caught and 468 missed.

---

### F1-score

**Plain English:** “A single score that punishes you if either precision or recall is weak.”

**Formula in words:** Harmonic mean of precision and recall  
= 2 × (precision × recall) ÷ (precision + recall).

**Why harmonic mean?** The ordinary average can hide a bad side. Example: precision 1.0 and recall 0.01 averages to ~0.5, but F1 is ~0.02 — correctly saying the model is almost useless.

**Intuition:** F1 is a compromise metric when you care about both false alarms and misses, but do not have dollar costs for each error yet.

**Role in this project:** We scan many probability cutoffs on the **training** set and pick the threshold that maximises F1. That threshold is then frozen and applied to the test set.

**How to read a value:** F1 ranges from 0 to 1. Higher is better, but only at a stated threshold. Always read F1 together with the precision and recall that produced it.

**This project (selected model, test):** **0.5212** (precision 0.44 × recall 0.65 balanced together).

---

### ROC curve and ROC-AUC

**ROC curve (intuition):** Plot true positive rate (recall) against false positive rate as you slide the decision threshold from strict to loose. A good model bows toward the top-left (high recall, low false-alarm rate).

**ROC-AUC (plain English):** “If I pick one random defaulter and one random non-defaulter, how often does the model give the defaulter the higher risk score?”

**Scale:**
- **0.5** = no better than coin-flip ranking.
- **~0.7–0.8** = useful discrimination for many tabular problems.
- **1.0** = perfect separation.

**Strength:** Summarises ranking quality without committing to one threshold.

**Weakness with imbalance:** Because non-defaulters are common, ROC-AUC can look “pretty good” even when precision on the rare default class is mediocre. That is why this project leads with PR-AUC.

**This project (selected model):** CV **0.7844**; hold-out **0.7748** — solid ranking, not near-perfect.

---

### Precision–recall curve and PR-AUC (Average Precision)

**PR curve (intuition):** Plot precision against recall as the threshold slides. It focuses on performance for the **positive class** (default).

**PR-AUC / Average Precision (plain English):** “Across different levels of how aggressively we hunt for defaulters, how high does precision stay on average?”

**Critical difference from ROC-AUC:** A useless model’s PR-AUC sits near the **positive base rate** (~0.22 here), not near 0.5. So 0.55 PR-AUC is a real lift over ~0.22, even though the number looks “smaller” than a 0.78 ROC-AUC.

**Why it is our primary metric:**
1. Default is the minority class (~22%).
2. The business cares about finding defaulters without drowning in false alarms.
3. PR-AUC rewards models that keep precision up while recall rises.

**How to read a value on this dataset:**
- ~0.22 ≈ no skill (base rate).
- ~0.55–0.56 = selected / top models here.
- Higher is better; compare models on the same split.

**This project (selected model):** CV **0.5583 ± 0.0080**; hold-out **0.5554**.

---

### Decision threshold

**Plain English:** The cutoff that turns a probability into a yes/no label.  
Example: if threshold = 0.43, then P(default) ≥ 0.43 → “default”, else “no default.”

**Why 0.5 is not sacred:** With class weighting and imbalance, calibrated “best” cutoffs often move. 0.5 assumes equal concern for both error types and well-calibrated probabilities.

**What we did:** On training data, try many cutoffs → choose the one with best F1 → use that same cutoff on the test set (no peeking at test labels to retune).

**How to use in business:** A bank could replace F1-tuning with cost-aware tuning, e.g. minimise  
`(cost of miss × FN) + (cost of false alarm × FP)`.

**This project (selected model):** **0.430**.

---

### Confusion matrix terms (with project counts)

| Term | Plain English | Formula link | Selected model (test, N = 6,000) |
|---|---|---|---|
| **TP** | Caught a real defaulter | Helps recall & precision | 859 |
| **FP** | False alarm on a good customer | Hurts precision | 1,110 |
| **TN** | Correctly left a good customer alone | Helps accuracy | 3,563 |
| **FN** | Missed a real defaulter | Hurts recall | 468 |

**How to read the matrix:** Rows are truth; columns are predictions. Moving the threshold shifts mass between columns (more/less “predicted default”).

---

## 2. Validation and scoring protocol

These are not “scores” themselves; they are the rules for computing scores fairly.

### Stratified train/test split

**Plain English:** Shuffle the data, hold out 20% as a final exam, and keep the default rate similar in both piles so the exam is realistic.

**Why stratify?** A non-stratified split might accidentally put only 15% defaults in test and 25% in train. Then metrics jump around for the wrong reason.

**This project:** 80% train / 20% test; `random_state=42` (reproducible shuffle). Train defaults 22.12%; test 22.11%.

**Mental model:** Train = practice problems used to fit the model. Test = sealed exam used once for the report numbers labeled `test_*`.

---

### Stratified k-fold cross-validation (CV)

**Plain English:** Inside the training set, rotate which slice is used for temporary validation so you do not rely on one lucky split.

**How 5-fold works:**
1. Split train into 5 stratified folds.
2. Train on 4 folds, score on the held-out fold.
3. Repeat so each fold is validation once.
4. Average the five scores (`cv_*_mean`) and look at spread (`cv_*_std`).

**Why it matters:** Model selection (which algorithm + feature set wins) uses CV so we are less likely to crown a model that only lucked out on one validation slice.

**This project:** 5-fold; primary ranking key = `cv_average_precision_mean` (CV PR-AUC). Small `cv_*_std` means more stable performance across folds.

---

### Hold-out / test metrics

**Plain English:** After you pick a winner using CV on train, measure it one more time on the sealed 20% test set.

**Leaderboard prefixes:**
- `cv_…` — cross-validation on train (selection).
- `test_…` — final exam (`test_pr_auc`, `test_roc_auc`, `test_precision`, `test_recall`, `test_f1`).

**Healthy pattern:** CV and test tell a similar story. A huge CV→test drop suggests overfitting or an unstable pipeline.

---

### class_weight = 'balanced'

**Plain English:** During training, treat each default example as “more important” because defaults are rarer, so the model is pushed to care about them.

**What it does *not* do:** It does not create new data. It reweights the loss so mistakes on the minority class cost more.

**Side effect:** Predicted probabilities can shift, so a 0.5 cutoff may no longer be the best operating point — another reason we tune the threshold.

**This project:** Logistic regression & hist gradient boosting use `balanced`; random forest uses `balanced_subsample` (rebalance within each tree’s sample).

---

## 3. Models compared

All three models output a **risk score** (probability of default). They differ in *how* they combine features.

### Logistic Regression

**Plain English:** Assume risk rises/falls smoothly with a weighted sum of features (after a logistic curve maps that sum into a probability between 0 and 1).

**What it is good at:**
- Transparency: each coefficient says “holding other features fixed, this feature pushes default risk up or down.”
- Strong baseline when relationships are roughly linear/monotonic.

**What it struggles with:**
- Interactions (e.g. “late *and* high utilization”) unless you engineer them.
- Highly collinear inputs (six bill columns) — coefficients become unstable.

**Needs:** Numeric features scaled (`StandardScaler`) so coefficients are comparable.

**Role here:** Interpretable baseline. CV PR-AUC ~**0.51** — clearly behind the tree ensembles.

**When to prefer it:** Regulatory / explanation-heavy settings, or when you need a simple scorecard-like story.

---

### Random Forest

**Plain English:** Grow many decision trees on slightly different views of the data, then average their votes. Each tree asks a sequence of yes/no questions (“Is PAY_1 ≥ 2?”, “Is utilization high?”).

**Bagging (intuition):** One tree overfits quirks. Many trees on bootstrap samples + random feature choices cancel quirks and keep shared signal.

**What it is good at:**
- Non-linear patterns and interactions automatically.
- Mixed feature scales (no required standardisation).
- Redundant features (collinear bills) — trees can pick one and ignore others.

**What to watch:**
- Less directly interpretable than logistic coefficients (we use impurity/importance rankings instead).
- Can be biased toward features with many split points if importances are read naively.

**Role here:** **Selected final model** with `kitchen_sink`.  
**Selected scores:** CV PR-AUC **0.5583 ± 0.0080**; test PR-AUC **0.5554**; recall **0.6473**; precision **0.4363**.  
**Top drivers:** `MAX_DELINQUENCY`, `PAY_1`, `RECENT_LATE`, `MONTHS_DELINQUENT` — matches the EDA story that repayment behaviour dominates.

---

### Histogram Gradient Boosting (`HistGradientBoostingClassifier`)

**Plain English:** Build trees **one after another**. Each new tree focuses on customers the current ensemble still gets wrong (**boosting**). “Histogram” means continuous features are binned for speed.

**Boosting vs bagging:**
- Random forest = many independent trees averaged (parallel idea).
- Gradient boosting = a chain of trees that correct residuals (sequential idea).

**What it is good at:**
- Often strong predictive accuracy on tabular data.
- Captures complex non-linear structure.

**What to watch:**
- Can overfit if run too long / too deep without regularisation.
- Harder to explain than a single logistic model.

**Role here:** Very close runner-up (kitchen_sink CV PR-AUC **0.5577**, essentially tied with the forest within noise). Useful as a confirmation that tree ensembles beat linear baselines on this problem.

---

### Side-by-side intuition

| Model | Learns shape | Needs scaling? | Interpretability | Result here |
|---|---|---|---|---|
| Logistic regression | Mostly linear | Yes | Highest (coefficients) | Weakest PR-AUC |
| Random forest | Piecewise non-linear | No | Medium (importances) | **Winner** |
| Hist gradient boosting | Flexible non-linear | No | Medium (permutation / etc.) | Near-tie runner-up |

---

## 4. Feature sets (model input configurations)

A **feature set** is simply “which columns does the model see?” Same algorithm can look strong or weak depending on the inputs.

### Why we compare three sets

The EDA argued: start from repayment behaviour; only add complexity if it earns a measurable lift on held-out clients. The three sets are that experiment.

### baseline_eda

**Idea:** Small, interpretable, EDA-faithful set.

**Includes:** `MONTHS_DELINQUENT`, `PAY_1`, `PAID_SHARE`, `INACTIVE`, `RECENT_LATE`, `DISTANT_LATE`, banded `AGE_BAND`.

**Why these:**
- Late-month counts and recent status were the strongest signals.
- Paid-share uses amounts (safe from the PAY-scale inconsistency).
- Inactive accounts were *high* risk in EDA, not low.
- Age is U-shaped, so banding beats a raw linear age term.

**Best CV PR-AUC on this set:** **0.5412**.

### extended

**Idea:** Add a few financially meaningful extras without dumping every collinear bill.

**Adds:** `PAY_2`, `MAX_DELINQUENCY`, `LIMIT_BAL`, `UTILIZATION`, `PAY_AMT1`.

**Best CV PR-AUC:** **0.5551** (+0.0139 vs baseline) — extras earned a clear but modest lift.

### kitchen_sink

**Idea:** Give tree models the full financial history panel and let them choose.

**Includes:** limits, age, utilization, all PAY / BILL / PAY_AMT columns, engineered delinquency features.  
**Excludes:** SEX, EDUCATION, MARRIAGE (weak EDA effect sizes + fairness / protected-attribute concerns).

**Best CV PR-AUC:** **0.5583** (+0.0171 vs baseline). Used by the selected random forest.

**How to talk about it in the report:** Trees can tolerate collinear bills better than logistic regression; the lift over `extended` is small, so the win is “full financial behaviour,” not “demographics.”

---

## 5. EDA statistical scores

These scores answer: “Is this variable related to default, and *how strongly*?” They guided feature engineering before any classifier was trained.

### Why both a test *and* an effect size?

With **N = 30,000**, almost any tiny difference can get p &lt; .001.  
A significant p-value only says “probably not exactly zero.”  
An **effect size** says “is the difference big enough to care about for prediction or policy?”

---

### Chi-square test of independence

**Plain English:** “Do default rates differ across categories of this variable more than random noise would allow?”

**Setup:** Build a contingency table (e.g. EDUCATION levels × default yes/no). Chi-square measures how far observed cell counts sit from the counts you’d expect if default were independent of EDUCATION.

**Outputs:**
- **chi-square statistic** — larger ⇒ bigger departure from independence.
- **p-value** — probability of seeing such a departure if there were truly no association.
- **degrees of freedom** — depends on table size.

**Assumption to remember:** Expected counts in cells should generally be ≥ 5. Sparse PAY_1 tail categories violated this; demographics were fine.

**How to use:** A significant chi-square is a green light to *look* at the variable — not automatic proof it belongs in the model.

---

### Cramer's V

**Plain English:** “On a 0–1-ish scale, how strong is the categorical association?”

**Why it exists:** Companion effect size for chi-square so huge samples do not fool you.

**Rough reading guide (rules of thumb):**
- ~0.0–0.1: weak
- ~0.1–0.3: moderate
- &gt; ~0.3: relatively strong for social/financial tabular associations

**This project:**
- SEX / EDUCATION / MARRIAGE: V &lt; **0.1** (statistically significant, practically weak).
- PAY_1: V = **0.423** (dominant categorical signal — belongs in modeling).

---

### Welch's t-test

**Plain English:** “Do defaulters and non-defaulters have different *average* values of this numeric variable?”

**Why Welch (not Student’s classic t):** Welch does **not** assume the two groups share the same variance. That matters because the default group is smaller and financial variables are skewed.

**Limitation:** Compares means. For right-skewed money variables, means are pulled by extreme tails — so we also report a rank test and an effect size.

---

### Mann–Whitney U test

**Plain English:** “If I pick one defaulter and one non-defaulter, is one group systematically higher on this variable?”

**Why use it:** It compares **distributions via ranks**, not means, so extreme bills/payments do not dominate as badly.

**Link to AUC:** Mann–Whitney U is mathematically tied to ROC-AUC for a single feature. That is why univariate AUCs in the EDA are a clean “separation” summary.

---

### Cohen's d

**Plain English:** “How large is the gap between group means, in standard-deviation units?”

**Formula in words:** (mean_defaulters − mean_non_defaulters) ÷ (pooled standard deviation).

**Rough guide:**
- |d| ≈ 0.2 small  
- |d| ≈ 0.5 medium  
- |d| ≈ 0.8 large  

**This project:** Nothing reached 0.5. LIMIT_BAL was largest in magnitude (~**−0.38**: defaulters had lower limits). Age had a significant p-value but d ≈ **0.03** — a textbook “significant but tiny” result.

---

### Spearman correlation

**Plain English:** “Do higher ranks on this feature tend to go with higher (or lower) chance of default?”

**Vs Pearson:** Spearman uses ranks, so it captures steady up/down relationships even if they are not straight lines, and it is less enslaved to extreme values.

**Scale:** −1 to +1.  
- Near 0: little monotonic association.  
- Positive: higher feature ↔ more default (e.g. late statuses).  
- Negative: higher feature ↔ less default (e.g. larger payments / higher limit).

**This project:** `MONTHS_DELINQUENT` ~**0.39**, `PAY_1` ~**0.29**; `PAY_AMT*` and `LIMIT_BAL` negative. Bill columns correlate strongly *with each other* (redundancy), not strongly with the target.

---

### VIF (Variance Inflation Factor)

**Plain English:** “If I try to use this feature in a regression, how much is it ‘double-counting’ information already in the other features?”

**How to read:**
- ~1: essentially independent of the other predictors.
- &gt; 5: caution — coefficients get noisy.
- &gt; 10: serious multicollinearity — linear models become unstable.

**This project:** BILL_AMT* VIFs peak around **15.5**; LIMIT_BAL and AGE near **1**. Practical takeaway: do not put all six bill columns into logistic regression; use utilization or one bill, or let a tree model handle redundancy.

---

### Univariate AUC (single-feature AUC)

**Plain English:** “If this one column were the entire model, how well would it rank defaulters above non-defaulters?”

**Scale:** Same as ROC-AUC (0.5 = useless, 1.0 = perfect).

**Why it is useful before modeling:** Builds a fair leaderboard of raw signals without yet choosing an algorithm. Correlation can miss U-shaped patterns; univariate AUC (rank-based) can also miss them if higher≠riskier in one direction only — which is why age looked useless as a raw score (AUC ≈ 0.50) even though banded age rates are U-shaped.

**This project:**
- `MONTHS_DELINQUENT` **0.726** (strongest single signal; engineered, not raw).
- Repayment block high; demographics near the bottom.
- AGE ≈ **0.504** as a linear/rank score — band it or drop it, don’t treat it as a straight line.

---

## 6. How to read the modeling leaderboard

1. **Select** with CV PR-AUC (`cv_average_precision_mean`) — best for imbalance.  
2. **Check stability** with `cv_average_precision_std` — smaller is steadier.  
3. **Confirm** on hold-out `test_*` metrics — watch for big CV→test drops.  
4. **Report operating point** with precision, recall, F1 at the **tuned threshold**, not AUC alone.  
5. **Prefer simplicity** when PR-AUC gaps are tiny; here `kitchen_sink` beat `baseline_eda` by ~0.017 PR-AUC, so the fuller financial set was justified for the tree model.  
6. **Tell the business story in recall/precision language:** “We catch about 65% of defaulters; about 44% of our alarms are real,” not only “AUC = 0.78.”

---

## 7. Mini FAQ

**Q: Why is accuracy lower than the majority baseline but the model still useful?**  
Because usefulness here is catching defaulters (recall) and ranking risk (PR-AUC/ROC-AUC), not maximising overall correctness.

**Q: Why does PR-AUC look “lower” than ROC-AUC?**  
Different baselines. ROC-AUC chance level is 0.5; PR-AUC chance level is the positive rate (~0.22). Compare each metric to its own baseline.

**Q: Why exclude SEX from models?**  
EDA: tiny effect size. Lending: protected attribute / fairness exposure. Weak predictive gain is not worth the risk.

**Q: Why not use a PAY_1 − PAY_6 “improvement” feature?**  
EDA showed status code `1` is basically present only in PAY_1. Differencing months partly measures a recording quirk, not true behavioural improvement.
