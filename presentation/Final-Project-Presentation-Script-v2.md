# Final Project Presentation Script

**Team 10 · AAI-500 · Predicting Credit Card Default**  
**Length:** 1257 spoken words ≈ **8:59** at 140 wpm (brief requires 8–10 minutes, do not exceed 10)  
**Audience:** non-technical business leaders  
**Record as:** `Final-Project-Presentation-Team-10.mp4`

---

## Why there are two scripts

`Final-Project-Presentation-Script.md` is the first version and stays in place. This v2 is an
alternative, not a replacement — pick one as a team.

What changed and why:

- **Length.** v1 aims at ~7.5 minutes. The brief asks for **8–10 minutes**, and under 8 counts
  against us on the rubric the same way as over 10. v2 runs 8:59.
- **Speakers.** v1 splits the talking four ways. Prasad has no commits in this repository and his
  assigned Introduction was, by the report's own note, compiled by the team. v2 splits the talking
  three ways, in equal thirds (33.7% / 33.5% / 32.8%), and gives each person the section they
  actually worked on. If Prasad does present, move slides 1–3 to him and the thirds still hold.
- **Visuals.** v2 puts one of our own figures on 10 of the 13 slides, with the file path named for
  each, rather than describing a chart in words.
- **Numbers.** Every figure in v2 is taken from `reports/Final-Project-Report-Team-10.md`,
  `reports/eda_findings.md` or `reports/model_findings.md`. None are approximated.

A matching deck is in this folder: `Final-Project-Presentation-Team-10.pptx` (this script is in
every slide's notes field) and `Final-Project-Presentation-Team-10-slides.pdf`.

---

## How to use this

1. Speak the **Script** lines. Put only the **On screen** lines on the slide — never read a bullet aloud word for word.
2. Slides, figures and these notes are already in `Final-Project-Presentation-Team-10.pptx` (every slide's notes field).
3. Rehearse once out loud with a timer. Reading in your head runs ~25% fast.
4. There is about one minute of headroom under the 10-minute ceiling. It is insurance, not room to improvise.

## Speaking split

Equal thirds, and each person presents the work they actually did.

| Speaker | Slides | Words | Time | Share | Covers |
| --- | --- | --- | --- | --- | --- |
| **Siddhant** | 1–4 | 424 | 3.03 min | 33.7% | Opening, business problem, the data, data quality |
| **Eshwar** | 5–8 | 421 | 3.01 min | 33.5% | The coding flaw, what predicts default, what does not, signal ranking |
| **Umesh** | 9–12 | 412 | 2.94 min | 32.8% | Model choice, performance, limits, recommendations |
| *Credits* | 13 | — | ~0:15 | — | Each member reads their own line |

---

# Siddhant — slides 1–4 (3.03 min)

## Slide 1 — Can we see a missed payment coming?

*0:00–0:35 · 86 words*

**On screen**

- *Subtitle:* Predicting credit card default  ·  AAI-500 Final Team Project
- Team 10 — Prasad Pidaparty · Siddhant Bhura · Eshwar Reddy M · Umesh Sharma

**Script (Siddhant):**

> Good morning. Credit card lenders lose money when customers stop paying, and the valuable moment is always before the miss, while the bank can still offer a payment plan, review a limit, or set money aside. Our question was whether a customer's recent card history can tell us who is about to miss next month's payment. Over the next nine minutes we will show you what the data said, how well a model works, and what a bank should and should not do with the answer.

---

## Slide 2 — The decision we are supporting

*0:35–1:25 · 110 words*

**On screen**

- *Subtitle:* Not a verdict on a customer — a ranked list for a team with limited time
- A collections team can call a fixed number of customers each month
- Calling the wrong ones wastes the budget; missing the right ones costs the balance
- So the model's job is to ORDER customers by risk, not to pass judgement
- Every design choice in this project follows from that one sentence

**Script (Siddhant):**

> It matters that we were precise about the decision before touching the data. A lender's collections team can only contact so many people in a month. Calling the wrong customers wastes that budget, and missing the right ones costs the entire outstanding balance, so the two kinds of mistake are not equally expensive. The job of this model is therefore not to deliver a verdict on any individual customer. It is to order customers by risk, so that a fixed and limited amount of human attention lands where it is worth the most. Every design choice we made follows from that one sentence, including which measures we refused to report.

---

## Slide 3 — What we looked at

*1:25–2:15 · 97 words*

**On screen**

- *Subtitle:* 30,000 customers · six months of history · one in five missed the next payment
- Public UCI dataset from a Taiwanese bank, April to September 2005
- Credit limit, demographics, six monthly bills, six monthly payments, six lateness codes
- 6,636 of 30,000 customers defaulted — 22.1%
- That imbalance sets a trap we will come back to at the end

**Visual:** `reports/figures/01_class_balance.png`

**Script (Siddhant):**

> We used a public dataset of thirty thousand credit card customers from a Taiwanese bank, covering six months of account history: the credit limit, some demographics, six monthly bills, six monthly payments, a record of how late each payment was, and crucially what actually happened the month after. Six thousand six hundred of them defaulted, which is twenty-two percent, or roughly one customer in five. Hold on to that number. It is the reason we rejected the most obvious way of judging a model, and it sets a trap we will close in the final two minutes.

---

## Slide 4 — Is the data trustworthy?

*2:15–3:05 · 131 words*

**On screen**

- *Subtitle:* We checked rather than assumed — and we checked our own cleaning too
- No missing values and no duplicate records anywhere in 30,000 rows
- Fixed: headings on the wrong row, a misnamed column, undocumented category codes
- Extreme amounts were kept, not capped, so over-limit customers stay visible
- Re-ran every conclusion on the untouched raw file — the answers did not move

**Visual:** `reports/figures/12_cleaning_validation.png`

**Script (Siddhant):**

> Before trusting any finding we had to trust the data. There were no missing values and no duplicate records anywhere in the thirty thousand rows, and we verified both rather than assuming. Three structural problems needed fixing: column headings sitting on the wrong row, a column named so that it breaks the ordering of the six months, and a few hundred customers carrying category codes that appear nowhere in the documentation. We also made one deliberate choice. We kept the extreme amounts rather than capping them, which is why over-limit customers stay visible: seven percent of the book, defaulting at thirty percent against twenty-two overall. Then we re-ran every conclusion on the untouched raw file. The answers did not move, so our findings describe credit behaviour, not our own cleaning choices.

---

# Eshwar — slides 5–8 (3.01 min)

## Slide 5 — The flaw we found before the model did

*3:05–4:00 · 117 words*

**On screen**

- *Subtitle:* The six monthly lateness codes are not the same ruler
- "One month late" appears 3,688 times in September and ZERO times in April and May
- A naive trend feature says customers who "improved" default at 26% vs 16% for stable
- They never improved — the code changed meaning underneath them
- So we built no trend feature, and used only measures immune to the problem

**Visual:** `reports/figures/05_pay_scale_anomaly.png`

**Script (Eshwar):**

> One finding changed what we built. The six monthly payment-status records are supposed to use a single coding system. They do not. The code meaning one month late appears three thousand six hundred times in the most recent month and not once in the two oldest. That nearly produced a serious error. Measure whether a customer's status improved across the six months, and the improving customers default at twenty-six percent against sixteen for the stable ones, so improvement appears to predict failure. It does not. Those customers were never improving; the code changed meaning underneath them. So we built no trend feature. Without that check we would have shipped a filing convention and called it customer behaviour.

---

## Slide 6 — What actually predicts a missed payment

*4:00–4:55 · 114 words*

**On screen**

- *Subtitle:* Recent payment behaviour, and almost nothing else
- Current on the last payment: 13.8% default.  One month late: 34%.  Two months: 69%
- Count of late months runs 11.7% (never) to 70.3% (all six) with no reversals
- Paid nothing against the bill: 51.8%.  Cleared it: 13.9%
- Late recently 41.6% vs late long ago 26.0% — recency carries ~1.6x the risk

**Visual:** `reports/figures/04_repayment_history.png`

**Script (Eshwar):**

> The strongest predictor is also the most intuitive: how late the customer was on their most recent payment. Customers who are current default about fourteen percent of the time. One month behind, thirty-four percent. Two months behind, sixty-nine percent. So a customer two months behind is more likely than not to default, and that is a policy you could write tomorrow without any model at all. Two refinements sharpen it. Counting how many of the six months were late gives a clean ladder from twelve percent to seventy, with no reversals. And how much they paid matters as much as whether they were late: paying nothing means fifty-two percent, clearing the bill means fourteen.

---

## Slide 7 — What does NOT predict it

*4:55–5:45 · 103 words*

**On screen**

- *Subtitle:* Statistically significant, practically useless — and a fairness risk
- Age, sex, education, marital status: all significant at p < .001, all effect sizes under 0.08
- Age on its own scores 0.504 — indistinguishable from a coin flip
- Because risk is U-shaped: 26.7% in the early 20s, 19.4% in the low 30s, 26.8% past 60
- We excluded demographics from the final model entirely

**Visual:** `reports/figures/03_demographics.png`

**Script (Eshwar):**

> What turned out not to matter is just as useful. Age, sex, education and marital status are all statistically significant, because with thirty thousand records almost anything is, but every effect size is tiny. Age alone scores barely above a coin flip, and the reason is instructive: risk is U-shaped, falling from twenty-seven percent in the early twenties to nineteen in the low thirties and climbing back past sixty. As a straight line the halves cancel and the variable looks worthless. We excluded demographics entirely. They buy almost no accuracy and carry real fairness exposure, since sex is a protected attribute in lending.

---

## Slide 8 — Ranking every signal honestly

*5:45–6:35 · 87 words*

**On screen**

- *Subtitle:* The dataset does not ship with its own best predictor — we had to build it
- Each variable scored alone, on its own, against the outcome
- Best single signal is the COUNT of late months, at 0.726 — and it is engineered, not given
- The repayment block fills the top of the list; demographics fill the bottom
- The six bill columns repeat each other (correlations 0.80–0.95), so we keep one

**Visual:** `reports/figures/10_predictive_power.png`

**Script (Eshwar):**

> To decide what deserved a place we scored every variable on its own against the outcome. Two things stood out. First, the strongest single signal is the count of late months, and that column does not exist in the data as delivered — we engineered it. The dataset does not ship with its own best predictor. Second, there is a clean gap rather than a gradual decline: the repayment measures fill the top of the list and the demographics fill the bottom, with very little in between.

---

# Umesh — slides 9–12 (2.94 min)

## Slide 9 — How we chose the model

*6:35–7:25 · 105 words*

**On screen**

- *Subtitle:* Nine combinations, judged on ranking quality — never on accuracy
- Three model families × three input sets, each validated on data it had never seen
- A transparent statistical model, two tree-based methods, plus a neural network
- Scored on how well each ORDERS customers by risk, not on accuracy
- Tree-based methods won; the top few are close enough to be a practical tie

**Visual:** `reports/report_figures/model_03_leaderboard.png`

**Script (Umesh):**

> We did not pick a model and hope. We tested three families across three sets of inputs, nine combinations, each validated on customers it had never seen, then added a neural network as a fourth. Two decisions made the comparison fair. We refused to score on accuracy, for reasons about to become obvious, and scored instead on how well each model orders customers by risk. And we tuned the decision cut-off rather than accepting a default. The tree-based methods beat the transparent statistical model, and the top few finished close enough that we treat them as a tie and chose the one easiest to explain.

---

## Slide 10 — How well it works

*7:25–8:15 · 106 words*

**On screen**

- *Subtitle:* On 6,000 customers the model had never seen
- Found 738 of 1,327 real defaulters — 55.6%
- Flagged 1,411 customers; 738 of them genuinely defaulted — 52.3%
- Flagging 1,411 AT RANDOM would have caught about 312
- So the same amount of effort catches 2.4x as many real defaulters

**Visual:** `reports/report_figures/model_02_confusion_matrix.png`

**Script (Umesh):**

> So how well does it work. On six thousand customers the model had never seen, it found seven hundred and thirty-eight of the one thousand three hundred and twenty-seven people who went on to default. Of the customers it flagged, just over half genuinely defaulted. Here is the number that matters to a business: flagging the same fourteen hundred customers at random would have caught about three hundred and twelve real defaulters. Our model catches seven hundred and thirty-eight. For the same amount of human effort, the team reaches roughly two and a half times as many of the people who were actually about to default.

---

## Slide 11 — The trap, and what the model cannot do

*8:15–9:05 · 107 words*

**On screen**

- *Subtitle:* Why we never once reported accuracy
- Our accuracy is 79.0%. Always answering "no default" scores 77.9%
- By the measure most people reach for first, the model is barely worth running
- By the measure that matches the decision, it is clearly worth running
- It cannot say a given customer WILL default — 70% flagged still has a 30% chance of paying

**Visual:** `reports/report_figures/model_01_roc_pr_curves.png`

**Script (Umesh):**

> And now the trap closes. Our accuracy is seventy-nine percent. Always answering no default scores seventy-seven point nine. By the measure most people reach for first, this model is barely better than doing nothing, which is exactly why we never reported it. By the measure that matches the decision, it is clearly worth running. That gap is the most important thing we learned. Two honest limits. It cannot tell you a particular customer will default; one flagged at seventy percent still has a thirty percent chance of paying, so this triggers a call, never an automatic closure. And this is one bank, one market, twenty years ago.

---

## Slide 12 — What we recommend

*9:05–9:45 · 94 words*

**On screen**

- *Subtitle:* Four things a lender should do with this
- Deploy it as a ranked monthly outreach list — never as an automated decision
- Re-set the cut-off from the bank's own costs; ours is tuned for statistical balance
- Fix the lateness coding — it would make five more months of history usable
- Treat "two months behind" as a trigger on its own; that group defaults more often than not

**Visual:** `reports/report_figures/model_04_feature_importance.png`

**Script (Umesh):**

> Four recommendations. Deploy it as a ranked monthly outreach list, not an automated decision, because the model orders risk well but is not good enough to act unsupervised. Re-set the cut-off from the bank's own costs; ours is tuned for statistical balance rather than for money. Fix the lateness coding we showed earlier, which would make five more months of history usable and is the cheapest improvement available. And treat two months behind as a trigger in its own right, since that group defaults more often than not whatever the model says. Thank you.

---

# Slide 13 — Who did what

*Required by the brief: one slide naming each member and their contributions.*

| Team member | Contribution |
| --- | --- |
| Siddhant Bhura | Data cleaning and preparation · neural network analysis · technical report |
| Eshwar Reddy M | Exploratory data analysis · regression assumption testing · presentation |
| Umesh Sharma | Model development, comparison and validation · reproducibility pipeline |
| Prasad Pidaparty | Dataset selection and business problem definition (project plan stage) |

**Script:** each member reads their own line, then: *"Thank you — questions welcome."*

---

## Checks before recording

- [ ] Confirm the team number. Canvas lists this group as **Group 4**; the repo and report say **Team 10**. The filenames depend on it.
- [ ] Settle which model run to report. `reports/model_findings.md` names random forest; the report names histogram gradient boosting. This script uses the **report's** numbers.
- [ ] Export MP4, check the audio, keep the total under 10:00.
- [ ] One member submits the report PDF and the MP4 separately on Canvas.
