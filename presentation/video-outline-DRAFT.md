# Video outline and script: DRAFT for Eshwar to adopt or replace

Team 10, AAI-500 Final Project. Target length 8 to 10 minutes. Audience: non-technical.

## Notes for the team (not spoken)

- Presenter split is a proposal so that all four members speak, as the assignment requires (equal share). The first draft gave Prasad nothing and Umesh the most. Now, by spoken words: Prasad slides 1, 2 and 9 (305 words), Siddhant slides 3 and 7 (382), Eshwar slides 4, 5 and 10 (320), Umesh slides 6 and 8 (376), plus a short credits line each on slide 11. That is about 2:00 to 2:30 each at 150 words a minute. The team should confirm or swap. Nothing in this file says anyone recorded anything yet.
- Every number comes from `reports/Final-Project-Report-Team-10.md`, which draws on `reports/eda_findings.md` and a fresh run of `src/modeling.py` in the pinned environment.
- Model numbers (slides 6 to 9) are from that fresh run: histogram gradient boosting, exam-set PR-AUC 0.5499. The saved `reports/model_findings.md` is from an earlier run that picked random forest (0.5554). The difference traces to the scikit-learn version: 1.7.2 reproduces the saved file, 1.9.1 (pinned) gives the fresh result. The two top models are within noise of each other, so the script does not call either one "the best". The team decision on which run to quote is pending. If the team picks the earlier run, slides 6 to 9 and the report must both change.
- The 50 percent and 14 percent figures on slide 4 compare "one or more months late" against "current". The exact one-month and two-month rates are 33.95 and 69.14 percent. The older EDA text labels 50.29 percent as "one month late", which is not correct.
- Slide 3 sentence on duplicates: 35 rows repeat another row's features and outcome, and were kept.
- The neural network (slide 7) is a scikit-learn MLPClassifier, hidden layers (128, 64), on the `extended` inputs. Its numbers match report Sections 4.5 and 5.7 and come from `reports/neural_network/model_comparison.csv` and `neural_network_results.md`. The training-check figures there use the same folds the network settings were chosen on, so they are slightly generous. The "simple formula" is logistic regression.
- Credits on slide 11 state assigned roles only. Each member should approve their own line.
- Spoken script is about 1430 words, roughly 9.5 minutes at 150 words per minute. This is an estimate, not a timed read.

## Slide plan

| # | Slide | Presenter | Time |
|---|---|---|---|
| 1 | The question | Prasad | 0:45 |
| 2 | The data | Prasad | 0:45 |
| 3 | Cleaning the data | Siddhant | 1:00 |
| 4 | What predicts default | Eshwar | 1:00 |
| 5 | What does not, and a surprise | Eshwar | 0:50 |
| 6 | How we built the model | Umesh | 0:55 |
| 7 | The neural network | Siddhant | 0:55 |
| 8 | How well it works | Umesh | 1:00 |
| 9 | Limits | Prasad | 0:50 |
| 10 | Recommendations | Eshwar | 0:40 |
| 11 | Who did what | Each member reads their own line | 0:30 |

Approximate speaking time per member, from the table (words in brackets): Prasad 2:20 (305), Siddhant 2:35 (382), Eshwar 2:20 (320), Umesh 2:30 (376), plus about 10 seconds each on slide 11. These are planning figures. Adjust after a timed read.

## Slide 1. The question (Prasad)

Visual: one sentence on screen: "Can we spot who will miss next month's payment?" Small icon of a credit card.

Script:
"Credit card companies lose money when customers stop paying. If a bank knew who was likely to miss next month's payment, it could reach out early. It could offer a payment plan, or lower a credit limit. Today we show how well that can be done from past billing and payment records. We are Team 10. The team has four members: Prasad, Siddhant, Eshwar and Umesh."

## Slide 2. The data (Prasad)

Visual: three numbers: 30,000 customers, 23 facts about each, 22% defaulted. Note: Taiwan, 2005.

Script:
"We used a public dataset of 30,000 credit card customers from a bank in Taiwan in 2005. For each customer we have 23 facts. These include age, credit limit, and six months of bills and payments. We also know whether they missed the following month's payment. About 22 in every 100 customers did. That is roughly one in four or five. Remember this number, because it matters later. The data covers April to September 2005. Our job is to turn that history into an early warning."

## Slide 3. Cleaning the data (Siddhant)

Visual: checklist. Missing values: 0. Duplicate IDs: 0. Extreme values: preserved (32,425 outside IQR fences, not trimmed). Unclear codes: 399 rows regrouped.

Script:
"Before any analysis, we checked the data for problems. Think of it as inspecting ingredients. First, nothing was missing. Every customer had a complete record. Second, no customer appeared twice, which we checked by ID and by full row. A few customers share an identical profile, 35 in all, but they have different IDs, so we kept them.

Third, some bills and payments were extremely large. That is normal for money data. We counted how many values sit outside the usual boxplot fences — about 32,425 cells, roughly 8 percent of the dollar columns — but we did not trim or delete them. Keeping the tails means over-limit and very large payment accounts stay distinguishable for the models.

Fourth, some education and marriage entries used codes the data guide does not define. 399 rows had these. We placed them in the existing 'other' group.

The regrouping is a judgment call. After cleaning, the script confirms all 30,000 customers remain and that every financial amount matches the raw file."

## Slide 4. What predicts default (Eshwar)

Visual: bar chart of default rate by number of late months, 0 to 6 (11.7% to 70.3%). Use `reports/figures/04_repayment_history.png` or redraw from the table in the report, Section 3.4.

Script:
"The strongest clue is simple: has the customer paid late before? Customers who were current on their latest payment defaulted about 14 percent of the time. Customers who were one or more months late defaulted 50 percent of the time. Exactly one month late was 34 percent, and exactly two months late was 69 percent.

We also counted how many of the six months each customer was late. With no late months, 11.7 percent defaulted. With six late months, 70.3 percent did. Risk climbs with every extra late month.

Recent trouble matters more than old trouble. Customers late only in the last two months defaulted at 41.6 percent. Customers late only in the two oldest months defaulted at 26.0 percent."

## Slide 5. What does not, and a surprise (Eshwar)

Visual: left, small bars for sex, education, marriage (weak). Right, "paid nothing: 51.8%" versus "paid in full: 15.4%".

Script:
"What does not help much? Who the customer is. Sex, education and marital status all show some difference in default rates, but the differences are small. Age is odd. Customers under 25 and over 60 default more than those in their thirties. So age matters, but not as a straight line.

How much a customer paid matters far more. Customers who paid nothing against their bill defaulted 51.8 percent of the time. Those who paid it in full defaulted 15.4 percent of the time.

One more surprise. Customers with no activity at all, no bill and no payment for six months, defaulted at 36.6 percent. Inactive does not mean safe."

## Slide 6. How we built the model (Umesh)

Visual: simple flow: history in, risk score out. Train on 80 percent, test on hidden 20 percent.

Script:
"A model is a set of rules learned from past customers. We showed it 80 percent of the customers, along with what happened to them. Then we tested it on the other 20 percent, which it had never seen. That is like studying from past exams and then sitting a new one.

A trap here: accuracy. A model that said 'nobody will default' would be right 78 percent of the time and find no one at risk. So we measured how many defaulters it finds, and how often its warnings are right.

We compared four kinds of model on three sets of inputs. A simple scoring formula did worst. Three others scored almost the same: gradient boosting, where each new tree corrects the last, a random forest, where many trees vote, and a neural network, which comes next. Picking between them is a judgment call. We went with gradient boosting.

The inputs that mattered most were the ones we found earlier: the latest payment status and how many months a customer was late. Next came the worst status in the six months, the latest bill, and how much of the credit limit is in use. In short, the model learned mostly what the exploratory work had already shown."

## Slide 7. The neural network (Siddhant)

Visual: three lines of scores on the hidden customers: neural network 0.553, gradient boosting 0.550, random forest 0.556. Caption: "A tie." Source: `reports/neural_network/model_comparison.csv`, figure `reports/neural_network/fig_nn_vs_best_tree_curves.png`.

Script:
"We also tested a neural network, because the team asked whether one belongs here. Think of a panel of small judges. Each judge looks at a few facts, like late months or credit use, and passes an opinion to the next row of judges. Over many rounds on the training customers, the panel learns how much each judge should count, until the last row gives a risk score. Ours had two rows.

It tied the tree models. On the hidden customers its ranking score was 0.553. Gradient boosting scored 0.550 and the random forest 0.556. In our training checks it was a little lower, 0.552 against 0.560. Those gaps are smaller than the 0.006 to 0.008 spread between checks, so we call it a tie. It did clearly beat the simple formula, which scored about 0.51 in training checks.

Because the gap is that small, the choice came from the training checks, where gradient boosting scored highest. Gradient boosting is also easier to explain to a non-technical reader, but that is a secondary point. So we kept gradient boosting."

## Slide 8. How well it works (Umesh)

Visual: the 6,000 test customers as a four-box grid: 1,327 defaulted, 738 caught, 589 missed, 673 false alarms.

Script:
"Here are the results on the 6,000 customers the model had never seen. 1,327 of them actually defaulted. The model caught 738, about 56 percent. It missed 589.

It also raised 673 false alarms. So when it flags someone, it is right about 52 percent of the time. A random guess would be right only about 22 percent of the time.

On a ranking score called PR-AUC, the model scored 0.55 where random guessing scores about 0.22. In short, the model ranks risky customers well above random, but it is not a crystal ball.

Its overall accuracy is 79 percent, barely above the 78 percent from saying 'nobody defaults'. That is why accuracy is not our main measure.

One caution. These percentages come from one run. When we changed how the trimming limits and the cutoff were set, the share found moved between 54 and 57 percent, and the share of correct flags between 51 and 53 percent. Read them as roughly half and roughly half."

## Slide 9. Limits (Prasad)

Visual: four short bullets.

Script:
"There are limits you should know about. First, the data is from one bank in Taiwan in 2005. We cannot claim it describes banks today.

Second, we set the trimming limits using all 30,000 customers, including the test ones. Redoing it with training customers only lowered the ranking score slightly, from 0.550 to 0.547.

Third, the top models scored almost the same. An earlier saved run picked the random forest. The difference traces to the scikit-learn version (1.7.2 versus 1.9.1). The team has not yet decided which run to report.

Fourth, fairness. We left out sex, education and marital status, but that does not guarantee equal treatment. Customers with only a high school education who would have paid were wrongly flagged 19 percent of the time. For graduate school customers it was 11 percent. We did not test whether that gap is due to chance. A bank should use a score like this to prompt a human review, not an automatic refusal."

## Slide 10. Recommendations (Eshwar)

Visual: three bullets.

Script:
"What would we suggest? Use the model to give each account a risk score and rank accounts, rather than making a yes or no call. Choose the cutoff using the bank's own costs for a missed defaulter versus a false alarm. Before real use, retest on recent customers, keep sex, education and marital status out unless a fairness review says otherwise, and repeat the group comparison on new data. Any bank should also check the results on its own customers, because habits differ by country and year."

## Slide 11. Who did what (all)

Visual: four names with roles, one line each. Roles are the ones in the team plan and the report's "Who wrote what" paragraph. Each member should confirm their own line.

Script (each member reads their own line):
"Prasad: assigned the Introduction, covering dataset research and the business problem.
Siddhant: data cleaning and preparation, and the neural network part of model selection and analysis.
Eshwar: exploratory analysis and the presentation content.
Umesh: modeling, model comparison, validation, and GitHub coordination.
Thank you for watching."

## Check before recording

- [ ] Confirm the presenter split with all four members (Prasad currently has slides 1, 2 and 9; Eshwar 4, 5 and 10).
- [x] The report has matching neural network sections (4.5, 5.7, Appendix 8.8). Slide 7 numbers were taken from them.
- [ ] The team confirms the team quotes the fresh gradient boosting run. If not, update slides 6 to 9 and the report together.
- [ ] Run a timed read. Target 8 to 10 minutes.
- [ ] Confirm every member approves the wording of their credit line on slide 11.
- [ ] Record the video and name the file `Final-Project-Presentation-Team-10.mp4`. It does not exist yet.
