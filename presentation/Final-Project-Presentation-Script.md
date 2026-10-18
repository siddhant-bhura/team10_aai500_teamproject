# Final Project Presentation Script (Executive / Non-Technical)

**Team 10 · AAI-500 · Predicting Credit Card Default**  
**Target length:** 6–9 minutes (aim for **~7.5 minutes**)  
**Audience:** executives, business stakeholders, non-technical listeners  
**Tone:** clear, confident, few jargon words; every number has a “so what”

**Suggested file name when recorded:** `Final-Project-Presentation-Team-10.mp4`

---

## How to use this document

1. One person can narrate the whole video, **or** the three members can split slides as marked below (assignment requires all four to speak).
2. Put only the **on-screen text** on slides. Speak the **script**; do not read bullet walls.
3. Practice once with a timer. At ~150 words/minute, the full script is about **1,150 words ≈ 7.5–8 minutes**. Trim the optional “cut if short on time” lines if needed.
4. Prefer figures from `reports/figures/` and `reports/neural_network/` over dense tables.



### Presenter split (equal share)


| Slides      | Presenter                   | Approx. time |
| ----------- | --------------------------- | ------------ |
| 1           | Umesh                       | ~1:30        |
| 2-3–4       | Siddhant                    | ~1:45        |
| 5–6         | Eshwar                      | ~2:00        |
| 7–8         | Umesh                       | ~2:00        |
| 9 + credits | All (each reads own credit) | ~0:45        |


---



## Timing overview (9 slides)


| #   | Slide title                       | Time | Presenter |
| --- | --------------------------------- | ---- | --------- |
| 1   | The business question             | 0:50 | Umesh     |
| 2   | What we looked at                 | 0:40 | Sidhant   |
| 3   | Is the data trustworthy?          | 0:55 | Siddhant  |
| 4   | What actually predicts risk       | 1:10 | Eshwar    |
| 5   | What does *not* matter much       | 0:50 | Eshwar    |
| 6   | How we tested the models          | 0:55 | Umesh     |
| 7   | How well it works (plain numbers) | 1:10 | Umesh     |
| 8   | Limits + what we recommend        | 1:10 | Sidhant   |
| 9   | Who did what / thank you          | 0:30 | Umesh     |


**Optional cut (if over 9 minutes):** shorten Slide 6 neural-network paragraph to one sentence; drop the “inactive accounts” line on Slide 5.

---



## Slide 1 — The business question

**On screen**

- Title: *Can we see a missed payment coming?*
- One line: *Early warning → earlier action*
- Team 10 · USD AAI-500

**Visual:** credit card or simple “risk thermometer” icon. No charts yet.

**Script (Umesh):**

> Credit card lenders lose money when customers stop paying. The valuable moment is *before* the miss — when the bank can still offer a payment plan, review a limit, or set money aside.
>
> Our question is simple: using a customer’s recent card history, can we flag who is likely to miss *next* month’s payment?
>
> We are Team 10: Prasad, Siddhant, Eshwar, and Umesh. In the next few minutes we will show what the data says, how well a model works, and what a bank should — and should not — do with the result.

---



## Slide 2 — What we looked at

**On screen (three big numbers)**

| 30,000 | 6 months | ~22% |
| customers | of bills & payments | missed the next payment |

- Source note (small): Taiwanese bank · 2005 · public UCI dataset

**Script (Sidhant):**

> We used a public dataset of thirty thousand credit card customers from a bank in Taiwan in two thousand five. For each person we have about two dozen facts: age and profile, credit limit, and six months of bills, payments, and how late they were.
>
> About twenty-two in every hundred customers missed the following month’s payment. That is roughly one in four or five. Remember that number — it shapes how we judge success later.
>
> Our job was to turn that history into an early-warning score, not a cold automatic “yes or no.”

---



## Slide 3 — Is the data trustworthy?

**On screen (checklist)**

- No missing values  
- No repeated customer IDs  
- Extreme dollar amounts **kept as-is** (not trimmed)  
- A few unclear category codes cleaned up (~399 rows)

**Visual:** simple green checkmarks. Optional tiny footnote: “Over-limit accounts stay visible.”

**Script (Siddhant):**

> Before any modeling, we checked the data the way you would inspect ingredients.
>
> Nothing was missing. No customer ID appeared twice. A handful of people share the same profile under different IDs; we kept them, because they look like different customers.
>
> Some bills and payments are extremely large. That is normal for money data. We did **not** chop those values down. Keeping the extremes means high-balance and over-limit accounts stay visible — and those over-limit customers default more often.
>
> A few education and marriage codes were undocumented. We placed about four hundred of those rows into the existing “other” group. After cleaning, all thirty thousand customers remain.

---



## Slide 4 — What actually predicts risk

**On screen**

- Headline: *Late payment is the signal*
- Two comparison numbers:
  - Current on latest payment → **~14%** default  
  - Late on latest payment → **~50%** default
- Climbing line or bars: 0 late months **12%** → 6 late months **70%**
- Caption: *Recent trouble matters more than old trouble*

**Visual:** `reports/figures/04_repayment_history.png` (or a redrawn simpler bar chart).

**Script (Eshwar):**

> The strongest clue is not who the customer *is*. It is whether they have been paying late.
>
> Customers who were current on their latest payment defaulted about fourteen percent of the time. Customers already late defaulted about fifty percent of the time — roughly three and a half times higher.
>
> If we count how many of the six months they were late, risk climbs almost every step: about twelve percent with no late months, up to about seventy percent with six late months.
>
> Timing matters too. Trouble in the *most recent* months is more dangerous than the same kind of trouble from months earlier.
>
> For a business audience, the takeaway is blunt: **watch repayment behaviour first.**

---



## Slide 5 — What does *not* matter much (and one surprise)

**On screen (two columns)**

**Left — weak signals**  
Sex · Education · Marital status → small differences  

**Right — stronger money behaviour**  
Paid nothing on the bill → **~52%** default  
Paid in full → **~15%** default  

**Surprise box:** Inactive for six months → **~37%** default (not “safe”)

**Script (Eshwar):**

> Who the customer is — sex, education, marital status — shows only small differences. Age is not a straight story either: younger and older customers look riskier than people in their thirties.
>
> How much they actually repay matters far more. Customers who paid nothing against an outstanding bill defaulted about fifty-two percent of the time. Those who cleared the bill defaulted about fifteen percent of the time.
>
> One surprise for operations teams: accounts with *no* bills and *no* payments for six months are not quiet and safe. They defaulted about thirty-seven percent of the time — well above the overall average. Zero activity is a red flag, not a green light.

---



## Slide 6 — How we tested the models

**On screen (simple flow)**

```
Past customers  →  Learn patterns  →  Hidden customers (exam)
     80%                                   20%
```

- We compared several approaches (simple formula, tree models, neural network)
- We **do not** use plain “accuracy” as the headline score

**Script (Umesh):**

> A model is a set of patterns learned from past customers. We trained on eighty percent of the data, then tested on twenty percent the model had never seen — like studying old exams and sitting a new one.
>
> Accuracy is a trap here. A system that always says “nobody will default” would look about seventy-eight percent correct — and would catch *zero* people at risk. So we asked better questions: how many future missed payments do we catch, and how often are our warnings right?
>
> We compared a simple scoring formula, tree-based methods, and a neural network. The trees and the network landed in a near-tie. The simple formula trailed. We selected a tree-based method that is easier to explain to a business audience, without giving up ranking quality.

---



## Slide 7 — How well it works (plain numbers)

**On screen (one clear scorecard)**

On **6,000** customers held back for the final exam:


|                                    |                      |
| ---------------------------------- | -------------------- |
| Actually missed payment            | **1,327**            |
| Caught by the model                | **~738 (~56%)**      |
| Missed by the model                | **~589**             |
| False alarms                       | **~673**             |
| When it flags someone, it is right | **~52% of the time** |


- Random guessing would be right only about **22%** of the time when it “flags”  
- Ranking quality: well above chance, **not** a crystal ball  
- Overall accuracy ~79% — barely above the “predict nobody” baseline — so we do **not** lead with accuracy

**Visual (optional):** four-box confusion sketch, or `reports/figures/model_02_confusion_matrix.png` simplified.

**Script (Umesh):**

> Here is the result in business language. Of about thirteen hundred people who went on to miss a payment, the model caught roughly half to a little more than half. It also raised several hundred false alarms. When it raises a flag, it is right about half the time — far better than the twenty-two percent base rate, but nowhere near perfect.
>
> So this is a **ranking and triage tool**, not a verdict. It helps a collections or risk team decide *whom to call first*. It should not refuse credit by itself.
>
> One caution: the exact percentages move a little when we change how we set the yes/no cutoff. Treat them as “about half caught, about half of flags correct,” not as a guarantee.

---



## Slide 8 — Limits and recommendations

**On screen**

**Limits**

1. One bank, one country, 2005 — not “all banks today”
2. Top models nearly tied — judgment call on which to deploy
3. Fairness: we excluded sex / education / marital status from the model, but gaps can still appear indirectly — human review required

**Recommendations**

1. Rank accounts by risk; act by tier (outreach, payment plan, limit review)
2. Set the cutoff from **business costs** (cost of a miss vs cost of a false alarm)
3. Monitor repayment behaviour operationally; retest on *your* recent customers before live use

**Script (Siddhant opens limits; Prasad or Umesh closes recommendations — or one speaker for time):**

> Limits first. This is one bank in Taiwan in two thousand five. We cannot claim it describes every market today. The best models scored almost the same, so choosing among them is a business judgment, not a landslide. And even though we kept sex, education, and marital status out of the model, other data can still act as stand-ins. A bank should use this to prompt a human review, not an automatic refusal.
>
> What we recommend: use the score to **rank** accounts and focus effort. Set the action threshold from your own costs — missing a defaulter versus bothering a good customer. Keep an eye on late payments and unpaid balances day to day. Before going live, retest on your own recent customers. Habits differ by country and year.

---



## Slide 9 — Who did what / thank you

**On screen**


| Member           | Contribution                                          |
| ---------------- | ----------------------------------------------------- |
| Prasad Pidaparty | Business problem, dataset framing, introduction       |
| Siddhant Bhura   | Data cleaning & preparation; neural network work      |
| Eshwar Reddy M   | Exploratory analysis; presentation content            |
| Umesh Sharma     | Modeling, comparison, validation; GitHub coordination |


**Script (each person reads their own line, then together):**

> Prasad: business problem and introduction.  
> Siddhant: data cleaning and the neural network comparison.  
> Eshwar: exploratory analysis and presentation content.  
> Umesh: modeling, validation, and repository coordination.  
>
> Thank you for watching. Questions welcome.

---



## Full spoken script (single narrator version)

Use this if one person records the whole video. Same content, continuous flow.

---

Credit card lenders lose money when customers stop paying. The valuable moment is before the miss — when the bank can still offer a payment plan, review a limit, or set money aside.

Our question is simple: using a customer’s recent card history, can we flag who is likely to miss next month’s payment? We are Team 10: Prasad, Siddhant, Eshwar, and Umesh.

We used a public dataset of thirty thousand credit card customers from a bank in Taiwan in two thousand five. For each person we have about two dozen facts: profile, credit limit, and six months of bills, payments, and lateness. About twenty-two in every hundred missed the next payment. Our job was to turn that history into an early-warning score.

Before modeling, we checked the data. Nothing was missing. No customer ID appeared twice. Some dollar amounts are extreme; we kept them as-is so over-limit accounts stay visible. A few unclear category codes were placed in “other.” All thirty thousand customers remain.

The strongest clue is late payment. Customers current on the latest payment defaulted about fourteen percent of the time; those already late, about fifty percent. Risk climbs with more late months — from about twelve percent with none, to about seventy percent with six. Recent trouble is worse than old trouble. Watch repayment behaviour first.

Who the customer is — sex, education, marital status — shows only small differences. How much they repay matters more: paying nothing against a bill goes with about fifty-two percent default; paying in full, about fifteen percent. And inactive accounts are not safe — about thirty-seven percent defaulted after six quiet months.

We trained on eighty percent of customers and tested on twenty percent never seen. Accuracy is a trap: always predicting “no default” looks about seventy-eight percent right and catches nobody. We compared a simple formula, tree models, and a neural network. Trees and the network nearly tied; the simple formula trailed. We chose a tree approach that is easier to explain without giving up ranking quality.

On six thousand held-back customers, about thirteen hundred actually defaulted. The model caught roughly half to a little over half, with false alarms as well. When it flags someone, it is right about half the time — far better than the twenty-two percent base rate, but not a crystal ball. Use it to triage, not to refuse credit alone.

Limits: one bank, one country, two thousand five. Top models nearly tied. Fairness still needs human review. Recommendations: rank accounts by risk; set thresholds from business costs; monitor lateness day to day; retest on your own recent customers before live use.

Umesh: business problem and introduction. Siddhant: cleaning and neural network. Eshwar: exploratory analysis and presentation. Umesh: modeling and validation. Thank you.

---



## Recording checklist

- [ ] Timed dry run lands between **6 and 9 minutes** (target **7:30**)
- [ ] Slides show big numbers; no dense tables on camera
- [ ] Figures used: repayment history (`reports/figures/04_repayment_history.png`), optional confusion / PR curves
- [ ] Script matches **outliers preserved** (do **not** say values were “trimmed” or “capped”)
- [ ] Avoid leading with “accuracy %” as the success metric
- [ ] End with recommendations an executive can act on
- [ ] All four members speak (or confirm single-narrator exception with the instructor)
- [ ] Export as `Final-Project-Presentation-Team-10.mp4`

---



## Speaker tips (executive audience)

1. **Lead with the decision**, not the algorithm: early outreach vs. ignore.
2. Say **“risk score / ranking”** instead of “classifier / PR-AUC” unless you briefly translate (“ranking quality well above chance”).
3. One metaphor max: “study on past exams, take a new exam” for train/test.
4. Pause after the **14% vs 50%** contrast — that is the memorable beat.
5. Close on **what to do Monday**: rank, cost-based cutoff, human review, retest on own book.

---



## Relationship to older draft

This file is the **6–9 minute executive** recording script. Keep `video-outline-DRAFT.md` for internal notes if useful; **record from this script**.