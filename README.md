# Predicting Credit Card Default (AAI-500 Final Team Project, Team 10)

This project is a part of the AAI-500 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

-- Project Status: Active (report, code and notebook are written; the presentation video is not recorded yet, see Deliverables and status)

TODO (team): confirm the final GitHub repository URL. The `git clone` line below uses the `origin` remote of the local repository (siddhant-bhura/team10_aai500_teamproject). The team status form lists siddhant-bhura/aai500-final-team-project. Nothing has been pushed or checked online.

## Installation

Python 3.14.6 was used. Other recent Python 3 versions should work but are untested.

```
git clone https://github.com/siddhant-bhura/team10_aai500_teamproject.git
cd team10_aai500_teamproject          # the repository root is the final-project folder
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run everything, in order (about 4.5 minutes on an 18-core Mac, about 7 minutes in a clean scratch copy on the same machine):

```
make all            # or: python run_all.py --no-report   (on a machine without make)
```

`make all` runs cleaning, EDA, modeling, the neural network and the robustness checks inside a temporary copy of the repository, then copies the fresh outputs to `reproduced/` (ignored by git). No tracked file is overwritten, including the teammates' `reports/model_findings.md` and `reports/figures/`.
It prints, per output folder, how many files are identical to the committed copy. In our check, the cleaned CSV, the EDA findings, 13 of 17 figures and all neural network files matched byte for byte. `model_findings.md` and the four `model_*` figures differ (see Results).

Optional flags: plain `python run_all.py` also rebuilds the PDF (it writes into `reports/`); `--no-report` skips that, `--skip-nn` skips the 3-minute network search.
The PDF step is optional and needs extra packages: `pip install -r requirements-pdf.txt && playwright install chromium`. Without them it prints a message and is skipped. The committed `reports/Final-Project-Report-Team-10.pdf` is the submission file, so you do not need this step. `playwright install chromium` can fail behind a corporate proxy (it failed for us with a download error on a clean machine). We got the PDF step working with `playwright==1.63.0`, which is the version pinned in `requirements-pdf.txt`.
Use another interpreter with `make all PYTHON=/path/to/python`.

Single steps with make. Each one runs in the same temporary copy as `make all`, so no tracked file is overwritten:


| Target            | Runs                                      | Writes                                                                                                |
| ----------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `make all`        | all five steps (same as `make reproduce`) | `reproduced/` only                                                                                    |
| `make reproduce`  | all five steps                            | `reproduced/` only                                                                                    |
| `make clean-data` | `src/data_cleaning.py`                    | `reproduced/data/processed/` only                                                                     |
| `make eda`        | `src/exploratory_analysis.py`             | `reproduced/reports/` (findings, figures) only                                                        |
| `make model`      | `src/modeling.py`                         | `reproduced/reports/` (findings, figures) only                                                        |
| `make nn`         | `src/neural_network.py`                   | `reproduced/reports/neural_network/` only                                                             |
| `make robustness` | `src/robustness_checks.py`                | prints only (log in `reproduced/logs/`)                                                               |
| `make report`     | `tools/build_report.py`                   | `reports/Final-Project-Report-Team-10.pdf` (skips with exit code 0 if mistune/playwright are missing) |


Each single step uses the files already in the repository (for example the committed cleaned CSV) as its input, not the output of an earlier step. `python run_all.py --only nn` does the same without make.

Running the scripts directly (below) writes INTO the repository and DOES overwrite tracked files, so use that on purpose:

```
python src/data_cleaning.py        # data/processed/credit_card_default_cleaned.csv (a few seconds)
python src/exploratory_analysis.py # reports/eda_findings.md, figures 01-13 (about 3 s)
python src/modeling.py             # reports/model_findings.md, model_01-04 figures (about 1 min)
python src/neural_network.py       # reports/neural_network/ only (about 3 min, deterministic)
python src/neural_network.py --selfcheck  # tiny check of the oversampling helper (instant)
python src/robustness_checks.py    # prints only; checks behind report Sections 5.5 and 5.6
```

Scripts resolve paths from their own location, so they run from any directory. Modeling needs the cleaned CSV and stops with a message if it is missing. The neural network imports `modeling.py`. Seeds are fixed (42). Model numbers can shift slightly between library versions, so use the pinned versions in `requirements.txt`. With the pinned 1.9.1 a fresh run does not reproduce the committed model file, because that file came from scikit-learn 1.7.2 (see Results).

To open the notebook: `pip install jupyterlab`, then `jupyter lab notebooks/final_analysis.ipynb` (Jupyter is not needed for the scripts). It imports the scripts in `src/` and sets `RERUN_FULL_SEARCH = False` so the network search is read from saved files; set it to `True` to re-run the 3-minute search.

## Project Intro/Objective

A credit card bank loses money when customers stop paying. If it can see a missed payment coming, it can offer a payment plan, review a limit or set money aside early.
The objective is to predict, from a customer's profile and six months of bills, payments and repayment status, whether the customer will miss next month's payment, and to say how reliable that prediction is.
Defaults are the minority class (22.1%), so the main score is PR-AUC (precision-recall area), reported next to ROC-AUC, precision, recall and F1.

## Partner(s)/Contributor(s)


| Member           | Work                                                                                                                                  | Report section                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Prasad Pidaparty | Assigned the Introduction. The report's Section 1 was compiled by the team from the status form and dataset documentation.            | Introduction                                                                        |
| Siddhant Bhura   | Data cleaning (`src/data_cleaning.py`); neural network (`src/neural_network.py`); robustness checks; one-command run; report assembly | Data Cleaning and Preparation; neural network parts of Model Selection and Analysis |
| Eshwar Reddy M   | Exploratory analysis (`src/exploratory_analysis.py`), presentation content                                                            | Exploratory Data Analysis                                                           |
| Umesh Sharma     | Modeling (`src/modeling.py`), GitHub coordination                                                                                     | Model Selection, Model Analysis                                                     |


No outside partner.

## Methods Used

- Data manipulation and cleaning (IQR capping, category recoding, self-checks)
- Descriptive and inferential statistics (chi-square with Cramer's V, rank correlations, effect sizes, variance inflation factors)
- Data visualization
- Machine learning (logistic regression, random forest, histogram gradient boosting) with stratified train/test split and 5-fold cross-validation
- Deep learning (multilayer perceptron, scikit-learn)
- Model evaluation on imbalanced data (PR-AUC, threshold choice, permutation importance)
- Ethics for AI (group error-rate comparison, discussion of protected attributes and explainability in credit scoring)



## Technologies

Python, pandas, NumPy, scikit-learn, SciPy, statsmodels, Matplotlib, seaborn, Jupyter (notebook only). The report PDF is built with mistune and Playwright (optional, see `requirements-pdf.txt`). Exact versions are in `requirements.txt`.

## Project Description

**Data source.** UCI Machine Learning Repository, "Default of Credit Card Clients" (Yeh, 2009), [https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). A Taiwanese bank, April to September 2005. The raw file is in `data/raw/`.

**Size.** 30,000 customers, 25 columns in the raw file (an `ID`, 23 input columns, and the outcome). 6,636 customers (22.12%) defaulted.

**Data dictionary.**


| Column(s)                    | Meaning                                                                                                                                                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ID`                         | Row identifier                                                                                                                                                                                                                 |
| `LIMIT_BAL`                  | Credit limit, New Taiwan dollars                                                                                                                                                                                               |
| `SEX`                        | 1 male, 2 female                                                                                                                                                                                                               |
| `EDUCATION`                  | 1 graduate school, 2 university, 3 high school, 4 other (undocumented codes 0, 5, 6 are regrouped into 4 by the cleaning step)                                                                                                 |
| `MARRIAGE`                   | 1 married, 2 single, 3 other (undocumented code 0 is regrouped into 3)                                                                                                                                                         |
| `AGE`                        | Age in years                                                                                                                                                                                                                   |
| `PAY_1` to `PAY_6`           | Repayment status for September back to April 2005. -2 and 0 are not described in the documentation; 1 is one month late, 2 two months late, and so on. The raw file calls the first one `PAY_0`; the cleaning step renames it. |
| `BILL_AMT1` to `BILL_AMT6`   | Bill amount for each of the same months                                                                                                                                                                                        |
| `PAY_AMT1` to `PAY_AMT6`     | Amount paid in each of the same months                                                                                                                                                                                         |
| `default payment next month` | Outcome: 1 defaulted in October 2005, 0 did not                                                                                                                                                                                |


**Questions and hypotheses.** Does repayment history predict default better than who the customer is? Do tree models beat logistic regression because the relationships are not straight lines? Does a neural network beat the tree models? Does the chosen model treat groups differently even though it never sees sex, education or marital status?

**Analysis and modeling.**

1. Cleaning: no missing values or repeated IDs. 32,425 extreme money values were capped (no rows dropped) and 399 undocumented education and marriage codes were regrouped.
2. EDA: repayment history is the strongest signal. Customers one or more months late on the latest payment defaulted at 50.29%, against 13.83% for current customers. Demographics have small effects.
3. Modeling: three input sets and four model families compared by 5-fold cross-validated PR-AUC on 24,000 training customers, then scored once on 6,000 held-back customers. The two tree models and the neural network are within noise of each other and all beat logistic regression.
4. Robustness and fairness: refitting the caps on training rows only, taking the cut-off from out-of-fold predictions, and error rates by sex and education.

**Roadblocks.**

- The raw Excel file has a placeholder first header row.
- The six repayment-status columns are not recorded the same way (status 1 appears 3,688 times in the latest month and almost never in the earlier five), so month-to-month trend features were avoided.
- PyTorch and Keras are not installed in the project environment, so the neural network is a scikit-learn MLP.
- A fresh run of `src/modeling.py` selected histogram gradient boosting, while the committed `reports/model_findings.md` (an earlier run) selected random forest. The cause is the scikit-learn version: 1.7.2 reproduces the committed file exactly, while `requirements.txt` pins 1.9.1.



## Results

Full write-up: `reports/Final-Project-Report-Team-10.pdf` (source `reports/Final-Project-Report-Team-10.md`). The report PDF includes the code-output appendix.

- The report quotes a fresh run of `src/modeling.py`. It selected histogram gradient boosting, with exam-set PR-AUC 0.5499, 738 of 1,327 defaulters found and 52.3% of flags correct. These two percentages moved by a few points in the robustness checks (report Section 5.5).
- The committed `reports/model_findings.md` and `reports/figures/model_*.png` come from an earlier run that selected random forest (PR-AUC 0.5554). The two models are within fold-to-fold noise. The figures matching the report are in `reports/report_figures/`. The team still has to agree on one run and commit its output.
- Neural network (`reports/neural_network/`): tied the tree models and did not beat them. It beats logistic regression on every input set.

Best version of each model (highest cross-validated PR-AUC across the three input sets). The cut-off for F1, recall and precision comes from out-of-fold predictions for every row, so the rows compare fairly. Source: `reports/neural_network/model_comparison.csv`.


| Model (input set)                      | CV PR-AUC | CV std | Test PR-AUC | Test ROC-AUC | Test F1 | Test recall | Test precision |
| -------------------------------------- | --------- | ------ | ----------- | ------------ | ------- | ----------- | -------------- |
| Hist. gradient boosting (kitchen_sink) | 0.5600    | 0.0060 | 0.5499      | 0.7760       | 0.5402  | 0.5546      | 0.5265         |
| Random forest (kitchen_sink)           | 0.5585    | 0.0073 | 0.5558      | 0.7750       | 0.5415  | 0.5780      | 0.5093         |
| Neural network (extended)              | 0.5520    | 0.0077 | 0.5527      | 0.7692       | 0.5419  | 0.5365      | 0.5473         |
| Logistic regression (baseline_eda)     | 0.5101    | 0.0121 | 0.5081      | 0.7311       | 0.5224  | 0.5320      | 0.5131         |


The first three rows are within one fold-to-fold spread (0.006 to 0.008), so we call them a tie. No-skill PR-AUC is the default rate, about 0.22.

## Repository structure

```
final-project/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-pdf.txt                (optional, PDF rebuild only)
├── requirements-dev.txt                (flake8, for the PEP 8 check)
├── Makefile                            (make all: whole pipeline, see Installation)
├── run_all.py                          (same, for machines without make)
├── data/
│   ├── raw/default of credit card clients.xls
│   └── processed/credit_card_default_cleaned.csv
├── notebooks/final_analysis.ipynb      (executed; scripts' output plus the neural network section)
├── src/
│   ├── data_cleaning.py
│   ├── exploratory_analysis.py
│   ├── modeling.py
│   ├── neural_network.py               (MLP added to the model comparison)
│   ├── robustness_checks.py            (read-only checks, report Sections 5.5 and 5.6)
│   └── visualization.py                (stub, unused)
├── tools/build_report.py               (markdown to PDF; needs mistune and playwright)
├── reports/
│   ├── Final-Project-Report-Team-10.md / .pdf   (final report)
│   ├── eda_findings.md
│   ├── model_findings.md               (earlier run, see Results)
│   ├── figures/                        (13 EDA and 4 model figures from the scripts)
│   ├── report_figures/                 (model and neural network figures matching the report)
│   └── neural_network/                 (results md, comparison and search CSVs, 3 figures)
├── presentation/
│   └── video-outline-DRAFT.md          (script draft; the video itself is not produced)
└── reproduced/                         (created by make all; ignored by git)
```



## Deliverables and status

- [x] Final report: `reports/Final-Project-Report-Team-10.pdf`. Open items: the team confirms the Introduction text and which model run is quoted.
- [ ] Presentation: `Final-Project-Presentation-Team-10.mp4`, 8 to 10 minutes, every member presents an equal share, one slide naming each member and their contributions. Not recorded. Only a script draft exists (`presentation/video-outline-DRAFT.md`).
- [x] Code for cleaning, EDA, modeling and the neural network, a one-command run, and the executed notebook `notebooks/final_analysis.ipynb`.
- [ ] Peer evaluation forms (Assignment 7.1, each member individually).
- [ ] One member submits the PDF and MP4 on Canvas by Mon 19 Oct 2026, 11:59 PM Pacific. No extensions.

Code style: PEP 8 checked with a 79-character limit for `data_cleaning.py`, `neural_network.py`, `robustness_checks.py`, `run_all.py` and `tools/build_report.py`. `exploratory_analysis.py` and `modeling.py` are teammates' files and were not touched. They have late-import warnings and long lines, and `modeling.py` has one unused f-string. Full PEP 8 compliance is therefore not claimed for the whole repository.

## License

MIT License for the code (see `LICENSE`). The dataset is from the UCI Machine Learning Repository and keeps its own license (shown on the UCI dataset page). Check it before redistributing the data.

## Acknowledgments

- Dataset: Yeh, I. C. (2009), UCI Machine Learning Repository, [https://doi.org/10.24432/C55S3H](https://doi.org/10.24432/C55S3H). Original study: Yeh, I. C., and Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
- **AI tool disclosure.** Different team members used a mix of AI Code Assistants to save time and produce better code quality of the Solution. e.g ClaudeCode, GitHiub CoPilot etc.

