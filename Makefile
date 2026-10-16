# AI-tool disclosure: written with the help of Claude Code (Anthropic). See README.md.
# Single-run entry point. `make all` runs cleaning, EDA, modeling, the neural
# network and the robustness checks in order, in a scratch copy of the repo,
# then copies the outputs to reproduced/. Tracked files are not overwritten.
# `make report` is a separate, explicit step that rebuilds the report PDF.
# Use another interpreter with:  make all PYTHON=/path/to/python
# No make installed? Run:  python run_all.py
PYTHON ?= python3

.PHONY: all reproduce report clean-data eda model nn robustness

all: reproduce

# Whole pipeline in a scratch copy. Outputs go to reproduced/ only.
reproduce:
	$(PYTHON) run_all.py --no-report

# Rebuilds reports/Final-Project-Report-Team-10.pdf (writes into reports/).
# Prints a message and exits 0 if mistune/playwright are not installed.
report:
	$(PYTHON) tools/build_report.py

# Single steps. Each one runs in a scratch copy, like `make all`, so tracked
# files are never overwritten. Fresh outputs go to reproduced/ and a summary
# says how many match the repo copy. Each step reads the files already in
# the repo (for example the cleaned CSV), not the output of another step.
clean-data:
	$(PYTHON) run_all.py --only cleaning
eda:
	$(PYTHON) run_all.py --only eda
model:
	$(PYTHON) run_all.py --only modeling
nn:
	$(PYTHON) run_all.py --only nn
robustness:
	$(PYTHON) run_all.py --only robustness
