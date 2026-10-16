"""Run the whole project end to end without touching tracked teammate files.

Order: cleaning -> EDA -> modeling -> neural network -> robustness checks.
Then (optional) rebuild the report PDF.

Why a scratch copy: data_cleaning.py, exploratory_analysis.py and modeling.py
write over files that are tracked in git (the cleaned CSV, eda_findings.md,
model_findings.md, figures). This script copies the repo to a temporary
folder, runs every step there, and copies the fresh outputs to
``reproduced/`` in the repo. Nothing tracked is changed. It uses only the
standard library, so it works on machines without ``make``.

AI-tool disclosure: written with the help of Claude Code (Anthropic); the code
was run and its output checked. See README.md.

Usage (from the repo root):
    python run_all.py              # all steps, then report build if possible
    python run_all.py --no-report  # skip the PDF build
    python run_all.py --skip-nn    # skip the neural network (about 3 min)
    python run_all.py --only nn    # one step (cleaning, eda, modeling, nn,
                                   # robustness), still in a scratch copy;
                                   # outputs go to reproduced/
"""
import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# (short name for --only, label, script, outputs worth keeping relative to
# the repo root). The robustness checks only print, so they have no outputs.
STEPS = [
    ("cleaning", "cleaning", "src/data_cleaning.py",
     ["data/processed/credit_card_default_cleaned.csv"]),
    ("eda", "EDA", "src/exploratory_analysis.py",
     ["reports/eda_findings.md", "reports/figures"]),
    ("modeling", "modeling", "src/modeling.py",
     ["reports/model_findings.md", "reports/figures"]),
    ("nn", "neural network", "src/neural_network.py",
     ["reports/neural_network"]),
    ("robustness", "robustness checks", "src/robustness_checks.py", []),
]
IGNORE = shutil.ignore_patterns(
    ".git", "reproduced", "__pycache__", ".venv", "venv", "*.pdf")


def run(label, script, cwd, log_dir):
    t0 = time.time()
    print(f"[{label}] {script} ...", flush=True)
    res = subprocess.run([sys.executable, script], cwd=cwd,
                         capture_output=True, text=True)
    (log_dir / f"{Path(script).stem}.log").write_text(
        res.stdout + "\n--- stderr ---\n" + res.stderr, encoding="utf-8")
    secs = time.time() - t0
    if res.returncode != 0:
        print(res.stderr[-2000:])
        sys.exit(f"[{label}] FAILED (exit {res.returncode}) after {secs:.0f}s")
    print(f"[{label}] ok in {secs:.0f}s", flush=True)


def compare(fresh, committed):
    """Return (same, differ, new) file counts for fresh vs committed."""
    same = differ = new = 0
    files = [fresh] if fresh.is_file() else sorted(
        p for p in fresh.rglob("*") if p.is_file())
    for f in files:
        rel = f.relative_to(fresh.parent if fresh.is_file() else fresh)
        old = committed if fresh.is_file() else committed / rel
        if not old.exists():
            new += 1
        elif filecmp.cmp(f, old, shallow=False):
            same += 1
        else:
            differ += 1
    return same, differ, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--only", choices=[st[0] for st in STEPS],
                    help="run one step in the scratch copy (no PDF build)")
    args = ap.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="aai500_run_"))
    work = scratch / "repo"
    shutil.copytree(ROOT, work, ignore=IGNORE)
    logs = ROOT / "reproduced" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"Scratch copy: {work}\nPython: {sys.executable}\n")

    steps = [st for st in STEPS if args.only in (None, st[0])]
    if args.skip_nn:
        for st in steps:
            if "neural" in st[2]:
                print(f"[{st[1]}] skipped")
        steps = [st for st in steps if "neural" not in st[2]]
    for _, label, script, _ in steps:
        run(label, script, work, logs)
    outputs = list(dict.fromkeys(o for st in steps for o in st[3]))

    print("\nCopying fresh outputs to reproduced/ ...")
    for rel in outputs:
        src = work / rel
        if not src.exists():
            continue
        dst = ROOT / "reproduced" / rel
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        s, d, n = compare(src, ROOT / rel)
        print(f"  {rel}: {s} identical to the repo copy, {d} different, "
              f"{n} new")
    shutil.rmtree(scratch, ignore_errors=True)
    if any(st[0] == "modeling" for st in steps):
        print("\nmodel_findings.md and the model_* figures may differ "
              "depending on the scikit-learn version (see README, Results).")

    if args.no_report or args.only:
        return
    print("\n[report] tools/build_report.py ...")
    res = subprocess.run([sys.executable, str(ROOT / "tools/build_report.py")])
    if res.returncode != 0:
        sys.exit("[report] FAILED")


if __name__ == "__main__":
    main()
