"""Reproduce every table, figure, and reported number in the paper.

    python code/run_all.py              # steps 03-07, from the committed data
    python code/run_all.py --download   # also re-download raw data (01-02)
    python code/run_all.py --from 05    # rerun exhibits and checks only
    python code/run_all.py --optimizer newton   # exact MLE instead of BFGS

Steps 01 and 02 are skipped by default because the raw data are committed and
re-downloading needs a TBA API key. Each step runs in its own process and the
pipeline stops at the first failure.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

DOWNLOAD_STEPS = ["01_download_events.py", "02_download_data.py"]
STEPS = [
    "03_build_sample.py",
    "04_estimate_models.py",
    "05_make_tables.py",
    "06_make_figures.py",
    "07_check_paper_numbers.py",
]

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--download", action="store_true",
                    help="also re-download raw data (needs TBA_API_KEY)")
parser.add_argument("--from", dest="start", metavar="NN",
                    help="start at the step with this number, e.g. 05")
parser.add_argument("--optimizer", choices=["bfgs", "newton"],
                    help="logit optimizer (default bfgs, as in the paper)")
args = parser.parse_args()
env = dict(os.environ)
if args.optimizer:
    env["FRC_OPTIMIZER"] = args.optimizer

steps = (DOWNLOAD_STEPS if args.download else []) + STEPS
if args.start:
    steps = [s for s in steps if s[:2] >= args.start.zfill(2)]

for script in steps:
    print(f"\n{'=' * 70}\n  {script}\n{'=' * 70}", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE,
                            env=env)
    if result.returncode != 0:
        sys.exit(f"\n{script} failed with exit code {result.returncode}")
    print(f"  ({time.time() - started:.0f}s)", flush=True)

print(f"\n{'=' * 70}\nDone. Tables: output/tables/  Figures: output/figures/\n"
      f"Check report: output/paper_number_check.csv")
