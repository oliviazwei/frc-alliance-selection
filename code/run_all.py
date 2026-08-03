"""Reproduce every number, table, and figure in the paper from the included data.

    python code/run_all.py

Runs steps 03-05 (build sample -> estimate -> exhibits). The download steps 01
and 02 are skipped by default because the raw data are already in the
repository and re-downloading requires a TBA API key; pass --download to run
them as well.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = ["03_build_sample.py", "04_estimate.py", "05_make_exhibits.py"]
DOWNLOAD_STEPS = ["01_download_events.py", "02_download_data.py"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--download", action="store_true",
                    help="also re-download raw data (needs TBA_API_KEY)")
args = parser.parse_args()

steps = (DOWNLOAD_STEPS if args.download else []) + STEPS

for script in steps:
    print(f"\n{'=' * 70}\n  {script}\n{'=' * 70}", flush=True)
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
    if result.returncode != 0:
        sys.exit(f"\n{script} failed with exit code {result.returncode}")

print(f"\n{'=' * 70}\nDone. Compile the paper with:\n"
      f"  cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main")
