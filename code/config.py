"""Shared paths and constants for the replication package.

Every script imports this module, so the pipeline can be run from any working
directory. Change the sample period or the excluded event types here rather
than in the individual scripts.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Optional: read TBA_API_KEY (and anything else) from a git-ignored .env file
# in the repository root. Variables already set in the shell take precedence.
if (ROOT / ".env").exists():
    for _line in (ROOT / ".env").read_text().splitlines():
        _key, _sep, _value = _line.partition("=")
        if _sep and not _key.strip().startswith("#"):
            os.environ.setdefault(_key.strip(), _value.strip().strip("'\""))

DATA_RAW = ROOT / "data" / "raw"
DATA_DERIVED = ROOT / "data" / "derived"
OUTPUT = ROOT / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

for _d in (DATA_RAW, DATA_DERIVED, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Raw inputs (committed; re-created by 01 and 02)
EVENTS_CSV = DATA_RAW / "events.csv"            # TBA event metadata
TEAM_EVENTS_CSV = DATA_RAW / "team_events.csv"  # Statbotics team-event EPA
ALLIANCES_CSV = DATA_RAW / "alliances.csv"      # TBA playoff alliances

# Derived outputs (committed; re-created by 03 and 04)
SAMPLE_CSV = DATA_DERIVED / "analysis_sample.csv"
MODELS_CSV = DATA_DERIVED / "model_estimates.csv"
CONTRASTS_CSV = DATA_DERIVED / "coefficient_contrasts.csv"
DIAGNOSTICS_CSV = DATA_DERIVED / "diagnostics.csv"
CORRELATIONS_CSV = DATA_DERIVED / "component_correlations.csv"

# Written by 07
PAPER_CHECK_CSV = OUTPUT / "paper_number_check.csv"

# 2021 is omitted: the in-person season was canceled by the COVID-19 pandemic.
YEARS = [y for y in range(2010, 2026) if y != 2021]

# Event types with no first-pick decision to model. Championship Finals
# (Einstein) hold no alliance selection -- division-winning alliances advance
# intact. Festival of Champions was a one-off invitational.
EXCLUDED_EVENT_TYPES = ["Championship Finals", "Festival of Champions"]

# Display order and labels used in the paper's tables.
TYPE_ORDER = [
    "Regional",
    "District",
    "District Championship",
    "District Championship Division",
    "Championship Division",
]
TYPE_LABELS = {
    "Regional": "Regional",
    "District": "District",
    "District Championship": "District Championship",
    "District Championship Division": "District Championship Div.",
    "Championship Division": "Championship Division",
    "ALL": "All events combined",
}

COMPONENTS = ["epa_auto", "epa_teleop", "epa_endgame"]

# Qualification-rank representations compared in Table 6.
RANK_FORMS = {
    "linear_percentile": ["rank_pct"],
    "negative_log_rank": ["neg_log_rank"],
    "inverse_rank": ["inverse_rank"],
    "rank_2_to_8_indicators": [f"rank_{r}" for r in range(2, 9)],
}

# Seeds for the three 25% random choice-set reductions (Supplementary Table A3).
RANDOM_DROP_SEEDS = (104729, 130363, 155921)
RANDOM_DROP_SHARE = 0.25

# Gauss-Hermite nodes for the random-slope mixed logit.
QUADRATURE_POINTS = 20

# Optimizer for the conditional and pooled logits. "bfgs" (statsmodels) is what
# produced the published estimates and reproduces every printed number.
# "newton" iterates to the exact maximum likelihood estimate; see README.md for
# how the two differ. Override per run: python code/run_all.py --optimizer newton
OPTIMIZER = os.environ.get("FRC_OPTIMIZER", "bfgs")

# A fit counts as converged when its Newton decrement -- approximately how far
# its log likelihood is below the maximum -- is below this value. BFGS stops
# up to 0.017 log-likelihood units short (the rank-indicator joint model), which
# moves pseudo-R^2 by about 1e-5; Newton reaches the maximum (< 1e-12).
CONVERGENCE_TOLERANCE = 0.05

TBA_API = "https://www.thebluealliance.com/api/v3"
STATBOTICS_API = "https://api.statbotics.io/v3"
