"""Shared paths and constants for the replication package.

All scripts locate files relative to this file, so they can be run from any
working directory.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_DERIVED = ROOT / "data" / "derived"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"

for _d in (DATA_RAW, DATA_DERIVED, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

# Raw inputs
EVENTS_CSV = DATA_RAW / "events.csv"           # TBA event metadata
TEAM_EVENTS_CSV = DATA_RAW / "team_events.csv"  # Statbotics team-event EPA
ALLIANCES_CSV = DATA_RAW / "alliances.csv"      # TBA playoff alliances

# Derived outputs
SAMPLE_CSV = DATA_DERIVED / "analysis_sample.csv"
RESULTS_CSV = DATA_DERIVED / "results.csv"

# 2021 is omitted: the in-person season was canceled by the COVID-19 pandemic.
YEARS = [y for y in range(2010, 2026) if y != 2021]

# Event types with no first-pick decision to model. Championship Finals
# (Einstein) hold no alliance selection -- division-winning alliances advance
# intact. Festival of Champions was a one-off invitational.
EXCLUDED_EVENT_TYPES = ["Championship Finals", "Festival of Champions"]

# Display order and short labels used in the paper's tables.
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
}

TBA_API = "https://www.thebluealliance.com/api/v3"
STATBOTICS_API = "https://api.statbotics.io/v3"
