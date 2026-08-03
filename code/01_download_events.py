"""Step 1: download the full list of official FRC events from The Blue Alliance.

Requires a free TBA API key, supplied via the TBA_API_KEY environment variable
(see README). This step is optional for replication -- data/raw/events.csv is
included in the repository.

Output: data/raw/events.csv
"""

import os
import sys

import pandas as pd
import requests

from config import EVENTS_CSV, TBA_API, YEARS

KEY = os.environ.get("TBA_API_KEY")
if not KEY:
    sys.exit(
        "TBA_API_KEY is not set. Get a free key at "
        "https://www.thebluealliance.com/account and export it, e.g.\n"
        "    export TBA_API_KEY=your_key_here        (macOS/Linux)\n"
        "    $env:TBA_API_KEY = 'your_key_here'      (Windows PowerShell)"
    )

KEEP = [
    "key", "year", "start_date", "end_date", "name", "event_code",
    "event_type", "event_type_string", "city", "state_prov", "country",
]

rows = []
session = requests.Session()
for year in YEARS:
    r = session.get(f"{TBA_API}/events/{year}",
                    headers={"X-TBA-Auth-Key": KEY}, timeout=60)
    r.raise_for_status()
    rows.extend(r.json())
    print(f"{year}: {len(r.json())} events", flush=True)

df = pd.DataFrame(rows)[KEEP]
# Drop events that were scheduled but canceled (mostly spring 2020).
df = df[~df["name"].str.contains("(Cancelled)", na=False, regex=False)]
df.to_csv(EVENTS_CSV, index=False)
print(f"\nSaved {EVENTS_CSV} ({len(df):,} events)")
