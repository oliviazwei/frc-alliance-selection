"""Step 2: download team-event EPA (Statbotics) and playoff alliances (TBA).

Two important details, both of which are easy to get wrong:

1. Statbotics' Python client defaults to `limit=100`, which silently truncates
   each season to its first 100 events -- roughly 40% of recent seasons. This
   script calls the REST API directly and paginates with limit=1000 & offset,
   so every event is retrieved.

2. `epa.stats.pre_elim` is the team's EPA at the end of qualification matches,
   i.e. the information the alliance captain actually had at the moment of the
   draft. `epa.breakdown.*` is measured at the end of the event and therefore
   partly reflects playoff matches played *after* selection. Both are kept so
   the paper can check that the results are not driven by that feedback.

Requires TBA_API_KEY for the alliance data (see README). This step is optional
for replication -- both output files are included in the repository.

Outputs: data/raw/team_events.csv, data/raw/alliances.csv
"""

import os
import sys
import time

import pandas as pd
import requests

from config import (ALLIANCES_CSV, EVENTS_CSV, STATBOTICS_API, TBA_API,
                    TEAM_EVENTS_CSV, YEARS)

KEY = os.environ.get("TBA_API_KEY")
if not KEY:
    sys.exit("TBA_API_KEY is not set -- see README (needed for alliance data).")

session = requests.Session()


def get_json(url, headers=None, tries=4):
    """GET with retries. Returns None on 404 (event has no data yet)."""
    for attempt in range(tries):
        try:
            r = session.get(url, headers=headers, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


# --------------------------------------------------------------------------
# Statbotics: team-event rows, paginated so nothing is truncated
# --------------------------------------------------------------------------
rows = []
for year in YEARS:
    offset, n_year = 0, 0
    while True:
        batch = get_json(
            f"{STATBOTICS_API}/team_events?year={year}&limit=1000&offset={offset}"
        )
        if not batch:
            break
        for te in batch:
            epa = te.get("epa") or {}
            breakdown = epa.get("breakdown") or {}
            stats = epa.get("stats") or {}
            qual = (te.get("record") or {}).get("qual") or {}
            rows.append({
                "event": te.get("event"),
                "year": te.get("year"),
                "team": te.get("team"),
                "week": te.get("week"),
                "epa_total": breakdown.get("total_points"),
                "epa_auto": breakdown.get("auto_points"),
                "epa_teleop": breakdown.get("teleop_points"),
                "epa_endgame": breakdown.get("endgame_points"),
                "epa_start": stats.get("start"),
                "epa_pre_elim": stats.get("pre_elim"),
                "qual_rank": qual.get("rank"),
                "num_teams": qual.get("num_teams"),
            })
        n_year += len(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    print(f"[statbotics] {year}: {n_year:,} team-event rows", flush=True)

team_events = pd.DataFrame(rows)
team_events.to_csv(TEAM_EVENTS_CSV, index=False)
print(f"[statbotics] saved {TEAM_EVENTS_CSV}: {len(team_events):,} rows, "
      f"{team_events['event'].nunique():,} events\n")

# --------------------------------------------------------------------------
# The Blue Alliance: playoff alliance composition, one call per event
# --------------------------------------------------------------------------
events = pd.read_csv(EVENTS_CSV)
events = events[events["year"].isin(YEARS)]


def team_number(key):
    """'frc254' -> 254. Offseason B-teams ('frc1234B') have no Statbotics row."""
    if not key:
        return None
    digits = key[3:]
    return int(digits) if digits.isdigit() else None


alliance_rows = []
headers = {"X-TBA-Auth-Key": KEY}
for i, (key, year) in enumerate(zip(events["key"], events["year"]), start=1):
    payload = get_json(f"{TBA_API}/event/{key}/alliances", headers=headers)
    if not payload:
        continue
    for rank, alliance in enumerate(payload, start=1):
        picks = (alliance.get("picks") or [])[:4]
        nums = [team_number(t) for t in picks] + [None] * (4 - len(picks))
        alliance_rows.append({
            "event": key,
            "year": year,
            "rank": rank,
            "team1": nums[0],   # captain
            "team2": nums[1],   # first pick
            "team3": nums[2],
            "team4": nums[3],
            # teams that declined an invitation from this alliance
            "declines": ";".join(alliance.get("declines") or []),
        })
    if i % 200 == 0:
        print(f"[tba] {i:,}/{len(events):,} events, "
              f"{len(alliance_rows):,} alliance rows", flush=True)

alliances = pd.DataFrame(alliance_rows)
alliances.to_csv(ALLIANCES_CSV, index=False)
print(f"[tba] saved {ALLIANCES_CSV}: {len(alliances):,} alliance rows across "
      f"{alliances['event'].nunique():,} events")
