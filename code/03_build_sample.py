"""Step 3: build the estimation sample.

The unit of observation is a team-event: one row per team available to the top
alliance captain at a given event. The outcome `pick1` marks the team the
captain actually selected as its first pick.

Construction:
  * keep the rank-1 (top) alliance at each event, which must have a recorded
    captain and first pick;
  * inner-join Statbotics team-event rows, so only events with both EPA data
    and an observed alliance selection enter;
  * drop the captain itself -- it cannot pick itself, so it is not in the
    choice set;
  * drop event types with no first-pick decision (see config);
  * add two within-event transforms used by the paper:
      epa_z    -- EPA standardized within event (mean 0, sd 1)
      rank_pct -- qualification-rank percentile, 1 = top seed, 0 = bottom.

Output: data/derived/analysis_sample.csv
"""

import pandas as pd

from config import (ALLIANCES_CSV, EVENTS_CSV, EXCLUDED_EVENT_TYPES,
                    SAMPLE_CSV, TEAM_EVENTS_CSV, TYPE_ORDER)

team_events = pd.read_csv(TEAM_EVENTS_CSV)
alliances = pd.read_csv(ALLIANCES_CSV)
events = pd.read_csv(EVENTS_CSV)

# Top alliance at each event: captain (team1) and first pick (team2).
top = (
    alliances[alliances["rank"] == 1]
    .dropna(subset=["team1", "team2"])
    .drop_duplicates("event")[["event", "year", "team1", "team2"]]
)

df = team_events.merge(top, on=["event", "year"], how="inner")
df = df[df["team"] != df["team1"]].copy()          # captain is not in the choice set
df["pick1"] = (df["team"] == df["team2"]).astype(int)

df = df.merge(
    events[["key", "event_type_string"]].rename(columns={"key": "event"}),
    on="event", how="left",
)
df = df[~df["event_type_string"].isin(EXCLUDED_EVENT_TYPES)]

# Within-event standardized EPA.
grp = df.groupby("event")["epa_total"]
df["epa_z"] = (df["epa_total"] - grp.transform("mean")) / grp.transform("std")

# Qualification-rank percentile: 1 = top-ranked, 0 = bottom-ranked.
n_teams = df.groupby("event")["num_teams"].transform("max")
n_teams = n_teams.fillna(df.groupby("event")["qual_rank"].transform("max"))
df["rank_pct"] = 1 - (df["qual_rank"] - 1) / (n_teams - 1)

cols = [
    "event", "year", "event_type_string", "team", "week", "pick1",
    "epa_total", "epa_auto", "epa_teleop", "epa_endgame",
    "epa_start", "epa_pre_elim", "epa_z",
    "qual_rank", "num_teams", "rank_pct",
]
df = df[cols].sort_values(["year", "event", "team"]).reset_index(drop=True)
df.to_csv(SAMPLE_CSV, index=False)

print(f"Saved {SAMPLE_CSV}")
print(f"  observations : {len(df):,}")
print(f"  events       : {df['event'].nunique():,}")
print(f"  unique teams : {df['team'].nunique():,}")
print(f"  first picks  : {df['pick1'].sum():,}")
print("\nEvents by type:")
print(df.groupby("event_type_string")["event"].nunique()
        .reindex(TYPE_ORDER).astype(int).to_string())
