================================================================================
CODE DIRECTORY -- Performance Metrics and Alliance Selection in FRC
================================================================================

This file explains what each script does, what it reads, and what it writes.
For the project overview, data dictionary, and quick start, see the README.md
in the repository root.

All scripts locate files relative to code/config.py, so they can be run from
any working directory. Run them individually, or run everything with:

    python code/run_all.py                 (steps 03-05; no API key needed)
    python code/run_all.py --download      (also steps 01-02; needs API key)


--------------------------------------------------------------------------------
PIPELINE
--------------------------------------------------------------------------------

    [TBA API]                    [Statbotics API]
        |                               |
        v                               v
  01_download_events.py  ------> 02_download_data.py
        |                               |
        v                               v
   data/raw/events.csv    data/raw/team_events.csv
                          data/raw/alliances.csv
                    \           |           /
                     v          v          v
                    03_build_sample.py
                             |
                             v
              data/derived/analysis_sample.csv
                       /             \
                      v               v
             04_estimate.py    05_make_exhibits.py
                      |               |
                      v               |
        data/derived/results.csv ---->|
                                      v
                          paper/tables/*.tex
                          paper/figures/*.pdf

Steps 01 and 02 download the raw data and are NOT required for replication --
their outputs are committed to the repository. Run them only to refresh the
data or extend it to a new season. Steps 03-05 reproduce every number in the
paper from the committed data.


--------------------------------------------------------------------------------
FILES
--------------------------------------------------------------------------------

config.py
    Shared paths and constants imported by every other script. Defines the
    directory layout, the file names above, the sample period (2010-2025
    excluding 2021), the event types excluded from the analysis, and the
    display order and labels used in the paper's tables.

    Edit this file -- not the individual scripts -- to change the sample
    period or the set of event types.

01_download_events.py
    Downloads metadata for every official FRC event from The Blue Alliance:
    event key, year, dates, city/state/country, and event type. Drops events
    whose name marks them as canceled (mostly spring 2020).

    Reads : TBA API                        Needs : TBA_API_KEY
    Writes: data/raw/events.csv            Runtime: under a minute

02_download_data.py
    Downloads the two main data sources.

    (a) Statbotics team-event rows: one per team per event, with overall EPA,
        its autonomous/teleoperated/endgame components, EPA at the end of
        qualification matches, and the team's qualification rank.
    (b) The Blue Alliance playoff alliances: captain and first/second/third
        pick for every alliance, plus any declined invitations.

    Reads : Statbotics API, TBA API, data/raw/events.csv
    Writes: data/raw/team_events.csv, data/raw/alliances.csv
    Needs : TBA_API_KEY (Statbotics requires no key)
    Runtime: roughly 10-20 minutes -- part (b) makes one request per event,
             about 3,000 calls

03_build_sample.py
    Builds the estimation sample. This is the single source of truth for how
    the sample is defined; 04 and 05 both read its output rather than
    re-deriving it.

    Steps:
      - keep the rank-1 (top) alliance at each event, requiring a recorded
        captain and first pick;
      - inner-join the Statbotics team-event rows, so only events with both
        EPA data and an observed alliance selection survive;
      - drop the captain itself (it cannot pick itself, so it is not in the
        choice set);
      - drop event types with no first-pick decision (see config.py);
      - flag the chosen team with pick1 = 1;
      - add epa_z (EPA standardized within event) and rank_pct (qualification
        rank as a within-event percentile, 1 = top seed, 0 = bottom seed).

    Reads : data/raw/*.csv
    Writes: data/derived/analysis_sample.csv    Runtime: seconds

    Expected output: 81,686 observations, 1,939 events, 6,951 teams.

04_estimate.py
    Estimates all eight specifications reported in the paper -- pooled logit
    and conditional logit, each for overall EPA and for the EPA components,
    plus the decision-time EPA check, the standardized-EPA version, and the
    EPA-versus-ranking horse race. Every specification is run once on the full
    sample and once per event type.

    Reads : data/derived/analysis_sample.csv
    Writes: data/derived/results.csv     Runtime: a few minutes

    Output format: one row per estimated coefficient, identified by spec,
    event_type ("ALL" or a specific type), and var; with coef, se, p, n,
    n_events, and pseudo_r2.

05_make_exhibits.py
    Generates every table and figure in the paper. Regression tables are
    written directly from results.csv, so the numbers in the paper cannot
    drift from the estimates.

    Reads : data/derived/analysis_sample.csv, data/derived/results.csv,
            data/raw/team_events.csv (for Figure 1)
    Writes: paper/tables/tab1_event_types.tex        (Table 1)
            paper/tables/tab2_descriptives.tex       (Table 2)
            paper/tables/tab3_pooled_logit.tex       (Table 3)
            paper/tables/tab4_conditional_logit.tex  (Table 4)
            paper/tables/tab5_components.tex         (Table 5)
            paper/tables/tab6_horserace.tex          (Table 6)
            paper/figures/fig_team_event_counts.pdf/.eps  (Figure 1)
            paper/figures/fig_epa_z_hist.pdf/.eps         (Figure 2)
            paper/figures/fig_pick_rank.pdf/.eps          (Figure 3)
    Runtime: seconds

    The script also prints several lines tagged [text check]. These are
    statistics quoted in the body of the paper that cannot be injected into
    LaTeX automatically -- the mean standardized EPA of first picks, the share
    of events where the pick is the highest-EPA candidate, the distribution of
    the pick's qualification rank, and the range of the teleoperated share of
    EPA. If the data are ever refreshed, check these against the prose.

run_all.py
    Runs 03, 04, and 05 in order, stopping if any step fails. Pass --download
    to run 01 and 02 first.


--------------------------------------------------------------------------------
IMPLEMENTATION NOTES
--------------------------------------------------------------------------------

Statbotics pagination.
    The Statbotics Python client defaults to limit=100, which silently
    truncates each season to its first 100 events -- roughly 40 percent of
    recent seasons. 02_download_data.py calls the REST API directly and
    paginates with limit=1000 and an offset, so all events are retrieved. If
    you rewrite the download using the client library, pass an explicit limit.

Timing of the EPA measure.
    epa_total is measured at the END of the event and therefore partly
    reflects playoff matches played after alliance selection -- teams picked
    by the top alliance tend to advance far, which would mechanically inflate
    the estimated coefficient. epa_pre_elim is EPA at the end of the
    qualification matches, i.e. the information the captain actually had. The
    clogit_pre_elim specification is the robustness check; its coefficient is
    slightly larger, so the results are not driven by that feedback.

Component EPA coverage.
    Statbotics reports the autonomous/teleoperated/endgame breakdown only from
    the 2016 season onward. Any specification using the components therefore
    covers 2016-2025 and has a smaller sample (58,994 observations, 1,470
    events). This is handled automatically by prep() in 04_estimate.py, which
    drops rows with missing regressors and then keeps only events that still
    contain exactly one observed pick and at least two candidates.

Excluded event types.
    Championship Finals (Einstein) hold no alliance selection -- the
    division-winning alliances advance intact, so no first-pick choice ever
    occurs there. Festival of Champions was a one-off invitational. Offseason
    events are absent for a different reason: Statbotics does not serve
    historical EPA for them, so they never enter the raw data at all.

Pseudo-R-squared.
    The two model families are scored against different null models: an
    intercept-only logit for the pooled specifications, and a uniform-random
    pick from the event's choice set for the conditional specifications. Each
    is the natural benchmark for its own model, but the two are NOT comparable
    to each other -- do not read the jump from about 0.04 to about 0.53 as a
    thirteenfold improvement in fit.

Standard errors.
    Pooled logits cluster by event, because exactly one team per event can be
    selected and rows within an event are mechanically dependent. Conditional
    logits treat the event as the unit of observation, so the likelihood
    already accounts for that dependence and default standard errors apply.
    Neither model clusters by team, although teams recur across events; a
    two-way or team-clustered variant would be a reasonable robustness check.


--------------------------------------------------------------------------------
EXTENDING TO A NEW SEASON
--------------------------------------------------------------------------------

1. Add the year to YEARS in config.py.
2. Set TBA_API_KEY and run: python code/run_all.py --download
3. Check the [text check] lines printed by 05_make_exhibits.py against the
   corresponding numbers in the body of paper/main.tex, and update the prose
   where they have moved.
4. Recompile the paper:
       cd paper
       pdflatex main && bibtex main && pdflatex main && pdflatex main


--------------------------------------------------------------------------------
TROUBLESHOOTING
--------------------------------------------------------------------------------

"TBA_API_KEY is not set"
    Only steps 01 and 02 need it. To replicate the paper, run
    "python code/run_all.py" without --download. To download, get a free key
    at https://www.thebluealliance.com/account and set it:
        export TBA_API_KEY=your_key_here       (macOS/Linux)
        $env:TBA_API_KEY = 'your_key_here'     (Windows PowerShell)

ModuleNotFoundError
    pip install -r requirements.txt

A conditional logit prints "[skip]"
    That event type had too few observations or no variation in the outcome
    for the specification. The remaining rows are unaffected.

Numbers differ from the paper
    Confirm data/raw/ has not been re-downloaded -- Statbotics revises EPA
    estimates over time, so a fresh download of the same seasons can shift
    results slightly. The committed raw files are the ones used in the paper.
