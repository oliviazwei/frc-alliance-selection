# Expected Points Added and Alliance Selection in the FIRST Robotics Competition

Replication package for **Olivia Wei, *Expected Points Added and Alliance
Selection in the FIRST Robotics Competition*** (revised manuscript, 2026).

The package contains the public data snapshot and the complete Python pipeline
that rebuilds the estimation sample, estimates every model, and regenerates
every table and figure in the paper and its supplementary appendix. A final
step checks **every table cell and every number quoted in the text** against
the manuscript and fails if anything differs.

```bash
pip install -r requirements.txt
python code/run_all.py
```

No API key is needed to reproduce the paper.

---

## Contents

- [The study in brief](#the-study-in-brief)
- [Main results](#main-results)
- [Quick start](#quick-start)
- [Where each exhibit comes from](#where-each-exhibit-comes-from)
- [Repository layout](#repository-layout)
- [Pipeline](#pipeline)
- [Data](#data)
- [Methods in brief](#methods-in-brief)
- [Data details that matter](#data-details-that-matter)
- [Re-downloading or extending the data](#re-downloading-or-extending-the-data)
- [Requirements](#requirements)
- [Limitations](#limitations)
- [Citation](#citation)
- [License and data terms](#license-and-data-terms)

---

## The study in brief

After qualification matches at an FRC event, the highest-ranked teams become
alliance captains and invite partners for the playoffs. The top-ranked captain
invites first, before any other captain removes a team from the pool, so its
accepted first-round partner is the cleanest choice to study.

The paper links alliance records from **The Blue Alliance (TBA)** to team
ratings from **Statbotics** for **1,939 official events, 2010–2025** (2021 is
excluded because the in-person season was canceled). For each event it compares
the team that the top captain's alliance actually accepted with every other team
that was available, and tests three hypotheses:

1. Teams with higher **Expected Points Added (EPA)** are more likely to be the
   accepted partner.
2. EPA's **autonomous, teleoperated, and endgame** components have different
   associations with selection once their different scales are considered.
3. EPA adds more information about the recorded choice than **qualification
   rank** does, under several ways of representing rank.

The main model is a **conditional logit** grouped by event, which compares
candidates only with other teams at the same event, so differences in scoring
scales across seasons and games cancel out.

The results describe **selection patterns**, not a causal rule: the public data
do not show whether captains consulted EPA, record declined invitations, or say
whether EPA-aligned picks won more playoff matches.

## Main results

All conditional-logit intervals are 95% confidence intervals with standard
errors clustered by team.

| Finding | Estimate | Exhibit |
|---|---|---|
| One more end-of-event EPA point | β = 0.213 [0.103, 0.323]; odds × ≈1.24 | Table 4A |
| Same, using EPA at the end of qualifications | β = 0.224 [0.108, 0.341] | Table 4A |
| One within-event SD of EPA | β = 2.490 [2.150, 2.829] | Table 4B |
| Autonomous / teleoperated / endgame EPA (per point, 2016–2025) | 0.275 [0.157, 0.393] / 0.258 [0.051, 0.465] / 0.101 [−0.021, 0.222] | Table 5A |
| Pseudo-R² gain from adding rank to EPA (four rank forms) | +0.004 to +0.009 | Table 6 |
| Pseudo-R² gain from adding EPA to rank (four rank forms) | +0.216 to +0.280 | Table 6 |
| Accepted partner is the highest-EPA candidate | 55.25% of 1,915 events | Table 7 |
| Accepted partner is the highest available qualification rank | 34.62% | Table 7 |
| EPA + rank conditional logit picks the accepted partner | 57.18% | Table 7 |
| Random-slope mixed logit, mean EPA slope | 2.608 [2.469, 2.748] | Supp. Table A3 |

The positive EPA association holds in every season (Supplementary Figure A1),
after dropping low-dispersion events, under random choice-set reductions, and in
the mixed logit. A Hausman–McFadden test rejects strict IIA coefficient stability
for the top-half choice set, which the paper treats as a limitation.

## Quick start

Requires Python 3.9 or newer.

```bash
git clone <https://github.com/oliviazwei/frc-alliance-selection>
cd frc-epa-alliance-selection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python code/run_all.py
```

`run_all.py` runs steps 03–07 from the committed raw data. The last lines should
read:

```
[07] 688 of 688 reported numbers reproduce exactly (651 table cells, 37 in-text values)
```

Useful options:

```bash
python code/run_all.py --from 05            # rebuild tables, figures, and checks only
python code/run_all.py --optimizer newton   # exact MLE, about 1 minute (see below)
python code/run_all.py --download           # re-download raw data first (needs TBA_API_KEY)
```

Every script can also be run on its own (for example `python code/04_estimate_models.py`)
from any working directory.

**Runtime.** Step 04 fits 116 logit models with clustered covariances plus two
mixed logits. With the default optimizer, which reproduces the published
numbers, it took about 5 minutes on a laptop (longer on a busy machine), because
`statsmodels`' conditional logit loops over events in Python. With
`--optimizer newton` the whole pipeline finishes in about a minute. All other
steps take seconds.

## Where each exhibit comes from

Tables are written by `code/05_make_tables.py` as Markdown (viewable on GitHub)
and LaTeX; [`output/tables/README.md`](output/tables/README.md) shows all of them
on one page. Figures are written by `code/06_make_figures.py` as PDF and 300-dpi
PNG.

| Paper exhibit | Content | Output file |
|---|---|---|
| Figure 1 | Procedure for forming playoff alliances | `output/figures/figure1_alliance_selection` |
| Table 1 | Events by event type | `output/tables/table1_event_types` |
| Table 2 | Descriptive statistics by season | `output/tables/table2_descriptives_by_season` |
| Table 3 | Pooled logit by event type | `output/tables/table3_pooled_logit` |
| Table 4 | Conditional logit: end-of-event, qualification-end, and standardized EPA | `output/tables/table4_conditional_logit` |
| Figure 2 | Standardized EPA, accepted partners vs. other candidates | `output/figures/figure2_epa_z_distribution` |
| Table 5 | EPA components (raw, within-event SD, within-event IQR) | `output/tables/table5_epa_components` |
| Figure 3 | Qualification rank of the accepted partner | `output/figures/figure3_partner_rank` |
| Table 6 | EPA vs. four representations of qualification rank | `output/tables/table6_epa_vs_rank` |
| Table 7 | Prediction benchmarks | `output/tables/table7_prediction_benchmarks` |
| Supp. Table A1 | Pooled component sign checks, correlations, contrasts | `output/tables/tableA1_component_checks` |
| Supp. Table A2 | Rank forms by event type | `output/tables/tableA2_rank_by_event_type` |
| Supp. Table A3 | Rank 2–8 exclusion and IIA checks (choice sets, mixed logit) | `output/tables/tableA3_rank_restriction_iia` |
| Supp. Table A4 | Sample flow, missing ranks, EPA-dispersion exclusions | `output/tables/tableA4_data_diagnostics` |
| Supp. Figure A1 | Standardized-EPA coefficient by season | `output/figures/figureA1_epa_by_season` |
| Supp. Figure A2 | Within-event EPA standard deviation by season | `output/figures/figureA2_event_epa_sd` |

`code/07_check_paper_numbers.py` compares every data cell of Tables 1–7 and
A1–A4, plus 37 numbers quoted in the abstract and text, with the manuscript. The
full comparison is saved to `output/paper_number_check.csv`.

## Repository layout

```
code/
  config.py                    paths, sample period, event types, rank forms, seeds
  choice_models.py             conditional, pooled, and mixed logit estimators;
                               clustered covariances; top-1 accuracy
  01_download_events.py        TBA event metadata              (optional; API key)
  02_download_data.py          Statbotics EPA + TBA alliances  (optional; API key)
  03_build_sample.py           raw data -> estimation sample
  04_estimate_models.py        every model in the paper and appendix
  05_make_tables.py            Tables 1-7 and A1-A4 (.md and .tex)
  06_make_figures.py           Figures 1-3 and A1-A2 (.pdf and .png)
  07_check_paper_numbers.py    compares all outputs with the manuscript
  run_all.py                   runs 03-07 (add --download for 01-02)
data/
  raw/                         events.csv, team_events.csv, alliances.csv
  derived/                     analysis_sample.csv and model outputs
output/
  tables/                      generated tables (+ README.md index)
  figures/                     generated figures
  paper_number_check.csv       results of step 07
```

## Pipeline

```
 The Blue Alliance API          Statbotics API
          |                           |
  01_download_events.py      02_download_data.py
          |                           |
          v                           v
  data/raw/events.csv      data/raw/team_events.csv, alliances.csv
                   \              |
                    v             v
                    03_build_sample.py
                            |
                data/derived/analysis_sample.csv
                            |
                    04_estimate_models.py
                            |
   data/derived/model_estimates.csv, coefficient_contrasts.csv,
           component_correlations.csv, diagnostics.csv
                 /                         \
     05_make_tables.py               06_make_figures.py
          |                                 |
   output/tables/                    output/figures/
                 \                         /
                  07_check_paper_numbers.py
                            |
              output/paper_number_check.csv
```

| Step | Reads | Writes |
|---|---|---|
| 01 | TBA API | `data/raw/events.csv` |
| 02 | Statbotics API, TBA API, `events.csv` | `data/raw/team_events.csv`, `data/raw/alliances.csv` |
| 03 | `data/raw/*.csv` | `data/derived/analysis_sample.csv` |
| 04 | raw data, analysis sample | `data/derived/model_estimates.csv`, `coefficient_contrasts.csv`, `component_correlations.csv`, `diagnostics.csv` |
| 05 | analysis sample, derived outputs | `output/tables/*.md`, `*.tex` |
| 06 | analysis sample, `model_estimates.csv` | `output/figures/*.pdf`, `*.png` |
| 07 | analysis sample, derived outputs, `output/tables/*.md` | `output/paper_number_check.csv` |

## Data

### Sources

- **The Blue Alliance** ([website](https://www.thebluealliance.com/),
  [API v3](https://www.thebluealliance.com/apidocs)): event metadata,
  qualification rankings, and playoff alliance members.
- **Statbotics** ([website](https://www.statbotics.io/),
  [documentation](https://statbotics.readthedocs.io/en/latest/),
  [EPA model](https://www.statbotics.io/blog/intro)): end-of-event EPA,
  qualification-end EPA, and the autonomous, teleoperated, and endgame
  components.

The committed files are the snapshot used in the paper. Statbotics revises its
historical estimates over time, so a fresh download can shift results slightly.

### Raw files (`data/raw/`)

| File | Rows | One row per | Key fields |
|---|---:|---|---|
| `events.csv` | 3,318 | TBA event | `key`, `year`, dates, location, `event_type_string` |
| `team_events.csv` | 84,933 | team at an event (1,978 events, 7,007 teams) | EPA measures, `qual_rank`, `num_teams` |
| `alliances.csv` | 19,664 | playoff alliance | `rank` (alliance number), `team1` captain, `team2` first pick, `team3`, `team4`, `declines` |

### From raw data to the analysis sample

| Step | Events |
|---|---:|
| Official events with Statbotics team data, 2010–2025 excl. 2021 | 1,978 |
| − no valid top-alliance accepted partner | 25 |
| − Championship Finals (division winners advance without a new selection) | 13 |
| − Festival of Champions | 1 |
| **Analysis sample** | **1,939** |

The analysis sample has **81,686** team–event rows (candidates), **6,951**
unique teams, and exactly one accepted partner per event. The top captain is
removed from its own choice set. Historical offseason events never enter
because Statbotics does not provide EPA for them.

Model-specific samples are subsets of these rows:

| Sample | Rows | Events | Used for |
|---|---:|---:|---|
| Full | 81,686 | 1,939 | Tables 3–4, Figure 2, season and dispersion checks |
| Components available (2016–2025) | 58,994 | 1,470 | Table 5, Supp. Table A1 |
| EPA and rank both available | 81,296 | 1,915 | Tables 6–7, Figure 3, Supp. Tables A2–A3 |

### Codebook: `data/derived/analysis_sample.csv`

| Variable | Description |
|---|---|
| `event` | TBA event key, e.g. `2019casj` |
| `year` | Season |
| `event_type_string` | Regional, District, District Championship, District Championship Division, or Championship Division |
| `team` | FRC team number |
| `week` | Competition week (Statbotics) |
| `pick1` | 1 if TBA records this team as the top alliance's accepted first-round partner, else 0 |
| `epa_total` | End-of-event EPA in game points (includes playoff matches); the primary exposure |
| `epa_auto`, `epa_teleop`, `epa_endgame` | End-of-event EPA components in game points; 2016 onward only |
| `epa_start` | EPA entering the event |
| `epa_pre_elim` | EPA at the end of qualification matches, before alliance selection |
| `epa_z` | `epa_total` standardized within event: (EPA − event mean) / event SD (paper Eq. 2) |
| `qual_rank` | Official qualification rank (1 = top seed); missing for 390 rows in 36 events |
| `num_teams` | Number of ranked teams at the event (Statbotics) |
| `rank_pct` | 1 − (rank − 1)/(N − 1); 1 = top seed, 0 = bottom (paper Eq. 3) |
| `neg_log_rank` | −log(rank) |
| `inverse_rank` | 1/rank |

### Model outputs (`data/derived/`)

| File | One row per | Key columns |
|---|---|---|
| `model_estimates.csv` | model × sample × event type × covariance × variable | `coef`, `se`, `ci_low`, `ci_high`, `p`, `odds_ratio`, `n`, `n_events`, `loglike`, `pseudo_r2`, `top1_accuracy`, `newton_decrement`, `converged` |
| `coefficient_contrasts.csv` | component difference × scale × covariance | `estimate`, `se`, `ci_low`, `ci_high`, `p` |
| `component_correlations.csv` | component pair × scale | `correlation` |
| `diagnostics.csv` | section × metric × subgroup | sample flow, missing ranks, benchmarks, Hausman test, AICs, and other counts |

`covariance` is one of `conventional`, `team_cluster`, or `event_team_two_way`
for conditional logits; `event_cluster`, `team_cluster`, or `event_team_two_way`
for pooled logits; and `numerical_hessian` for the mixed logit. The paper
reports `event_cluster` for Table 3 and `team_cluster` for the conditional
models unless a note says otherwise.

To pull a number yourself:

```python
import pandas as pd
m = pd.read_csv("data/derived/model_estimates.csv")
m.query("model == 'end_epa' and sample == 'full' and event_type == 'ALL' "
        "and covariance == 'team_cluster'")[["coef", "ci_low", "ci_high", "pseudo_r2"]]
```

Main `model` keys: `pooled_end_epa` (Table 3); `end_epa`, `qual_end_epa`,
`epa_z` (Table 4 and robustness); `components_raw`, `components_event_sd`,
`components_event_iqr` (Table 5); `pooled_components_*` (Supp. Table A1);
`epa_only`, `rank_only_<form>`, `epa_plus_<form>` (Tables 6–7, A2–A3);
`mixed_epa_z*` (Supp. Table A3). The `sample` column distinguishes the full
sample from subsets such as `rank_complete`, `rank_9_or_worse`, `season_2019`,
or `random_drop_25pct_seed_104729`.

## Methods in brief

**Pooled logit (Table 3).** A descriptive benchmark on all team–event rows,
$\Pr(y_{ei}=1)=\Lambda(\alpha+\beta\,\mathrm{EPA}_{ei})$, with standard errors
clustered by event. Its pseudo-R² is relative to an intercept-only logit.

**Conditional logit (Tables 4–7).** For event $e$ with available candidates
$C_e$ (all teams except the captain),

$$\Pr(\text{team } i \text{ is chosen}) = \frac{\exp(\beta' x_{ei})}{\sum_{j \in C_e} \exp(\beta' x_{ej})}.$$

Only within-event differences in $x$ identify $\beta$, which is equivalent to a
logit with a separate fixed effect for each event. Covariance matrices use the
exact analytic Hessian; see [Optimizer](#optimizer-and-exact-maximum-likelihood)
for how the coefficients are computed.

**Exposures.** Raw end-of-event EPA is primary; qualification-end EPA checks
timing; within-event standardized EPA (`epa_z`) supports comparisons across
games. Dividing by the event SD changes the effective raw-point slope at each
event, so the standardized model is not just the raw model in new units.
Components are compared per raw point, per within-event SD, and per
within-event IQR.

**Rank representations (Table 6).** Linear percentile, −log(rank), 1/rank, and
indicators for ranks 2–8 (rank 9 or worse as reference). All EPA-versus-rank
comparisons use the same 81,296-row, 1,915-event sample, and the primary
comparison is the incremental pseudo-R², not coefficient shrinkage.

**Fit.** McFadden's pseudo-R² $= 1 - \ln L_{\text{model}} / \ln L_0$, where for
conditional models $L_0$ gives every candidate in an event an equal chance.
Pooled and conditional pseudo-R² values are not comparable with each other.

**Inference.** Teams appear at many events, so conditional-logit intervals use
sandwich standard errors clustered by team, built from candidate-level score
contributions with the usual $G/(G-1)\cdot(N-1)/(N-K)$ correction. Event/team
two-way clustered results are also stored. If a small event-type two-way
covariance has a negative diagonal entry, its SE is left blank and the case is
logged in `diagnostics.csv` (section `covariance`); none of these enter the
paper's tables.

**IIA checks (Supp. Table A3).** (i) Re-estimate after keeping only the
top-ranked half of each choice set, with a Hausman–McFadden test; (ii) three
seeded random 25% reductions of each choice set (the accepted partner is always
kept); (iii) a mixed logit with an event-level normal random EPA slope,
integrated by 20-point Gauss–Hermite quadrature and compared by AIC.

**Other robustness.** Season-by-season estimates (Supp. Figure A1); excluding
events with EPA SD below 1 point or in the bottom 1% or 5% of dispersion (Supp.
Table A4); excluding candidates ranked 2–8, which keeps only the 271 events
whose accepted partner was ranked 9 or worse (Supp. Table A3).

### Optimizer and exact maximum likelihood

The published estimates were computed with `statsmodels`' BFGS optimizer
(`ConditionalLogit` and `Logit` with `method="bfgs"`). That is the default here
(`OPTIMIZER` in `code/config.py`), so every printed number reproduces exactly.

BFGS stops when the gradient is small, not at the exact maximum. To check how
much that matters, `--optimizer newton` replaces it with a damped
Newton–Raphson that iterates to the exact maximum likelihood estimate. With the
exact MLE, 666 of the 688 checked numbers are unchanged. The other 22 move by
0.001–0.003 in the third decimal, mostly in standardized-EPA coefficients and
confidence bounds. The largest changes are the joint EPA coefficient with
1/rank in Table 6 (2.329 → 2.332) and two top-1 rates there (57.2 → 57.3 and
56.9 → 56.8). No sign, significance call, or conclusion changes. Running
`--optimizer newton` makes step 07 list each of these differences.

Every model in `model_estimates.csv` also records its Newton decrement,
$g'(-H)^{-1}g/2$, which approximates how far its log likelihood is below the
maximum. Under BFGS the largest is 0.017 (the joint EPA and rank-indicator
model), which moves pseudo-R² by about $10^{-5}$. Under Newton every pooled and
conditional logit is below $10^{-12}$; the two mixed logits, which use L-BFGS-B
in both modes, are below $10^{-8}$. Step 04 stops with an error if any
decrement exceeds `CONVERGENCE_TOLERANCE` (0.05).

## Data details that matter

**Statbotics pagination.** The Statbotics Python client defaults to
`limit=100`, which silently truncates each season to its first 100 events
(roughly 40% of recent seasons). `02_download_data.py` calls the REST API
directly and pages through results with `limit=1000` and `offset`.

**Timing of EPA.** `epa_total` is measured at the end of the event, so it
partly reflects playoff matches played after selection. `epa_pre_elim` stops at
the end of qualifications, the information available at selection time. The two
give similar estimates (0.213 vs. 0.224), so playoff updates do not drive the
result.

**Declines and invitation order.** TBA's `declines` field is empty throughout
this snapshot, so the data identify the accepted partner but not the captain's
first invitation. A team ranked 2–8 may decline because it can captain its own
alliance; the rank 2–8 exclusion is a sensitivity check, not a reconstruction of
the invitation sequence.

**2018 components.** In the 2018 source data, mean total EPA (132.542) and the
mean of component sums (133.004) differ by −0.462. The difference is in the
Statbotics snapshot, not introduced by this code (Supp. Table A4).

**District Championship Divisions.** Their end-of-event and qualification-end
coefficients agree to three decimals (0.187109 vs. 0.187101). This is a rounding
coincidence: only 9 of 2,203 underlying EPA pairs are identical (Supp. Table A4).

## Re-downloading or extending the data

Only needed to refresh the snapshot or add a season. Get a free key at
[thebluealliance.com/account](https://www.thebluealliance.com/account), then:

```bash
export TBA_API_KEY=your_key_here          # macOS/Linux
$env:TBA_API_KEY = "your_key_here"        # Windows PowerShell
python code/run_all.py --download
```

Statbotics needs no key. The alliance download makes one request per event
(about 3,000 calls, 10–20 minutes). Copy `.env.example` to `.env` if you prefer
to keep the key in a file; `.env` is git-ignored.

To add a season, extend `YEARS` in `code/config.py` and rerun with `--download`.
Step 07 will then report differences from the published numbers, which is
expected: it checks the paper's snapshot, not new data.

## Requirements

Python 3.9+ with `pandas`, `numpy`, `scipy`, `statsmodels`, and `matplotlib`;
`requests` only for the download steps. See `requirements.txt` for minimum
versions and the versions last used to reproduce the paper. The figures use
Times New Roman when installed and fall back to DejaVu Serif.

## Limitations

These come from the paper and apply to anything built on this code:

- The public data omit scouting, reliability, defense, driver skill, strategic
  fit, and prior partnerships; EPA does not measure defense or breakdown risk.
- Declined invitations and invitation order are not observed.
- The conditional logit assumes IIA; the top-half Hausman–McFadden test rejects
  strict coefficient stability, although the EPA association stays positive in
  the random-slope model.
- Only the top alliance's accepted first-round partner is studied.
- Agreement with recorded picks is not evidence that EPA-aligned picks win more
  playoff matches, so the results are not a recipe for building pick lists.

## Citation

If you use this code or data, please cite the paper:

```bibtex
@unpublished{wei2026epa,
  author = {Wei, Olivia},
  title  = {Expected Points Added and Alliance Selection in the {FIRST} Robotics Competition},
  year   = {2026},
  note   = {Revised manuscript. Replication package: https://github.com/oliviazwei/frc-alliance-selection}
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## License and data terms

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The
paper text is © the author. Data retrieved from The Blue Alliance and Statbotics
remain subject to those services' terms; please respect their API usage
guidelines when re-downloading.

This project is not affiliated with or endorsed by *FIRST*®, The Blue Alliance,
or Statbotics. FIRST® and FIRST® Robotics Competition are registered trademarks
of FIRST.
