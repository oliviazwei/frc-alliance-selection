"""Step 7: check every reported number against the revised manuscript.

Two kinds of checks:
  * every data cell of Tables 1-7 and Supplementary Tables A1-A4, comparing
    output/tables/*.md with the values printed in the manuscript (transcribed
    below), and
  * numbers quoted in the abstract and text but not printed in a table.

Writes output/paper_number_check.csv and exits with status 1 if anything
differs, so `python code/run_all.py` fails loudly when the data or code
change the paper's results.
"""

import sys

import numpy as np
import pandas as pd

from config import (CONTRASTS_CSV, DIAGNOSTICS_CSV, MODELS_CSV,
                    PAPER_CHECK_CSV, RANK_FORMS, SAMPLE_CSV, TABLES)

# ---------------------------------------------------------------------------
# Tables as printed in the manuscript (data rows only, cells split on "|")
# ---------------------------------------------------------------------------
PAPER_TABLES = {
    "table1_event_types": ["""
Regional | 791
District | 893
District Championship | 99
District Championship Div. | 54
Championship Division | 102
Total | 1,939
"""],
    "table2_descriptives_by_season": ["""
2010 | 49 | 1,611 | 1.2 | 1.0 | – | – | –
2011 | 55 | 1,824 | 11.7 | 10.8 | – | – | –
2012 | 68 | 2,207 | 7.6 | 5.9 | – | – | –
2013 | 78 | 2,437 | 19.8 | 13.4 | – | – | –
2014 | 102 | 2,694 | 31.2 | 18.5 | – | – | –
2015 | 117 | 2,871 | 27.0 | 17.4 | – | – | –
2016 | 134 | 3,113 | 25.5 | 9.2 | 7.1 | 14.3 | 4.1
2017 | 161 | 3,327 | 72.8 | 24.8 | 8.5 | 35.6 | 28.7
2018 | 176 | 3,613 | 132.5 | 28.4 | 17.6 | 100.0 | 15.5
2019 | 191 | 3,757 | 17.1 | 6.4 | 4.1 | 8.9 | 4.0
2020 | 52 | 1,956 | 27.2 | 14.1 | 9.3 | 6.3 | 11.5
2022 | 182 | 3,059 | 16.8 | 10.5 | 4.9 | 7.1 | 4.9
2023 | 183 | 3,288 | 28.8 | 13.8 | 7.9 | 14.4 | 6.3
2024 | 189 | 3,473 | 16.6 | 9.3 | 6.1 | 8.9 | 1.6
2025 | 202 | 3,689 | 30.4 | 18.0 | 6.6 | 20.3 | 3.5
"""],
    "table3_pooled_logit": ["""
Regional | 0.0149 (0.0007) | 0.049 | 37,044 | 791
District | 0.0125 (0.0005) | 0.039 | 30,584 | 893
District Championship | 0.0097 (0.0014) | 0.026 | 4,155 | 99
District Championship Div. | 0.0094 (0.0013) | 0.028 | 2,203 | 54
Championship Division | 0.0116 (0.0010) | 0.041 | 7,700 | 102
All events combined | 0.0122 (0.0004) | 0.038 | 81,686 | 1,939
"""],
    "table4_conditional_logit": ["""
Regional | 0.220 [0.062, 0.379] | 0.557 | 0.230 [0.069, 0.392] | 0.551 | 37,044 | 791
District | 0.227 [0.067, 0.388] | 0.548 | 0.236 [0.067, 0.404] | 0.544 | 30,584 | 893
District Championship | 0.149 [-0.165, 0.463] | 0.358 | 0.197 [-0.270, 0.664] | 0.410 | 4,155 | 99
District Championship Div. | 0.187 [-0.424, 0.799] | 0.443 | 0.187 [-0.415, 0.790] | 0.445 | 2,203 | 54
Championship Division | 0.170 [-0.140, 0.480] | 0.456 | 0.166 [-0.138, 0.470] | 0.450 | 7,700 | 102
All events combined | 0.213 [0.103, 0.323] | 0.531 | 0.224 [0.108, 0.341] | 0.530 | 81,686 | 1,939
""", """
Regional | 2.368 [1.901, 2.835] | 0.634 | 46.8 | 47 | 37,044 | 791
District | 2.681 [2.123, 3.238] | 0.622 | 34.2 | 36 | 30,584 | 893
District Championship | 2.082 [1.375, 2.788] | 0.465 | 42.0 | 44 | 4,155 | 99
District Championship Div. | 2.365 [0.729, 4.002] | 0.520 | 40.8 | 39 | 2,203 | 54
Championship Division | 2.615 [1.308, 3.922] | 0.535 | 75.5 | 74 | 7,700 | 102
All events combined | 2.490 [2.150, 2.829] | 0.610 | 42.1 | 39 | 81,686 | 1,939
"""],
    "table5_epa_components": ["""
Regional | 0.263 [0.096, 0.429] | 0.262 [-0.067, 0.591] | 0.101 [-0.085, 0.287] | 0.590 | 22,675 | 501
District | 0.315 [0.123, 0.506] | 0.282 [-0.007, 0.571] | 0.127 [-0.061, 0.315] | 0.593 | 25,625 | 757
District Championship | 0.168 [-0.104, 0.440] | 0.185 [-0.391, 0.761] | 0.022 [-0.273, 0.318] | 0.370 | 3,266 | 84
District Championship Div. | 0.229 [-0.284, 0.741] | 0.205 [-0.537, 0.947] | 0.039 [-0.838, 0.916] | 0.461 | 2,203 | 54
Championship Division | 0.230 [-0.130, 0.590] | 0.215 [-0.372, 0.802] | 0.035 [-0.371, 0.440] | 0.483 | 5,225 | 74
All events combined | 0.275 [0.157, 0.393] | 0.258 [0.051, 0.465] | 0.101 [-0.021, 0.222] | 0.564 | 58,994 | 1,470
""", """
Raw EPA points | 0.275 [0.157, 0.393] | 0.258 [0.051, 0.465] | 0.101 [-0.021, 0.222] | 0.564 | 1,470
One within-event SD | 0.671 [0.537, 0.806] | 1.851 [1.551, 2.150] | 0.521 [0.397, 0.646] | 0.619 | 1,470
One within-event IQR | 0.790 [0.606, 0.974] | 2.297 [1.854, 2.741] | 0.539 [0.406, 0.672] | 0.607 | 1,470
"""],
    "table6_epa_vs_rank": ["""
Linear rank percentile | 0.616 | 0.408 | 0.624 | +0.009 | +0.216 | 2.215 [2.088, 2.342] | 57.2
-log(rank) | 0.616 | 0.404 | 0.622 | +0.007 | +0.218 | 2.253 [1.664, 2.843] | 57.5
1/rank | 0.616 | 0.340 | 0.620 | +0.004 | +0.280 | 2.329 [2.000, 2.659] | 57.2
Indicators for ranks 2–8 | 0.616 | 0.378 | 0.622 | +0.006 | +0.243 | 2.297 [1.961, 2.634] | 56.9
"""],
    "table7_prediction_benchmarks": ["""
Uniform random choice (event-specific) | 49.1 | 1,915 | 2.57
Choose highest-EPA candidate | 1058 | 1,915 | 55.25
Choose highest available qualification rank | 663 | 1,915 | 34.62
Always choose qualification rank 2 | 661 | 1,915 | 34.52
EPA-only conditional logit | 1058 | 1,915 | 55.25
EPA plus linear-rank conditional logit | 1095 | 1,915 | 57.18
"""],
    "tableA1_component_checks": ["""
Raw components, no season controls | 0.255 [0.237, 0.272] | -0.023 [-0.026, -0.020] | 0.009 [0.004, 0.015] | 58,994
Raw components plus season indicators | 0.186 [0.165, 0.208] | 0.086 [0.077, 0.096] | 0.041 [0.025, 0.057] | 58,994
Components standardized within season | 0.477 [0.401, 0.553] | 0.954 [0.885, 1.024] | 0.215 [0.152, 0.277] | 58,994
""", """
Auto and teleop | 0.825 | 0.685
Auto and endgame | 0.454 | 0.502
Teleop and endgame | 0.493 | 0.551
""", """
Raw points | 0.017 [-0.138, 0.173] | 0.175 [0.079, 0.271] | 0.157 [0.020, 0.295]
Within-event SD | -1.179 [-1.407, -0.951] | 0.150 [0.034, 0.266] | 1.330 [1.113, 1.546]
Within-event IQR | -1.508 [-1.873, -1.142] | 0.251 [0.092, 0.410] | 1.758 [1.403, 2.114]
"""],
    "tableA2_rank_by_event_type": ["""
Regional | 0.405 | 0.412 | 0.350 | 0.382 | 36,862 | 788
District | 0.440 | 0.434 | 0.369 | 0.424 | 30,522 | 892
District Championship | 0.335 | 0.324 | 0.247 | 0.308 | 4,009 | 79
District Championship Div. | 0.307 | 0.272 | 0.189 | 0.234 | 2,203 | 54
Championship Division | 0.315 | 0.302 | 0.219 | 0.226 | 7,700 | 102
All events combined | 0.408 | 0.404 | 0.340 | 0.378 | 81,296 | 1,915
""", """
Regional | 0.635 | 0.641 (+0.006) | 0.640 (+0.005) | 0.639 (+0.004) | 0.641 (+0.006)
District | 0.622 | 0.635 (+0.013) | 0.633 (+0.010) | 0.629 (+0.006) | 0.635 (+0.013)
District Championship | 0.554 | 0.564 (+0.010) | 0.560 (+0.006) | 0.557 (+0.003) | 0.570 (+0.016)
District Championship Div. | 0.520 | 0.521 (+0.002) | 0.520 (+0.000) | 0.522 (+0.002) | 0.531 (+0.011)
Championship Division | 0.535 | 0.544 (+0.009) | 0.538 (+0.003) | 0.535 (+0.000) | 0.540 (+0.005)
All events combined | 0.616 | 0.624 (+0.009) | 0.622 (+0.007) | 0.620 (+0.004) | 0.622 (+0.006)
"""],
    "tableA3_rank_restriction_iia": ["""
Complete EPA-and-rank sample | 0.616 | 0.408 | 0.624 | +0.009 | +0.216 | 81,296 | 1,915
Candidates ranked 9 or worse | 0.594 | 0.266 | 0.605 | +0.011 | +0.339 | 11,605 | 271
""", """
Same events, full choice set | 2.531 [2.178, 2.883] | 0.618 | 80,801 | 1,905
Same events, top rank half only | 2.421 [2.038, 2.805] | 0.539 | 39,869 | 1,905
Randomly drop 25%, seed 104729 | 2.550 [2.174, 2.925] | 0.642 | 61,842 | 1,939
Randomly drop 25%, seed 130363 | 2.490 [2.147, 2.833] | 0.632 | 61,697 | 1,939
Randomly drop 25%, seed 155921 | 2.553 [2.173, 2.933] | 0.642 | 61,649 | 1,939
""", """
Fixed EPA slope | 2.490 [2.402, 2.577] | – | 5579.7
Random EPA slope | 2.608 [2.469, 2.748] | 0.452 [0.289, 0.708] | 5576.5
Fixed EPA slope plus rank | 2.215 [2.113, 2.317] | – | 5335.1
Random EPA slope plus rank | 2.374 [2.222, 2.526] | 0.552 [0.388, 0.785] | 5327.7
"""],
    "tableA4_data_diagnostics": ["""
Events in pre-merge team-event data | 1,978
Events in analysis sample | 1,939
Excluded: no valid top-alliance accepted pick | 25
Excluded: Championship Finals | 13
Excluded: Festival of Champions | 1
Rows with missing qualification rank | 390
Events affected by any missing rank | 36
Events with selected partner rank missing | 24
Alliance rows with recorded declines | 0
DCD rows / events / first season | 2,203 / 54 / 2017
DCD exact end/qualification EPA matches | 9
DCD mean absolute difference between EPA measures | 1.203
2018 overall EPA mean | 132.542
2018 component-sum mean | 133.004
2018 mean total minus component sum | -0.462
""", """
2010 | District | 37 | 1
2010 | Regional | 166 | 3
2014 | District | 1 | 1
2015 | Regional | 1 | 1
2017 | District Championship | 11 | 1
2018 | District Championship | 17 | 2
2019 | District Championship | 16 | 2
2020 | District | 3 | 1
2022 | District | 9 | 2
2022 | District Championship | 22 | 3
2022 | Regional | 15 | 4
2023 | District | 6 | 1
2023 | District Championship | 27 | 4
2024 | District | 3 | 1
2024 | District Championship | 27 | 4
2025 | District | 3 | 1
2025 | District Championship | 26 | 4
""", """
All events | 2.490 [2.150, 2.829] | 0.610 | 81,686 | 1,939
Exclude EPA SD below 1 | 2.515 [2.167, 2.863] | 0.613 | 79,780 | 1,899
Exclude bottom 1 percent of EPA SD | 2.509 [2.161, 2.856] | 0.613 | 80,703 | 1,919
Exclude bottom 5 percent of EPA SD | 2.532 [2.176, 2.887] | 0.614 | 77,169 | 1,842
"""],
}


def normalize(cell):
    return "".join(str(cell).replace("−", "-").split())


def parse_paper(block):
    return [[c.strip() for c in line.split("|")]
            for line in block.strip().splitlines()]


def parse_markdown(path):
    """Data rows of each pipe table in a generated Markdown file."""
    panels, current = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            if current is None:
                current = []
                panels.append(current)
            current.append([c.strip() for c in line.strip("|").split("|")])
        else:
            current = None
    return [panel[2:] for panel in panels]      # drop header and rule rows


checks = []


def check(where, item, expected, reproduced):
    checks.append({
        "where": where, "item": item, "expected": str(expected),
        "reproduced": str(reproduced),
        "match": normalize(expected) == normalize(reproduced),
    })


for name, panels in PAPER_TABLES.items():
    generated = parse_markdown(TABLES / f"{name}.md")
    if len(generated) != len(panels):
        check(name, "number of panels", len(panels), len(generated))
        continue
    for p, (block, rows) in enumerate(zip(panels, generated)):
        expected_rows = parse_paper(block)
        panel = f"{name} panel {'ABC'[p]}" if len(panels) > 1 else name
        if len(expected_rows) != len(rows):
            check(panel, "number of rows", len(expected_rows), len(rows))
            continue
        for expected_row, row in zip(expected_rows, rows):
            for col, (expected, reproduced) in enumerate(zip(expected_row, row)):
                item = expected_row[0] if col == 0 else \
                    f"{expected_row[0]} / column {col + 1}"
                check(panel, item, expected, reproduced)


# ---------------------------------------------------------------------------
# Numbers quoted in the text
# ---------------------------------------------------------------------------
sample = pd.read_csv(SAMPLE_CSV)
models = pd.read_csv(MODELS_CSV)
contrasts = pd.read_csv(CONTRASTS_CSV)
diagnostics = pd.read_csv(DIAGNOSTICS_CSV).fillna({"subgroup": ""})


def est(model, sample_name, var, event_type="ALL", covariance="team_cluster"):
    rows = models[
        (models["model"] == model) & (models["sample"] == sample_name)
        & (models["event_type"] == event_type) & (models["var"] == var)
        & (models["covariance"] == covariance)
    ]
    assert len(rows) == 1, (model, sample_name, var, event_type, covariance)
    return rows.iloc[0]


def diag(section, metric, subgroup=""):
    rows = diagnostics[(diagnostics["section"] == section)
                       & (diagnostics["metric"] == metric)
                       & (diagnostics["subgroup"] == subgroup)]
    assert len(rows) == 1, (section, metric, subgroup)
    return rows.iloc[0]


def ci(row):
    return f"{row['coef']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


def r2(model, form=None, sample_name="rank_complete"):
    var = "epa_z" if form is None else RANK_FORMS[form][0]
    return est(model, sample_name, var, covariance="conventional")["pseudo_r2"]


def span(values, decimals=3):
    return f"{min(values):.{decimals}f}–{max(values):.{decimals}f}"


end = est("end_epa", "full", "epa_total")
epa_only = r2("epa_only")
rank_only = {f: r2(f"rank_only_{f}", f) for f in RANK_FORMS}
joint = {f: r2(f"epa_plus_{f}") for f in RANK_FORMS}
benchmark = {m: diag("prediction_benchmark", m)["value"] for m in
             ["highest_epa", "highest_available_rank", "qualification_rank_2",
              "epa_plus_linear_rank_conditional_logit"]}
picks = sample[sample["pick1"] == 1]
pick_rank = picks["qual_rank"].dropna()
pooled = [est("pooled_end_epa", "full", "epa_total", t, "event_cluster")
          for t in models.loc[models["model"] == "pooled_end_epa",
                              "event_type"].unique()]
by_year = sample.groupby("year")[["epa_total", "epa_teleop"]].mean().dropna()
teleop_share = by_year["epa_teleop"] / by_year["epa_total"]
mixed = est("mixed_epa_z", "full", "epa_z_mean", covariance="numerical_hessian")
aic = {g: diag("mixed_logit", "aic", g)["value"]
       for g in ["fixed_epa_z", "mixed_epa_z"]}
seed_coefs = [est("epa_z", s, "epa_z")["coef"] for s in models["sample"].unique()
              if s.startswith("random_drop_")]

text_checks = [
    ("Abstract", "events", "1,939", f"{sample['event'].nunique():,}"),
    ("Abstract", "end-of-event EPA coefficient", "0.213 [0.103, 0.323]",
     ci(end)),
    ("Abstract", "standardized EPA coefficient", "2.490 [2.150, 2.829]",
     ci(est("epa_z", "full", "epa_z"))),
    ("Abstract", "pseudo-R² gain from adding rank to EPA", "0.004–0.009",
     span([joint[f] - epa_only for f in RANK_FORMS])),
    ("Abstract", "pseudo-R² gain from adding EPA to rank", "0.216–0.280",
     span([joint[f] - rank_only[f] for f in RANK_FORMS])),
    ("Abstract", "common-sample events", "1,915",
     f"{int(est('epa_only', 'rank_complete', 'epa_z')['n_events']):,}"),
    ("Abstract", "EPA-plus-rank top-1 accuracy", "57.18%",
     f"{100 * benchmark['epa_plus_linear_rank_conditional_logit']:.2f}%"),
    ("Abstract", "highest-available-rank accuracy", "34.62%",
     f"{100 * benchmark['highest_available_rank']:.2f}%"),
    ("Sample", "pre-merge unique teams", "7,007",
     f"{int(diag('sample_flow', 'premerge_teams')['value']):,}"),
    ("Sample", "pre-merge events", "1,978",
     f"{int(diag('sample_flow', 'premerge_events')['value']):,}"),
    ("Sample", "team-event observations", "81,686", f"{len(sample):,}"),
    ("Sample", "unique teams", "6,951", f"{sample['team'].nunique():,}"),
    ("Sample", "2025 teams / events", "3,689 / 202",
     f"{sample.loc[sample['year'] == 2025, 'team'].nunique():,} / "
     f"{sample.loc[sample['year'] == 2025, 'event'].nunique()}"),
    ("Variables", "rows / events with missing rank", "390 / 36",
     f"{int(diag('missing_rank', 'missing_rows')['value'])} / "
     f"{int(diag('missing_rank', 'affected_events')['value'])}"),
    ("Variables", "complete-case rank sample rows", "81,296",
     f"{int(est('epa_only', 'rank_complete', 'epa_z')['n']):,}"),
    ("Results", "pooled coefficient range", "0.009–0.015",
     span([r["coef"] for r in pooled])),
    ("Results", "pooled pseudo-R² range", "0.03–0.05",
     span([r["pseudo_r2"] for r in pooled], 2)),
    ("Data", "season with largest / smallest teleop share of mean EPA",
     "2018 / 2020",
     f"{teleop_share.idxmax()} / {teleop_share.idxmin()}"),
    ("Results", "odds multiplier per EPA point", "1.24",
     f"{np.exp(end['coef']):.2f}"),
    ("Results", "DCD end / qualification-end coefficients (6 dp)",
     "0.187109 / 0.187101",
     f"{est('end_epa', 'full', 'epa_total', 'District Championship Division')['coef']:.6f} / "
     f"{est('qual_end_epa', 'full', 'epa_pre_elim', 'District Championship Division')['coef']:.6f}"),
    ("Figure 2", "non-selected candidates", "79,747",
     f"{(sample['pick1'] == 0).sum():,}"),
    ("Figure 2", "mean standardized EPA of accepted partners", "2.27",
     f"{picks['epa_z'].mean():.2f}"),
    ("Results", "highest-EPA accuracy", "55.25%",
     f"{100 * benchmark['highest_epa']:.2f}%"),
    ("Results", "component sample rows / events", "58,994 / 1,470",
     f"{int(est('components_raw', 'components', 'epa_auto')['n']):,} / "
     f"{int(est('components_raw', 'components', 'epa_auto')['n_events']):,}"),
    ("Results", "auto minus teleop (raw points)", "0.017 [-0.138, 0.173]",
     "{estimate:.3f} [{ci_low:.3f}, {ci_high:.3f}]".format(**contrasts[
         (contrasts["model"] == "components_raw")
         & (contrasts["contrast"] == "epa_auto_minus_epa_teleop")
         & (contrasts["covariance"] == "team_cluster")].iloc[0])),
    ("Results", "pooled teleop: raw / season FE / season z",
     "-0.023 / 0.086 / 0.954",
     " / ".join(f"{est(m, 'components', v)['coef']:.3f}" for m, v in [
         ("pooled_components_raw", "epa_teleop"),
         ("pooled_components_season_fe", "epa_teleop"),
         ("pooled_components_season_z", "epa_teleop_season_z")])),
    ("Figure 3", "accepted partner is rank 2 / rank 3", "34.52% / 19.37%",
     f"{100 * (pick_rank == 2).mean():.2f}% / "
     f"{100 * (pick_rank == 3).mean():.2f}%"),
    ("Results", "rank-only pseudo-R² range", "0.340–0.408",
     span(rank_only.values())),
    ("Results", "joint pseudo-R² range", "0.620–0.624", span(joint.values())),
    ("Results", "rank 2-8 restriction: events / add rank / add EPA",
     "271 / 0.011 / 0.339",
     f"{int(diag('rank_restriction', 'retained_events')['value'])} / "
     f"{r2('epa_plus_linear_percentile', None, 'rank_9_or_worse') - r2('epa_only', None, 'rank_9_or_worse'):.3f} / "
     f"{r2('epa_plus_linear_percentile', None, 'rank_9_or_worse') - r2('rank_only_linear_percentile', 'linear_percentile', 'rank_9_or_worse'):.3f}"),
    ("Robustness", "low-dispersion exclusions", "2.515 / 2.509 / 2.532",
     " / ".join(f"{est('epa_z', s, 'epa_z')['coef']:.3f}" for s in [
         "exclude_sd_below_1", "exclude_bottom_1pct_sd",
         "exclude_bottom_5pct_sd"])),
    ("Robustness", "top-half restriction: full -> restricted", "2.531 -> 2.421",
     f"{est('epa_z', 'same_events_full_set', 'epa_z')['coef']:.3f} -> "
     f"{est('epa_z', 'same_events_top_rank_half', 'epa_z')['coef']:.3f}"),
    ("Robustness", "random 25% reductions range", "2.490–2.553",
     span(seed_coefs)),
    ("Robustness", "Hausman-McFadden rejects at 5%", "True",
     str(diag("iia", "hausman_mcfadden_p")["value"] < 0.05)),
    ("Robustness", "mixed logit mean EPA coefficient", "2.608 [2.469, 2.748]",
     ci(mixed)),
    ("Robustness", "mixed logit AIC below fixed-slope AIC", "True",
     str(aic["mixed_epa_z"] < aic["fixed_epa_z"])),
    ("Data", "every model converged", "True",
     str(bool(models["converged"].all()))),
]
for where, item, expected, reproduced in text_checks:
    check(where, item, expected, reproduced)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
report = pd.DataFrame(checks)
report.to_csv(PAPER_CHECK_CSV, index=False)
failed = report[~report["match"]]
print(f"[07] {report['match'].sum():,} of {len(report):,} reported numbers "
      f"reproduce exactly ({len(report) - len(text_checks):,} table cells, "
      f"{len(text_checks)} in-text values)")
print(f"[07] wrote {PAPER_CHECK_CSV.relative_to(PAPER_CHECK_CSV.parents[1])}")
if not failed.empty:
    print("\n[07] MISMATCHES:")
    print(failed.to_string(index=False))
    sys.exit(1)
