"""Step 5: write every table in the paper and the supplementary appendix.

Each table is written twice under output/tables/:
  <name>.md   Markdown, readable directly on GitHub
  <name>.tex  LaTeX fragment (needs the booktabs package)
output/tables/README.md collects all tables on one page.

Every numerical cell is looked up in data/derived/*.csv by an explicit key
(model, sample, event type, covariance, variable), or computed directly from
the analysis sample for the descriptive tables, so the tables cannot drift
from the estimates.
"""

import re

import numpy as np
import pandas as pd

from config import (COMPONENTS, CONTRASTS_CSV, CORRELATIONS_CSV,
                    DIAGNOSTICS_CSV, MODELS_CSV, RANK_FORMS, RANDOM_DROP_SEEDS,
                    SAMPLE_CSV, TABLES, TYPE_LABELS, TYPE_ORDER)

sample = pd.read_csv(SAMPLE_CSV)
models = pd.read_csv(MODELS_CSV)
contrasts = pd.read_csv(CONTRASTS_CSV)
diagnostics = pd.read_csv(DIAGNOSTICS_CSV).fillna({"subgroup": ""})
correlations = pd.read_csv(CORRELATIONS_CSV)

EVENT_ROWS = TYPE_ORDER + ["ALL"]
TEAM = "team_cluster"
DASH = "–"
written = []


# ---------------------------------------------------------------------------
# Lookups and formatting
# ---------------------------------------------------------------------------
def est(model, sample_name, var, event_type="ALL", covariance=TEAM):
    rows = models[
        (models["model"] == model) & (models["sample"] == sample_name)
        & (models["event_type"] == event_type) & (models["var"] == var)
        & (models["covariance"] == covariance)
    ]
    if len(rows) != 1:
        raise KeyError(f"{model}/{sample_name}/{event_type}/{var}/{covariance}:"
                       f" {len(rows)} rows")
    return rows.iloc[0]


def diag(section, metric, subgroup=""):
    rows = diagnostics[
        (diagnostics["section"] == section) & (diagnostics["metric"] == metric)
        & (diagnostics["subgroup"] == subgroup)
    ]
    if len(rows) != 1:
        raise KeyError(f"{section}/{metric}/{subgroup}: {len(rows)} rows")
    return rows.iloc[0]


def ci(row, decimals=3):
    """'coef [low, high]' with blank bounds when the variance was invalid."""
    if not (np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])):
        return f"{row['coef']:.{decimals}f} [{DASH}, {DASH}]"
    return (f"{row['coef']:.{decimals}f} "
            f"[{row['ci_low']:.{decimals}f}, {row['ci_high']:.{decimals}f}]")


def count(value):
    return f"{int(round(value)):,}"


def num(value, decimals=1):
    return DASH if pd.isna(value) else f"{value:.{decimals}f}"


def sci(p):
    mantissa, exponent = f"{p:.0e}".split("e")
    return f"{mantissa} × 10^{int(exponent)}"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def to_latex(text):
    text = str(text)
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#")]:
        text = text.replace(old, new)
    text = text.replace("-log(rank)", r"$-\log(\mathrm{rank})$")
    text = re.sub(r"10\^(-?\d+)", r"$10^{\1}$", text)
    text = re.sub(r"(?<![\w.{])-(?=\d)", "$-$", text)
    for old, new in [("β̂", r"$\hat\beta$"), ("R²", r"$R^2$"),
                     ("Δ", r"$\Delta$"), ("×", r"$\times$"), ("−", "$-$"),
                     ("–", "--"), ("“", "``"), ("”", "''")]:
        text = text.replace(old, new)
    return text


def write_table(name, title, panels, note):
    """panels: list of (panel title or None, header cells, rows of cells)."""
    md = [f"### {title}", ""]
    tex = [r"\begin{table}[!htbp]", r"\centering",
           rf"\caption{{{to_latex(title.split('. ', 1)[1])}}}",
           rf"\label{{tab:{name}}}", r"\footnotesize"]
    for i, (panel, header, rows) in enumerate(panels):
        if panel:
            md += [f"*{panel}*", ""]
            tex += [rf"\textit{{{to_latex(panel)}}}\par\smallskip"]
        md += ["| " + " | ".join(header) + " |",
               "|" + "|".join([":---"] + ["---:"] * (len(header) - 1)) + "|"]
        md += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
        md += [""]
        tex += [rf"\begin{{tabular}}{{l{'c' * (len(header) - 1)}}}",
                r"\toprule",
                " & ".join(to_latex(h) for h in header) + r" \\",
                r"\midrule"]
        tex += [" & ".join(to_latex(c) for c in row) + r" \\" for row in rows]
        tex += [r"\bottomrule", r"\end{tabular}"]
        if i < len(panels) - 1:
            tex += [r"\vspace{0.8em}", ""]
    md += [f"*Note:* {note}", ""]
    tex += [r"\par\medskip", r"\begin{minipage}{0.95\linewidth}",
            rf"\textit{{Note:}} {to_latex(note)}",
            r"\end{minipage}", r"\end{table}", ""]
    (TABLES / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    (TABLES / f"{name}.tex").write_text("\n".join(tex), encoding="utf-8")
    written.append((name, title))
    print(f"[05] wrote output/tables/{name}.md|.tex")


# ---------------------------------------------------------------------------
# Table 1: events by type
# ---------------------------------------------------------------------------
events_by_type = sample.groupby("event_type_string")["event"].nunique()
write_table(
    "table1_event_types",
    "Table 1. Number of FRC events in the analysis sample by event type, "
    "2010–2025",
    [(None, ["Event type", "Number of events"],
      [[TYPE_LABELS[t], count(events_by_type[t])] for t in TYPE_ORDER]
      + [["Total", count(events_by_type.sum())]])],
    "The table reports the number of distinct FIRST Robotics Competition "
    "(FRC) events by event type in the analysis sample, 2010–2025 seasons "
    "(2021 is excluded because the pandemic canceled the in-person season). "
    "Event types follow FIRST/The Blue Alliance classifications. The sample "
    "includes every official event with Statbotics team-level EPA data and a "
    "recorded alliance selection. Championship Finals (Einstein) fields are "
    "excluded because no alliance selection takes place there: the "
    "division-winning alliances advance intact. Offseason events are excluded "
    "because Statbotics does not provide EPA data for past offseason "
    "competitions.",
)


# ---------------------------------------------------------------------------
# Table 2: descriptive statistics by season
# ---------------------------------------------------------------------------
by_year = sample.groupby("year").agg(
    events=("event", "nunique"), teams=("team", "nunique"),
    mean=("epa_total", "mean"), sd=("epa_total", "std"),
    auto=("epa_auto", "mean"), teleop=("epa_teleop", "mean"),
    endgame=("epa_endgame", "mean"),
)
gap_2018 = diag("component_reconciliation", "mean_total_minus_components",
                "2018")["value"]
write_table(
    "table2_descriptives_by_season",
    "Table 2. Descriptive statistics by season",
    [(None, ["Season", "Events", "Teams", "Mean", "SD", "Auto", "Teleop",
             "Endgame"],
      [[str(year), count(r.events), count(r.teams), num(r["mean"]),
        num(r.sd), num(r.auto), num(r.teleop), num(r.endgame)]
       for year, r in by_year.iterrows()])],
    "Statistics are computed over the estimation sample (team–event "
    "observations at events with a recorded alliance selection, excluding the "
    "top alliance's captain). “Events” is the number of events and “Teams” "
    "the number of unique teams in each season. Mean and SD describe overall "
    "EPA; Auto, Teleop, and Endgame are component means. EPA is measured in "
    "points of that season's game, so levels are not comparable across "
    "seasons. Statbotics reports the components from 2016 onward; dashes "
    "denote seasons without component data. In 2018, the unrounded overall "
    f"mean is {diag('component_reconciliation', 'overall_epa_mean', '2018')['value']:.3f} "
    "and the mean of the row-level component sums is "
    f"{diag('component_reconciliation', 'component_sum_mean', '2018')['value']:.3f}, "
    f"a {gap_2018:.3f} difference present in the source snapshot. The 2020 "
    "season was cut short and the 2021 season was canceled.",
)


# ---------------------------------------------------------------------------
# Table 3: pooled logit by event type
# ---------------------------------------------------------------------------
rows = []
for t in EVENT_ROWS:
    r = est("pooled_end_epa", "full", "epa_total", t, "event_cluster")
    rows.append([TYPE_LABELS[t], f"{r['coef']:.4f} ({r['se']:.4f})",
                 f"{r['pseudo_r2']:.3f}", count(r["n"]), count(r["n_events"])])
write_table(
    "table3_pooled_logit",
    "Table 3. Logit estimates of the association between EPA and the "
    "probability of being the top alliance's first pick, by event type",
    [(None, ["Event type", "β̂ (SE)", "McFadden R²", "N", "Events"], rows)],
    "Entries report coefficients from separate descriptive logistic "
    "regressions of the accepted-partner indicator on event EPA, estimated by "
    "event type on the full 2010–2025 sample. Standard errors clustered by "
    "event are in parentheses because outcomes are linked within each event "
    "choice set. McFadden R² is relative to an intercept-only logit and is "
    "not comparable with the conditional-logit values in Tables 4–6.",
)


# ---------------------------------------------------------------------------
# Table 4: conditional logit, overall EPA
# ---------------------------------------------------------------------------
sizes = sample.groupby(["event", "event_type_string"]).size().reset_index(
    name="candidates")
panel_a, panel_b = [], []
for t in EVENT_ROWS:
    end = est("end_epa", "full", "epa_total", t)
    qual = est("qual_end_epa", "full", "epa_pre_elim", t)
    z = est("epa_z", "full", "epa_z", t)
    cands = sizes["candidates"] if t == "ALL" else \
        sizes.loc[sizes["event_type_string"] == t, "candidates"]
    panel_a.append([TYPE_LABELS[t], ci(end), f"{end['pseudo_r2']:.3f}",
                    ci(qual), f"{qual['pseudo_r2']:.3f}", count(end["n"]),
                    count(end["n_events"])])
    panel_b.append([TYPE_LABELS[t], ci(z), f"{z['pseudo_r2']:.3f}",
                    f"{cands.mean():.1f}", f"{cands.median():.0f}",
                    count(z["n"]), count(z["n_events"])])
dcd_end = est("end_epa", "full", "epa_total", "District Championship Division")
dcd_qual = est("qual_end_epa", "full", "epa_pre_elim",
               "District Championship Division")
write_table(
    "table4_conditional_logit",
    "Table 4. Conditional-logit estimates for the top alliance's accepted "
    "first partner",
    [("Panel A: End-of-event and qualification-end EPA",
      ["Event type", "End EPA β̂ [95% CI]", "End R²",
       "Qual.-end EPA β̂ [95% CI]", "Qual.-end R²", "N", "Events"], panel_a),
     ("Panel B: Within-event standardized EPA and choice-set size",
      ["Event type", "EPA z β̂ [95% CI]", "McFadden R²", "Mean candidates",
       "Median candidates", "N", "Events"], panel_b)],
    "Each row is a separate conditional logit grouped by event. Confidence "
    "intervals use team-clustered standard errors to account for repeated "
    "appearances of the same team. Panel A prints the McFadden pseudo-R² for "
    "both EPA timing measures. Panel B makes explicit that event-type "
    "pseudo-R² values have different average choice-set sizes. Standardizing "
    "EPA within event changes the effective raw-point slope by the event's "
    "EPA standard deviation; it is not only a change of units. District "
    "Championship Divisions begin in "
    f"{int(diag('dcd_verification', 'first_season')['value'])} and contain "
    f"{int(diag('dcd_verification', 'events')['value'])} events. Their end and "
    f"qualification-end coefficients are {dcd_end['coef']:.6f} and "
    f"{dcd_qual['coef']:.6f} before rounding; only "
    f"{int(diag('dcd_verification', 'exact_equal_epa_rows')['value'])} of "
    f"{count(diag('dcd_verification', 'rows')['value'])} underlying EPA pairs "
    "are exactly equal.",
)


# ---------------------------------------------------------------------------
# Table 5: EPA components
# ---------------------------------------------------------------------------
panel_a = []
for t in EVENT_ROWS:
    r = [est("components_raw", "components", c, t) for c in COMPONENTS]
    panel_a.append([TYPE_LABELS[t], ci(r[0]), ci(r[1]), ci(r[2]),
                    f"{r[0]['pseudo_r2']:.3f}", count(r[0]["n"]),
                    count(r[0]["n_events"])])
panel_b = []
for label, model, suffix in [
    ("Raw EPA points", "components_raw", ""),
    ("One within-event SD", "components_event_sd", "_event_sd"),
    ("One within-event IQR", "components_event_iqr", "_event_iqr"),
]:
    r = [est(model, "components_full_scale", c + suffix) for c in COMPONENTS]
    panel_b.append([label, ci(r[0]), ci(r[1]), ci(r[2]),
                    f"{r[0]['pseudo_r2']:.3f}", count(r[0]["n_events"])])
write_table(
    "table5_epa_components",
    "Table 5. EPA components and the accepted first partner, 2016–2025",
    [("Panel A: Raw-point conditional-logit estimates by event type",
      ["Event type", "Auto β̂ [95% CI]", "Teleop β̂ [95% CI]",
       "Endgame β̂ [95% CI]", "McFadden R²", "N", "Events"], panel_a),
     ("Panel B: Full-sample component effects on comparable scales",
      ["Scale of change", "Auto β̂ [95% CI]", "Teleop β̂ [95% CI]",
       "Endgame β̂ [95% CI]", "McFadden R²", "Events"], panel_b)],
    "Each row contains three coefficients from one conditional logit. "
    "Confidence intervals use team-clustered standard errors. Component data "
    "begin in 2016. Panel A separates the full-sample estimates from noisier "
    "event-type estimates; an interval containing zero is not statistically "
    "distinguishable from zero at the 5 percent level. Under team clustering, "
    "the full-sample auto and teleop intervals exclude zero, while the endgame "
    "interval includes zero. Panel B avoids interpreting raw-point "
    "coefficients as directly comparable weights by also scaling each "
    "component by its within-event standard deviation and interquartile range.",
)


# ---------------------------------------------------------------------------
# Table 6: EPA versus four rank representations
# ---------------------------------------------------------------------------
RANK_LABELS = {
    "linear_percentile": "Linear rank percentile",
    "negative_log_rank": "-log(rank)",
    "inverse_rank": "1/rank",
    "rank_2_to_8_indicators": "Indicators for ranks 2–8",
}


def rank_fit(model, t="ALL", form=None, covariance="conventional"):
    var = "epa_z" if form is None else RANK_FORMS[form][0]
    return est(model, "rank_complete", var, t, covariance)


rows = []
epa_only = rank_fit("epa_only")
for form, label in RANK_LABELS.items():
    rank_only = rank_fit(f"rank_only_{form}", form=form)
    joint = rank_fit(f"epa_plus_{form}", covariance=TEAM)
    rows.append([
        label, f"{epa_only['pseudo_r2']:.3f}", f"{rank_only['pseudo_r2']:.3f}",
        f"{joint['pseudo_r2']:.3f}",
        f"{joint['pseudo_r2'] - epa_only['pseudo_r2']:+.3f}",
        f"{joint['pseudo_r2'] - rank_only['pseudo_r2']:+.3f}",
        ci(joint), f"{100 * joint['top1_accuracy']:.1f}",
    ])
write_table(
    "table6_epa_vs_rank",
    "Table 6. EPA versus alternative representations of qualification rank",
    [(None, ["Rank form", "EPA-only R²", "Rank-only R²", "Joint R²",
             "Add rank ΔR²", "Add EPA ΔR²", "Joint EPA β̂ [95% CI]",
             "Top-1 (%)"], rows)],
    f"All columns use the same {count(epa_only['n'])} team-event observations "
    f"and {count(epa_only['n_events'])} events with complete EPA and rank "
    "data. R² is McFadden's pseudo-R² relative to a uniform pick. Confidence "
    "intervals use team-clustered standard errors. The fit gains, rather than "
    "coefficient shrinkage, are the primary comparison. The indicator "
    "specification includes separate indicators for ranks 2 through 8, with "
    "rank 9 or worse as the pooled reference category. EPA adds substantially "
    "more fit to every rank-only model than rank adds to EPA.",
)


# ---------------------------------------------------------------------------
# Table 7: prediction benchmarks
# ---------------------------------------------------------------------------
rows = []
for label, metric in [
    ("Uniform random choice (event-specific)", "uniform_random"),
    ("Choose highest-EPA candidate", "highest_epa"),
    ("Choose highest available qualification rank", "highest_available_rank"),
    ("Always choose qualification rank 2", "qualification_rank_2"),
    ("EPA-only conditional logit", "epa_only_conditional_logit"),
    ("EPA plus linear-rank conditional logit",
     "epa_plus_linear_rank_conditional_logit"),
]:
    d = diag("prediction_benchmark", metric)
    correct = (f"{d['numerator']:.1f}" if metric == "uniform_random"
               else str(int(d["numerator"])))
    rows.append([label, correct, count(d["denominator"]),
                 f"{100 * d['value']:.2f}"])
write_table(
    "table7_prediction_benchmarks",
    "Table 7. Prediction benchmarks on the common EPA-and-rank sample",
    [(None, ["Rule or model", "Correct picks", "Events", "Accuracy (%)"],
      rows)],
    f"Every benchmark uses the same {count(epa_only['n_events'])} events. The "
    "random benchmark is calculated event by event and then averaged. Because "
    "the EPA-only coefficient is positive, its top-1 prediction is "
    "mechanically the highest-EPA candidate. “Highest available qualification "
    "rank” allows for the empirically observed rank values, while the next row "
    "applies the literal rank-2 rule.",
)


# ---------------------------------------------------------------------------
# Supplementary Table A1: component sign checks, correlations, contrasts
# ---------------------------------------------------------------------------
panel_a = []
for label, model, suffix in [
    ("Raw components, no season controls", "pooled_components_raw", ""),
    ("Raw components plus season indicators", "pooled_components_season_fe", ""),
    ("Components standardized within season", "pooled_components_season_z",
     "_season_z"),
]:
    r = [est(model, "components", c + suffix) for c in COMPONENTS]
    panel_a.append([label, ci(r[0]), ci(r[1]), ci(r[2]), count(r[0]["n"])])


def correlation(scale, left, right):
    row = correlations[(correlations["scale"] == scale)
                       & (correlations["var1"] == left)
                       & (correlations["var2"] == right)]
    return f"{row.iloc[0]['correlation']:.3f}"


PAIRS = [("Auto and teleop", "epa_auto", "epa_teleop"),
         ("Auto and endgame", "epa_auto", "epa_endgame"),
         ("Teleop and endgame", "epa_teleop", "epa_endgame")]
panel_b = [[label, correlation("raw", a, b),
            correlation("within_season_z", a, b)] for label, a, b in PAIRS]


def contrast_ci(model, left, right):
    row = contrasts[(contrasts["model"] == model)
                    & (contrasts["contrast"] == f"{left}_minus_{right}")
                    & (contrasts["covariance"] == TEAM)].iloc[0]
    return (f"{row['estimate']:.3f} "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]")


panel_c = [[label] + [contrast_ci(model, a, b) for _, a, b in PAIRS]
           for label, model in [("Raw points", "components_raw"),
                                ("Within-event SD", "components_event_sd"),
                                ("Within-event IQR", "components_event_iqr")]]
write_table(
    "tableA1_component_checks",
    "Supplementary Table A1. Component-model robustness, correlations, and "
    "coefficient contrasts",
    [("Panel A: Pooled component models",
      ["Specification", "Auto β̂ [95% CI]", "Teleop β̂ [95% CI]",
       "Endgame β̂ [95% CI]", "N"], panel_a),
     ("Panel B: Pairwise component correlations",
      ["Pair", "Raw", "After within-season standardization"], panel_b),
     ("Panel C: Conditional-logit coefficient differences",
      ["Scale", "Auto − teleop [95% CI]", "Auto − endgame [95% CI]",
       "Teleop − endgame [95% CI]"], panel_c)],
    "Confidence intervals use team-clustered standard errors. Panel A directly "
    "tests the proposed cross-season explanation for the pooled teleop sign: "
    "all three raw coefficients become positive after season indicators are "
    "added, and all are positive after within-season standardization. Panel B "
    "shows substantial but imperfect collinearity. Panel C shows that the raw "
    "auto and teleop coefficients are not statistically distinguishable, while "
    "teleop has the largest association on standardized within-event scales.",
)


# ---------------------------------------------------------------------------
# Supplementary Table A2: rank forms by event type
# ---------------------------------------------------------------------------
panel_a, panel_b = [], []
for t in EVENT_ROWS:
    epa = rank_fit("epa_only", t)
    rank_r2, joint_cells = [], []
    for form in RANK_LABELS:
        rank_r2.append(f"{rank_fit(f'rank_only_{form}', t, form)['pseudo_r2']:.3f}")
        joint = rank_fit(f"epa_plus_{form}", t)
        joint_cells.append(f"{joint['pseudo_r2']:.3f} "
                           f"({joint['pseudo_r2'] - epa['pseudo_r2']:+.3f})")
    panel_a.append([TYPE_LABELS[t], *rank_r2, count(epa["n"]),
                    count(epa["n_events"])])
    panel_b.append([TYPE_LABELS[t], f"{epa['pseudo_r2']:.3f}", *joint_cells])
write_table(
    "tableA2_rank_by_event_type",
    "Supplementary Table A2. Rank-form robustness by event type",
    [("Panel A: Rank-only McFadden pseudo-R²",
      ["Event type", "Linear percentile", "-log(rank)", "1/rank",
       "Ranks 2–8", "N", "Events"], panel_a),
     ("Panel B: Joint-model pseudo-R² (gain from adding rank to EPA)",
      ["Event type", "EPA only", "Linear percentile", "-log(rank)", "1/rank",
       "Ranks 2–8"], panel_b)],
    "Every comparison within an event type uses a common complete-case "
    "sample. Parentheses in Panel B report the change in pseudo-R² from "
    "adding that rank representation to the EPA-only model. The pattern that "
    "EPA dominates rank-only fit is not confined to one event type.",
)


# ---------------------------------------------------------------------------
# Supplementary Table A3: rank restriction and IIA checks
# ---------------------------------------------------------------------------
panel_a = []
for label, sample_name in [("Complete EPA-and-rank sample", "rank_complete"),
                           ("Candidates ranked 9 or worse", "rank_9_or_worse")]:
    epa = est("epa_only", sample_name, "epa_z", covariance="conventional")
    rank = est("rank_only_linear_percentile", sample_name, "rank_pct",
               covariance="conventional")
    joint = est("epa_plus_linear_percentile", sample_name, "epa_z",
                covariance="conventional")
    panel_a.append([label, f"{epa['pseudo_r2']:.3f}",
                    f"{rank['pseudo_r2']:.3f}", f"{joint['pseudo_r2']:.3f}",
                    f"{joint['pseudo_r2'] - epa['pseudo_r2']:+.3f}",
                    f"{joint['pseudo_r2'] - rank['pseudo_r2']:+.3f}",
                    count(joint["n"]), count(joint["n_events"])])

panel_b = []
for label, sample_name in (
    [("Same events, full choice set", "same_events_full_set"),
     ("Same events, top rank half only", "same_events_top_rank_half")]
    + [(f"Randomly drop 25%, seed {s}", f"random_drop_25pct_seed_{s}")
       for s in RANDOM_DROP_SEEDS]
):
    r = est("epa_z", sample_name, "epa_z")
    panel_b.append([label, ci(r), f"{r['pseudo_r2']:.3f}", count(r["n"]),
                    count(r["n_events"])])


def aic(subgroup):
    return f"{diag('mixed_logit', 'aic', subgroup)['value']:.1f}"


H = "numerical_hessian"
panel_c = [
    ["Fixed EPA slope",
     ci(est("epa_z", "full", "epa_z", covariance="conventional")), DASH,
     aic("fixed_epa_z")],
    ["Random EPA slope",
     ci(est("mixed_epa_z", "full", "epa_z_mean", covariance=H)),
     ci(est("mixed_epa_z", "full", "epa_z_sd", covariance=H)),
     aic("mixed_epa_z")],
    ["Fixed EPA slope plus rank",
     ci(rank_fit("epa_plus_linear_percentile")), DASH,
     aic("fixed_epa_z_plus_linear_percentile")],
    ["Random EPA slope plus rank",
     ci(est("mixed_epa_z_plus_linear_percentile", "rank_complete",
            "epa_z_mean", covariance=H)),
     ci(est("mixed_epa_z_plus_linear_percentile", "rank_complete",
            "epa_z_sd", covariance=H)),
     aic("mixed_epa_z_plus_linear_percentile")],
]
restricted_events = int(diag("rank_restriction", "retained_events")["value"])
write_table(
    "tableA3_rank_restriction_iia",
    "Supplementary Table A3. Rank-decline sensitivity and IIA robustness",
    [("Panel A: Excluding candidates ranked 2–8",
      ["Sample", "EPA-only R²", "Rank-only R²", "Joint R²", "Add rank",
       "Add EPA", "N", "Events"], panel_a),
     ("Panel B: Choice-set perturbations",
      ["Choice set", "EPA z β̂ [95% CI]", "McFadden R²", "N", "Events"],
      panel_b),
     ("Panel C: Fixed- and random-coefficient logit",
      ["Model", "EPA mean β̂ [95% CI]", "EPA slope SD [95% CI]", "AIC"],
      panel_c)],
    "Panel A's restricted row is conditional on the accepted partner being "
    f"ranked 9 or worse; it retains only {restricted_events} of "
    f"{count(epa_only['n_events'])} events and cannot reconstruct unobserved "
    "invitation order. Panel B uses team-clustered intervals. The top-half "
    "Hausman-McFadden comparison rejects strict coefficient stability "
    f"(p = {sci(diag('iia', 'hausman_mcfadden_p')['value'])}), so IIA should "
    "be treated as a limitation. Panel C uses conventional intervals and "
    "integrates over event-level random EPA slopes using 20-point "
    "Gauss-Hermite quadrature; the central positive EPA relationship remains "
    "under this relaxation.",
)


# ---------------------------------------------------------------------------
# Supplementary Table A4: data integrity, missing ranks, EPA dispersion
# ---------------------------------------------------------------------------
def value(section, metric, subgroup="", decimals=None):
    v = diag(section, metric, subgroup)["value"]
    return f"{v:.{decimals}f}" if decimals is not None else count(v)


panel_a = [
    ["Events in pre-merge team-event data",
     value("sample_flow", "premerge_events")],
    ["Events in analysis sample", value("sample_flow", "analysis_events")],
    ["Excluded: no valid top-alliance accepted pick",
     value("sample_flow", "excluded_events", "no_valid_top_alliance_pick")],
    ["Excluded: Championship Finals",
     value("sample_flow", "excluded_events", "excluded_championship_finals")],
    ["Excluded: Festival of Champions",
     value("sample_flow", "excluded_events", "excluded_festival_of_champions")],
    ["Rows with missing qualification rank",
     value("missing_rank", "missing_rows")],
    ["Events affected by any missing rank",
     value("missing_rank", "affected_events")],
    ["Events with selected partner rank missing",
     value("missing_rank", "events_with_missing_selected_rank")],
    ["Alliance rows with recorded declines",
     value("declines", "alliance_rows_with_recorded_declines")],
    ["DCD rows / events / first season",
     f"{value('dcd_verification', 'rows')} / "
     f"{value('dcd_verification', 'events')} / "
     f"{int(diag('dcd_verification', 'first_season')['value'])}"],
    ["DCD exact end/qualification EPA matches",
     value("dcd_verification", "exact_equal_epa_rows")],
    ["DCD mean absolute difference between EPA measures",
     value("dcd_verification", "mean_abs_epa_difference", decimals=3)],
    ["2018 overall EPA mean",
     value("component_reconciliation", "overall_epa_mean", "2018", 3)],
    ["2018 component-sum mean",
     value("component_reconciliation", "component_sum_mean", "2018", 3)],
    ["2018 mean total minus component sum",
     value("component_reconciliation", "mean_total_minus_components", "2018",
           3)],
]

missing = diagnostics[(diagnostics["section"] == "missing_rank")
                      & (diagnostics["subgroup"] != "")]
missing = missing.pivot(index="subgroup", columns="metric",
                        values="value").reset_index()
missing[["year", "event_type"]] = missing["subgroup"].str.split("|", expand=True)
panel_b = [[r.year, r.event_type, count(r.missing_rows),
            count(r.affected_events)]
           for r in missing.sort_values(["year", "event_type"]).itertuples()]

panel_c = []
for label, sample_name in [
    ("All events", "full"),
    ("Exclude EPA SD below 1", "exclude_sd_below_1"),
    ("Exclude bottom 1 percent of EPA SD", "exclude_bottom_1pct_sd"),
    ("Exclude bottom 5 percent of EPA SD", "exclude_bottom_5pct_sd"),
]:
    r = est("epa_z", sample_name, "epa_z")
    panel_c.append([label, ci(r), f"{r['pseudo_r2']:.3f}", count(r["n"]),
                    count(r["n_events"])])
write_table(
    "tableA4_data_diagnostics",
    "Supplementary Table A4. Data-integrity, missing-rank, and EPA-dispersion "
    "checks",
    [("Panel A: Sample and arithmetic checks", ["Diagnostic", "Value"],
      panel_a),
     ("Panel B: Missing qualification ranks by season and event type",
      ["Season", "Event type", "Missing rows", "Affected events"], panel_b),
     ("Panel C: Sensitivity to small within-event EPA dispersion",
      ["Sample", "EPA z β̂ [95% CI]", "McFadden R²", "N", "Events"],
      panel_c)],
    "The 39-event attrition is fully reconciled in Panel A. Offseason events "
    "never enter the pre-merge data because historical Statbotics EPA is "
    "unavailable. DCD = District Championship Division. The 2018 "
    "total/component difference is present in the raw Statbotics snapshot "
    "rather than caused by table arithmetic. Panel C uses team-clustered "
    "intervals and shows that low-dispersion events do not drive the "
    "standardized-EPA result.",
)


# ---------------------------------------------------------------------------
# One-page index of every table
# ---------------------------------------------------------------------------
index = ["# Tables", "",
         "Generated by `code/05_make_tables.py`. Each table is also available "
         "as a LaTeX fragment (`.tex`) in this folder.", ""]
for name, _ in written:
    index += [(TABLES / f"{name}.md").read_text(encoding="utf-8"), ""]
(TABLES / "README.md").write_text("\n".join(index), encoding="utf-8")
print(f"[05] wrote output/tables/README.md ({len(written)} tables)")
