"""Step 4: estimate every model reported in the paper and its appendix.

Sections follow the order of the paper:

  Sample flow and data checks ......... text, Supplementary Table A4
  Pooled logit by event type .......... Table 3
  Conditional logit, overall EPA ...... Table 4, Figure A1 reference line
  EPA components ...................... Table 5, Supplementary Table A1
  EPA versus qualification rank ....... Table 6, Supplementary Table A2
  Prediction benchmarks ............... Table 7
  Excluding candidates ranked 2-8 ..... Supplementary Table A3, Panel A
  IIA: choice-set perturbations ....... Supplementary Table A3, Panel B
  IIA: random-slope mixed logit ....... Supplementary Table A3, Panel C
  Season-by-season estimates .......... Supplementary Figure A1
  Low-dispersion exclusions ........... Supplementary Table A4, Panel C

Outputs (all long format, keyed by the columns listed in README.md):
  data/derived/model_estimates.csv        coefficients, SEs, CIs, fit
  data/derived/coefficient_contrasts.csv  component coefficient differences
  data/derived/component_correlations.csv raw and within-season correlations
  data/derived/diagnostics.csv            counts, benchmarks, tests, AICs

Runtime: about 5 minutes with the default BFGS optimizer (statsmodels' conditional
logit loops over events in Python); under a minute with FRC_OPTIMIZER=newton.
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

from choice_models import (conditional_logit, contrast, mixed_logit,
                           pooled_logit, prepare_choice_sets, tidy)
from config import (ALLIANCES_CSV, COMPONENTS, CONTRASTS_CSV,
                    CONVERGENCE_TOLERANCE, CORRELATIONS_CSV, DIAGNOSTICS_CSV,
                    EVENTS_CSV, EXCLUDED_EVENT_TYPES, MODELS_CSV, OPTIMIZER,
                    QUADRATURE_POINTS, RANDOM_DROP_SEEDS, RANDOM_DROP_SHARE,
                    RANK_FORMS, SAMPLE_CSV, TEAM_EVENTS_CSV, TYPE_ORDER)

df = pd.read_csv(SAMPLE_CSV)
events = pd.read_csv(EVENTS_CSV)
team_events = pd.read_csv(TEAM_EVENTS_CSV)
alliances = pd.read_csv(ALLIANCES_CSV)

model_rows, contrast_rows, diagnostic_rows, correlation_rows = [], [], [], []


def diagnose(section, metric, value, subgroup="", numerator=np.nan,
             denominator=np.nan, note=""):
    diagnostic_rows.append({
        "section": section, "metric": metric, "subgroup": subgroup,
        "value": value, "numerator": numerator, "denominator": denominator,
        "note": note,
    })


def record(fit, model, sample, event_type="ALL", log_sigma_var=None):
    rows, invalid = tidy(fit, model, sample, event_type, log_sigma_var,
                         CONVERGENCE_TOLERANCE)
    model_rows.extend(rows)
    for covariance, count in invalid:
        diagnose("covariance", "negative_variance_diagonal", count,
                 subgroup=f"{model}|{sample}|{event_type}|{covariance}",
                 note="SE and CI left blank; no PSD adjustment applied.")
    return fit


def clogit(data, xcols, model, sample, event_type="ALL"):
    return record(conditional_logit(data, xcols, OPTIMIZER),
                  model, sample, event_type)


def pooled(data, xcols, model, sample, event_type="ALL", report_cols=None):
    return record(pooled_logit(data, xcols, report_cols, OPTIMIZER),
                  model, sample, event_type)


def by_event_type(data):
    """("ALL", data) followed by one subset per event type."""
    yield "ALL", data
    for event_type in TYPE_ORDER:
        yield event_type, data[data["event_type_string"] == event_type]


print(f"[04] optimizer: {OPTIMIZER}", flush=True)
print("[04] validating the analysis sample", flush=True)
assert not df.duplicated(["event", "team"]).any()
assert (df.groupby("event")["pick1"].sum() == 1).all()
assert (df.groupby("event").size() >= 2).all()


# ---------------------------------------------------------------------------
# Sample flow and data checks (Methods text; Supplementary Table A4)
# ---------------------------------------------------------------------------
diagnose("sample_flow", "premerge_events", team_events["event"].nunique())
diagnose("sample_flow", "premerge_teams", team_events["team"].nunique())
diagnose("sample_flow", "analysis_events", df["event"].nunique())
diagnose("sample_flow", "analysis_rows", len(df))
diagnose("sample_flow", "analysis_teams", df["team"].nunique())

top = (alliances[alliances["rank"] == 1]
       .dropna(subset=["team1", "team2"]).drop_duplicates("event"))
type_of = events.set_index("key")["event_type_string"].to_dict()
lost = sorted(set(team_events["event"]) - set(df["event"]))
reasons = pd.Series([
    "no_valid_top_alliance_pick" if e not in set(top["event"])
    else f"excluded_{type_of.get(e, '').lower().replace(' ', '_')}"
    if type_of.get(e) in EXCLUDED_EVENT_TYPES else "other"
    for e in lost
], dtype=str)
for reason, count in reasons.value_counts().items():
    diagnose("sample_flow", "excluded_events", int(count), subgroup=reason)

missing = df[df["qual_rank"].isna()]
diagnose("missing_rank", "missing_rows", len(missing))
diagnose("missing_rank", "affected_events", missing["event"].nunique())
diagnose("missing_rank", "events_with_missing_selected_rank",
         df.loc[(df["pick1"] == 1) & df["qual_rank"].isna(), "event"].nunique())
for (year, event_type), group in missing.groupby(["year", "event_type_string"]):
    subgroup = f"{int(year)}|{event_type}"
    diagnose("missing_rank", "missing_rows", len(group), subgroup=subgroup)
    diagnose("missing_rank", "affected_events", group["event"].nunique(),
             subgroup=subgroup)

component_rows = df.dropna(subset=COMPONENTS)
for year, group in component_rows.groupby("year"):
    component_sum = group[COMPONENTS].sum(axis=1)
    gap = group["epa_total"] - component_sum
    for metric, value in {
        "overall_epa_mean": group["epa_total"].mean(),
        "component_sum_mean": component_sum.mean(),
        "mean_total_minus_components": gap.mean(),
        "max_abs_gap": gap.abs().max(),
    }.items():
        diagnose("component_reconciliation", metric, value, subgroup=str(year))

dcd = df[df["event_type_string"] == "District Championship Division"]
dcd_gap = dcd["epa_total"] - dcd["epa_pre_elim"]
diagnose("dcd_verification", "rows", len(dcd))
diagnose("dcd_verification", "events", dcd["event"].nunique())
diagnose("dcd_verification", "first_season", dcd["year"].min())
diagnose("dcd_verification", "exact_equal_epa_rows",
         int(np.isclose(dcd_gap, 0, atol=1e-12).sum()), denominator=len(dcd))
diagnose("dcd_verification", "mean_abs_epa_difference", dcd_gap.abs().mean())
diagnose("dcd_verification", "all_components_present",
         int(dcd[COMPONENTS].notna().all().all()))
diagnose("dcd_verification", "all_ranks_present",
         int(dcd["qual_rank"].notna().all()))

declines = alliances["declines"].dropna().astype(str).str.strip()
diagnose("declines", "alliance_rows", len(alliances))
diagnose("declines", "alliance_rows_with_recorded_declines",
         int(declines.ne("").sum()),
         note="The snapshot records no invitation order or declines.")

event_sd = df.groupby("event")["epa_total"].std()
for q, value in event_sd.quantile([0, .01, .05, .25, .5, .75, .95, 1]).items():
    diagnose("epa_dispersion", "event_sd_quantile", value, subgroup=f"q{q:.2f}")


# ---------------------------------------------------------------------------
# Table 3: pooled logit by event type, event-clustered SEs
# ---------------------------------------------------------------------------
print("[04] Table 3: pooled logit", flush=True)
for event_type, group in by_event_type(df):
    pooled(group, ["epa_total"], "pooled_end_epa", "full", event_type)


# ---------------------------------------------------------------------------
# Table 4: conditional logit with end-of-event, qualification-end, and
# within-event standardized EPA
# ---------------------------------------------------------------------------
print("[04] Table 4: conditional logit, overall EPA", flush=True)
for event_type, group in by_event_type(df):
    clogit(group, ["epa_total"], "end_epa", "full", event_type)
    clogit(group, ["epa_pre_elim"], "qual_end_epa", "full", event_type)
    clogit(group, ["epa_z"], "epa_z", "full", event_type)


# ---------------------------------------------------------------------------
# Table 5 and Supplementary Table A1: EPA components (2016-2025)
# ---------------------------------------------------------------------------
print("[04] Table 5: EPA components", flush=True)
for event_type, group in by_event_type(df):
    clogit(group, COMPONENTS, "components_raw", "components", event_type)

comp = prepare_choice_sets(df, COMPONENTS)
sd_cols, iqr_cols, season_z_cols = [], [], []
for c in COMPONENTS:
    by_event = comp.groupby("event")[c]
    comp[f"{c}_event_sd"] = (
        (comp[c] - by_event.transform("mean")) / by_event.transform("std")
    )
    iqr = (by_event.transform(lambda s: s.quantile(.75))
           - by_event.transform(lambda s: s.quantile(.25)))
    comp[f"{c}_event_iqr"] = (
        (comp[c] - by_event.transform("median")) / iqr.replace(0, np.nan)
    )
    by_season = comp.groupby("year")[c]
    comp[f"{c}_season_z"] = (
        (comp[c] - by_season.transform("mean")) / by_season.transform("std")
    )
    sd_cols.append(f"{c}_event_sd")
    iqr_cols.append(f"{c}_event_iqr")
    season_z_cols.append(f"{c}_season_z")

scale_fits = {
    "components_raw": clogit(comp, COMPONENTS, "components_raw",
                             "components_full_scale"),
    "components_event_sd": clogit(comp, sd_cols, "components_event_sd",
                                  "components_full_scale"),
    "components_event_iqr": clogit(comp, iqr_cols, "components_event_iqr",
                                   "components_full_scale"),
}
for model, fit in scale_fits.items():
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        left, right = fit.xcols[i], fit.xcols[j]
        for covariance in fit.covariances:
            contrast_rows.append({
                "model": model,
                "contrast": f"{COMPONENTS[i]}_minus_{COMPONENTS[j]}",
                "left_var": left,
                "right_var": right,
                "covariance": covariance,
                **contrast(fit, left, right, covariance),
                "n": fit.n,
                "n_events": fit.n_events,
            })

# Supplementary Table A1, Panel A: why the pooled teleop sign reverses.
season_dummies = pd.get_dummies(comp["year"].astype(int), prefix="season",
                                drop_first=True, dtype=float)
comp = pd.concat([comp, season_dummies], axis=1)
pooled(comp, COMPONENTS, "pooled_components_raw", "components")
pooled(comp, COMPONENTS + list(season_dummies.columns),
       "pooled_components_season_fe", "components", report_cols=COMPONENTS)
pooled(comp, season_z_cols, "pooled_components_season_z", "components")

# Supplementary Table A1, Panel B: component correlations.
for scale, cols in [("raw", COMPONENTS), ("within_season_z", season_z_cols)]:
    corr = comp[cols].corr()
    for i, left in enumerate(cols):
        for right in cols:
            correlation_rows.append({
                "scale": scale, "var1": COMPONENTS[i],
                "var2": COMPONENTS[cols.index(right)],
                "correlation": corr.loc[left, right], "n": len(comp),
            })


# ---------------------------------------------------------------------------
# Table 6 and Supplementary Table A2: EPA versus four rank representations on
# a common complete-case sample
# ---------------------------------------------------------------------------
print("[04] Table 6: EPA versus qualification rank", flush=True)
rank_common = prepare_choice_sets(
    df, ["epa_z", "qual_rank", "rank_pct", "neg_log_rank", "inverse_rank"]
)
for r in range(2, 9):
    rank_common[f"rank_{r}"] = (rank_common["qual_rank"] == r).astype(float)

rank_fits = {}
for event_type, group in by_event_type(rank_common):
    rank_fits[event_type, "epa_only"] = clogit(
        group, ["epa_z"], "epa_only", "rank_complete", event_type)
    for form, cols in RANK_FORMS.items():
        rank_fits[event_type, f"rank_only_{form}"] = clogit(
            group, cols, f"rank_only_{form}", "rank_complete", event_type)
        rank_fits[event_type, f"epa_plus_{form}"] = clogit(
            group, ["epa_z", *cols], f"epa_plus_{form}", "rank_complete",
            event_type)


# ---------------------------------------------------------------------------
# Table 7: prediction benchmarks on the same common sample
# ---------------------------------------------------------------------------
picks = rank_common[rank_common["pick1"] == 1].set_index("event")
n_picks = len(picks)
sizes = rank_common.groupby("event").size()
hits = {
    "highest_epa": int(picks["epa_z"].eq(
        rank_common.groupby("event")["epa_z"].max()).sum()),
    "highest_available_rank": int(picks["qual_rank"].eq(
        rank_common.groupby("event")["qual_rank"].min()).sum()),
    "qualification_rank_2": int((picks["qual_rank"] == 2).sum()),
    "epa_only_conditional_logit": int(round(
        rank_fits["ALL", "epa_only"].top1 * n_picks)),
    "epa_plus_linear_rank_conditional_logit": int(round(
        rank_fits["ALL", "epa_plus_linear_percentile"].top1 * n_picks)),
}
diagnose("prediction_benchmark", "uniform_random", (1 / sizes).mean(),
         numerator=(1 / sizes).sum(), denominator=n_picks,
         note="Expected correct picks; event-specific 1/candidates averaged.")
for metric, count in hits.items():
    diagnose("prediction_benchmark", metric, count / n_picks,
             numerator=count, denominator=n_picks)


# ---------------------------------------------------------------------------
# Supplementary Table A3, Panel A: remove candidates ranked 2-8. Events whose
# accepted partner was ranked 2-8 lose their chosen alternative and drop out,
# so the retained events are those with a partner ranked 9 or worse.
# ---------------------------------------------------------------------------
print("[04] Supplementary Table A3: rank restriction and IIA", flush=True)
rank_9_plus = prepare_choice_sets(
    rank_common[~rank_common["qual_rank"].between(2, 8)], ["epa_z", "rank_pct"]
)
diagnose("rank_restriction", "retained_events", rank_9_plus["event"].nunique(),
         denominator=rank_common["event"].nunique())
clogit(rank_9_plus, ["epa_z"], "epa_only", "rank_9_or_worse")
clogit(rank_9_plus, ["rank_pct"], "rank_only_linear_percentile",
       "rank_9_or_worse")
clogit(rank_9_plus, ["epa_z", "rank_pct"], "epa_plus_linear_percentile",
       "rank_9_or_worse")


# ---------------------------------------------------------------------------
# Supplementary Table A3, Panel B: choice-set perturbations
# ---------------------------------------------------------------------------
top_half_events = rank_common.loc[
    (rank_common["pick1"] == 1) & (rank_common["rank_pct"] >= .5), "event"
]
same_events = rank_common[rank_common["event"].isin(set(top_half_events))]
full_set = clogit(same_events, ["epa_z"], "epa_z", "same_events_full_set")
top_half = clogit(same_events[same_events["rank_pct"] >= .5], ["epa_z"],
                  "epa_z", "same_events_top_rank_half")

# Hausman-McFadden test: (b_r - b_f)' [V_r - V_f]^-1 (b_r - b_f) ~ chi2(1).
b_diff = top_half.params - full_set.params
v_diff = (top_half.covariances["conventional"]
          - full_set.covariances["conventional"])
if v_diff[0, 0] > 0:
    stat = float(b_diff @ np.linalg.pinv(v_diff) @ b_diff)
    p_value = float(chi2.sf(stat, df=1))
else:
    stat = p_value = np.nan
diagnose("iia", "hausman_mcfadden_stat", stat)
diagnose("iia", "hausman_mcfadden_p", p_value)

for seed in RANDOM_DROP_SEEDS:
    rng = np.random.default_rng(seed)
    keep = (df["pick1"] == 1) | (rng.random(len(df)) >= RANDOM_DROP_SHARE)
    clogit(df[keep], ["epa_z"], "epa_z", f"random_drop_25pct_seed_{seed}")


# ---------------------------------------------------------------------------
# Supplementary Table A3, Panel C: event-level random EPA slope
# ---------------------------------------------------------------------------
print("[04] Supplementary Table A3: mixed logit", flush=True)
mixed = record(mixed_logit(df, "epa_z", quadrature_points=QUADRATURE_POINTS),
               "mixed_epa_z", "full", log_sigma_var="epa_z_sd")
mixed_rank = record(
    mixed_logit(rank_common, "epa_z", fixed_col="rank_pct",
                quadrature_points=QUADRATURE_POINTS),
    "mixed_epa_z_plus_linear_percentile", "rank_complete",
    log_sigma_var="epa_z_sd",
)
fixed_ll = {
    "fixed_epa_z": next(r["loglike"] for r in model_rows
                        if (r["model"], r["sample"], r["event_type"])
                        == ("epa_z", "full", "ALL")),
    "fixed_epa_z_plus_linear_percentile":
        rank_fits["ALL", "epa_plus_linear_percentile"].loglike,
}
diagnose("mixed_logit", "aic", -2 * fixed_ll["fixed_epa_z"] + 2,
         subgroup="fixed_epa_z")
diagnose("mixed_logit", "aic", -2 * mixed.loglike + 4, subgroup="mixed_epa_z")
diagnose("mixed_logit", "aic",
         -2 * fixed_ll["fixed_epa_z_plus_linear_percentile"] + 4,
         subgroup="fixed_epa_z_plus_linear_percentile")
diagnose("mixed_logit", "aic", -2 * mixed_rank.loglike + 6,
         subgroup="mixed_epa_z_plus_linear_percentile")


# ---------------------------------------------------------------------------
# Supplementary Figure A1: standardized EPA by season
# ---------------------------------------------------------------------------
print("[04] Supplementary Figure A1 and Table A4: season and dispersion",
      flush=True)
for year, group in df.groupby("year"):
    clogit(group, ["epa_z"], "epa_z", f"season_{int(year)}")


# ---------------------------------------------------------------------------
# Supplementary Table A4, Panel C: drop events with little EPA dispersion
# ---------------------------------------------------------------------------
sd_by_row = df["event"].map(event_sd)
for sample, keep in {
    "exclude_sd_below_1": sd_by_row >= 1,
    "exclude_bottom_1pct_sd": sd_by_row >= event_sd.quantile(.01),
    "exclude_bottom_5pct_sd": sd_by_row >= event_sd.quantile(.05),
}.items():
    clogit(df[keep], ["epa_z"], "epa_z", sample)


# ---------------------------------------------------------------------------
# Write and validate
# ---------------------------------------------------------------------------
outputs = {
    MODELS_CSV: (pd.DataFrame(model_rows),
                 ["model", "sample", "event_type", "covariance", "var"]),
    CONTRASTS_CSV: (pd.DataFrame(contrast_rows),
                    ["model", "contrast", "covariance"]),
    DIAGNOSTICS_CSV: (pd.DataFrame(diagnostic_rows),
                      ["section", "metric", "subgroup"]),
    CORRELATIONS_CSV: (pd.DataFrame(correlation_rows),
                       ["scale", "var1", "var2"]),
}
for path, (frame, keys) in outputs.items():
    assert not frame.duplicated(keys).any(), f"Duplicate keys in {path.name}"
    frame.to_csv(path, index=False)
    print(f"[04] saved {path.relative_to(path.parents[2])}: {len(frame):,} rows")

models_out = outputs[MODELS_CSV][0]
assert np.isfinite(models_out["coef"]).all()
blank_se = models_out["se"].isna().sum()
flagged = sum(r["value"] for r in diagnostic_rows
              if r["section"] == "covariance")
assert blank_se == flagged, "Blank SEs must match flagged covariances"

fits = models_out.drop_duplicates(["model", "sample", "event_type"])
print(f"[04] {len(fits)} models; largest Newton decrement "
      f"{fits['newton_decrement'].max():.1e} (tolerance "
      f"{CONVERGENCE_TOLERANCE:.0e})")
unconverged = fits[~fits["converged"]]
if not unconverged.empty:
    print(unconverged[["model", "sample", "event_type", "newton_decrement"]]
          .to_string(index=False))
    sys.exit("[04] some models did not converge")
