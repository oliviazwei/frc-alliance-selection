"""Step 4: estimate every specification reported in the paper.

Two families of models:

  Pooled logit (the benchmark, Table 3 of the paper)
      Pr(pick=1) = Lambda(alpha + beta'x), one Bernoulli trial per team-event.
      Standard errors are clustered by event, because exactly one team per
      event can be selected and the rows within an event are dependent.

  Conditional logit (the paper's preferred model, Tables 4-6)
      The captain chooses exactly one team from the event's field, so the
      likelihood contribution of an event is the choice probability of the
      team actually picked. Only within-event variation identifies beta, which
      absorbs cross-season differences in game scoring scales.

McFadden pseudo-R-squared uses each model's natural null: an intercept-only
logit for the pooled models, and a uniform-random pick from the choice set for
the conditional models. The two are therefore not directly comparable.

Output: data/derived/results.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

from config import RESULTS_CSV, SAMPLE_CSV, TYPE_ORDER

df = pd.read_csv(SAMPLE_CSV)
results = []


def prep(data, xcols):
    """Rows usable for a given specification.

    Drops missing/infinite regressors (component EPA exists only from 2016),
    then keeps events that still contain exactly one observed pick and at
    least two candidates -- the conditional logit needs a real choice.
    """
    d = data.dropna(subset=list(xcols)).copy()
    d = d[np.isfinite(d[list(xcols)]).all(axis=1)]
    d = d[d.groupby("event")["pick1"].transform("sum") == 1]
    d = d[d.groupby("event")["pick1"].transform("size") >= 2]
    return d


def record(spec, etype, res, xcols, n, n_events, pseudo_r2):
    for v in xcols:
        results.append({
            "spec": spec,
            "event_type": etype,
            "var": v,
            "coef": res.params[v],
            "se": res.bse[v],
            "p": res.pvalues[v],
            "n": n,
            "n_events": n_events,
            "pseudo_r2": pseudo_r2,
        })


def fit_pooled(data, xcols, spec, etype):
    d = prep(data, xcols)
    if len(d) == 0 or d["pick1"].nunique() < 2:
        return
    X = sm.add_constant(d[list(xcols)])
    try:
        res = sm.Logit(d["pick1"], X).fit(
            disp=False, cov_type="cluster", cov_kwds={"groups": d["event"]})
    except Exception as exc:                      # pragma: no cover
        print(f"  [skip] {spec} / {etype}: {exc}")
        return
    record(spec, etype, res, xcols, int(res.nobs), d["event"].nunique(),
           res.prsquared)


def fit_conditional(data, xcols, spec, etype):
    d = prep(data, xcols)
    if len(d) == 0 or d["pick1"].nunique() < 2:
        return
    try:
        res = ConditionalLogit(d["pick1"], d[list(xcols)],
                               groups=d["event"]).fit(method="bfgs",
                                                      maxiter=500, disp=False)
    except Exception as exc:                      # pragma: no cover
        print(f"  [skip] {spec} / {etype}: {exc}")
        return
    # Null model: pick uniformly at random from each event's choice set.
    ll0 = -np.log(d.groupby("event").size()).sum()
    record(spec, etype, res, xcols, len(d), d["event"].nunique(),
           1 - res.llf / ll0)


SPECS = [
    # (name, estimator, regressors)
    ("pooled_total",        fit_pooled,      ["epa_total"]),
    ("pooled_components",   fit_pooled,      ["epa_auto", "epa_teleop", "epa_endgame"]),
    ("clogit_total",        fit_conditional, ["epa_total"]),
    ("clogit_pre_elim",     fit_conditional, ["epa_pre_elim"]),
    ("clogit_components",   fit_conditional, ["epa_auto", "epa_teleop", "epa_endgame"]),
    ("clogit_z",            fit_conditional, ["epa_z"]),
    ("clogit_rank_only",    fit_conditional, ["rank_pct"]),
    ("clogit_horserace",    fit_conditional, ["epa_z", "rank_pct"]),
]

for spec, fitter, xcols in SPECS:
    print(f"Estimating {spec} ...", flush=True)
    fitter(df, xcols, spec, "ALL")
    for etype, sub in df.groupby("event_type_string"):
        fitter(sub, xcols, spec, etype)

out = pd.DataFrame(results)
out.to_csv(RESULTS_CSV, index=False)


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


for spec, _, _ in SPECS:
    sub = out[out["spec"] == spec]
    if sub.empty:
        continue
    print(f"\n===== {spec} =====")
    for etype in ["ALL"] + TYPE_ORDER:
        rows = sub[sub["event_type"] == etype]
        if rows.empty:
            continue
        cells = "  ".join(
            f"{r['var']}={r['coef']:.4f} ({r['se']:.4f}){stars(r['p'])}"
            for _, r in rows.iterrows()
        )
        info = rows.iloc[0]
        print(f"{etype:32s} {cells}  | R2={info['pseudo_r2']:.3f} "
              f"N={info['n']:,} events={info['n_events']:,}")

print(f"\nSaved {RESULTS_CSV}")
