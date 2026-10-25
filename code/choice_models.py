"""Estimators shared by the analysis scripts.

conditional_logit()
    McFadden conditional logit grouped by event (Equation 5 of the paper).
    The captain picks exactly one team from the event's choice set, so only
    within-event differences identify the coefficients. Returns conventional,
    team-clustered, and event/team two-way clustered covariance matrices.
    optimizer="bfgs" uses statsmodels' ConditionalLogit, as in the published
    analysis; optimizer="newton" iterates to the exact maximum.

pooled_logit()
    Binary logit on all team-event rows (Equation 4). Returns event-clustered,
    team-clustered, and event/team two-way clustered covariance matrices.

mixed_logit()
    Conditional logit with a normally distributed, event-level random slope on
    one regressor, integrated by Gauss-Hermite quadrature. It relaxes the common
    slope and therefore does not impose marginal IIA in the same way.

tidy()
    Converts a fitted model to long-format rows (one per variable and
    covariance type) with standard errors, 95% intervals, and p-values.

Clustered covariances are sandwich estimators built from alternative-level
score contributions, with the usual G/(G-1) * (N-1)/(N-K) small-sample factor.
The two-way estimator is V_event + V_team - V_(event x team).

Every fit reports its Newton decrement, g' (-H)^-1 g / 2, which bounds how far
its log likelihood is from the maximum. It is a scale-free convergence check
that does not rely on the optimizer's own stopping rule.
"""

from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.tools.numdiff import approx_fprime, approx_hess

Z_CRIT = norm.ppf(0.975)


@dataclass
class Fit:
    """A fitted model and everything the tables need from it."""

    xcols: list
    params: np.ndarray
    covariances: dict
    loglike: float
    pseudo_r2: float
    n: int
    n_events: int
    newton_decrement: float
    data: pd.DataFrame = field(repr=False)
    top1: float = np.nan
    tie_adjusted_top1: float = np.nan


def prepare_choice_sets(data, xcols):
    """Rows usable for a conditional-choice specification.

    Drops rows with missing or infinite regressors (component EPA exists only
    from 2016; a few qualification ranks are missing), then keeps events that
    still contain exactly one accepted partner and at least two candidates.
    """
    xcols = list(xcols)
    d = data.dropna(subset=xcols).copy()
    d = d[np.isfinite(d[xcols]).all(axis=1)]
    d = d[d.groupby("event")["pick1"].transform("sum") == 1]
    d = d[d.groupby("event")["pick1"].transform("size") >= 2]
    return d.sort_values(["event", "team"]).reset_index(drop=True)


def _cluster_meat(scores, groups):
    summed = (
        pd.DataFrame(scores)
        .assign(_cluster=np.asarray(groups))
        .groupby("_cluster", sort=False).sum()
        .to_numpy()
    )
    n, k = scores.shape
    n_clusters = len(summed)
    if n_clusters <= 1 or n <= k:
        raise ValueError("Not enough clusters for a clustered covariance")
    correction = n_clusters / (n_clusters - 1) * (n - 1) / (n - k)
    return correction * (summed.T @ summed)


def _sandwiches(data, scores, bread):
    """Event-, team-, and two-way clustered covariance matrices."""
    event = np.asarray(data["event"].astype(str), dtype=str)
    team = np.asarray(data["team"].astype(str), dtype=str)
    event_team = np.char.add(np.char.add(event, "|"), team)
    meat_event = _cluster_meat(scores, event)
    meat_team = _cluster_meat(scores, team)
    meat_both = _cluster_meat(scores, event_team)
    return {
        "event_cluster": bread @ meat_event @ bread,
        "team_cluster": bread @ meat_team @ bread,
        "event_team_two_way": bread @ (meat_event + meat_team - meat_both) @ bread,
    }


def _symmetric(covariances):
    return {name: (c + c.T) / 2 for name, c in covariances.items()}


def top1_accuracy(data, scores):
    """Share of events whose accepted partner has the highest score.

    Returns the raw hit rate and a tie-adjusted rate that divides each hit by
    the number of candidates sharing the event maximum.
    """
    work = data[["event", "pick1"]].copy()
    work["score"] = np.asarray(scores)
    maximum = work.groupby("event")["score"].transform("max")
    work["is_top"] = np.isclose(work["score"], maximum, atol=1e-10)
    picked = work[work["pick1"] == 1].set_index("event")
    tie_sizes = work[work["is_top"]].groupby("event").size()
    hit = float(picked["is_top"].mean())
    tie_adjusted = float((picked["is_top"] / tie_sizes).fillna(0).mean())
    return hit, tie_adjusted


def newton_decrement(gradient, hessian):
    """g' (-H)^-1 g / 2 for the gradient and Hessian of a log likelihood."""
    return float(gradient @ np.linalg.pinv(-hessian) @ gradient / 2)


def uniform_null_loglike(data):
    """Log likelihood of picking uniformly at random within each event."""
    return -np.log(data.groupby("event").size()).sum()


def _event_blocks(data):
    """Start index and size of each event's rows (data sorted by event)."""
    codes, _ = pd.factorize(data["event"], sort=False)
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    return starts, np.diff(np.r_[starts, len(data)])


def conditional_loglike(x, y, params, starts, sizes):
    """Log likelihood, probabilities, index, gradient, and exact Hessian.

    gradient = sum_i x_i (y_i - p_i)
    Hessian  = -sum_i p_i (x_i - xbar_e)(x_i - xbar_e)',  xbar_e = sum_e p_i x_i
    """
    eta = x @ params
    maxima = np.maximum.reduceat(eta, starts)
    weights = np.exp(eta - np.repeat(maxima, sizes))
    denominators = np.add.reduceat(weights, starts)
    p = weights / np.repeat(denominators, sizes)
    loglike = float(y @ eta - (maxima + np.log(denominators)).sum())
    xbar = np.add.reduceat(x * p[:, None], starts)
    centered = x - np.repeat(xbar, sizes, axis=0)
    gradient = x.T @ (y - p)
    hessian = -(centered * p[:, None]).T @ centered
    return loglike, p, eta, gradient, hessian


def _newton(x, y, starts, sizes, tolerance=1e-12, maxiter=100):
    """Damped Newton-Raphson from zero; the log likelihood is concave."""
    params = np.zeros(x.shape[1])
    state = conditional_loglike(x, y, params, starts, sizes)
    for _ in range(maxiter):
        loglike, _, _, gradient, hessian = state
        step = np.linalg.solve(-hessian, gradient)
        if gradient @ step / 2 < tolerance:
            return params
        t = 1.0
        while True:
            trial = conditional_loglike(x, y, params + t * step, starts, sizes)
            if trial[0] >= loglike or t < 1e-10:
                break
            t /= 2
        params, state = params + t * step, trial
    raise RuntimeError("Newton-Raphson did not converge")


def conditional_logit(data, xcols, optimizer="bfgs",
                      covariance_types=("conventional", "team_cluster",
                                        "event_team_two_way")):
    xcols = list(xcols)
    d = prepare_choice_sets(data, xcols)
    if d.empty:
        raise ValueError(f"No usable choice sets for {xcols}")
    x = d[xcols].to_numpy(float)
    y = d["pick1"].to_numpy(float)
    starts, sizes = _event_blocks(d)

    if optimizer == "bfgs":
        model = ConditionalLogit(d["pick1"], d[xcols], groups=d["event"])
        params = np.asarray(model.fit(method="bfgs", maxiter=500,
                                      disp=False).params)
    elif optimizer == "newton":
        params = _newton(x, y, starts, sizes)
    else:
        raise ValueError(f"Unknown optimizer {optimizer!r}")

    loglike, probabilities, eta, gradient, hessian = conditional_loglike(
        x, y, params, starts, sizes)
    bread = np.linalg.pinv(-hessian)
    covariances = _sandwiches(d, x * (y - probabilities)[:, None], bread)
    covariances["conventional"] = bread
    covariances = _symmetric({k: covariances[k] for k in covariance_types})

    top1, tie_adjusted = top1_accuracy(d, eta)
    return Fit(
        xcols=xcols,
        params=params,
        covariances=covariances,
        loglike=loglike,
        pseudo_r2=float(1 - loglike / -np.log(sizes).sum()),
        n=len(d),
        n_events=len(starts),
        newton_decrement=newton_decrement(gradient, hessian),
        data=d,
        top1=top1,
        tie_adjusted_top1=tie_adjusted,
    )


def pooled_logit(data, xcols, report_cols=None, optimizer="bfgs"):
    """Binary logit with an intercept; pseudo-R^2 is against intercept-only."""
    xcols = list(xcols)
    report_cols = list(report_cols or xcols)
    d = data.dropna(subset=xcols).copy()
    d = d[np.isfinite(d[xcols]).all(axis=1)].reset_index(drop=True)
    X = sm.add_constant(d[xcols], has_constant="add")
    x = X.to_numpy(float)
    y = d["pick1"].to_numpy(float)

    model = sm.Logit(d["pick1"], X)
    result = model.fit(method=optimizer, maxiter=500, disp=False)
    params = np.asarray(result.params)

    probabilities = 1 / (1 + np.exp(-(x @ params)))
    hessian = np.asarray(model.hessian(params))
    bread = np.linalg.pinv(-hessian)
    covariances = _sandwiches(d, x * (y - probabilities)[:, None], bread)

    keep = [list(X.columns).index(c) for c in report_cols]
    return Fit(
        xcols=report_cols,
        params=params[keep],
        covariances=_symmetric(
            {k: c[np.ix_(keep, keep)] for k, c in covariances.items()}
        ),
        loglike=float(result.llf),
        pseudo_r2=float(result.prsquared),
        n=len(d),
        n_events=d["event"].nunique(),
        newton_decrement=newton_decrement(np.asarray(model.score(params)),
                                          hessian),
        data=d,
    )


def mixed_logit(data, xcol, fixed_col=None, quadrature_points=20):
    """Conditional logit with an event-level random slope on `xcol`.

    beta_e ~ Normal(mean, sigma^2) is drawn once per event and integrated out
    with Gauss-Hermite quadrature. `fixed_col` optionally adds one regressor
    with a common slope. Parameters are (mean, log sigma[, fixed slope]); the
    covariance comes from the numerical Hessian of the log likelihood.
    """
    required = [xcol] + ([fixed_col] if fixed_col else [])
    d = prepare_choice_sets(data, required)
    x = d[xcol].to_numpy(float)
    z = d[fixed_col].to_numpy(float) if fixed_col else np.zeros(len(d))
    y = d["pick1"].to_numpy(float)

    codes, _ = pd.factorize(d["event"], sort=False)
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    sizes = np.diff(np.r_[starts, len(d)])
    nodes, weights = hermgauss(quadrature_points)
    weights = weights / np.sqrt(np.pi)
    log_weights = np.log(weights)

    def event_log_prob(beta, gamma):
        eta = beta * x + gamma * z
        maxima = np.maximum.reduceat(eta, starts)
        log_denominator = maxima + np.log(
            np.add.reduceat(np.exp(eta - np.repeat(maxima, sizes)), starts)
        )
        return np.add.reduceat(y * eta, starts) - log_denominator

    def unpack(theta):
        return theta[0], np.exp(theta[1]), (theta[2] if fixed_col else 0.0)

    def loglike(theta):
        mean, sigma, gamma = unpack(theta)
        draws = np.vstack([
            event_log_prob(beta, gamma)
            for beta in mean + np.sqrt(2) * sigma * nodes
        ])
        return float(logsumexp(log_weights[:, None] + draws, axis=0).sum())

    start = np.array([2.5, np.log(0.5)] + ([2.0] if fixed_col else []))
    bounds = [(-20, 20), (-7, 3)] + ([(-20, 20)] if fixed_col else [])
    opt = minimize(lambda t: -loglike(t), start, method="L-BFGS-B",
                   bounds=bounds)
    if not opt.success:
        warnings.warn(f"Mixed logit did not fully converge: {opt.message}")

    theta = opt.x
    hessian = -approx_hess(theta, lambda t: -loglike(t))
    covariance = np.linalg.pinv(-hessian)
    llf = loglike(theta)

    # Top-1 accuracy from the probabilities averaged over the slope draws.
    mean, sigma, gamma = unpack(theta)
    probabilities = np.zeros(len(d))
    for w, beta in zip(weights, mean + np.sqrt(2) * sigma * nodes):
        eta = beta * x + gamma * z
        centered = eta - np.repeat(np.maximum.reduceat(eta, starts), sizes)
        denom = np.repeat(np.add.reduceat(np.exp(centered), starts), sizes)
        probabilities += w * np.exp(centered) / denom
    top1, tie_adjusted = top1_accuracy(d, probabilities)

    names = [f"{xcol}_mean", f"{xcol}_sd"] + ([fixed_col] if fixed_col else [])
    return Fit(
        xcols=names,
        params=theta,
        covariances=_symmetric({"numerical_hessian": covariance}),
        loglike=llf,
        pseudo_r2=float(1 - llf / -np.log(sizes).sum()),
        n=len(d),
        n_events=len(starts),
        newton_decrement=newton_decrement(approx_fprime(theta, loglike),
                                          hessian),
        data=d,
        top1=top1,
        tie_adjusted_top1=tie_adjusted,
    )


def tidy(fit, model, sample, event_type="ALL", log_sigma_var=None,
         tolerance=0.05):
    """Long-format rows for one fitted model.

    A fit counts as converged when its Newton decrement is below `tolerance`.

    Returns (rows, invalid) where `invalid` lists covariance types with a
    negative variance on the diagonal. Those standard errors and intervals are
    left blank rather than forced positive.

    `log_sigma_var` names a parameter estimated on the log scale (the mixed
    logit's slope SD); it is reported as sigma with a delta-method SE and an
    interval obtained by exponentiating the log-scale interval.
    """
    rows, invalid = [], []
    for cov_name, cov in fit.covariances.items():
        variances = np.diag(cov)
        if (variances < 0).any():
            invalid.append((cov_name, int((variances < 0).sum())))
        ses = np.sqrt(np.where(variances >= 0, variances, np.nan))
        for var, est, se in zip(fit.xcols, fit.params, ses):
            ok = np.isfinite(se)
            lo = est - Z_CRIT * se if ok else np.nan
            hi = est + Z_CRIT * se if ok else np.nan
            p = 2 * norm.sf(abs(est / se)) if ok and se > 0 else np.nan
            if var == log_sigma_var:
                est, se, lo, hi, p = (np.exp(est), np.exp(est) * se,
                                      np.exp(lo), np.exp(hi), np.nan)
            rows.append({
                "model": model,
                "sample": sample,
                "event_type": event_type,
                "covariance": cov_name,
                "var": var,
                "coef": est,
                "se": se,
                "ci_low": lo,
                "ci_high": hi,
                "p": p,
                "odds_ratio": np.exp(est) if abs(est) < 700 else np.nan,
                "n": fit.n,
                "n_events": fit.n_events,
                "loglike": fit.loglike,
                "pseudo_r2": fit.pseudo_r2,
                "top1_accuracy": fit.top1,
                "tie_adjusted_top1": fit.tie_adjusted_top1,
                "newton_decrement": fit.newton_decrement,
                "converged": fit.newton_decrement < tolerance,
            })
    return rows, invalid


def contrast(fit, left, right, covariance):
    """Estimate, SE, 95% CI, and p-value for beta_left - beta_right."""
    weights = np.zeros(len(fit.xcols))
    weights[fit.xcols.index(left)] = 1
    weights[fit.xcols.index(right)] = -1
    estimate = float(weights @ fit.params)
    variance = float(weights @ fit.covariances[covariance] @ weights)
    se = np.sqrt(variance) if variance >= 0 else np.nan
    ok = np.isfinite(se)
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - Z_CRIT * se if ok else np.nan,
        "ci_high": estimate + Z_CRIT * se if ok else np.nan,
        "p": 2 * norm.sf(abs(estimate / se)) if ok and se > 0 else np.nan,
    }
