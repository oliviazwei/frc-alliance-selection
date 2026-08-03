"""Step 5: build every table and figure in the paper.

Regression tables are written directly from data/derived/results.csv, so the
numbers in the paper cannot drift from the estimates. LaTeX fragments land in
paper/tables/ and are pulled in by paper/main.tex with \\input.

Outputs:
  paper/figures/fig_team_event_counts.pdf|.eps   (Figure 1)
  paper/figures/fig_epa_z_hist.pdf|.eps          (Figure 2)
  paper/figures/fig_pick_rank.pdf|.eps           (Figure 3)
  paper/tables/tab1_event_types.tex   ... tab6_horserace.tex
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (FIGURES, RESULTS_CSV, SAMPLE_CSV, TABLES, TEAM_EVENTS_CSV,
                    TYPE_LABELS, TYPE_ORDER)

df = pd.read_csv(SAMPLE_CSV)
res = pd.read_csv(RESULTS_CSV)

ROWS = [(t, TYPE_LABELS[t]) for t in TYPE_ORDER] + [("ALL", "All events combined")]


def thous(n):
    """Thousands separator that survives LaTeX math mode."""
    return f"{int(n):,}".replace(",", "{,}")


def num(v, dec=1):
    return "--" if pd.isna(v) else f"{v:.{dec}f}"


def write(name, text):
    (TABLES / name).write_text(text, encoding="utf-8")
    print(f"  wrote paper/tables/{name}")


# ==========================================================================
# Figure 1: participation over time (all official events, not just the sample)
# ==========================================================================
team_events = pd.read_csv(TEAM_EVENTS_CSV)
years = range(2010, 2026)
teams_per_year = team_events.groupby("year")["team"].nunique().reindex(years)
events_per_year = team_events.groupby("year")["event"].nunique().reindex(years)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(years, teams_per_year.values, color="blue", marker="o")
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of teams", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax2 = ax1.twinx()
ax2.plot(years, events_per_year.values, color="red", marker="s")
ax2.set_ylabel("Number of events", color="red")
ax2.tick_params(axis="y", labelcolor="red")
plt.title("FRC: Number of teams and events by year (2010–2025)")
plt.tight_layout()
for ext in ("pdf", "eps"):
    plt.savefig(FIGURES / f"fig_team_event_counts.{ext}")
plt.close(fig)
print(f"  wrote paper/figures/fig_team_event_counts.pdf "
      f"({team_events['team'].nunique():,} teams, "
      f"{team_events['event'].nunique():,} events)")

# ==========================================================================
# Table 1: analysis-sample events by type
# ==========================================================================
counts = df.groupby("event_type_string")["event"].nunique()
rows = "\n".join(
    f"        {TYPE_LABELS[t]:31s} & {thous(counts[t])} \\\\" for t in TYPE_ORDER
)
write("tab1_event_types.tex", rf"""\begin{{table}}[h!]
    \centering
\begin{{threeparttable}}
    \caption{{Number of FRC events in the analysis sample by event type, 2010--2025}}
    \label{{tab:event_types}}

\begin{{tabular}}
{{@{{\extracolsep{{125pt}}}}@{{\kern\tabcolsep}}l r}}
\toprule
        Event type                      & Number of events \\
        \hline
{rows}
\midrule
        Total                           & {thous(counts.sum())} \\
\bottomrule
    \end{{tabular}}
{{\footnotesize
\begin{{tablenotes}}
\item \textsc{{Note}}: The table reports the number of distinct FIRST Robotics Competition (FRC) events by event type in the analysis sample, 2010--2025 seasons (2021 is excluded because the pandemic canceled the in-person season). Event types follow FIRST/The Blue Alliance classifications. The sample includes every official event with Statbotics team-level EPA data and a recorded alliance selection. Championship Finals (Einstein) fields are excluded because no alliance selection takes place there: the division-winning alliances advance intact. Offseason events are excluded because Statbotics does not provide EPA data for past offseason competitions.
\end{{tablenotes}}
}}
\end{{threeparttable}}

\end{{table}}
""")

# ==========================================================================
# Table 2: descriptive statistics by season
# ==========================================================================
by_year = df.groupby("year").agg(
    teams=("team", "nunique"),
    events=("event", "nunique"),
    epa_mean=("epa_total", "mean"),
    epa_sd=("epa_total", "std"),
    auto_mean=("epa_auto", "mean"),
    teleop_mean=("epa_teleop", "mean"),
    endgame_mean=("epa_endgame", "mean"),
)
season_rows = "\n".join(
    f"        {int(y)} & {thous(r['teams'])} & {int(r['events'])} & "
    f"{num(r['epa_mean'])} & {num(r['epa_sd'])} & {num(r['auto_mean'])} & "
    f"{num(r['teleop_mean'])} & {num(r['endgame_mean'])} \\\\"
    for y, r in by_year.iterrows()
)
write("tab2_descriptives.tex", rf"""\begin{{table}}[h!]
    \centering
\begin{{threeparttable}}
    \caption{{Descriptive statistics by season}}
    \label{{tab:desc_by_season}}
{{\footnotesize
\begin{{tabular}}
{{@{{\extracolsep{{8pt}}}}@{{\kern\tabcolsep}}lccccccc}}
\toprule
        & & & \multicolumn{{2}}{{c}}{{Overall EPA}} & \multicolumn{{3}}{{c}}{{Component means}} \\
\cmidrule{{4-5}} \cmidrule{{6-8}}
        Season & Teams & Events & Mean & SD & Auto & Teleop & Endgame \\[0.2em]
        \hline
{season_rows}
\bottomrule
    \end{{tabular}}
}}
    \vspace{{0.5em}}
    \footnotesize\emph{{Notes:}} Statistics are computed over the estimation sample (team--event observations at events with a recorded alliance selection, excluding the top alliance's captain). ``Teams'' is the number of unique teams and ``Events'' the number of events in each season. EPA is measured in points of that season's game, so levels are not comparable across seasons. Statbotics reports the autonomous, teleoperated, and endgame components from the 2016 season onward; dashes denote seasons without component data. The 2020 season was cut short by the COVID-19 pandemic and the 2021 season was canceled.

\end{{threeparttable}}
\end{{table}}
""")

teleop_share = (by_year["teleop_mean"] / by_year["epa_mean"]).dropna()
print(f"  [text check] teleop share of mean EPA: "
      f"min {teleop_share.min():.2f} ({int(teleop_share.idxmin())}), "
      f"max {teleop_share.max():.2f} ({int(teleop_share.idxmax())})")

# ==========================================================================
# Figure 2: within-event standardized EPA, first picks vs. the field
# ==========================================================================
field = df.loc[df["pick1"] == 0, "epa_z"].dropna()
picks = df.loc[df["pick1"] == 1, "epa_z"].dropna()
bins = np.arange(-4.0, 5.25, 0.25)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(field.clip(-4, 5), bins=bins, density=True, color="blue", alpha=0.45,
        label="All candidate teams")
ax.hist(picks.clip(-4, 5), bins=bins, density=True, color="red", alpha=0.55,
        label="First picks of the top alliance")
ax.set_xlabel("Within-event standardized EPA")
ax.set_ylabel("Density")
ax.legend(loc="upper left")
plt.tight_layout()
for ext in ("pdf", "eps"):
    plt.savefig(FIGURES / f"fig_epa_z_hist.{ext}")
plt.close(fig)

top_share = (
    df.dropna(subset=["epa_z"])
      .assign(is_top=lambda d: d.groupby("event")["epa_z"].transform("max") == d["epa_z"])
      .query("pick1 == 1")["is_top"].mean()
)
print(f"  wrote paper/figures/fig_epa_z_hist.pdf")
print(f"  [text check] mean z of first picks = {picks.mean():.2f}; "
      f"pick is highest-EPA candidate in {top_share*100:.1f}% of events")

# ==========================================================================
# Figure 3: qualification rank of the first pick
# ==========================================================================
pick_rank = df.loc[df["pick1"] == 1, "qual_rank"].dropna()
MAX_BIN = 12
shares = [(str(r), (pick_rank == r).mean() * 100) for r in range(2, MAX_BIN + 1)]
shares.append((f"{MAX_BIN + 1}+", (pick_rank > MAX_BIN).mean() * 100))

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([lbl for lbl, _ in shares], [v for _, v in shares], color="blue")
ax.set_xlabel("Qualification rank of the first pick")
ax.set_ylabel("Percent of first picks")
plt.tight_layout()
for ext in ("pdf", "eps"):
    plt.savefig(FIGURES / f"fig_pick_rank.{ext}")
plt.close(fig)
print(f"  wrote paper/figures/fig_pick_rank.pdf")
print(f"  [text check] first pick is rank 2 in {(pick_rank == 2).mean()*100:.0f}%, "
      f"median rank {pick_rank.median():.0f}, "
      f"outside top 5 in {(pick_rank > 5).mean()*100:.0f}%, "
      f"outside top 8 in {(pick_rank > 8).mean()*100:.0f}% "
      f"(n={len(pick_rank):,})")

# ==========================================================================
# Regression tables
# ==========================================================================
def cell(spec, etype, var, dec=3):
    r = res[(res["spec"] == spec) & (res["event_type"] == etype) & (res["var"] == var)]
    if r.empty:
        return "--"
    r = r.iloc[0]
    return f"{r['coef']:.{dec}f}\\,({r['se']:.{dec}f})"


def stat(spec, etype, field_, dec=3):
    r = res[(res["spec"] == spec) & (res["event_type"] == etype)]
    if r.empty:
        return "--"
    v = r.iloc[0][field_]
    return thous(v) if field_ in ("n", "n_events") else f"{v:.{dec}f}"


def body(row_fn):
    lines = []
    for key, label in ROWS:
        if key == "ALL":
            lines.append(r" \\")
            lines.append(r"\midrule")
        lines.append(f"        {label:31s} & " + row_fn(key) + r" \\")
    return "\n".join(lines)


# --- Table 3: pooled logit, total EPA -------------------------------------
write("tab3_pooled_logit.tex", rf"""\begin{{table}}[h!]
    \centering
\begin{{threeparttable}}

    \caption{{Logit estimates of the effect of EPA on the probability of being the top alliance's first pick, by event type}}
    \label{{tab:logit_by_event_type}}

\begin{{tabular}}
{{@{{\extracolsep{{18pt}}}}@{{\kern\tabcolsep}}lcccc}}
\toprule

        Event type                      & $\hat\beta_{{\text{{EPA}}}}$ &  $R^2$ & $N$ & Events \\[0.2em]
        \hline
{body(lambda k: " & ".join([
    cell("pooled_total", k, "epa_total", 4),
    stat("pooled_total", k, "pseudo_r2"),
    stat("pooled_total", k, "n"),
    stat("pooled_total", k, "n_events")]))}

\bottomrule
    \end{{tabular}}
    \vspace{{0.5em}}
    \footnotesize\emph{{Notes:}} Entries report coefficients from separate logistic regressions of a dummy for being selected as the first pick by the top-ranked alliance captain on the team's event-level EPA, estimated separately by event type on the full 2010--2025 sample. Standard errors clustered by event are reported in parentheses. $R^2$ is McFadden's pseudo-$R^2$. All coefficients are significant at the 1 percent level.

\end{{threeparttable}}
\end{{table}}
""")

# --- Table 4: conditional logit, total EPA ---------------------------------
write("tab4_conditional_logit.tex", rf"""\begin{{table}}[h!]
    \centering
\begin{{threeparttable}}

    \caption{{Conditional logit estimates: the top alliance's first pick as a choice among the teams at the event}}
    \label{{tab:clogit_total}}

{{\scriptsize
\begin{{tabular}}
{{lccccccc}}
\toprule
        Event type                      & $\hat\beta_{{\text{{EPA}}}}$ & $\hat\beta_{{\text{{EPA}}}}^{{\,\text{{pre}}}}$ & $\hat\beta_{{\text{{EPA}}}}^{{\,z}}$ & $R^2$ & $R^2$ ($z$) & $N$ & Events \\[0.2em]
        \hline
{body(lambda k: " & ".join([
    cell("clogit_total", k, "epa_total"),
    cell("clogit_pre_elim", k, "epa_pre_elim"),
    cell("clogit_z", k, "epa_z"),
    stat("clogit_total", k, "pseudo_r2"),
    stat("clogit_z", k, "pseudo_r2"),
    stat("clogit_total", k, "n"),
    stat("clogit_total", k, "n_events")]))}
\bottomrule
    \end{{tabular}}
}}
    \vspace{{0.5em}}
    \footnotesize\emph{{Notes:}} Each entry is from a separate conditional (McFadden) logit in which the top-ranked alliance captain at each event chooses exactly one first pick from the teams at that event (the captain itself is excluded from the choice set). ``End-of-event EPA'' ($\hat\beta_{{\text{{EPA}}}}$) uses the team's EPA at the end of the event; ``pre-playoff EPA'' ($\hat\beta_{{\text{{EPA}}}}^{{\,\text{{pre}}}}$) uses the team's EPA at the end of the qualification matches, i.e., the information actually available at the moment of alliance selection; the within-event $z$-score ($\hat\beta_{{\text{{EPA}}}}^{{\,z}}$) standardizes end-of-event EPA to mean zero and unit standard deviation within each event. Standard errors in parentheses. $R^2$ is McFadden's pseudo-$R^2$ relative to a uniform-random pick from the choice set, reported for the end-of-event EPA and $z$-score specifications (the pre-playoff specification fits almost identically to the first column). All coefficients are significant at the 1 percent level.

\end{{threeparttable}}
\end{{table}}
""")

# --- Table 5: components, pooled vs conditional ----------------------------
def comp_row(spec, k):
    return " & ".join([
        cell(spec, k, "epa_auto"), cell(spec, k, "epa_teleop"),
        cell(spec, k, "epa_endgame"), stat(spec, k, "pseudo_r2"),
        stat(spec, k, "n"),
    ])


write("tab5_components.tex", rf"""\begin{{table}}[h!]
    \centering
\begin{{threeparttable}}
    \caption{{EPA components and the probability of being the top alliance's first pick, by event type}}
    \label{{tab:logit_by_event_type2}}
{{\footnotesize
\begin{{tabular}}
{{@{{\extracolsep{{2pt}}}}@{{\kern\tabcolsep}}lccccc}}
\toprule
        Event type                      & $ \widehat\beta_{{\text{{auto}}}} $ & $ \widehat\beta_{{\text{{teleop}}}} $ & $ \widehat\beta_{{\text{{endgame}}}} $ & $R^2$ & $N$ \\[0.2em]
\midrule
\multicolumn{{6}}{{l}}{{\emph{{Panel A: pooled logit (cross-event comparison)}}}} \\[0.2em]
{body(lambda k: comp_row("pooled_components", k))}
\midrule
\multicolumn{{6}}{{l}}{{\emph{{Panel B: conditional logit (within-event choice)}}}} \\[0.2em]
{body(lambda k: comp_row("clogit_components", k))}
\bottomrule
    \end{{tabular}}
}}
    \vspace{{0.5em}}
    \footnotesize\emph{{Notes:}} Each row reports a separate regression of the first-pick outcome on the autonomous, teleoperated, and endgame components of EPA. Panel~A pools all team--event observations within an event type in a standard logit (standard errors clustered by event); Panel~B estimates a conditional logit in which the captain chooses one team from the event's field, so that only within-event differences in the components matter. Statbotics reports the component breakdown of EPA from the 2016 season onward, so both panels cover 2016--2025. $R^2$ is McFadden's pseudo-$R^2$ (in Panel~B, relative to a uniform-random pick). The sign reversal on teleoperated EPA between the two panels reflects the pooling of raw point scales across seasons in Panel~A; see the text.

\end{{threeparttable}}
\end{{table}}
""")

# --- Table 6: EPA vs qualification rank ------------------------------------
def hr_row(k):
    return " & ".join([
        cell("clogit_z", k, "epa_z"), stat("clogit_z", k, "pseudo_r2"),
        cell("clogit_rank_only", k, "rank_pct", 2),
        stat("clogit_rank_only", k, "pseudo_r2"),
        cell("clogit_horserace", k, "epa_z"),
        cell("clogit_horserace", k, "rank_pct", 2),
        stat("clogit_horserace", k, "pseudo_r2"),
        stat("clogit_horserace", k, "n"),
    ])


write("tab6_horserace.tex", rf"""\begin{{table}}[h!]
    \centering
\begin{{threeparttable}}
    \caption{{Horse race: EPA versus the official qualification ranking (conditional logit)}}
    \label{{tab:horserace}}
{{\scriptsize
\begin{{tabular}}
{{lcccccccc}}
\toprule
        & \multicolumn{{2}}{{c}}{{EPA only}} & \multicolumn{{2}}{{c}}{{Rank only}} & \multicolumn{{3}}{{c}}{{EPA and rank}} & \\
\cmidrule{{2-3}} \cmidrule{{4-5}} \cmidrule{{6-8}}
        Event type                      & $\hat\beta_{{\text{{EPA}}}}^{{\,z}}$ & $R^2$ & $\hat\beta_{{\text{{rank}}}}$ & $R^2$ & $\hat\beta_{{\text{{EPA}}}}^{{\,z}}$ & $\hat\beta_{{\text{{rank}}}}$ & $R^2$ & $N$ \\[0.2em]
        \hline
{body(hr_row)}
\bottomrule
    \end{{tabular}}
}}
    \vspace{{0.5em}}
    \footnotesize\emph{{Notes:}} Conditional logit estimates of the top captain's first-pick choice. $\text{{EPA}}^{{\,z}}$ is end-of-event EPA standardized within event; ``rank'' is the team's qualification-rank percentile within the event, rescaled so that 1 denotes the top-ranked and 0 the bottom-ranked team. Standard errors in parentheses. $R^2$ is McFadden's pseudo-$R^2$ relative to a uniform-random pick. $N$ refers to the joint specification, which requires a non-missing qualification rank; the EPA-only sample is marginally larger. All coefficients are significant at the 1 percent level except the rank coefficient in the joint specification for District Championship Divisions ($p>0.10$) and District Championships ($p<0.05$).

\end{{threeparttable}}
\end{{table}}
""")

print("\nAll exhibits written.")
