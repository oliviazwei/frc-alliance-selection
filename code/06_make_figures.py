"""Step 6: draw every figure in the paper and the supplementary appendix.

Outputs (PDF for print, PNG at 300 dpi for Word and the web):
  output/figures/figure1_alliance_selection.pdf|.png   Figure 1
  output/figures/figure2_epa_z_distribution.pdf|.png   Figure 2
  output/figures/figure3_partner_rank.pdf|.png         Figure 3
  output/figures/figureA1_epa_by_season.pdf|.png       Supplementary Figure A1
  output/figures/figureA2_event_epa_sd.pdf|.png        Supplementary Figure A2
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from config import FIGURES, MODELS_CSV, SAMPLE_CSV

sample = pd.read_csv(SAMPLE_CSV)
models = pd.read_csv(MODELS_CSV)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#595959"
EDGE = "#333333"
LIGHT_BLUE = "#DCEAF4"
LIGHT_ORANGE = "#F7E4D7"
LIGHT_GRAY = "#F2F2F2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
})


def save(fig, name):
    for ext, options in [("pdf", {}), ("png", {"dpi": 300})]:
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight",
                    pad_inches=0.04, **options)
    plt.close(fig)
    print(f"[06] wrote output/figures/{name}.pdf|.png")


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure 1: procedure for forming playoff alliances
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.2, 5.7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x, y, w, h, text, face=LIGHT_BLUE, lw=1.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=face, edgecolor=EDGE, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=14, linespacing=1.15)


def arrow(start, end, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2,
        color=EDGE, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))


def label(x, y, text, size=12.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=EDGE,
            linespacing=1.1)


box(0.015, 0.67, 0.16, 0.17, "Qualification\nmatches")
box(0.205, 0.67, 0.16, 0.17, "Official\nrankings")
box(0.395, 0.67, 0.16, 0.17, "Top teams\nbecome alliance\ncaptains")
box(0.585, 0.67, 0.16, 0.17, "Alliance 1\ninvites eligible\nteam")
ax.add_patch(Polygon([[0.875, 0.65], [0.955, 0.755], [0.875, 0.86],
                      [0.795, 0.755]], closed=True, facecolor=LIGHT_GRAY,
                     edgecolor=EDGE, linewidth=1.2))
ax.text(0.875, 0.755, "Invitation\naccepted?", ha="center", va="center",
        fontsize=14)
box(0.745, 0.29, 0.23, 0.18, "Accepted first\npartner analyzed",
    LIGHT_ORANGE, 1.7)
box(0.455, 0.29, 0.23, 0.18, "Invitation declined\nNot recorded\nin the data",
    LIGHT_GRAY)

for x0, x1 in [(0.175, 0.205), (0.365, 0.395), (0.555, 0.585),
               (0.745, 0.795)]:
    arrow((x0, 0.755), (x1, 0.755))
arrow((0.90, 0.655), (0.88, 0.475))
label(0.915, 0.565, "Yes")
arrow((0.83, 0.67), (0.66, 0.475), dashed=True)
label(0.735, 0.62, "No")
arrow((0.57, 0.475), (0.66, 0.665), dashed=True, rad=-0.35)
label(0.475, 0.585, "Invite another\neligible team")
ax.text(0.5, 0.10, "Solid path: recorded alliance formation. "
        "Dashed path: invitation history not observed.",
        ha="center", va="center", fontsize=12, color=GRAY)
save(fig, "figure1_alliance_selection")


# ---------------------------------------------------------------------------
# Figure 2: within-event standardized EPA, selected versus not selected
# ---------------------------------------------------------------------------
selected = sample.loc[sample["pick1"] == 1, "epa_z"]
not_selected = sample.loc[sample["pick1"] == 0, "epa_z"]
bins = np.arange(-3.25, 5.26, 0.25)
fig, ax = plt.subplots(figsize=(9.2, 5.3))
ax.hist(not_selected, bins=bins, density=True, histtype="step", linewidth=2.0,
        color=BLUE, label=f"Candidates not selected (N={len(not_selected):,})")
ax.hist(selected, bins=bins, density=True, histtype="step", linewidth=2.2,
        linestyle="--", color=ORANGE,
        label=f"Accepted first partners (N={len(selected):,})")
ax.axvline(selected.mean(), color=ORANGE, linewidth=1.0, linestyle=":")
ax.text(selected.mean() - 0.08, ax.get_ylim()[1] * 0.9,
        f"Partner mean = {selected.mean():.2f}", color=ORANGE, fontsize=9,
        ha="right")
ax.set_xlim(-3.25, 5.1)
ax.set_xlabel("Within-event standardized EPA")
ax.set_ylabel("Density")
ax.legend(loc="upper left", frameon=False)
clean_axis(ax)
save(fig, "figure2_epa_z_distribution")


# ---------------------------------------------------------------------------
# Figure 3: qualification rank of the accepted first partner
# ---------------------------------------------------------------------------
pick_rank = sample.loc[sample["pick1"] == 1, "qual_rank"].dropna()
labels = [str(r) for r in range(2, 13)] + ["13 or worse"]
shares = [100 * (pick_rank == r).mean() for r in range(2, 13)]
shares.append(100 * (pick_rank >= 13).mean())
fig, ax = plt.subplots(figsize=(9.4, 5.3))
bars = ax.bar(np.arange(len(labels)), shares,
              color=[ORANGE] + [BLUE] * 10 + ["#B3B3B3"],
              edgecolor=EDGE, linewidth=0.6)
bars[-1].set_hatch("///")
for bar, share in zip(bars, shares):
    ax.text(bar.get_x() + bar.get_width() / 2, share + 0.7, f"{share:.1f}",
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(np.arange(len(labels)), labels)
ax.tick_params(axis="x", rotation=28)
ax.set_ylim(0, max(shares) + 6)
ax.set_xlabel("Qualification rank of accepted first partner")
ax.set_ylabel("Share of first partners (percent)")
clean_axis(ax)
save(fig, "figure3_partner_rank")


# ---------------------------------------------------------------------------
# Supplementary Figure A1: standardized-EPA coefficient by season
# ---------------------------------------------------------------------------
team = models[(models["model"] == "epa_z") & (models["covariance"] == "team_cluster")
              & (models["event_type"] == "ALL")]
season = team[team["sample"].str.startswith("season_")].copy()
season["year"] = season["sample"].str.removeprefix("season_").astype(int)
season = season.sort_values("year")
pooled = team[team["sample"] == "full"].iloc[0]
fig, ax = plt.subplots(figsize=(9.4, 5.3))
ax.errorbar(season["year"], season["coef"],
            yerr=[season["coef"] - season["ci_low"],
                  season["ci_high"] - season["coef"]],
            fmt="o", color=BLUE, ecolor=BLUE, capsize=3, markersize=5,
            linewidth=1.2, label="Season estimate with 95% CI")
ax.axhline(pooled["coef"], color=ORANGE, linestyle="--", linewidth=1.3,
           label=f"All seasons combined ({pooled['coef']:.2f})")
ax.set_xticks(season["year"])
ax.tick_params(axis="x", rotation=35)
ax.set_ylim(0, max(5.8, season["ci_high"].max() + 0.25))
ax.set_xlabel("Season")
ax.set_ylabel("Conditional-logit coefficient on within-event EPA z-score")
ax.legend(loc="upper left", frameon=False)
clean_axis(ax)
save(fig, "figureA1_epa_by_season")


# ---------------------------------------------------------------------------
# Supplementary Figure A2: within-event EPA standard deviation by season
# ---------------------------------------------------------------------------
event_sd = sample.groupby(["year", "event"])["epa_total"].std().reset_index()
years = sorted(event_sd["year"].unique())
fig, ax = plt.subplots(figsize=(9.4, 5.3))
parts = ax.boxplot([event_sd.loc[event_sd["year"] == y, "epa_total"]
                    for y in years],
                   tick_labels=[str(y) for y in years], showfliers=False,
                   patch_artist=True, widths=0.6)
for patch in parts["boxes"]:
    patch.set_facecolor(LIGHT_BLUE)
    patch.set_edgecolor(BLUE)
for median in parts["medians"]:
    median.set_color(ORANGE)
    median.set_linewidth(1.6)
for line in parts["whiskers"] + parts["caps"]:
    line.set_color(BLUE)
ax.axhline(1, color=GRAY, linestyle="--", linewidth=1.0,
           label="EPA SD = 1 point")
ax.set_yscale("log")
ax.tick_params(axis="x", rotation=35)
ax.set_xlabel("Season")
ax.set_ylabel("Within-event EPA standard deviation (points, log scale)")
ax.legend(loc="upper left", frameon=False)
clean_axis(ax)
save(fig, "figureA2_event_epa_sd")
