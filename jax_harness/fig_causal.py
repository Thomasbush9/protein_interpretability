"""Figure: the model uses the direction, it does not merely contain it.

Everything else in this study is correlational. This is the intervention: add
alpha * PC2 to the final pair representation, hand the modified trunk state to
the structure module, and read what changes.

Effect size cannot answer the question, and the figure is built so that it never
looks like it does. Any vector of the same norm moves the output about as much.
What separates a direction the model USES from one that merely disturbs it is
SIGN STRUCTURE: PC2 is the broadening axis, so +alpha should broaden and -alpha
should sharpen, and the response should be ODD in alpha. A direction with no
privileged orientation has an even response. So the statistic plotted is

    odd(a) = [f(+a) - f(-a)] / 2a

and never the raw magnitude.

  A  per protein, PC2 against the eight random directions drawn in that same
     protein. The comparison is always within a protein, so differences in
     chain length or representation scale cannot leak into it.
  B  the two summaries that make it a test rather than an ordering: how often
     PC2 ranks first out of nine, against the 1/9 expected by chance, and the
     same for PC1. PC1 is a real component -- substitution volume -- but not the
     stability axis, so it is the control that decides whether this is about PC2
     or about components in general.

Colour: PC2 is the accent, its own controls are grey, and PC1 is the second slot
so it never reads as a variant of PC2.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
_have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_sans = next((f for f in ("Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial")
              if f in _have), None)
plt.rcParams.update({
    **({"font.family": "sans-serif", "font.sans-serif": [_sans]} if _sans else {}),
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "font.size": 9, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
})
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
C_REF = "#8a8885"
HALO = dict(boxstyle="round,pad=0.18", facecolor=SURF, edgecolor="none",
            alpha=0.92)

ap = argparse.ArgumentParser()
ap.add_argument("--steer", required=True)
ap.add_argument("--metric", default="d_sd_site")
ap.add_argument("--out", required=True)
a = ap.parse_args()

S = json.load(open(a.steer))
M = S["metrics"][a.metric]
rows = M["per_assay"]


def tidy(ax, title, sub=None):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=22 if sub else 6)
    if sub:
        ax.annotate(sub, (0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", fontsize=8.4, color=INK2,
                    va="bottom", ha="left")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(13.6, 5.0))
gs = fig.add_gridspec(1, 2, wspace=0.26, top=0.80, bottom=0.15,
                      width_ratios=[1.45, 1.0])

# ---- A: PC2 against its own controls, per protein ------------------------
ax = fig.add_subplot(gs[0, 0])
order = np.argsort([abs(r["pc2"]) for r in rows])
ys = np.arange(len(rows))
for i, oi in enumerate(order):
    r = rows[oi]
    # The archive keeps only the best control per assay, which is the one the
    # ranking turns on; drawing it as the bar end states the comparison exactly.
    ax.plot([0, r["random_max"]], [ys[i], ys[i]], color=C_REF, lw=5.0,
            solid_capstyle="butt", alpha=0.45, zorder=2)
    ax.scatter([abs(r["pc2"])], [ys[i]], s=58, color=SLOT[0], zorder=5,
               edgecolor=SURF, linewidth=1.3)
ax.set_yticks(ys, [rows[oi]["assay"] for oi in order], fontsize=8.4)
ax.set_xlabel("|odd component| per unit alpha  —  sign-structured response")
ax.set_xlim(0, None)
top = len(rows) - 1
ax.annotate("PC2", (abs(rows[order[-1]]["pc2"]), top), xytext=(0, 13),
            textcoords="offset points", ha="center", fontsize=8.8,
            color=SLOT[0], bbox=HALO, zorder=6)
ax.annotate("best of 8 random", (rows[order[0]]["random_max"], 0),
            xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
            fontsize=8.6, color=INK2, bbox=HALO, zorder=6)
ax.set_ylim(-0.9, len(rows) + 0.3)
tidy(ax, "A  PC2 against its own controls, protein by protein",
     f"{M['label']}; controls redrawn inside each protein")

# ---- B: the test --------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
n = M["n_assays"]
chance = M["p_first_each"] * n
bars = [("PC2", M["pc2_first"], SLOT[0]), ("PC1 (control)", M["pc1_beats"], SLOT[1])]
xs = np.arange(len(bars))
for x, (lab, v, c) in zip(xs, bars):
    ax.bar(x, v, width=0.5, color=c, zorder=3)
    # Counts sit INSIDE tall bars and beside short ones, so neither lands on
    # the chance line the eye is meant to compare them against.
    if v > chance * 2:
        ax.annotate(f"{v}/{n}", (x, v), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=10.0,
                    color=SURF, fontweight="semibold", zorder=7)
    else:
        ax.annotate(f"{v}/{n}", (x + 0.28, v), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=10.0, color=INK, bbox=HALO, zorder=7)
ax.axhline(chance, color=C_REF, lw=1.5, ls=(0, (4, 3)), zorder=4)
ax.annotate(f"chance {chance:.1f}", (0.99, chance), xycoords=("axes fraction", "data"),
            xytext=(0, 8), textcoords="offset points", ha="right", fontsize=8.4,
            color=C_REF, bbox=HALO, zorder=6)
ax.set_xticks(xs, [b[0] for b in bars], fontsize=9.2)
ax.set_ylim(0, max(b[1] for b in bars) + 2.2)
ax.set_ylabel(f"proteins where it ranks first of {int(1/M['p_first_each'])}")
# A permutation p of exactly 0 means "no draw beat the observation", not zero
# probability, so it is reported as a bound rather than as 0.
pr = M["p_rank"]
pr_txt = "< 5e-06" if pr == 0 else f"= {pr:.1e}"
ax.annotate(f"exact binomial (rank-first, 1/9)  p = {M['p_sign']:.1e}\n"
            f"mean rank {M['mean_norm_rank']:.2f} of 1.0  (chance 0.50)\n"
            f"permutation  p {pr_txt}",
            (0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
            fontsize=8.8, color=INK, bbox=HALO, zorder=6)
tidy(ax, "B  Not an ordering — a test",
     "PC1 is a real component but not the stability axis")

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
