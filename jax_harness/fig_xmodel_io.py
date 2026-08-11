"""Figure: does "internal beats output" hold in more than one architecture?

The headline result is twelve proteins in Boltz-2. This asks whether the same
comparison survives in OpenFold3 and Protenix, which differ in depth (48 and 16
Pairformer blocks against Boltz-2's 64), in distogram binning and in alignment
handling.

  A  each model's internal probe against its own emitted quantities. The
     baseline that matters is pLDDT: it is the model's OWN uncertainty head, so
     it is the strongest form of "the model already tells you this". Every bar
     is a within-model comparison on identical rows and identical variants --
     the archives are checked for matching variant IDs before any cross-model
     row is printed.
  B  the gap, with the ASSAY as the independent unit. Four assays is not many,
     so the panel draws the intervals rather than reporting a point estimate
     that would imply more proteins than there are. They all clear zero even so.

TM to wild type is deliberately absent. It needs `tmtools`, which is not in the
analysis container, and substituting a different structural metric under the
same name is how a number nobody computed ends up in a paper. The pLDDT
comparison is the stronger one anyway.

Colour: one slot per model, held across both panels. Grey is reference only.
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
NICE = {"boltz2": "Boltz-2", "of3": "OpenFold3", "protenix": "Protenix"}
BARS = [("internal (128-dim, final layer)", "internal\n(128-dim)"),
        ("pLDDT", "pLDDT\n(chain)"), ("pLDDT@site", "pLDDT\nat site")]

ap = argparse.ArgumentParser()
ap.add_argument("--xio", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
D = json.load(open(a.xio))
MODELS = D["models"]


def tidy(ax, title, sub=None):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=22 if sub else 6)
    if sub:
        ax.annotate(sub, (0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", fontsize=8.4, color=INK2,
                    va="bottom", ha="left")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(13.2, 4.9))
gs = fig.add_gridspec(1, 2, wspace=0.28, top=0.78, bottom=0.17,
                      width_ratios=[1.35, 1.0])

# ---- A ------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
xs = np.arange(len(BARS))
w = 0.26
for i, m in enumerate(MODELS):
    vals = [D["spearman"][m][k] for k, _ in BARS]
    ax.bar(xs + (i - 1) * w, vals, width=w * 0.86, color=SLOT[i], zorder=3,
           label=f"{NICE[m]} ({D['layers'][m]} layers)")
ax.set_xticks(xs, [lab for _, lab in BARS], fontsize=8.8)
ax.set_ylim(0, 0.72)
ax.set_ylabel("Spearman vs measured stability")
ax.legend(frameon=False, fontsize=8.6, loc="upper right")
tidy(ax, "A  Internal beats the model's own outputs, in all three",
     f"{len(D['assays'])} assays x {D['splits']} splits, "
     f"held-out positions, identical variants")

# ---- B ------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ys = np.arange(len(MODELS))[::-1]
for m, y in zip(MODELS, ys):
    g = D["internal_minus"]["pLDDT"][m]
    i = MODELS.index(m)
    ax.plot([g["ci_lo"], g["ci_hi"]], [y, y], color=SLOT[i], lw=2.8,
            solid_capstyle="round", zorder=4)
    ax.scatter([g["gap"]], [y], s=66, color=SLOT[i], zorder=5,
               edgecolor=SURF, linewidth=1.4)
    ax.annotate(f"{g['gap']:+.3f}   {g['wins']}/{g['splits']} splits",
                (g["ci_hi"], y), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=8.4, color=INK, zorder=6)
ax.axvline(0, color=C_REF, lw=1.4, ls=(0, (4, 3)), zorder=2)
ax.set_yticks(ys, [NICE[m] for m in MODELS], fontsize=9.2)
ax.set_xlim(-0.05, 0.62)
ax.set_xlabel("internal minus pLDDT")
tidy(ax, "B  Every interval clears zero",
     f"assay is the unit; only {len(D['assays'])} of them, so intervals stay wide")

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
