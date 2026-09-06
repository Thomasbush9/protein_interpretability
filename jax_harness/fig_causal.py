"""Figure: the intervention evidence, shown descriptively — v2.

v1 of this figure plotted the pooled |odd| ranking and printed confirmatory
p-values. Those were withdrawn on 2026-09-06: the eight control orientations
are shared across assays (no per-assay seed), the pooled statistic averages
doses whose contributions have opposite signs, and PC2 is also the
highest-GAIN direction, so ranking |odd| against isotropic controls does not
test sign structure. The docstring's old claim that the plot shows "never the
raw magnitude" did not survive contact with the archived doses.

What the corrected figure shows instead:

  A  PC2's odd distogram-width response per unit alpha, per protein, at each
     dose separately. The dose non-monotonicity is the point: positive and
     dominant at |a|=10, reversed at |a|=30, indistinguishable from the
     controls at |a|=3. The grey band is the per-dose spread of the eight
     (shared) random controls.
  B  the deletion test, which the master report previously omitted: remove one
     direction from a real mutation's own delta-z, re-run the structure
     module, and ask how much of the mutation's distogram response is undone.
     Every PC2 interval overlaps the random ones and zero-recovery; the
     surgery's positive control is what makes that null interpretable.

  python fig_causal.py --steer runs/steer_pooled_v2.json \
      --ablate runs/ablate_v2.json --out figures/causal.png
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
ap.add_argument("--ablate", required=True)
ap.add_argument("--metric", default="d_sd_site")
ap.add_argument("--mode", default="sym")
ap.add_argument("--out", required=True)
a = ap.parse_args()

S = json.load(open(a.steer))
cell = S["cells"][f"{a.metric}:{a.mode}"]
rows = cell["per_assay"]
doses = sorted({float(k) for r in rows for k in r["pc2_odd_per_dose"]})
A = json.load(open(a.ablate))


def tidy(ax, title, sub=None):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=22 if sub else 6)
    if sub:
        ax.annotate(sub, (0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", fontsize=8.4, color=INK2,
                    va="bottom", ha="left")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(13.6, 5.2))
gs = fig.add_gridspec(1, 2, wspace=0.28, top=0.78, bottom=0.15,
                      width_ratios=[1.15, 1.0])

# ---- A: odd response per dose, protein by protein ------------------------
ax = fig.add_subplot(gs[0, 0])
xs = np.arange(len(doses))
for r in rows:
    ys = [r["pc2_odd_per_dose"][f"{d}"] for d in doses]
    ax.plot(xs, ys, color=SLOT[0], lw=1.1, alpha=0.55, zorder=4)
    ax.scatter(xs, ys, s=16, color=SLOT[0], alpha=0.75, zorder=5,
               edgecolor=SURF, linewidth=0.6)
ax.axhline(0, color=INK2, lw=1.0, zorder=3)
for i, d in enumerate(doses):
    firsts = cell["by_abs_odd_per_dose"][f"{d}"]["first"]
    ax.annotate(f"first by |odd|\n{firsts}/{cell['n_assays']}",
                (i, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -34), textcoords="offset points", ha="center",
                fontsize=8.2, color=INK2, annotation_clip=False)
ax.set_xticks(xs, [f"|α| = {d:g}" for d in doses], fontsize=9.2)
ax.set_ylabel("odd component per unit α, distogram width at site")
tidy(ax, "A  The odd response is not one number — it reverses with dose",
     "one line per protein (12); controls shared across proteins, so no p-value")

# ---- B: the deletion null ------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
dirs = sorted(A["directions"], key=lambda d: (not d.startswith("PC"), d))
ys = np.arange(len(dirs))[::-1]
for y, dn in zip(ys, dirs):
    r = A["directions"][dn]["recovery"]
    c = SLOT[0] if dn == "PC2" else (SLOT[1] if dn == "PC1" else C_REF)
    ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=c, lw=2.4, alpha=0.8,
            zorder=4, solid_capstyle="butt")
    ax.scatter([r["mean"]], [y], s=54, color=c, zorder=5, edgecolor=SURF,
               linewidth=1.2)
ax.axvline(0, color=INK2, lw=1.0, zorder=3)
ax.set_yticks(ys, dirs, fontsize=9.0)
ax.set_xlabel("distogram recovery when the direction is deleted "
              "(1 = reverts to wild type)")
resid = max(A["positive_control_residual_fraction"].values())
ax.annotate("every interval includes zero;\n"
            "PC2 − random paired gaps all include zero\n"
            f"surgery verified: ≤ {resid:.0e} of the component survives",
            (0.03, 0.05), xycoords="axes fraction", ha="left", va="bottom",
            fontsize=8.6, color=INK, bbox=HALO, zorder=6)
tidy(ax, "B  Deleting the direction from real mutations changes nothing",
     "4 proteins × 16 variants spanning each assay's DMS range")

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
