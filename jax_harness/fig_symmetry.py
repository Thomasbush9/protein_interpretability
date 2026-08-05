"""Figure: the internal/output comparison with the dimensional asymmetry removed.

The published comparison gave the internal side 256 features and the output side
ten. The SVD study widened that gap by showing internal reaches +0.73 on its raw
channels, which made the asymmetry a genuine threat to the headline claim. These
panels run one protocol over both sides and show what happens.

  A  held-out Spearman against the number of directions kept, every block on the
     same axes. Internal is a single fixed layer; the output blocks run up to
     1741 dimensions.
  B  the paired per-assay difference at a pre-specified k, with the win count.
  C  the answer to the obvious objection. "Output does badly at 1741 dimensions"
     could just be the curse of dimensionality with n = 250. It is not: the
     output side was described at five sizes spanning ten to 1741 dimensions and
     its ceiling is flat across all of them, while internal at 128 sits far
     above. Plotting ceiling against dimensionality is what distinguishes an
     estimator problem from an information problem, so it gets its own panel.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

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
C_INT, C_ALL, C_RICH, C_DISP, C_GEO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8885"

INTERNAL = "internal dz (last layer)"
STYLE = {
    INTERNAL:                       (C_INT,  2.6, "-",  "internal Δz (1 layer, 128)"),
    "output all (max generosity)":  (C_ALL,  2.0, "-",  "output: everything (1741)"),
    "output rich (published)":      (C_RICH, 2.0, "-",  "output: rich (10, published)"),
    "output displacement":          (C_DISP, 1.6, "-",  "output: displacement (~68)"),
    "output coordinates":           (C_GEO,  1.4, "--", "output: coordinates (~207)"),
    "output pair distances":        (C_GEO,  1.4, ":",  "output: pair distances (~1479)"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--symmetry", required=True)
ap.add_argument("--k-report", type=int, default=32)
ap.add_argument("--out", required=True)
a = ap.parse_args()

S = json.load(open(a.symmetry))
CV = S["components_view"]
ks = sorted(int(k) for k in CV[INTERNAL])

fig = plt.figure(figsize=(14.6, 4.8))
gs = fig.add_gridspec(1, 3, wspace=0.30, width_ratios=[1.25, 1.0, 1.0])

# --- A: the curves ---------------------------------------------------------
axA = fig.add_subplot(gs[0, 0])
for bn, (col, lw, ls, lab) in STYLE.items():
    if bn not in CV:
        continue
    m = np.array([CV[bn][str(k)]["mean"] for k in ks])
    lo = np.array([CV[bn][str(k)]["ci_lo"] for k in ks])
    hi = np.array([CV[bn][str(k)]["ci_hi"] for k in ks])
    if bn == INTERNAL:
        axA.fill_between(ks, lo, hi, color=col, alpha=0.13, lw=0, zorder=2)
    axA.plot(ks, m, color=col, lw=lw, ls=ls, zorder=4, label=lab,
             solid_capstyle="round")
axA.set_xscale("log", base=2)
axA.set_xticks(ks); axA.set_xticklabels([str(k) for k in ks])
axA.set_xlabel("directions kept (k)")
axA.set_ylabel("held-out Spearman vs DMS")
axA.set_title("A  One protocol, both sides", color=INK, fontsize=10,
              loc="left", pad=8)
axA.legend(frameon=False, fontsize=7.4, ncol=2, loc="lower center",
           bbox_to_anchor=(0.5, -0.52))
axA.grid(True, color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)

# --- B: paired differences -------------------------------------------------
axB = fig.add_subplot(gs[0, 1])
gaps = S["paired_gaps"]
items = [(bn, gaps[f"{bn} @ k={a.k_report}"]) for bn in STYLE
         if bn != INTERNAL and f"{bn} @ k={a.k_report}" in gaps]
items.sort(key=lambda kv: kv[1]["gap"])
y = np.arange(len(items))
for i, (bn, g) in enumerate(items):
    col = STYLE[bn][0]
    axB.plot([g["ci_lo"], g["ci_hi"]], [i, i], color=col, lw=2.8, zorder=4,
             solid_capstyle="round")
    axB.scatter([g["gap"]], [i], s=52, color=col, zorder=5, edgecolor=SURF,
                linewidth=1.8)
    axB.annotate(f"{g['gap']:+.3f}   {g['wins']}/{g['n']}", xy=(g["ci_hi"], i),
                 xytext=(8, 0), textcoords="offset points", va="center",
                 fontsize=7.8, color=INK2)
axB.axvline(0, color=INK, lw=1.2, zorder=3)
axB.set_yticks(y)
axB.set_yticklabels([STYLE[bn][3].replace("output: ", "") for bn, _ in items],
                    fontsize=7.8)
axB.set_xlabel("internal − output (paired, per assay)")
axB.set_title(f"B  Every assay, every block\n     at k = {a.k_report}",
              color=INK, fontsize=9.5, loc="left", pad=8)
axB.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)
axB.margins(x=0.32)

# --- C: is it an estimator problem or an information problem? --------------
axC = fig.add_subplot(gs[0, 2])
dims = S["protocol"]["dims"]
first = next(iter(dims))
for bn, (col, lw, ls, lab) in STYLE.items():
    if bn not in CV:
        continue
    d = float(np.mean([dims[n][bn] for n in dims]))
    best = max(CV[bn][str(k)]["mean"] for k in ks)
    axC.scatter([d], [best], s=110 if bn == INTERNAL else 62, color=col,
                zorder=5, edgecolor=SURF, linewidth=1.8)
    # The two largest output blocks sit close in both dimensions and in score,
    # so their labels are placed explicitly rather than by a shared rule.
    dx, dy, ha = {"output pair distances": (-6, -16, "right"),
                  "output all (max generosity)": (6, 11, "left"),
                  "output coordinates": (0, -16, "center"),
                  "output displacement": (0, -16, "center"),
                  }.get(bn, (0, 11, "center"))
    axC.annotate(lab.split(" (")[0].replace("output: ", ""), xy=(d, best),
                 xytext=(dx, dy), textcoords="offset points", fontsize=7.2,
                 color=col, ha=ha)
out_best = [max(CV[bn][str(k)]["mean"] for k in ks) for bn in CV if bn != INTERNAL]
axC.axhspan(min(out_best), max(out_best), color=C_GEO, alpha=0.13, lw=0, zorder=1)
axC.annotate("every output description,\n10 to 1741 dimensions",
             xy=(1741, max(out_best)), xytext=(-4, 12), textcoords="offset points",
             fontsize=7.2, color=C_GEO, ha="right")
axC.set_xscale("log")
axC.set_xlabel("dimensions given to the block")
axC.set_ylabel("best held-out Spearman over k")
axC.set_ylim(0, max(CV[INTERNAL][str(k)]["mean"] for k in ks) * 1.25)
axC.set_title("C  Output saturates regardless\n     of how richly it is described",
              color=INK, fontsize=9.5, loc="left", pad=8)
axC.grid(True, color=GRID, lw=0.8, zorder=0); axC.set_axisbelow(True)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
