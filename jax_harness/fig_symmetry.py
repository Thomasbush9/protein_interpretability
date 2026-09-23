"""Figure: the internal/output comparison with the dimensional asymmetry removed.

The published comparison gave the internal side 256 features and the output side
ten. The SVD study widened that gap by showing internal reaches +0.73 on its raw
channels, which made the asymmetry a genuine threat to the headline claim. These
panels run one protocol over both sides and show what happens.

  A  held-out Spearman against the number of directions kept, every block on the
     same axes. Internal is a single fixed layer; the output blocks run to
     whatever the widest archived block is.
  B  the paired per-assay difference at a pre-specified k, with the win count.
  C  the answer to the obvious objection. "Output does badly at ~1800
     dimensions" could just be the curse of dimensionality with n = 250. It is
     not: the output side is described at several sizes spanning ten to the
     widest block and its ceiling is flat across all of them, while internal at
     128 sits far above. Plotting ceiling against dimensionality is what
     distinguishes an estimator problem from an information problem, so it gets
     its own panel.

Every dimension count in this figure is READ FROM the archive. An earlier
version hardcoded 1741 in the legend and in panel C's annotation, and carried no
STYLE entry for the two per-residue pLDDT blocks, so pointing it at a newer run
silently dropped those two series and kept printing the old width. Blocks
present in the data but absent from STYLE now raise instead of vanishing.
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
C_PLD = "#9d5bd2"

INTERNAL = "internal dz (last layer)"
# (colour, linewidth, linestyle, label template). "{d}" is filled from the
# archived per-assay dimension counts -- never typed in.
STYLE = {
    INTERNAL:                       (C_INT,  2.6, "-",  "internal Δz (1 layer, {d})"),
    "output all (max generosity)":  (C_ALL,  2.0, "-",  "output: everything ({d})"),
    "output pLDDT + rich":          (C_PLD,  2.2, "-",  "output: pLDDT + rich (~{d})"),
    "output pLDDT per residue":     (C_PLD,  1.6, "--", "output: pLDDT per residue (~{d})"),
    "output rich (published)":      (C_RICH, 2.0, "-",  "output: rich ({d}, published)"),
    "output displacement":          (C_DISP, 1.6, "-",  "output: displacement (~{d})"),
    "output coordinates":           (C_GEO,  1.4, "--", "output: coordinates (~{d})"),
    "output pair distances":        (C_GEO,  1.4, ":",  "output: pair distances (~{d})"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--symmetry", required=True)
ap.add_argument("--k-report", type=int, default=32)
ap.add_argument("--out", required=True)
a = ap.parse_args()

S = json.load(open(a.symmetry))
CV = S["components_view"]
ks = sorted(int(k) for k in CV[INTERNAL])

# A block the archive scores but STYLE does not know about would previously be
# dropped without a word -- which is how a figure built from a run containing
# per-residue pLDDT went out showing the five older blocks. Fail instead.
unstyled = [bn for bn in CV if bn not in STYLE]
if unstyled:
    raise SystemExit(
        f"{a.symmetry} scores blocks with no STYLE entry: {unstyled}\n"
        "Add them to STYLE -- silently dropping them is how the stale figure "
        "in report_svd happened."
    )

dims = S["protocol"]["dims"]
DIM = {bn: int(round(float(np.mean([dims[n][bn] for n in dims if bn in dims[n]]))))
       for bn in CV}
LABEL = {bn: STYLE[bn][3].format(d=DIM[bn]) for bn in CV}
OUT_DIMS = [DIM[bn] for bn in CV if bn != INTERNAL]

fig = plt.figure(figsize=(14.6, 4.8))
gs = fig.add_gridspec(1, 3, wspace=0.30, width_ratios=[1.25, 1.0, 1.0])

# --- A: the curves ---------------------------------------------------------
axA = fig.add_subplot(gs[0, 0])
for bn, (col, lw, ls, _tmpl) in STYLE.items():
    if bn not in CV:
        continue
    m = np.array([CV[bn][str(k)]["mean"] for k in ks])
    lo = np.array([CV[bn][str(k)]["ci_lo"] for k in ks])
    hi = np.array([CV[bn][str(k)]["ci_hi"] for k in ks])
    if bn == INTERNAL:
        axA.fill_between(ks, lo, hi, color=col, alpha=0.13, lw=0, zorder=2)
    axA.plot(ks, m, color=col, lw=lw, ls=ls, zorder=4, label=LABEL[bn],
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
axB.set_yticklabels([LABEL[bn].replace("output: ", "") for bn, _ in items],
                    fontsize=7.8)
axB.set_xlabel("internal − output (paired, per assay)")
axB.set_title(f"B  Every assay, every block\n     at k = {a.k_report}",
              color=INK, fontsize=9.5, loc="left", pad=8)
axB.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)
axB.margins(x=0.32)

# --- C: is it an estimator problem or an information problem? --------------
axC = fig.add_subplot(gs[0, 2])
for bn, (col, lw, ls, _tmpl) in STYLE.items():
    if bn not in CV:
        continue
    d = DIM[bn]
    best = max(CV[bn][str(k)]["mean"] for k in ks)
    axC.scatter([d], [best], s=110 if bn == INTERNAL else 62, color=col,
                zorder=5, edgecolor=SURF, linewidth=1.8)
    # Several blocks sit close in both dimensions and in score, so their labels
    # are placed explicitly rather than by a shared rule.
    # Two pairs of blocks land on top of each other here -- displacement and
    # per-residue pLDDT are both ~69 dimensions, and coordinates and pair
    # distances both sit on the floor of the band -- so every label is placed
    # by hand and the collisions are checked by eye after each rebuild.
    dx, dy, ha = {"output pair distances": (9, -14, "left"),
                  "output all (max generosity)": (0, 12, "center"),
                  "output coordinates": (-9, -14, "right"),
                  "output displacement": (0, -17, "center"),
                  "output pLDDT per residue": (11, -3, "left"),
                  "output pLDDT + rich": (0, 12, "center"),
                  }.get(bn, (0, 11, "center"))
    axC.annotate(LABEL[bn].split(" (")[0].replace("output: ", ""), xy=(d, best),
                 xytext=(dx, dy), textcoords="offset points", fontsize=7.2,
                 color=col, ha=ha)
out_best = [max(CV[bn][str(k)]["mean"] for k in ks) for bn in CV if bn != INTERNAL]
axC.axhspan(min(out_best), max(out_best), color=C_GEO, alpha=0.13, lw=0, zorder=1)
axC.annotate(f"every output description,\n{min(OUT_DIMS)} to {max(OUT_DIMS)} dimensions",
             xy=(max(OUT_DIMS), max(out_best)), xytext=(-4, 32),
             textcoords="offset points", fontsize=7.2, color=C_GEO, ha="right")
axC.margins(x=0.20)
axC.set_xscale("log")
axC.set_xlabel("dimensions given to the block")
axC.set_ylabel("best held-out Spearman over k")
axC.set_ylim(0, max(CV[INTERNAL][str(k)]["mean"] for k in ks) * 1.25)
axC.set_title("C  Output saturates regardless\n     of how richly it is described",
              color=INK, fontsize=9.5, loc="left", pad=8)
axC.grid(True, color=GRID, lw=0.8, zorder=0); axC.set_axisbelow(True)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
