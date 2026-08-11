"""Figure: a frozen direction applied to sixteen proteins it never saw.

  A  every held-out assay, |rho| of frozen PC2 against DMS, with the frozen
     chemistry baseline on the same row. Sorted, so the four non-stability
     assays sort themselves rather than being placed apart by the figure.
  B  the blocks, summarised over assays, for the two phenotype groups.
  C  the confound. |PC2 rho| falls with chain length across the sixteen, and the
     non-stability assays are the long ones -- so the phenotype contrast and a
     length contrast are partly the same contrast. ENVZ is the test: at 60
     residues it sits inside the stability panel's length range and still falls
     0.47 below its neighbours, and the longest assay of all outscores it.

Colour encodes the phenotype group and nothing else -- the blocks in B are
already separated by position, so painting them too would be decoration that has
to pass a contrast check for no gain. The two group colours were checked for
CVD separation before use (worst-case OKLab dE 24.7 across deuter/protan/tritan,
against a floor of 8); an earlier draft used the house green against the house
grey for the baselines, which fails both the normal-vision floor and deuteranopia.
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
C_STAB, C_NON, C_BASE = "#2a78d6", "#eb6834", "#8a8885"

ap = argparse.ArgumentParser()
ap.add_argument("--heldout", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

H = json.load(open(a.heldout))
rows = H["per_assay"]
names = sorted(rows, key=lambda n: -abs(rows[n]["pc2_transductive"]))
col = {n: (C_NON if rows[n]["group"] == "non-stability" else C_STAB) for n in names}
short = {n: n.split("_")[0] for n in names}

fig = plt.figure(figsize=(15.0, 5.4))
gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.15, 1.15, 0.95])

# --- A: every held-out assay ----------------------------------------------
axA = fig.add_subplot(gs[0, 0])
y = np.arange(len(names))[::-1]
for i, n in enumerate(names):
    r = rows[n]
    axA.plot([abs(r["chem_frozen"]), abs(r["pc2_transductive"])], [y[i], y[i]],
             color=GRID, lw=1.6, zorder=2, solid_capstyle="round")
    axA.scatter([abs(r["chem_frozen"])], [y[i]], s=30, facecolor=SURF,
                edgecolor=C_BASE, linewidth=1.5, zorder=4)
    axA.scatter([abs(r["pc2_transductive"])], [y[i]], s=54, color=col[n],
                zorder=5, edgecolor=SURF, linewidth=1.4)
wb = abs(H["pc2_within_basis"]["mean"])
axA.axvline(wb, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=3)
axA.annotate(f"PC2 inside the basis  {wb:.2f}", xy=(wb, y[0] + 0.7),
             xytext=(-5, 0), textcoords="offset points", fontsize=7.2,
             color=INK2, ha="right")
axA.set_yticks(y)
axA.set_yticklabels([short[n] for n in names], fontsize=7.6)
axA.set_xlabel("|Spearman| vs DMS   (nothing fitted on these proteins)")
axA.set_title("A  Sixteen proteins outside the basis", color=INK, fontsize=10,
              loc="left", pad=8)
axA.set_xlim(0, 0.95)
axA.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
axA.set_axisbelow(True)
h = [plt.Line2D([], [], ls="", marker="o", ms=7, mfc=C_STAB, mec=SURF,
                label="stability (12)"),
     plt.Line2D([], [], ls="", marker="o", ms=7, mfc=C_NON, mec=SURF,
                label="other phenotype (4)"),
     plt.Line2D([], [], ls="", marker="o", ms=6, mfc=SURF, mec=C_BASE, mew=1.5,
                label="frozen chemistry (17)")]
axA.legend(handles=h, frameon=False, fontsize=7.4, loc="lower right")

# --- B: blocks, by group ---------------------------------------------------
axB = fig.add_subplot(gs[0, 1])
SA = H["summary_abs"]
order = ["pc2_transductive", "pc2_inductive", "dz_frozen", "dz_within",
         "chem_frozen", "random_absmax", "random_absmean", "pc1_transductive"]
order = [k for k in order if k in SA]
yb = np.arange(len(order))[::-1]
for i, k in enumerate(order):
    for g, c, dy in (("stability (12)", C_STAB, 0.17),
                     ("non-stability (4)", C_NON, -0.17)):
        s = SA[k].get(g)
        if not s:
            continue
        axB.plot([s["ci_lo"], s["ci_hi"]], [yb[i] + dy] * 2, color=c, lw=2.6,
                 zorder=4, solid_capstyle="round")
        axB.scatter([s["mean"]], [yb[i] + dy], s=46, color=c, zorder=5,
                    edgecolor=SURF, linewidth=1.4)
        # Direct-label the top block only, so B carries its own identity
        # without a second legend box repeating panel A's.
        if i == 0:
            axB.annotate(g.split(" (")[0], xy=(s["ci_hi"], yb[i] + dy),
                         xytext=(7, 0), textcoords="offset points",
                         fontsize=7.2, color=c, va="center")
axB.set_yticks(yb)
axB.set_yticklabels([SA[k]["label"].replace(" frozen", "") for k in order],
                    fontsize=7.6)
axB.set_xlabel("|Spearman| vs DMS, pooled over assays")
axB.set_title("B  Every block is frozen on the twelve", color=INK, fontsize=10,
              loc="left", pad=8)
axB.set_xlim(0, 0.95)
axB.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
axB.set_axisbelow(True)

# --- C: the length confound ------------------------------------------------
axC = fig.add_subplot(gs[0, 2])
for n in names:
    axC.scatter([rows[n]["n_res"]], [abs(rows[n]["pc2_transductive"])],
                s=52, color=col[n], zorder=5, edgecolor=SURF, linewidth=1.4)
lm = H.get("length_matched", {}).get("ENVZ_ECOLI_Ghose_2023")
if lm and lm.get("neighbours"):
    axC.annotate(f"ENVZ, {lm['n_res']} aa\n{lm['gap']:+.2f} vs {len(lm['neighbours'])}"
                 f" stability\nassays of the same size",
                 xy=(lm["n_res"], lm["self"]), xytext=(14, -4),
                 textcoords="offset points", fontsize=7.2, color=C_NON,
                 va="top",
                 arrowprops=dict(arrowstyle="-", color=C_NON, lw=0.9))
    axC.plot([lm["n_res"] - 9, lm["n_res"] + 9], [lm["neighbour_mean"]] * 2,
             color=C_STAB, lw=1.8, ls=(0, (3, 2)), zorder=4)
axC.set_xlabel("chain length (residues)")
axC.set_ylabel("|Spearman| vs DMS")
axC.set_title("C  Length does not explain it", color=INK, fontsize=10,
              loc="left", pad=8)
axC.set_ylim(0, 0.95)
axC.grid(True, color=GRID, lw=0.8, zorder=0)
axC.set_axisbelow(True)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
