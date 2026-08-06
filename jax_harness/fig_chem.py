"""Figure: the mutation subspace is not re-encoded substitution chemistry.

Chemistry is the deciding baseline the audit named, and the SVD results made it
sharper rather than softer: PC1 correlates with volume change at -0.80 and even
PC2 carries -0.53. So the deflationary reading -- the model has learned which
amino acid was substituted, and chemistry predicts stability -- has to be
answered head on.

  A  what each feature block achieves under one estimator
  B  the increments in both directions. "PC2 adds to chemistry" and "chemistry
     adds to PC2" are different questions and only asking the first would be
     advocacy.
  C  PC2 alone, transferred: one number per variant, basis AND sign taken from
     the other eleven proteins, nothing fitted on the held-out one.

Blue is the internal side, orange chemistry, green the combinations.
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
C_INT, C_CHEM, C_MIX, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8885"

ap = argparse.ArgumentParser()
ap.add_argument("--chem", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
S = json.load(open(a.chem))

fig = plt.figure(figsize=(14.4, 4.8))
gs = fig.add_gridspec(1, 3, wspace=0.62, width_ratios=[1.2, 1.0, 0.95])

# --- A: blocks -------------------------------------------------------------
axA = fig.add_subplot(gs[0, 0])
ORDER = [("chemistry (17)", "chemistry (17)", C_CHEM),
         ("dz residualised on chemistry (128)", "Δz minus chemistry (128)", C_MIX),
         ("PC2 alone (1)", "PC2 alone (1)", C_INT),
         ("PC1-4 (4)", "PC1–4 (4)", C_INT),
         ("chemistry + PC2 (18)", "chemistry + PC2 (18)", C_MIX),
         ("full dz (128)", "full Δz (128)", C_INT),
         ("chemistry + full dz (145)", "chemistry + full Δz (145)", C_MIX)]
y = np.arange(len(ORDER))[::-1]
for yy, (k, lab, col) in zip(y, ORDER):
    b = S["blocks"][k]
    axA.plot([b["ci_lo"], b["ci_hi"]], [yy, yy], color=col, lw=2.8, zorder=4,
             solid_capstyle="round")
    axA.scatter([b["mean"]], [yy], s=54, color=col, zorder=5, edgecolor=SURF,
                linewidth=1.7)
    axA.annotate(f"{b['mean']:+.3f}", xy=(b["ci_hi"], yy), xytext=(7, 0),
                 textcoords="offset points", va="center", fontsize=7.8, color=INK2)
axA.axvline(S["blocks"]["chemistry (17)"]["mean"], color=C_CHEM, lw=1.1, ls="--",
            zorder=2)
axA.set_yticks(y); axA.set_yticklabels([l for _, l, _ in ORDER], fontsize=8)
axA.set_xlabel("held-out Spearman vs DMS")
axA.set_title("A  One scalar from the shared basis beats\n     seventeen chemistry "
              "descriptors", color=INK, fontsize=9.5, loc="left", pad=8)
axA.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)
axA.margins(x=0.20)

# --- B: increments ---------------------------------------------------------
axB = fig.add_subplot(gs[0, 1])
# Short labels: the left column of panel B sits next to panel A's value
# annotations and long text collides across the gap.
INC = [("PC2 adds to chemistry", "PC2 over chem.", C_INT),
       ("PC1-4 add to chemistry", "PC1–4 over chem.", C_INT),
       ("full dz adds to chemistry", "full Δz over chem.", C_INT),
       ("chemistry adds to PC2", "chem. over PC2", C_CHEM),
       ("chemistry-residual dz beats chemistry", "Δz−chem. vs chem.", C_MIX)]
y = np.arange(len(INC))[::-1]
for yy, (k, lab, col) in zip(y, INC):
    g = S["increments"][k]
    sig = np.isfinite(g["ci_lo"]) and (g["ci_lo"] > 0 or g["ci_hi"] < 0)
    axB.plot([g["ci_lo"], g["ci_hi"]], [yy, yy], color=col if sig else C_REF,
             lw=2.8, zorder=4, solid_capstyle="round")
    axB.scatter([g["gap"]], [yy], s=54, color=col if sig else C_REF, zorder=5,
                edgecolor=SURF, linewidth=1.7)
    axB.annotate(f"{g['gap']:+.3f}  {g['wins']}/{g['n']}", xy=(g["ci_hi"], yy),
                 xytext=(7, 0), textcoords="offset points", va="center",
                 fontsize=7.6, color=INK2)
axB.axvline(0, color=INK, lw=1.2, zorder=3)
axB.set_yticks(y)
axB.set_yticklabels([lab for _, lab, _ in INC], fontsize=7.8)
axB.set_xlabel("paired difference in Spearman")
axB.set_title("B  Both directions asked; grey = interval\n     includes zero",
              color=INK, fontsize=9.5, loc="left", pad=8)
axB.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)
axB.margins(x=0.34)

# --- C: PC2 alone, transferred --------------------------------------------
axC = fig.add_subplot(gs[0, 2])
lo = S["pc2_alone_loao"]
pa = lo["per_assay"]
names = sorted(pa, key=lambda n: pa[n])
x = np.arange(len(names))
axC.barh(x, [pa[n] for n in names], 0.68, color=C_INT, alpha=0.9, zorder=3)
axC.axvline(lo["mean"], color=INK, lw=1.4, ls="--", zorder=5)
axC.annotate(f"pooled {lo['mean']:+.3f}", xy=(lo["mean"], len(names) - 0.4),
             xytext=(4, 0), textcoords="offset points", fontsize=7.8, color=INK)
axC.set_yticks(x); axC.set_yticklabels(names, fontsize=7.2)
axC.set_xlabel("Spearman on the held-out assay")
axC.set_title("C  PC2 alone, basis and sign from\n     the other eleven proteins",
              color=INK, fontsize=9.5, loc="left", pad=8)
axC.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axC.set_axisbelow(True)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
