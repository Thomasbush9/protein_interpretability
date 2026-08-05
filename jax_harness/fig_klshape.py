"""Figure: is the divergence relocation or broadening, and which does the probe use?

Panel A is descriptive -- how the divergence splits per assay. Panels B and C
carry the claim: the probe re-fit on each half, at matched dimensionality, with
assay dots behind the bars and assay-level bootstrap intervals.

Palette is the project's validated categorical set (worst adjacent CVD dE 9.5,
above the 8 floor); blue is the published KL features, orange relocation, green
broadening, so the same colour means the same thing across every panel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
C_KL, C_SHIFT, C_SPREAD, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8885"

ap = argparse.ArgumentParser()
ap.add_argument("--klshape", required=True)
ap.add_argument("--probe", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

ks = json.load(open(a.klshape))
pr = json.load(open(a.probe))

fig = plt.figure(figsize=(14.2, 8.6))
gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.32)

# --- A: how the divergence splits, per assay -------------------------------
axA = fig.add_subplot(gs[0, :2])
assays = sorted(ks["per_assay"], key=lambda k: -ks["per_assay"][k]["shift_share"])
sh = np.array([ks["per_assay"][k]["shift_share"] for k in assays])
x = np.arange(len(assays))
axA.bar(x, sh, 0.68, color=C_SHIFT, alpha=0.9, zorder=3, label="relocation (shift)")
axA.bar(x, 1 - sh, 0.68, bottom=sh, color=C_SPREAD, alpha=0.9, zorder=3,
        label="broadening (spread)")
axA.axhline(0.5, color=INK, lw=1.2, ls="--", zorder=5)
axA.set_xticks(x); axA.set_xticklabels(assays, fontsize=7.5, rotation=30, ha="right")
axA.set_ylabel("share of the symmetric KL")
axA.set_ylim(0, 1)
axA.legend(frameon=False, fontsize=8, ncol=2, loc="lower center",
           bbox_to_anchor=(0.5, -0.42))
p = ks["pooled"]
axA.set_title("A  The divergence is not mostly one thing: relocation "
              f"{100*p['shift_share']['mean']:.0f}% vs broadening "
              f"{100*p['spread_share']['mean']:.0f}%   (final layer)",
              color=INK, fontsize=10, loc="left", pad=8)
axA.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)

# --- B: the probe re-fit on each half --------------------------------------
axB = fig.add_subplot(gs[0, 2])
ORDER = ["internal", "kl_only", "shift", "spread", "dmu", "dsd"]
LAB = {"internal": "internal\n(256)", "kl_only": "KL\n(128)", "shift": "shift\n(128)",
       "spread": "spread\n(128)", "dmu": "|d mu|\n(128)", "dsd": "d sigma\n(128)"}
COL = {"internal": C_REF, "kl_only": C_KL, "shift": C_SHIFT, "spread": C_SPREAD,
       "dmu": C_SHIFT, "dsd": C_SPREAD}
rng = np.random.default_rng(0)
for i, k in enumerate(ORDER):
    b = pr["blocks"][k]
    axB.bar(i, b["mean"], 0.66, color=COL[k],
            alpha=0.9 if k in ("kl_only", "shift", "spread") else 0.55, zorder=3)
    if np.isfinite(b["ci_lo"]):
        axB.plot([i, i], [b["ci_lo"], b["ci_hi"]], color=INK, lw=2.0, zorder=5,
                 solid_capstyle="butt")
    v = list(b["per_assay"].values())
    axB.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=11, zorder=6,
                color=INK, alpha=0.5, edgecolor=SURF, linewidth=0.5)
axB.set_xticks(range(len(ORDER)))
axB.set_xticklabels([LAB[k] for k in ORDER], fontsize=7)
axB.set_ylabel("Spearman on held-out positions")
axB.set_title("B  Re-fit on each half\n     (dots = assays)", color=INK,
              fontsize=9.5, loc="left", pad=8)
axB.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)

# --- C: the paired gaps ----------------------------------------------------
axC = fig.add_subplot(gs[1, :2])
G = [("KL minus shift", "kl_only - shift", C_SHIFT),
     ("KL minus spread", "kl_only - spread", C_SPREAD),
     ("shift minus spread", "shift - spread", C_KL),
     ("internal minus shift+spread", "internal - shift+spread", C_REF)]
y = np.arange(len(G))
for i, (lab, key, col) in enumerate(G):
    g = pr["gaps"][key]
    axC.plot([g["ci_lo"], g["ci_hi"]], [i, i], color=col, lw=2.6, zorder=3,
             solid_capstyle="round")
    axC.scatter([g["gap"]], [i], s=52, color=col, zorder=4, edgecolor=SURF, linewidth=2)
    axC.annotate(f"{g['gap']:+.3f}  ({g['wins']}/{g['splits']})",
                 xy=(g["ci_hi"], i), xytext=(8, 0), textcoords="offset points",
                 va="center", fontsize=8, color=INK2)
axC.axvline(0, color=INK, lw=1.2, zorder=2)
axC.set_yticks(y); axC.set_yticklabels([g[0] for g in G], fontsize=8.5)
axC.set_xlabel("paired difference in Spearman (positive = first block wins)")
axC.set_title("C  A gap whose interval crosses zero means the two halves are "
              "interchangeable for the probe", color=INK, fontsize=10,
              loc="left", pad=8)
axC.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axC.set_axisbelow(True)
axC.margins(x=0.30)

# --- D: does the mutant get less certain? ----------------------------------
axD = fig.add_subplot(gs[1, 2])
ds = np.array([ks["per_assay"][k]["d_sigma_mean"] for k in assays])
o = np.argsort(ds)
axD.barh(np.arange(len(assays)), ds[o], 0.7, color=C_SPREAD, alpha=0.9, zorder=3)
axD.axvline(0, color=INK, lw=1.2, zorder=4)
axD.set_yticks(np.arange(len(assays)))
axD.set_yticklabels([assays[i] for i in o], fontsize=7)
axD.set_xlabel("mean sigma(mutant) - sigma(WT), angstrom")
m = ks["pooled"]["d_sigma_mean"]
axD.set_title(f"D  Mutants are broader, but barely\n     pooled {m['mean']:+.3f} A "
              f"[{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}]  (bin = 0.32 A)",
              color=INK, fontsize=9.5, loc="left", pad=8)
axD.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axD.set_axisbelow(True)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
