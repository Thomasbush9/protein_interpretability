"""Figure: PC2 is an amplitude, not a location.

The panels are ordered as the analysis actually went, because the middle one is
a trap and the report is more useful if it shows the trap rather than only the
conclusion.

  A  the width change concentrates near the mutated residue, for every variant,
     far above the noise floor the gym2/gym2s replicate provides.
  B  split variants by raw PC2 and the concentration looks like it depends on
     PC2 -- a clean monotone gradient with distance. It does not. Magnitude
     produces the same gradient on its own, and PC2 tracks magnitude at +0.54.
     Adjusting for it removes the effect.
  C  what each component does and does not carry, as pooled intervals: PC2 owns
     the amplitude and none of the components own the localisation.

Blue is PC2 throughout, grey the magnitude confound, green the adjusted
version. Panel B deliberately plots all three on one axis, since the whole
point is that two of them coincide and the third does not.
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
C_PC2, C_MAG, C_ADJ, C_REF, C_AMB = "#2a78d6", "#8a8885", "#1baf7a", "#8a8885", "#eb6834"

ap = argparse.ArgumentParser()
ap.add_argument("--pc2", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

S = json.load(open(a.pc2))
pf = S["profiles"]
B = np.array(pf["bins"])
mid = 0.5 * (B[:-1] + np.minimum(B[1:], B[-2] + 6))
lab = [f"{B[i]:.0f}–{B[i+1]:.0f}" for i in range(len(B) - 1)]
lab[-1] = f">{B[-2]:.0f}"

fig = plt.figure(figsize=(14.2, 4.6))
gs = fig.add_gridspec(1, 3, wspace=0.30)

# --- A: the perturbation is local ------------------------------------------
axA = fig.add_subplot(gs[0, 0])
allq = np.array(pf["by_magnitude_quartile"]).mean(0)
# The profile is normalised per variant by its own mean |dsigma|, while
# `profiles.noise` is in absolute angstroms -- dividing one by the other would
# mix units and understate the floor by more than an order of magnitude. The
# per-assay `noise_to_signal` field is already the ratio this axis needs.
ns = [v["noise_to_signal"] for v in S["per_assay"].values()
      if np.isfinite(v["noise_to_signal"])]
noise_rel = float(np.mean(ns))
x = np.arange(len(allq))
axA.plot(x, allq, color=C_PC2, lw=2.2, zorder=4)
axA.scatter(x, allq, s=30, color=C_PC2, zorder=5, edgecolor=SURF, linewidth=1.5)
axA.axhline(1.0, color=INK, lw=1.1, ls="--", zorder=3)
axA.annotate("no spatial preference", xy=(len(x) - 1, 1.0), xytext=(-4, 7),
             textcoords="offset points", fontsize=7.4, color=INK2, ha="right")
axA.fill_between(x, 0, noise_rel, color=C_AMB, alpha=0.30, lw=0, zorder=2)
axA.annotate(f"replicate noise floor ({noise_rel:.2f})", xy=(0, noise_rel),
             xytext=(4, 6), textcoords="offset points", fontsize=7.4, color=C_AMB)
rr = S["localisation radius ratio"]
axA.set_xticks(x); axA.set_xticklabels(lab, fontsize=7, rotation=35, ha="right")
axA.set_xlabel("distance from the mutated residue (Å)")
axA.set_ylabel("normalised |dσ|  (1 = variant's own mean)")
axA.set_ylim(0, max(allq) * 1.15)
axA.set_title(f"A  The width change is local\n     weighted radius = "
              f"{rr['mean']:.3f} of uniform [{rr['ci_lo']:.3f}, {rr['ci_hi']:.3f}]",
              color=INK, fontsize=9.5, loc="left", pad=8)
axA.grid(True, color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)

# --- B: the confound, and its removal --------------------------------------
axB = fig.add_subplot(gs[0, 1])
for key, sub, col, lw, ls, name in (
        ("by_magnitude_quartile", None, C_MAG, 2.0, "--", "magnitude alone"),
        ("by_quartile", "PC2", C_PC2, 2.2, "-", "PC2, raw"),
        ("by_quartile_mag_adjusted", "PC2", C_ADJ, 2.2, "-",
         "PC2, magnitude-adjusted")):
    P = np.array(pf[key] if sub is None else pf[key][sub])
    axB.plot(x, P[3] - P[0], color=col, lw=lw, ls=ls, zorder=4, label=name)
axB.axhline(0, color=INK, lw=1.1, zorder=3)
axB.set_xticks(x); axB.set_xticklabels(lab, fontsize=7, rotation=35, ha="right")
axB.set_xlabel("distance from the mutated residue (Å)")
axB.set_ylabel("top quartile − bottom quartile")
axB.legend(frameon=False, fontsize=7.6, loc="lower left")
# Title states what the green curve actually does: it is not flat, it simply
# stops being monotone, and the residual excursion in the last bin is left
# visible rather than described away.
axB.set_title("B  The raw PC2 gradient is magnitude in\n     disguise; adjusting "
              "destroys the monotone trend", color=INK, fontsize=9.5,
              loc="left", pad=8)
axB.grid(True, color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)

# --- C: what each component carries ----------------------------------------
axC = fig.add_subplot(gs[0, 2])
n_pc = S["protocol"]["n_pc"]
rows, cols_, labs = [], [], []
for c in range(n_pc):
    rows.append(S[f"PC{c+1} vs perturbation magnitude"]); cols_.append(C_PC2)
    labs.append(f"PC{c+1}  amplitude")
for c in range(n_pc):
    rows.append(S[f"PC{c+1} vs radius ratio | magnitude"]); cols_.append(C_ADJ)
    labs.append(f"PC{c+1}  localisation")
y = np.arange(len(rows))[::-1]
for yy, g, col in zip(y, rows, cols_):
    axC.plot([g["ci_lo"], g["ci_hi"]], [yy, yy], color=col, lw=2.6, zorder=4,
             solid_capstyle="round")
    axC.scatter([g["mean"]], [yy], s=44, color=col, zorder=5, edgecolor=SURF,
                linewidth=1.6)
axC.axvline(0, color=INK, lw=1.2, zorder=3)
axC.axhline(n_pc - 0.5, color=GRID, lw=1.0, zorder=2)
axC.set_yticks(y); axC.set_yticklabels(labs, fontsize=7.6)
axC.set_xlabel("Spearman (assay-level 95% interval)")
axC.set_title("C  PC2 owns the amplitude;\n     no component owns the location",
              color=INK, fontsize=9.5, loc="left", pad=8)
axC.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axC.set_axisbelow(True)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
