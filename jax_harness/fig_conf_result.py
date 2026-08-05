"""Figure: the XCL1 conformational-axis result.

Panel A is the precondition -- whether wild type's distogram carries mass on both
states at all. Panel B is the claim: a signed, two-sided projection, shown both
with and without the crosslinked pair, because for V21C/V59C that pair alone
could produce the whole effect. Panel C shows where in the trunk it happens.
Panel D asks the same question of the emitted structure.

Direction is a diverging encoding -- blue toward Ltn10, orange toward Ltn40,
neutral at zero -- and never a single ramp, because the sign is the result.
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
C_A, C_B, C_N = "#2a78d6", "#eb6834", "#8a8885"      # Ltn10 / Ltn40 / neutral

ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

R = json.load(open(a.res))
d = np.load(a.run, allow_pickle=True)
names = [str(x) for x in d["names"]]
layer = R["layer"]

fig = plt.figure(figsize=(14.4, 8.8))
gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.46)

# --- A: is wild type carrying both states? ---------------------------------
axA = fig.add_subplot(gs[0, 0])
import pi_conf                                        # noqa: E402
c = pi_conf.bin_centers()
p = d["p_wt_pairs"][layer].astype(np.float64)
da, db = d["d_a_pairs"], d["d_b_pairs"]
mA = (p * (np.abs(c[None, :] - da[:, None]) <= 1.5)).sum(1)
mB = (p * (np.abs(c[None, :] - db[:, None]) <= 1.5)).sum(1)
axA.scatter(mA, mB, s=9, color=C_N, alpha=0.45, edgecolor="none", zorder=3)
both = (mA > 0.10) & (mB > 0.10)
axA.scatter(mA[both], mB[both], s=11, color=C_A, alpha=0.8, edgecolor="none", zorder=4)
axA.axhline(0.10, color=INK2, lw=0.9, ls="--"); axA.axvline(0.10, color=INK2, lw=0.9, ls="--")
axA.set_xlabel("WT mass near the Ltn10 distance")
axA.set_ylabel("WT mass near the Ltn40 distance")
axA.set_title(f"A  Precondition: {100*both.mean():.0f}% of axis pairs\n"
              f"     carry >10% on BOTH states", color=INK, fontsize=9.5,
              loc="left", pad=8)
axA.grid(True, color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)

# --- B: the signed projection ----------------------------------------------
axB = fig.add_subplot(gs[0, 1:])
V = [n for n in names if n != "WT"]
yv = np.arange(len(V))
spec = R.get("specificity", {})
for i, nm in enumerate(V):
    r = R["projection"][nm]
    # what a RANDOM direction of the same movement would have given
    s = spec.get(nm)
    if s and s.get("null_sd"):
        axB.add_patch(plt.Rectangle(
            (s["null_mu"] - 2 * s["null_sd"], i - 0.34),
            4 * s["null_sd"], 0.68, facecolor=C_N, alpha=0.20, zorder=1,
            edgecolor="none",
            label="permuted axis (±2 SD)" if i == 0 else None))
    for (val, lo, hi), off, alpha, lab in (
            ((r["proj"], *r["ci"]), +0.16, 0.35, "all axis pairs"),
            ((r["proj_excl"], *r["ci_excl"]), -0.16, 1.0, "crosslink excluded")):
        col = C_A if val > 0 else C_B
        axB.plot([lo, hi], [i + off, i + off], color=col, lw=2.6, alpha=alpha,
                 zorder=3, solid_capstyle="round")
        axB.scatter([val], [i + off], s=46, color=col, alpha=alpha, zorder=4,
                    edgecolor=SURF, linewidth=1.6,
                    label=lab if i == 0 else None)
axB.axvline(0, color=INK, lw=1.4, zorder=5)
axB.set_yticks(yv)
# the verdict rides on the tick label rather than floating in the panel, where
# it collided with the legend
axB.set_yticklabels(
    [f"{n}\n→{'Ltn10' if R['projection'][n]['expected_sign'] > 0 else 'Ltn40'}"
     f" · {R['projection'][n]['verdict']}" for n in V], fontsize=7.6)
axB.set_xlabel("signed projection onto the axis   "
               "(negative = toward Ltn40, positive = toward Ltn10)")
axB.set_title("B  Two-sided prediction: V21C/V59C must go one way and A36C/A49C "
              "the other\n     (A36C/A49C is the clean test; its crosslink "
              "carries no axis information)",
              color=INK, fontsize=10, loc="left", pad=8)
axB.legend(frameon=False, fontsize=7.5, loc="upper left")
axB.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)
axB.margins(x=0.22, y=0.12)

# --- C: where in the trunk ---------------------------------------------------
axC = fig.add_subplot(gs[1, :2])
# style says what KIND of variant it is: cysteine (the disulfide), its serine
# control, or the single non-disulfide point mutation
STYLE = {"V21C_V59C": ("-", 2.0), "A36C_A49C": ("-", 2.0),
         "V21S_V59S": ("--", 1.3), "A36S_A49S": ("--", 1.3),
         "W55D": (":", 2.0)}
for k, nm in enumerate(names):
    if nm == "WT":
        continue
    v = np.array(R["per_layer"][nm])
    exp = R["projection"][nm]["expected_sign"]
    col = C_A if exp > 0 else C_B
    ls, lw = STYLE.get(nm, ("-", 1.5))
    axC.plot(v, color=col, lw=lw, ls=ls, alpha=0.9, label=nm, zorder=3)
axC.axhline(0, color=INK, lw=1.2, zorder=4)
axC.set_xlabel("Pairformer layer")
axC.set_ylabel("signed projection")
axC.set_title("C  Where along the trunk the movement accumulates   "
              "(solid = disulfide, dashed = serine control, dotted = W55D)\n"
              "     colour is the PREDICTED direction; every trace ends negative",
              color=INK, fontsize=9.5, loc="left", pad=8)
axC.legend(frameon=False, fontsize=7.5, ncol=3)
axC.grid(True, color=GRID, lw=0.8, zorder=0); axC.set_axisbelow(True)

# --- D: the emitted structure ------------------------------------------------
axD = fig.add_subplot(gs[1, 2])
ea = np.array([R["structure"][n]["err_a"] for n in names])
eb = np.array([R["structure"][n]["err_b"] for n in names])
x = np.arange(len(names))
axD.bar(x - 0.19, ea, 0.36, color=C_A, alpha=0.9, zorder=3, label="vs Ltn10")
axD.bar(x + 0.19, eb, 0.36, color=C_B, alpha=0.9, zorder=3, label="vs Ltn40")
axD.set_xticks(x); axD.set_xticklabels(names, fontsize=7, rotation=32, ha="right")
axD.set_ylabel("mean |d(pred) - d(ref)|, angstrom")
axD.set_title("D  The emitted structure,\n     same two references",
              color=INK, fontsize=9.5, loc="left", pad=8)
axD.legend(frameon=False, fontsize=7.5)
axD.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); axD.set_axisbelow(True)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
