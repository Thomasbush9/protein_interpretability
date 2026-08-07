"""Figure: depth, and the correction it forces on the cross-model result.

The cross-model comparison used each model's LAST layer and reported that
Boltz-2 sat at CKA ~0.47 from both others while OpenFold3 and Protenix agreed
at 0.80. Panel B shows that this is a property of the final layer and not of the
models: at every other matched fractional depth all three agree at 0.72-0.91,
and every pair falls off sharply only at l/L = 1. Comparing "the last of 64"
with "the last of 16" was the error.

  A  decodability against fractional depth -- severity is readable from the
     first eighth of the trunk at nearly the level it reaches at the end, so it
     is not something the pair stack builds up.
  B  cross-model agreement at matched depth, with the last-layer values that
     were previously reported marked.
  C  the severity direction alone. Total decodability is flat with depth but
     the single direction sharpens, so the trunk concentrates an already
     present signal rather than creating one.

Fractional depth is the only axis on which 64, 48 and 16 layers can be compared.
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
COL = {"boltz2": "#2a78d6", "of3": "#eb6834", "protenix": "#1baf7a"}
NICE = {"boltz2": "Boltz-2", "of3": "OpenFold3", "protenix": "Protenix"}
PCOL = {"boltz2|of3": "#2a78d6", "boltz2|protenix": "#eb6834",
        "of3|protenix": "#1baf7a"}

ap = argparse.ArgumentParser()
ap.add_argument("--depth", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
S = json.load(open(a.depth))
F = np.array(S["fracs"])

fig = plt.figure(figsize=(14.4, 4.6))
gs = fig.add_gridspec(1, 3, wspace=0.32)


def band(ax, rows, col, lab, lw=2.2):
    m = np.array([r["mean"] for r in rows])
    lo = np.array([r["ci_lo"] for r in rows])
    hi = np.array([r["ci_hi"] for r in rows])
    ax.fill_between(F, lo, hi, color=col, alpha=0.12, lw=0, zorder=2)
    ax.plot(F, m, color=col, lw=lw, zorder=4, marker="o", ms=3.6, label=lab)
    return m


# --- A: decodability -------------------------------------------------------
axA = fig.add_subplot(gs[0, 0])
for m_ in COL:
    band(axA, S["decodability_by_depth"][m_], COL[m_],
         f"{NICE[m_]} ({S['n_layers'][m_]} layers)")
axA.set_xlabel("fractional depth  l / L")
axA.set_ylabel("held-out Spearman vs DMS")
axA.legend(frameon=False, fontsize=7.4, loc="lower right")
axA.set_title("A  Severity is readable from the first\n     eighth of the trunk",
              color=INK, fontsize=9.5, loc="left", pad=8)
axA.grid(True, color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)

# --- B: the correction -----------------------------------------------------
axB = fig.add_subplot(gs[0, 1])
for k, rows in S["cka_by_depth"].items():
    m = band(axB, rows, PCOL[k], NICE_k := k.replace("boltz2", "Boltz-2")
             .replace("of3", "OF3").replace("protenix", "PTX").replace("|", " · "))
    axB.scatter([F[-1]], [m[-1]], s=80, facecolor="none", edgecolor=PCOL[k],
                linewidth=2.0, zorder=6)
axB.annotate("previously reported\n(last layer only)", xy=(F[-1], 0.47),
             xytext=(-10, -6), textcoords="offset points", ha="right",
             fontsize=7.4, color=INK2)
axB.set_xlabel("fractional depth  l / L")
axB.set_ylabel("cross-model CKA")
axB.set_ylim(0.35, 1.0)
axB.legend(frameon=False, fontsize=7.4, loc="lower left")
axB.set_title("B  The asymmetry was a last-layer\n     artifact, not a fact about "
              "the models", color=INK, fontsize=9.5, loc="left", pad=8)
axB.grid(True, color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)

# --- C: the severity direction --------------------------------------------
axC = fig.add_subplot(gs[0, 2])
for m_ in COL:
    band(axC, S["severity_direction_by_depth"][m_], COL[m_], NICE[m_])
axC.set_xlabel("fractional depth  l / L")
axC.set_ylabel("Spearman, single direction")
axC.legend(frameon=False, fontsize=7.4, loc="upper left")
axC.set_title("C  One direction sharpens with depth\n     while the total stays flat",
              color=INK, fontsize=9.5, loc="left", pad=8)
axC.grid(True, color=GRID, lw=0.8, zorder=0); axC.set_axisbelow(True)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
