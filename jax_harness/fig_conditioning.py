"""Figure: how far toward the sampler the mutation signal stays readable.

The project has measured the trunk against what the model emitted, and never
anything in between. This puts the diffusion-conditioning tensors -- the last
thing the structure module sees -- on the same axis as both.

  A  score against capacity. Every block is reduced to k training-fold
     principal components and scored by the identical protein-held-out
     protocol, so blocks of 384, 128 and 37 channels are compared at matched
     capacity rather than at native width, where the widest would win on width.
     The two lines that matter converge: the trunk pair row and the token-pair
     conditioning meet, while emitted geometry stays flat and low.
  B  the paired differences at k = 32, assay-clustered. Above the rule, what
     the trunk beats; below it, what the conditioning beats. The gap that does
     NOT exclude zero is the informative one.

Colour separates three families and nothing else: the trunk in blue, the
conditioning tensors in green, what was emitted in grey. Within the
conditioning family the token-pair channel is solid and the two atom-derived
channels are lighter, because that distinction is the finding -- signal lives
in the pair channel.

  python fig_conditioning.py --cond ../runs/conditioning_probe.json \
      --out ../figures/conditioning.png
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
C_TRUNK, C_COND, C_EMIT = "#2a78d6", "#1baf7a", "#8a8885"
C_TRUNK_2, C_COND_2, C_COND_3 = "#9dc0ea", "#8ed7bc", "#c9e8db"
HALO = dict(boxstyle="round,pad=0.18", facecolor=SURF, edgecolor="none",
            alpha=0.92)

# label, key, colour, linestyle, label nudge in points -- ordered by distance
# from the decoder. Every series is named on the plot rather than in a legend,
# so the nudges exist to stop two lines that land a few thousandths apart from
# printing their names on top of each other.
SERIES = [
    ("trunk single s",           "trunk_s",  C_TRUNK_2, (0, (4, 2)), -9),
    ("trunk pair row z",         "trunk_z",  C_TRUNK,   "-",          0),
    ("conditioning: token pair", "cond_ttb", C_COND,    "-",         +1),
    ("conditioning: c",          "cond_c",   C_COND_2,  (0, (4, 2)), +9),
    ("conditioning: q",          "cond_q",   C_COND_3,  (0, (1, 2)),  0),
    ("emitted geometry",         "geometry", C_EMIT,    "-",          0),
]

ap = argparse.ArgumentParser()
ap.add_argument("--cond", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

D = json.load(open(a.cond))
KS = [int(k) for k in D["ks"]]
W = D["widths"]
n_assays = len(D["assays"])


def tidy(ax, title, sub=None):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=22 if sub else 6)
    if sub:
        ax.annotate(sub, (0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", fontsize=8.4, color=INK2,
                    va="bottom", ha="left")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(14.4, 5.3))
gs = fig.add_gridspec(1, 2, wspace=0.34, top=0.78, bottom=0.16,
                      width_ratios=[1.20, 1.0])

# ---- A: score against matched capacity ----------------------------------
ax = fig.add_subplot(gs[0, 0])
xs = np.arange(len(KS))
for lab, key, col, ls, dy in SERIES:
    ys, xk = [], []
    for i, k in enumerate(KS):
        c = D["curve"].get(key, {}).get(str(k))
        if c is None:
            continue
        xk.append(i)
        ys.append(c["mean"])
    lead = key in ("trunk_z", "cond_ttb")
    lw = 2.2 if lead else 1.4
    ax.plot(xk, ys, color=col, lw=lw, ls=ls, zorder=5 if lead else 4,
            solid_capstyle="round")
    ax.scatter(xk[-1:], ys[-1:], s=30 if lead else 16, color=col, zorder=6,
               edgecolor=SURF, linewidth=1.1)
    ax.annotate(lab, (xk[-1], ys[-1]), xytext=(8, dy),
                textcoords="offset points", ha="left", va="center",
                fontsize=8.5 if lead else 8.1, color=col,
                fontweight="semibold" if lead else "normal", zorder=7)
ax.set_xlim(-0.2, len(KS) - 1 + 2.35)
ax.set_xticks(xs, [str(k) for k in KS], fontsize=9)
ax.set_xlabel("k training-fold principal components  (matched capacity)")
ax.set_ylabel("Spearman, held-out protein")
ax.axhline(0, color=INK2, lw=0.9, zorder=3)
tidy(ax, "A  The signal is still readable at the decoder entrance",
     f"{n_assays} assays, leave-one-assay-out; native widths "
     f"{W['cond_ttb']}/{W['trunk_z']}/{W['geometry']} are not comparable")

# ---- B: paired differences at k = 32 -------------------------------------
ax = fig.add_subplot(gs[0, 1])
K = 32
rows = []
for lab, key, col, _ls, _dy in SERIES:
    if key == "trunk_z":
        continue
    g = D["gaps"].get(f"k={K}: trunk_z - {key}")
    if g:
        rows.append((f"trunk z  −  {lab}", g, C_TRUNK))
for key, lab in (("cond_ttb", "conditioning: token pair"),):
    g = D["gaps"].get(f"k={K}: {key} - geometry")
    if g:
        rows.append((f"{lab}  −  emitted", g, C_COND))
ys = np.arange(len(rows))[::-1]
for y, (lab, g, col) in zip(ys, rows):
    crosses = not (g["ci_lo"] > 0 or g["ci_hi"] < 0)
    c = C_EMIT if crosses else col
    ax.plot([g["ci_lo"], g["ci_hi"]], [y, y], color=c, lw=2.4, alpha=0.85,
            solid_capstyle="butt", zorder=4)
    ax.scatter([g["gap"]], [y], s=52, color=c, zorder=5, edgecolor=SURF,
               linewidth=1.2)
    ax.annotate(f"{g['wins']}/{g['n']}", (g["ci_hi"], y), xytext=(7, 0),
                textcoords="offset points", va="center", fontsize=8.2,
                color=INK2)
    if crosses:
        ax.annotate("indistinguishable", (g["ci_hi"], y), xytext=(38, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.4, color=INK, zorder=7)
ax.axvline(0, color=INK2, lw=1.1, zorder=3)
ax.set_yticks(ys, [r[0] for r in rows], fontsize=8.8)
ax.set_xlabel("paired difference in Spearman at k = 32  (assay is the unit)")
ax.set_ylim(-0.7, len(rows) - 0.3)
tidy(ax, "B  What the trunk beats, and what it does not",
     "95% assay-clustered intervals; grey where the interval includes zero")

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
