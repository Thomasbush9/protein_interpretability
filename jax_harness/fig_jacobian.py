"""Figure: the Jacobian method and what it says about the pair path.

Three panels, method first, mechanism second, generality third:

  A  the reason the method exists. Effective rank of the bare weight matrices
     (grey reference lines) against the effective rank of the same layer's
     Jacobian, per operation. The weights and the operator disagree by a factor
     of about four, so a weight-space decomposition is describing a matrix the
     model never applies.
  B  why the operator is so much smaller. The SwiGLU gate leaves roughly a
     seventh of the hidden units live, and the live count tracks the rank across
     depth. Both are drawn as a FRACTION of their own ceiling (512 units, 128
     dimensions) on a single axis -- a second y-scale would let any two curves
     be made to agree.
  C  that none of this is a property of one protein. Cross-assay agreement of
     the top-k subspaces, per operation, against the k/128 a random subspace
     gives.

Palette is the documented categorical theme in fixed slot order for the five
operations -- the same colour is the same operation in every panel here and in
the companion PC2 figure -- with grey reserved for reference levels and never
used as a series. Three of the five slots sit below 3:1 on this surface, so the
figure carries a legend and the same numbers appear as tables in the report.
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
NICE = {"tri_mul_out": "tri mul out", "tri_mul_in": "tri mul in",
        "tri_att_start": "tri att start", "tri_att_end": "tri att end",
        "transition_z": "transition (MLP)"}
HALO = dict(boxstyle="round,pad=0.18", facecolor=SURF, edgecolor="none",
            alpha=0.92)

ap = argparse.ArgumentParser()
ap.add_argument("--ops", required=True)
ap.add_argument("--gate", required=True)
ap.add_argument("--wsvd", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

O, G, WS = (json.load(open(a.ops)), json.load(open(a.gate)),
            json.load(open(a.wsvd)))
OPS = O["ops"]
COL = {o: SLOT[i] for i, o in enumerate(OPS)}
L, DIM = O["layers"], O["dim"]
x = np.arange(L)


def tidy(ax, title, sub=None):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=22 if sub else 6)
    if sub:
        ax.annotate(sub, (0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", fontsize=8.4, color=INK2,
                    va="bottom", ha="left")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def note(ax, xy, text, color, ha="center", fs=8.4):
    ax.annotate(text, xy, color=color, fontsize=fs, ha=ha, va="center",
                bbox=HALO, zorder=6)


fig = plt.figure(figsize=(15.2, 4.9))
gs = fig.add_gridspec(1, 3, wspace=0.28, top=0.76, bottom=0.14)
handles = [plt.Line2D([], [], color=COL[o], lw=2.6, label=NICE[o]) for o in OPS]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.062, 0.985),
           frameon=False, ncol=5, fontsize=9.2, handlelength=1.5,
           columnspacing=2.0)

# ---- A ------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
for o in OPS:
    ax.plot(x, O["eff_rank_by_layer"][o], color=COL[o], lw=2.0,
            solid_capstyle="round")
for m in ("fc1", "fc2", "fc3"):
    v = float(np.median(WS["eff_rank"][m]))
    ax.axhline(v, color=C_REF, lw=1.0, ls=(0, (4, 3)))
    ax.text(1.0, v + 1.8, f"{m} weights  {v:.0f}", color=C_REF, fontsize=7.8)
ax.set_xlim(0, L - 1)
ax.set_ylim(0, 100)
ax.set_xlabel("Pairformer layer")
ax.set_ylabel("effective rank  (of 128)")
tidy(ax, "A  The weights are not the operator",
     "bare weights sit at 73-94; every Jacobian sits far below")

# ---- B ------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
live = 100 * np.array(G["live_by_layer"]) / 512
rank = 100 * np.array(G["rank_by_layer"]) / 128
ax.plot(x, live, color=SLOT[0], lw=2.0, solid_capstyle="round")
ax.plot(x, rank, color=SLOT[1], lw=2.0, solid_capstyle="round")
note(ax, (34, 6.0), "live hidden units (% of 512)", SLOT[0])
note(ax, (20, 28.0), "Jacobian rank (% of 128)", SLOT[1])
ax.set_xlim(0, L - 1)
ax.set_ylim(0, 32)
ax.set_xlabel("Pairformer layer")
ax.set_ylabel("percent of own ceiling")
tidy(ax, "B  The SwiGLU gate is the mechanism",
     f"r = {G['corr']:.2f} across (assay, layer); one axis, no dual scale")

# ---- C ------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
xs = np.arange(len(OPS))
out = [O["agreement"][o][0] for o in OPS]
inn = [O["agreement"][o][1] for o in OPS]
ax.bar(xs - 0.19, out, width=0.34, color=[COL[o] for o in OPS], zorder=3)
ax.bar(xs + 0.19, inn, width=0.34, color=[COL[o] for o in OPS], zorder=3,
       alpha=0.45)
rb = O["k"] / DIM
ax.axhline(rb, color=C_REF, lw=1.4, ls=(0, (4, 3)), zorder=4)
note(ax, (-0.46, rb), f"random {rb:.3f}", C_REF, ha="left", fs=8.0)
for i in range(len(OPS)):
    ax.text(i - 0.19, out[i] + 0.02, "out", ha="center", fontsize=7.4, color=INK2)
    ax.text(i + 0.19, inn[i] + 0.02, "in", ha="center", fontsize=7.4, color=INK2)
ax.set_xticks(xs, [NICE[o].replace(" ", "\n", 1) for o in OPS], fontsize=8.0)
ax.set_ylim(0, 1.06)
ax.set_ylabel(f"mean cos$^2$, top-{O['k']} subspace")
tidy(ax, "C  Every operation is protein-general",
     "principal angles between assays; 12 unrelated folds")

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
