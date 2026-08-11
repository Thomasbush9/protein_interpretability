"""Figure: the Jacobian study -- what the five z-path operations actually do.

Six panels, in the order the argument has to be believed:

  A  the weights say one thing and the operating point says another. Effective
     rank of the bare weight matrices (grey reference lines) against the
     effective rank of the same layer's Jacobian. This is the panel that
     justifies redoing the analysis at all.
  B  why: the SwiGLU gate leaves only a seventh of the hidden units live, and
     the live count tracks the rank across depth. Both are drawn as a FRACTION
     of their own ceiling (512 units, 128 dimensions) rather than on two axes --
     a second y-scale would let any two curves be made to agree.
  C  how much of the site row each operation moves, and how much of PC2 it adds.
     transition_z is the largest mover; the triangle operations are an order
     down.
  D  the negative result. |percentile - 0.5| against the matched null, per
     operation and component. A percentile drawn from noise is uniform, so 0.25
     -- not zero -- is the chance level, and it is drawn.
  E  twelve unrelated folds put every operation in nearly the same subspace,
     against the k/128 a random subspace would give.
  F  where the PC directions actually sit in the transition's Jacobian, against
     the same random baseline. Below the line means the operation barely reads
     or writes that component.

Palette is the documented categorical theme in fixed slot order (blue, orange,
aqua, yellow, magenta) for the five operations -- the same colour means the same
operation in every panel -- with grey reserved for reference levels and never
used as a series. The set passes the adjacent-pair CVD and normal-vision gates;
three slots sit below 3:1 on the light surface, so every series is also directly
labelled and the same numbers appear as tables in the report.
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
# Documented categorical theme, fixed slot order. Grey is a reference level.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
C_REF = "#8a8885"
NICE = {"tri_mul_out": "tri mul out", "tri_mul_in": "tri mul in",
        "tri_att_start": "tri att start", "tri_att_end": "tri att end",
        "transition_z": "transition (MLP)"}

ap = argparse.ArgumentParser()
ap.add_argument("--jac", required=True)
ap.add_argument("--ops", required=True)
ap.add_argument("--gate", required=True)
ap.add_argument("--wsvd", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

J, O = json.load(open(a.jac)), json.load(open(a.ops))
G, WS = json.load(open(a.gate)), json.load(open(a.wsvd))
OPS = O["ops"]
COL = {o: SLOT[i] for i, o in enumerate(OPS)}
L, DIM = O["layers"], O["dim"]
x = np.arange(L)


def tidy(ax, title, sub=None):
    # Title sits above the subtitle, not on top of it: the title pad is in
    # points while an axes-fraction offset scales with panel height, so the two
    # cross over on tall panels unless the pad clears the subtitle explicitly.
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=22 if sub else 6)
    if sub:
        ax.annotate(sub, (0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", fontsize=8.4, color=INK2,
                    va="bottom", ha="left")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


# A surface-coloured halo behind in-plot text. Direct labels are mandatory here
# -- three of the five slots sit below 3:1 contrast on this surface -- so the
# label must stay legible even where it crosses the line it names.
HALO = dict(boxstyle="round,pad=0.18", facecolor=SURF, edgecolor="none",
            alpha=0.92)


def note(ax, xy, text, color, dx=0, dy=0, ha="center", fs=8.4):
    ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=fs, ha=ha, va="center", bbox=HALO,
                zorder=6)


fig = plt.figure(figsize=(15.2, 9.8))
gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.28, top=0.88)

# One legend for the whole figure: the same colour is the same operation in
# every panel, so repeating a five-entry key per panel would only crowd them.
handles = [plt.Line2D([], [], color=COL[o], lw=2.6, label=NICE[o]) for o in OPS]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.062, 0.955),
           frameon=False, ncol=5, fontsize=9.2, handlelength=1.5,
           columnspacing=2.0)

# ---- A: rank, weights vs operating point --------------------------------
ax = fig.add_subplot(gs[0, 0])
for o in OPS:
    ax.plot(x, O["eff_rank_by_layer"][o], color=COL[o], lw=2.0, solid_capstyle="round")
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

# ---- B: the gate is the mechanism ---------------------------------------
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
     f"r = {G['corr']:.2f} across (assay, layer); both as % of ceiling, one axis")

# ---- C: how much each operation moves, and PC2 gain ----------------------
ax = fig.add_subplot(gs[0, 2])
for o in OPS:
    ax.plot(x, O["gain_by_layer"][o]["PC2"], color=COL[o], lw=2.0,
            solid_capstyle="round")
ax.axhline(0, color=C_REF, lw=1.0)
# Only the MLP is separable by eye here; labelling the four flat curves would
# stack four labels inside a 0.02-tall band. The legend carries the rest.
note(ax, (23, -0.105), "transition (MLP)", COL["transition_z"], ha="left")
note(ax, (34, 0.012), "the four triangle operations", INK2)
ax.set_xlim(0, L - 1)
ax.set_xlabel("Pairformer layer")
ax.set_ylabel("PC2 gain  (added coordinate per unit)")
tidy(ax, "C  Only the MLP moves PC2 appreciably",
     "total multiplier on the coordinate is 1 + gain")

# ---- D: the negative result ---------------------------------------------
ax = fig.add_subplot(gs[1, 0])
nd = O["null_departure"]
chance = O["null_departure_chance"]
npc = len(nd[OPS[0]])
w = 0.15
xs = np.arange(npc)
for i, o in enumerate(OPS):
    ax.bar(xs + (i - (len(OPS) - 1) / 2) * w, nd[o], width=w * 0.86,
           color=COL[o], label=NICE[o], zorder=3)
ax.axhline(chance, color=C_REF, lw=1.4, ls=(0, (4, 3)), zorder=4)
note(ax, (-0.46, chance), f"chance {chance:.2f}", C_REF, ha="left", fs=8.0)
ax.set_xticks(xs, [f"PC{c+1}" for c in range(npc)])
ax.set_ylim(0, 0.46)
ax.set_ylabel("|percentile - 0.5|")
tidy(ax, "D  No operation singles out the stability axis",
     "a percentile from noise is uniform, so 0.25 is chance -- not 0")

# ---- E: cross-assay agreement -------------------------------------------
ax = fig.add_subplot(gs[1, 1])
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
tidy(ax, "E  Every operation is protein-general",
     "principal angles between assays; 12 unrelated folds")

# ---- F: where the PCs sit in the transition's Jacobian -------------------
ax = fig.add_subplot(gs[1, 2])
ks = np.array(J["ks"])
base = ks / DIM
for side, ls, alpha in (("out", "-", 1.0), ("in", (0, (3, 2)), 0.75)):
    for c in (0, 1):
        y = np.array(J["capture_last_layer"][side][c])
        ax.plot(ks, y, color=SLOT[c], lw=2.0, ls=ls, alpha=alpha,
                solid_capstyle="round")
ax.plot(ks, base, color=C_REF, lw=1.6, ls=(0, (4, 3)))
ax.set_xscale("log", base=2)
ax.set_xticks(ks, [str(k) for k in ks])
note(ax, (26, 0.30), "PC1", SLOT[0], ha="right", fs=8.8)
note(ax, (54, 0.52), "PC2", SLOT[1], ha="right", fs=8.8)
note(ax, (18, 0.36), "random", C_REF, ha="right")
ax.text(0.03, 0.96, "solid = write side,  dashed = read side", color=INK2,
        fontsize=8.2, transform=ax.transAxes, va="top")
ax.set_xlim(1, 128)
ax.set_ylim(0, 1.04)
ax.set_xlabel("subspace dimension k")
ax.set_ylabel("fraction of the component captured")
tidy(ax, "F  PC1 and PC2 sit below chance in the MLP",
     "the transition barely reads or writes the stability axis")

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
