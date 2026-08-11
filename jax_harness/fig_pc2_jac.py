"""Figure: is the stability axis computed in the Pairformer?

Order matters here. Two of these panels exist because the first version of this
analysis was wrong, and the correction has to be visible before the result that
depends on it.

  A  the mutation basis rotates almost completely with depth. Fitted
     independently at each layer, the top-k subspace overlaps the LAST layer's
     at barely above chance until roughly layer 45. Every depth profile that
     projects onto the last layer's PC2 is therefore measuring a direction the
     shallow layers do not use -- which is what panel B corrects.
  B  departure from the matched null, per operation and component, in each
     layer's OWN basis. Chance is 0.25, not zero: a percentile drawn from noise
     is uniform. Four of five operations clear it; none of them clears it for
     PC2 more than for the other components, which is the actual answer to the
     question in the title.
  C  the linearisation is not assumed. Pushing the archived mutation difference
     through each layer's own Jacobian reproduces the model's next-layer
     difference at cosine ~0.98 (one-step). Chaining 63 of them without
     re-reading the archive decays to ~0.6, which bounds the composed picture
     without touching the per-layer one.
  D  where PC1 and PC2 sit inside the transition's Jacobian at the last layer,
     against the k/128 a random direction gives.
  E  where the rotation in panel A actually happens, and which operation does
     it: per-layer, per-operation, concentrated 4x in the second half.
  F  and whether any operation rotates the subspace MORE than its own size
     predicts. Each bar pair is the measured rotation beside a null that keeps
     the operator's exact singular spectrum and randomises only its singular
     vectors. Every operation falls short of its own null, which is the panel's
     point: the subspace is comparatively preserved, not steered.

Palette is the documented categorical theme in fixed slot order, the same
operation-to-colour mapping as the method figure. Grey is reference levels only.

Panels B, E and F plot operations and use that palette. Panels A, C and D
deliberately break out of it: they plot things that are not operations -- basis
comparisons, prediction modes, components -- and reusing slot 1 there would make
blue mean "tri mul out" in panel B and "one-step prediction" in panel C. They
use an achromatic pair (#0b0b0b,
#7d7b77) separated by lightness, dash and direct labels instead, which is also
the clearest signal that those series are not operations. That pair fails the
categorical validator's chroma floor and lightness band by construction --
those checks exist to stop a HUE reading as grey, and these are meant to. What
matters for legibility is measured and passes: OKLab dE 43.4 between the two
under normal, protan, deutan and tritan vision (floors 15 and 8), and contrast
19.2 and 4.1 against the surface (floor 3).
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
ap.add_argument("--ops", required=True, help="per-layer-basis pooled ops")
ap.add_argument("--jac", required=True)
ap.add_argument("--basis", required=True)
ap.add_argument("--comp", required=True)
ap.add_argument("--rot", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

O, J = json.load(open(a.ops)), json.load(open(a.jac))
B, C = json.load(open(a.basis)), json.load(open(a.comp))
RT = json.load(open(a.rot))
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


fig = plt.figure(figsize=(15.4, 9.6))
gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.30, top=0.86)
# The operation palette belongs to the panels that plot operations -- B, E and
# F. Panels A, C and D plot entirely different entities (basis comparisons,
# prediction modes, components), so reusing slot 1 and slot 2 there would mean
# blue was "tri mul out" in one panel and "one-step prediction" in the next.
# Those panels get a neutral pair separated by weight and dash instead, and the
# legend names the panels it actually covers.
handles = [plt.Line2D([], [], color=COL[o], lw=2.6, label=NICE[o]) for o in OPS]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.975),
           frameon=False, ncol=5, fontsize=9.0, handlelength=1.5,
           columnspacing=1.8, title="operations (panels B, E, F)",
           title_fontproperties={"size": 8.4}, alignment="left")
C_A, C_B = INK, "#7d7b77"

# ---- A: the basis rotates ------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
rot = np.array(B["rot_vs_last"])
ax.plot(x, rot, color=C_A, lw=2.2, solid_capstyle="round")
ident = np.array(B["identity_vs_last"])
ax.plot(x, ident[:, 1] ** 2, color=C_B, lw=2.0, ls=(0, (3, 2)))
ax.axhline(B["random_baseline"], color=C_REF, lw=1.4, ls=(0, (4, 3)))
note(ax, (30, B["random_baseline"] - 0.045),
     f"unrelated bases  {B['random_baseline']:.3f}", C_REF)
note(ax, (18, 0.29), f"top-{B['k']} subspace", C_A)
note(ax, (40, 0.13), "PC2 alone (cos$^2$)", C_B)
ax.set_xlim(0, L - 1)
ax.set_ylim(0, 1.04)
ax.set_xlabel("Pairformer layer")
ax.set_ylabel(f"agreement with the layer-{L-1} basis")
tidy(ax, "A  The mutation basis rotates with depth",
     "so a gain measured against the last layer's PC2 is meaningless at depth")

# ---- B: departure from the null, corrected basis -------------------------
ax = fig.add_subplot(gs[0, 1])
nd = O["null_departure"]
chance = O["null_departure_chance"]
npc = len(nd[OPS[0]])
w = 0.15
xs = np.arange(npc)
for i, o in enumerate(OPS):
    ax.bar(xs + (i - (len(OPS) - 1) / 2) * w, nd[o], width=w * 0.86,
           color=COL[o], zorder=3)
ax.axhline(chance, color=C_REF, lw=1.4, ls=(0, (4, 3)), zorder=4)
ax.axhline(0.5, color=C_REF, lw=1.0, ls=(0, (1, 2)), zorder=4)
note(ax, (-0.46, chance), f"chance {chance:.2f}", C_REF, ha="left", fs=8.0)
note(ax, (npc - 0.55, 0.5), "ceiling 0.50", C_REF, ha="right", fs=8.0)
ax.set_xticks(xs, [f"PC{c+1}" for c in range(npc)])
ax.set_ylim(0, 0.58)
ax.set_ylabel("|percentile - 0.5|")
tidy(ax, "B  The subspace is engaged; PC2 is not singled out",
     "each layer's own basis; PC2 is never the tallest bar for any operation")

# ---- C: the linearisation predicts ---------------------------------------
ax = fig.add_subplot(gs[0, 2])
one = np.array(C["one_step"]["cosine"])
free = np.array(C["free"]["cosine"])
ax.plot(x[1:], one[1:], color=C_A, lw=2.4, solid_capstyle="round")
ax.plot(x[1:], free[1:], color=C_B, lw=2.4, ls=(0, (5, 2)),
        solid_capstyle="round")
note(ax, (30, 0.945), "one-step (re-seeded each layer)", C_A)
note(ax, (41, 0.735), "free-running (63 chained)", C_B)
ax.set_xlim(1, L - 1)
ax.set_ylim(0.4, 1.03)
ax.set_xlabel("Pairformer layer")
ax.set_ylabel("cosine, predicted vs actual dz")
tidy(ax, "C  Each layer's Jacobian predicts the real response",
     "pushing archived mutation differences through the linearised layer")

# ---- D: placement --------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ks = np.array(J["ks"])
base = ks / DIM
# Plotted as a RATIO to the chance baseline rather than the raw fraction. Four
# near-flat curves hugging the bottom of a 0-1 axis are unreadable, and the
# claim is entirely about the distance from chance -- so make the axis carry it:
# 1.0 is chance, below 1.0 is what the panel is asserting.
# Components, not operations, so the neutral pair again -- never a slot colour.
for side, ls in (("out", "-"), ("in", (0, (3, 2)))):
    for c, cc in ((0, C_A), (1, C_B)):
        y = np.array(J["capture_last_layer"][side][c]) / base
        ax.plot(ks, y, color=cc, lw=2.1, ls=ls, solid_capstyle="round")
ax.axhline(1.0, color=C_REF, lw=1.6, ls=(0, (4, 3)))
ax.set_xscale("log", base=2)
ax.set_xticks(ks, [str(k) for k in ks])
note(ax, (8, 1.10), "chance", C_REF, ha="center")
note(ax, (10, 0.72), "PC1", C_A, ha="right", fs=8.8)
note(ax, (10, 0.44), "PC2", C_B, ha="right", fs=8.8)
ax.text(0.03, 0.96, "solid = write side,  dashed = read side", color=INK2,
        fontsize=8.2, transform=ax.transAxes, va="top")
ax.set_xlim(1, 128)
ax.set_ylim(0, 1.55)
ax.set_xlabel("subspace dimension k")
ax.set_ylabel("capture relative to chance")
tidy(ax, "D  PC1 and PC2 sit below chance in the MLP",
     f"transition's Jacobian, layer {L-1}, where the basis is exact")

# ---- E: where the rotation happens, and who does it ----------------------
# Operations again, so the slot palette is correct here.
ax = fig.add_subplot(gs[1, 1])
for o in OPS:
    ax.plot(x[1:], np.array(RT["rotation_by_layer"][o])[1:], color=COL[o],
            lw=2.0, solid_capstyle="round")
note(ax, (26, 0.0215), "transition (MLP)", COL["transition_z"])
note(ax, (30, 0.0045), "the other four", INK2)
ax.set_xlim(1, L - 1)
ax.set_xlabel("Pairformer layer")
ax.set_ylabel("subspace rotation contributed")
tidy(ax, "E  The rotation is late, and mostly the MLP",
     f"{RT['late_mean']/RT['early_mean']:.1f}x more per layer in the second half")

# ---- F: but no more than each operation's own size predicts --------------
ax = fig.add_subplot(gs[1, 2])
xs = np.arange(len(OPS))
real = [RT["rotation"][o] for o in OPS]
null = [RT["rotation_null_spectrum"][o] for o in OPS]
ax.bar(xs - 0.19, real, width=0.34, color=[COL[o] for o in OPS], zorder=3)
ax.bar(xs + 0.19, null, width=0.34, facecolor="none", zorder=3,
       edgecolor=[COL[o] for o in OPS], lw=1.6, hatch="////")
for i in range(len(OPS)):
    ax.text(i - 0.19, real[i] + 0.0004, "real", ha="center", fontsize=7.4,
            color=INK2)
    ax.text(i + 0.19, null[i] + 0.0004, "null", ha="center", fontsize=7.4,
            color=INK2)
ax.set_xticks(xs, [NICE[o].replace(" ", "\n", 1) for o in OPS], fontsize=8.0)
ax.set_ylabel("mean rotation per layer")
tidy(ax, "F  Every operation rotates it LESS than chance",
     "null keeps each operator's exact singular spectrum, randomises direction")

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
