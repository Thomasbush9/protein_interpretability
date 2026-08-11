"""Figure: what the model knows versus what it says.

The single most important claim in the project had no figure. This is it.

  A  leave-one-assay-out transfer, with the internal side given the 128 pair
     channels rather than per-layer magnitudes. An earlier version led with the
     magnitude probe, which discards the direction this whole project is about
     and cost roughly 0.12 Spearman.
     Each predictor is trained on eleven proteins
     and scored on the twelfth, so nothing here is fitted on the protein it is
     tested on. The internal probe is the only one that is not a description of
     the model's own output or of the substitution itself. Per-assay values are
     drawn as well as the pooled interval, because a pooled mean with twelve
     points behind it should show the twelve points.
  B  the same comparison paired within protein. Internal beats the emitted
     output in all twelve, which is the fact a mean and an interval cannot show
     -- a +0.25 average gap could in principle be four large wins and eight
     losses.
  C  where the output fails. Splitting the target into BETWEEN-position variance
     (which residue was mutated) and WITHIN-position variance (which substitution
     at a fixed residue) separates a trivially available signal from a hard one.
     Burial and packing already say a lot about the first. The second is the one
     that needs to know something about the specific amino acid exchanged, and it
     is where the emitted structure retains almost nothing while the internal
     representation retains most of its advantage.

Panel C is the honest counterweight to panel A. Substitution chemistry is close
to the internal probe on within-position variance, and the figure says so rather
than dropping the baseline that competes.

Palette is the documented categorical theme in fixed slot order; each predictor
keeps its colour across all three panels. Grey is reference levels only.
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
HALO = dict(boxstyle="round,pad=0.18", facecolor=SURF, edgecolor="none",
            alpha=0.92)

# predictor key -> (label, colour slot)
PRED = [("internal_vec", "internal, 128 pair channels", 0),
        ("internal", "internal, per-layer magnitudes", 4),
        ("chemistry", "substitution chemistry", 2),
        ("output_rich", "emitted structure (10 features)", 1),
        ("TM_to_WT", "TM score to wild type", 3)]
BW = [("internal dz (128, one layer)", "internal (128-dim)", 0),
      ("substitution chemistry (17)", "chemistry", 2),
      ("output rich (published, 10)", "emitted structure", 1)]

ap = argparse.ArgumentParser()
ap.add_argument("--transfer", required=True)
ap.add_argument("--bw", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

T, B = json.load(open(a.transfer)), json.load(open(a.bw))
P = T["predictors"]


def tidy(ax, title, sub=None):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=22 if sub else 6)
    if sub:
        ax.annotate(sub, (0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", fontsize=8.4, color=INK2,
                    va="bottom", ha="left")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(15.4, 5.2))
gs = fig.add_gridspec(1, 3, wspace=0.34, top=0.80, bottom=0.15,
                      width_ratios=[1.15, 1.0, 1.0])

# ---- A: the predictor comparison ----------------------------------------
ax = fig.add_subplot(gs[0, 0])
ys = np.arange(len(PRED))[::-1]
rng = np.random.default_rng(0)
for (key, label, slot), y in zip(PRED, ys):
    d = P[key]
    per = np.array(list(d["per_assay"].values()))
    ax.scatter(per, y + rng.uniform(-0.13, 0.13, per.size), s=18,
               color=SLOT[slot], alpha=0.35, linewidths=0, zorder=3)
    ax.plot([d["ci_lo"], d["ci_hi"]], [y, y], color=SLOT[slot], lw=2.6,
            solid_capstyle="round", zorder=4)
    ax.scatter([d["mean"]], [y], s=64, color=SLOT[slot], zorder=5,
               edgecolor=SURF, linewidth=1.4)
    ax.annotate(f"{d['mean']:+.3f}", (d["mean"], y), xytext=(0, 13),
                textcoords="offset points", ha="center", fontsize=8.6,
                color=INK, bbox=HALO, zorder=6)
ax.set_yticks(ys, [p[1] for p in PRED], fontsize=9.2)
ax.set_xlim(-0.05, 0.92)
ax.set_xlabel("Spearman on the held-out protein")
tidy(ax, "A  Internal beats everything the model emits",
     "leave-one-assay-out; dots are the 12 proteins, bar is the 95% interval")

# ---- B: paired, within protein ------------------------------------------
ax = fig.add_subplot(gs[0, 1])
names = list(P["internal"]["per_assay"])
ints = np.array([P["internal_vec"]["per_assay"][n] for n in names])
outs = np.array([P["output_rich"]["per_assay"][n] for n in names])
order = np.argsort(ints)
ys = np.arange(len(names))
for i, o in enumerate(order):
    ax.plot([outs[o], ints[o]], [ys[i], ys[i]], color=C_REF, lw=1.3, zorder=2)
ax.scatter(outs[order], ys, s=42, color=SLOT[1], zorder=4)
ax.scatter(ints[order], ys, s=42, color=SLOT[0], zorder=4)
ax.set_yticks(ys, [names[o] for o in order], fontsize=8.2)
# Direct labels on the top pair instead of a legend box: the colours already
# carry their meaning from panel A, and a legend here lands on the dumbbells.
top = len(names) - 1
ax.annotate("internal", (ints[order][-1], top), xytext=(0, 12),
            textcoords="offset points", ha="center", fontsize=8.6,
            color=SLOT[0], bbox=HALO, zorder=6)
ax.annotate("emitted", (outs[order][-1], top), xytext=(0, 12),
            textcoords="offset points", ha="center", fontsize=8.6,
            color=SLOT[1], bbox=HALO, zorder=6)
g = T["gaps"]["internal 128-dim - output-rich"]
ax.annotate(f"internal wins {g['wins']}/{g['n_assays']}\ngap {g['gap']:+.3f} "
            f"[{g['ci_lo']:+.3f}, {g['ci_hi']:+.3f}]",
            (0.99, -0.62), xycoords=("axes fraction", "data"),
            ha="right", va="center", fontsize=8.8, color=INK, bbox=HALO,
            zorder=6)
ax.set_xlim(-0.05, 0.92)
ax.set_ylim(-1.25, len(names) + 0.2)
ax.set_xlabel("Spearman on the held-out protein")
tidy(ax, "B  In every one of the twelve proteins",
     "same rows, same protocol, paired within protein")

# ---- C: between vs within position --------------------------------------
ax = fig.add_subplot(gs[0, 2])
xs = np.arange(2)
w = 0.26
for i, (key, label, slot) in enumerate(BW):
    d = B["blocks"][key]
    vals = [d["between"]["mean"], d["within"]["mean"]]
    lo = [d["between"]["mean"] - d["between"]["ci_lo"],
          d["within"]["mean"] - d["within"]["ci_lo"]]
    hi = [d["between"]["ci_hi"] - d["between"]["mean"],
          d["within"]["ci_hi"] - d["within"]["mean"]]
    ax.bar(xs + (i - 1) * w, vals, width=w * 0.86, color=SLOT[slot], zorder=3,
           label=label)
    ax.errorbar(xs + (i - 1) * w, vals, yerr=[lo, hi], fmt="none",
                ecolor=INK2, elinewidth=1.1, capsize=2.5, zorder=5)
ax.set_xticks(xs, ["between positions\n(which residue)",
                   "within position\n(which substitution)"], fontsize=8.8)
ax.set_ylim(0, 0.92)
ax.set_ylabel("Spearman")
ax.legend(frameon=False, fontsize=8.4, loc="upper right", ncol=1)
tidy(ax, "C  The output collapses on the hard half",
     "chemistry is competitive within position; the emitted structure is not")

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
