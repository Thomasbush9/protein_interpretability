"""Figure: does the sharpening result survive run-to-run drift?

`gym2_*` and `gym2s_*` are two independent executions of the identical variant
set, so the difference between them IS the inference noise. The three panels
ask, in order: does the classification reproduce, how far from zero does a
variant have to sit before the call is safe, and does the biological claim hold
once the ambiguous band is excluded rather than silently split.

Green is the confident subset, amber the ambiguous band, grey the reference
levels -- so in panel C the reader can see the effect grow rather than shrink
when the noisy variants are removed, which is the point of the panel.
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
C_OK, C_AMB, C_REF, C_BLUE = "#1baf7a", "#eda100", "#8a8885", "#2a78d6"

ap = argparse.ArgumentParser()
ap.add_argument("--drift", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

Dj = json.load(open(a.drift))
per = Dj["per_assay"]
names = sorted(per)

fig = plt.figure(figsize=(14.2, 5.0))
gs = fig.add_gridspec(1, 3, wspace=0.30)

# --- A: does the sharpening call reproduce? --------------------------------
axA = fig.add_subplot(gs[0, 0])
r1 = np.array([100 * per[n]["frac_sharpen_run1"] for n in names])
r2 = np.array([100 * per[n]["frac_sharpen_run2"] for n in names])
lim = (0, max(r1.max(), r2.max()) * 1.18)
axA.plot(lim, lim, color=C_REF, lw=1.2, ls="--", zorder=2)
axA.scatter(r1, r2, s=44, color=C_BLUE, zorder=4, edgecolor=SURF, linewidth=1.4)
# Several assays sit almost on top of each other (ILF3 and RCRO differ by under
# a point on both axes), so a single fixed offset overlaps their labels.
# Offsets alternate by rank along the diagonal, which separates neighbours
# without needing a layout solver.
for r, i in enumerate(np.argsort(r1 + r2)):
    dx, dy = ((6, 4), (6, -10), (-8, 6))[r % 3]
    axA.annotate(names[i], (r1[i], r2[i]), xytext=(dx, dy),
                 textcoords="offset points", fontsize=6.8, color=INK2,
                 ha="right" if dx < 0 else "left")
axA.set_xlim(lim); axA.set_ylim(lim)
axA.set_xlabel("% of variants sharpening, run 1")
axA.set_ylabel("run 2")
sg = Dj["sign agreement on dsd"]
axA.set_title("A  Two independent runs agree on the\n     sign for "
              f"{100*sg['mean']:.1f}% of variants", color=INK, fontsize=9.5,
              loc="left", pad=8)
axA.grid(True, color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)

# --- B: how many variants clear the noise? ---------------------------------
axB = fig.add_subplot(gs[0, 1])
o = np.argsort([per[n]["frac_confident"] for n in names])
nn = [names[i] for i in o]
conf = np.array([100 * per[n]["frac_confident"] for n in nn])
x = np.arange(len(nn))
axB.bar(x, conf, 0.68, color=C_OK, alpha=0.9, zorder=3, label="call is safe")
axB.bar(x, 100 - conf, 0.68, bottom=conf, color=C_AMB, alpha=0.9, zorder=3,
        label="inside the noise band")
fc = Dj["fraction clearing the noise band"]
axB.axhline(100 * fc["mean"], color=INK, lw=1.2, ls="--", zorder=5)
axB.set_xticks(x); axB.set_xticklabels(nn, fontsize=7, rotation=35, ha="right")
axB.set_ylabel(r"% of variants, $|d\sigma|$ vs $\pm2\sigma_{noise}$")
axB.set_ylim(0, 100)
axB.legend(frameon=False, fontsize=7.6, ncol=2, loc="lower center",
           bbox_to_anchor=(0.5, -0.34))
axB.set_title(f"B  {100*fc['mean']:.0f}% clear twice the drift\n"
              f"     [{100*fc['ci_lo']:.0f}, {100*fc['ci_hi']:.0f}]",
              color=INK, fontsize=9.5, loc="left", pad=8)
axB.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)

# --- C: does the claim survive dropping the ambiguous band? ----------------
axC = fig.add_subplot(gs[0, 2])
ga = Dj["DMS(sharpen) - DMS(broaden), all variants"]
gc = Dj["DMS(sharpen) - DMS(broaden), confident only"]
rng = np.random.default_rng(0)
for i, (lab, g, key, col) in enumerate(
        (("all variants", ga, "dms_gap_all", C_REF),
         ("noise band excluded", gc, "dms_gap_confident", C_OK))):
    v = [per[n][key] for n in names if np.isfinite(per[n].get(key, np.nan))]
    axC.scatter(np.full(len(v), i) + rng.uniform(-0.13, 0.13, len(v)), v,
                s=22, color=INK, alpha=0.42, zorder=4, edgecolor=SURF, linewidth=0.5)
    axC.plot([i, i], [g["ci_lo"], g["ci_hi"]], color=col, lw=3.0, zorder=5,
             solid_capstyle="round")
    axC.scatter([i], [g["mean"]], s=90, color=col, zorder=6, edgecolor=SURF,
                linewidth=2)
    axC.annotate(f"{g['mean']:+.3f}", xy=(i, g["mean"]), xytext=(14, -3),
                 textcoords="offset points", fontsize=8.4, color=col)
axC.axhline(0, color=INK, lw=1.2, zorder=2)
axC.set_xticks([0, 1]); axC.set_xticklabels(["all variants", "noise band\nexcluded"],
                                            fontsize=8.5)
axC.set_xlim(-0.5, 1.7)
axC.set_ylabel("mean DMS(sharpen) - DMS(broaden)")
axC.set_title("C  The effect grows when the\n     ambiguous variants are dropped",
              color=INK, fontsize=9.5, loc="left", pad=8)
axC.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); axC.set_axisbelow(True)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
