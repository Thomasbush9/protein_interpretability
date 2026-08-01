"""Beta dose-response: the intervention is connected, and it costs confidence."""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_have={f.name for f in matplotlib.font_manager.fontManager.ttflist}
_s=next((f for f in ("Nimbus Sans","DejaVu Sans","Helvetica","Arial") if f in _have),None)
plt.rcParams.update({**({"font.family":"sans-serif","font.sans-serif":[_s]} if _s else {}),
 "figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.edgecolor":GRID,
 "axes.labelcolor":INK2,"text.color":INK,"xtick.color":INK2,"ytick.color":INK2,
 "font.size":9,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
b = np.array([1.0, 1.5, 2.0, 3.0]); pl = np.array([0.921, 0.798, 0.736, 0.569])
axes[0].plot(b, pl, "-o", color="#eb6834", lw=2, ms=6, zorder=3)
axes[0].axhline(0.921, color=INK2, ls=":", lw=1)
axes[0].text(2.9, 0.928, "stock (beta=1)", fontsize=7.5, color=INK2, ha="right")
axes[0].set_xlabel("beta (scale on pair-derived attention biases)")
axes[0].set_ylabel("wild-type pLDDT")
axes[0].set_title("A  bias scaling IS connected — and costs confidence",
                  color=INK, fontsize=10, loc="left", pad=8)
# contrast with the no-op insertion point
bz = np.array([1.0, 1.5, 2.0]); cond = np.array([0.2847031, 0.2847031, 0.2847034])
ax = axes[1]
ax.plot(bz, cond, "-o", color="#2a78d6", lw=2, ms=6, zorder=3, label="z_trunk (absorbed)")
ax.set_ylim(0.2846, 0.2848)
ax.set_xlabel("beta (scale on z_trunk, before DiffusionConditioning)")
ax.set_ylabel("conditioning ||dq|| / ||q_wt||")
ax.set_title("B  ...but scaling z_trunk is a no-op\n     PairwiseConditioning opens with LayerNorm",
             color=INK, fontsize=10, loc="left", pad=8)
ax.legend(frameon=False, fontsize=8)
ax.ticklabel_format(useOffset=False, axis="y")
for a_ in axes:
    a_.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); a_.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/beta_diagnostic.png", dpi=170)
print("wrote figures/beta_diagnostic.png")
