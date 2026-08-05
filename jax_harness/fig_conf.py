"""Sanity figure for the conformational references: is the axis where it should be?

Plots |E[d_A] - E[d_B]| per residue pair for each fold-switch system, with pairs
outside the axis mask greyed out. This is a verification plot, not a result: a
misaligned residue numbering would show up here as structureless noise instead of
the blocks that a real domain rearrangement produces.

Magnitude gets a sequential single-hue ramp, light to dark. The greyed region is
"not on the axis" -- either beyond 22 A in both states, or moving by less than
2 A, or within three residues along the chain.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

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
RAMP = LinearSegmentedColormap.from_list(
    "pi_blue", ["#eef4fc", "#9dc2ee", "#2a78d6", "#17457c"])
RAMP.set_bad("#ecebe8")

ap = argparse.ArgumentParser()
ap.add_argument("--refs", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
                                  "prot_interp_files/data/conformations/refs")
ap.add_argument("--out", required=True)
a = ap.parse_args()

fams = ["XCL1", "RfaH", "KaiB", "Mad2"]
fig, axes = plt.subplots(1, 4, figsize=(15.4, 4.5))
for ax, fam in zip(axes, fams):
    d = np.load(Path(a.refs) / f"conf_{fam}.npz", allow_pickle=True)
    dd = np.abs(d["d_a"] - d["d_b"])
    mask, res = d["mask"], d["resnums"]
    shown = np.where(mask, dd, np.nan)
    im = ax.imshow(shown, cmap=RAMP, origin="lower", vmin=0, vmax=20,
                   interpolation="nearest",
                   extent=[res.min(), res.max(), res.min(), res.max()])
    frac = 100 * mask[np.triu_indices(len(res), k=3)].mean()
    ax.set_title(f"{fam}   {str(d['state_a'])} vs {str(d['state_b'])}\n"
                 f"{int(mask[np.triu_indices(len(res), k=3)].sum())} axis pairs "
                 f"({frac:.0f}% of {len(res)} residues)",
                 color=INK, fontsize=9.5, loc="left", pad=8)
    ax.set_xlabel("residue")
    if fam == fams[0]:
        ax.set_ylabel("residue")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelsize=7.5)

fig.suptitle("Conformational axis per residue pair:  |mean CB-CB distance in state A "
             "− state B|, angstroms.   Grey = off-axis "
             "(>22 Å in both states, or moving <2 Å, or |i−j|<3).",
             fontsize=10, color=INK, x=0.5, y=1.02)
fig.tight_layout()
fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
