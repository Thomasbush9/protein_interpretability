"""Dose series: how localisation on mutated residues evolves with depth and count."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_sans = next((f for f in ("Nimbus Sans","DejaVu Sans","Helvetica","Arial") if f in _have), None)
plt.rcParams.update({**({"font.family":"sans-serif","font.sans-serif":[_sans]} if _sans else {}),
    "figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.edgecolor":GRID,
    "axes.labelcolor":INK2,"text.color":INK,"xtick.color":INK2,"ytick.color":INK2,
    "font.size":9,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
# sequential magnitude (mutation count) -> one hue, light to dark
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]

ap = argparse.ArgumentParser()
ap.add_argument("--dose", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
d = json.loads(Path(a.dose).read_text())

core = sorted([k for k in d["mutants"] if "core" in k],
              key=lambda k: int(re.search(r"(\d+)$", k).group(1)))
surf = [k for k in d["mutants"] if "surface" in k]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for i, mid in enumerate(core):
    e = np.array(d["mutants"][mid]["enrichment_at_mutated"])
    n = int(re.search(r"(\d+)$", mid).group(1))
    axes[0].plot(e, color=RAMP[i % len(RAMP)], lw=2, label=f"core n={n}", zorder=3)
    axes[1].plot(e / e[0], color=RAMP[i % len(RAMP)], lw=2, zorder=3)
for mid in surf:
    e = np.array(d["mutants"][mid]["enrichment_at_mutated"])
    axes[0].plot(e, color="#eb6834", lw=2, ls="--", label="surface n=32", zorder=3)
    axes[1].plot(e / e[0], color="#eb6834", lw=2, ls="--", zorder=3)

axes[0].set_yscale("log")
from matplotlib.ticker import FuncFormatter, NullFormatter
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:g}"))
axes[0].yaxis.set_minor_formatter(NullFormatter())
axes[0].set_title("Localisation on mutated residues", color=INK, fontsize=10, loc="left", pad=8)
axes[0].set_ylabel("KL enrichment (x)  [log]")
axes[0].legend(frameon=False, fontsize=7, ncol=2)
axes[1].axhline(1.0, color=INK2, lw=0.8, ls=":")
axes[1].set_title("Same curves, normalised to layer 0 (shape only)", color=INK, fontsize=10,
                  loc="left", pad=8)
axes[1].set_ylabel("enrichment / enrichment at L0")
for ax in axes:
    ax.set_xlabel("Pairformer layer (0-63)")
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(a.out, dpi=170); print(f"wrote {a.out}")
