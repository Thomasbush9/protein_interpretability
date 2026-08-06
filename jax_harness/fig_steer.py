"""Figure: perturbing the pair representation, and what the model does about it.

Three claims, and the first one is a correction of an earlier reading in this
same project, so it is drawn rather than described.

  A  How far the emitted structure moves, against the sampler's own drift.
     Injecting into ONE row leaves the structure below the noise floor;
     injecting everywhere moves it by angstroms. An earlier version of this
     experiment used only the one-row lever and concluded the structure module
     was insensitive to z. It is not -- the lever was too local.
  B  Odd versus even response. A direction the model uses as a signed quantity
     would broaden at +alpha and sharpen at -alpha, giving a large odd
     component. Every direction here, PC2 included, is dominated by the even
     part: the model responds to how big the perturbation is, not to which way
     it points.
  C  PC2 against random directions of identical norm, under the global lever.
     If PC2 were a privileged channel it would sit outside the random spread.
     It sits inside it.

Blue is PC2 throughout, grey the random controls, amber the drift floor.
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
C_PC2, C_PC1, C_PC3, C_RND, C_DRIFT = "#2a78d6", "#eb6834", "#1baf7a", "#8a8885", "#eda100"
MODE_LAB = {"row": "one row of z", "sym": "row + column", "glob": "every pair"}
MODE_LS = {"row": ":", "sym": "--", "glob": "-"}

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--summary", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

d = np.load(a.run, allow_pickle=True)
S = json.load(open(a.summary))
dirs = np.array([str(x) for x in d["rec_dir"]])
modes = np.array([str(x) for x in d["rec_mode"]])
alpha, ca = d["alpha"], d["ca_rmsd"]
drift = S["drift"]["ca_rmsd"]
names = sorted(set(dirs))
rnd = [n for n in names if n.startswith("random")]

fig = plt.figure(figsize=(14.4, 4.6))
gs = fig.add_gridspec(1, 3, wspace=0.32)

# --- A: does the structure move at all? ------------------------------------
axA = fig.add_subplot(gs[0, 0])
mags = sorted({abs(x) for x in alpha if x != 0})
for mode in ("row", "sym", "glob"):
    if mode not in set(modes):
        continue
    y = [np.median(ca[(modes == mode) & (np.abs(alpha) == m)]) for m in mags]
    axA.plot(mags, y, color=C_PC2 if mode == "glob" else C_RND,
             lw=2.4 if mode == "glob" else 1.6, ls=MODE_LS[mode], zorder=4,
             marker="o", ms=4, label=f"inject into {MODE_LAB[mode]}")
axA.axhline(drift, color=C_DRIFT, lw=1.8, ls="--", zorder=3)
axA.annotate(f"sampler's own drift ({drift:.2f} Å)\n(same z, different diffusion key)",
             xy=(mags[0], drift), xytext=(2, 8), textcoords="offset points",
             fontsize=7.4, color=C_DRIFT)
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xticks(mags); axA.set_xticklabels([str(int(m)) for m in mags])
axA.set_xlabel(r"perturbation size $|\alpha|$  (1 = a typical mutation's $\|\Delta z\|$)")
axA.set_ylabel("superposed CA RMSD (Å)")
axA.legend(frameon=False, fontsize=7.6, loc="upper left")
axA.set_title("A  The structure module is not deaf to z —\n     but one row is "
              "too small a lever", color=INK, fontsize=9.5, loc="left", pad=8)
axA.grid(True, color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)

# --- B: signed or magnitude-driven? ----------------------------------------
axB = fig.add_subplot(gs[0, 1])
KEYS = [("d_sd_site", "distogram\nwidth"), ("d_plddt_site", "pLDDT at\nthe site"),
        ("ca_rmsd", "CA RMSD")]
w, x = 0.26, np.arange(len(KEYS))
for i, (nm, col) in enumerate((("PC2", C_PC2), ("PC1", C_PC1), ("PC3", C_PC3))):
    vals = [abs(S["glob"][k][nm]["ratio"]) for k, _ in KEYS if nm in S["glob"][k]]
    axB.bar(x + (i - 1) * w, vals, w * 0.9, color=col, alpha=0.9, zorder=3, label=nm)
for j, (k, _) in enumerate(KEYS):
    rr = [abs(S["glob"][k][n]["ratio"]) for n in rnd if n in S["glob"][k]]
    axB.plot([x[j] - 1.6 * w, x[j] + 1.6 * w], [max(rr)] * 2, color=INK, lw=1.2,
             ls="--", zorder=6)
axB.annotate("dashed = largest of the random directions", xy=(0, 0),
             xycoords="axes fraction", xytext=(2, -38), textcoords="offset points",
             fontsize=7.2, color=INK2)
axB.set_xticks(x); axB.set_xticklabels([l for _, l in KEYS], fontsize=8)
axB.set_ylabel("|odd| / |even| response")
axB.legend(frameon=False, fontsize=7.6, ncol=3, loc="upper right")
axB.set_title(r"B  Odd part is $\leq$7% of even for every" "\n     direction — "
              "response tracks size, not sign",
              color=INK, fontsize=9.5, loc="left", pad=8)
axB.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)

# --- C: is PC2 privileged? -------------------------------------------------
axC = fig.add_subplot(gs[0, 2])
rows = [("d_sd_site", "distogram width"), ("d_plddt_site", "pLDDT at the site"),
        ("ca_rmsd", "CA RMSD")]
for i, (k, lab) in enumerate(rows):
    g = S["glob"][k]
    rv = [abs(g[n]["even_mean"]) for n in rnd if n in g]
    axC.plot([min(rv), max(rv)], [i, i], color=C_RND, lw=7, alpha=0.4, zorder=3,
             solid_capstyle="round")
    axC.scatter(rv, [i] * len(rv), s=22, color=C_RND, zorder=4)
    axC.scatter([abs(g["PC2"]["even_mean"])], [i], s=92, color=C_PC2, zorder=6,
                edgecolor=SURF, linewidth=1.8)
    axC.set_yticks(range(len(rows)))
axC.set_yticklabels([l for _, l in rows], fontsize=8.5)
axC.set_xscale("log")
axC.set_xlabel("magnitude of response, global injection")
axC.annotate("PC2", xy=(abs(S["glob"]["ca_rmsd"]["PC2"]["even_mean"]), 2),
             xytext=(0, 13), textcoords="offset points", fontsize=8,
             color=C_PC2, ha="center")
axC.annotate("random directions,\nidentical norm", xy=(0.02, 0.06),
             xycoords="axes fraction", fontsize=7.4, color=C_RND)
axC.set_title("C  PC2 moves nothing more than a\n     random direction of equal norm",
              color=INK, fontsize=9.5, loc="left", pad=8)
axC.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axC.set_axisbelow(True)
axC.margins(y=0.28)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
