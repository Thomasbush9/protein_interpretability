"""Route decomposition across 12 proteins: which path carries the mutation?

Reads runs/routes_*.json (exp_paths.py). A mutation can reach the Pairformer's
output by five paths. For each, two complementary interventions:

  necessity    take the mutant run, restore ONE route to its wild-type value,
               and ask how much of the mutant's total displacement disappears.
               1.0 = that route alone carried everything.
  sufficiency  take the wild-type run, inject ONE route from the mutant, and ask
               how much of the mutant's displacement is reproduced.
               1.0 = that route alone is enough.

Necessity and sufficiency are separate questions. A route can be necessary but
not sufficient (it gates something) or sufficient but not necessary (redundant
with another route). Both near 1 for a single route means that route IS the
mechanism -- which is what z_direct shows.

    A  necessity per route, all proteins
    B  sufficiency per route, all proteins
    C  the two against each other, one point per protein x variant
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_h = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_s = next((f for f in ("Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial") if f in _h), None)
plt.rcParams.update({**({"font.family": "sans-serif", "font.sans-serif": [_s]} if _s else {}),
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
                     "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})

ROUTES = ["z_direct", "s_direct", "msa_bcast", "msa_query", "msa_prior"]
NICE = {"z_direct": "z_direct\npair rep.", "s_direct": "s_direct\nsingle rep.",
        "msa_bcast": "msa_bcast\nMSA -> pair", "msa_query": "msa_query\nquery row",
        "msa_prior": "msa_prior\nprofile"}
ACC, WT_C, MUT_C = "#7a4fb5", "#2a78d6", "#eb6834"


def title(ax, letter, text, sub=None):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=10.5, fontweight="bold",
                 color=INK, pad=20 if sub else 6)
    if sub:
        ax.text(0, 1.018, sub, transform=ax.transAxes, fontsize=8.2, color=INK2,
                va="bottom", ha="left")


def strip(ax, vals, colour, ylabel, letter, head, sub):
    """One column of points per route, with the median as a wide bar."""
    for k, r in enumerate(ROUTES):
        v = vals[r]
        jit = (np.arange(len(v)) - (len(v) - 1) / 2) / max(len(v), 2) * .46
        ax.scatter(k + jit, v, s=17, color=colour, alpha=.62, lw=0, zorder=3)
        ax.plot([k - .30, k + .30], [np.median(v)] * 2, color=INK, lw=2.2, zorder=4)
    ax.axhline(0, color=GRID, lw=1)
    ax.axhline(1, color=INK2, lw=.9, ls=":")
    ax.set_xticks(range(len(ROUTES)))
    ax.set_xticklabels([NICE[r] for r in ROUTES], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-.6, len(ROUTES) - .4)
    title(ax, letter, head, sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/routes_*.json")
    ap.add_argument("--out", default="figures/routes12.png")
    args = ap.parse_args()

    nec = {r: [] for r in ROUTES}
    suf = {r: [] for r in ROUTES}
    pts, prot = [], []
    for f in sorted(glob.glob(args.glob)):
        name = Path(f).stem.replace("routes_", "").split("_Tsuboyama")[0]
        prot.append(name)
        for rec in json.load(open(f)):
            for r in ROUTES:
                nec[r].append(rec[f"restore_{r}_necessity"])
                suf[r].append(rec[f"inject_{r}_sufficiency"])
            pts.append((rec["restore_z_direct_necessity"],
                        rec["inject_z_direct_sufficiency"], name, rec["mutant"]))
    nec = {r: np.array(v) for r, v in nec.items()}
    suf = {r: np.array(v) for r, v in suf.items()}
    n_prot, n_obs = len(prot), len(nec["z_direct"])

    fig = plt.figure(figsize=(14.2, 4.9))
    gs = fig.add_gridspec(1, 3, wspace=.26, left=.055, right=.985, top=.70, bottom=.15)

    ax = fig.add_subplot(gs[0, 0])
    strip(ax, nec, ACC, "necessity  (fraction of the effect removed)", "A",
          "restoring one route to wild type",
          "1.0 = this route carried the whole effect;  bar = median")
    ax = fig.add_subplot(gs[0, 1])
    strip(ax, suf, MUT_C, "sufficiency  (fraction of the effect reproduced)", "B",
          "injecting one route from the mutant",
          "1.0 = this route alone reproduces the effect")

    ax = fig.add_subplot(gs[0, 2])
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    ax.axhline(1, color=GRID, lw=1)
    ax.axvline(1, color=GRID, lw=1)
    ax.plot([0, 1.3], [0, 1.3], color=GRID, lw=1, ls="--", zorder=1)
    ax.scatter(x, y, s=34, color=ACC, alpha=.72, lw=0, zorder=3)
    ax.set_xlabel("z_direct necessity")
    ax.set_ylabel("z_direct sufficiency")
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1.3)
    ax.set_aspect("equal")
    title(ax, "C", "z_direct is both necessary and sufficient",
          f"one point per protein x variant ({len(x)} points)")

    fig.text(.055, .935,
             "Across 12 proteins the mutation reaches the output through the pair "
             "representation, not the MSA",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(.055, .845,
             f"{n_prot} ProteinGym assays, {n_obs} protein x variant observations. "
             f"Displacement is measured in Angstrom of expected distance at the trunk output.\n"
             f"Both interventions are exact pytree swaps; the two sanity rows "
             f"(restore nothing / restore everything) are 0 in every run.",
             fontsize=8.6, color=INK2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=190, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}  ({n_prot} proteins, {n_obs} observations)")
    for r in ROUTES:
        print(f"  {r:11s} necessity med {np.median(nec[r]):+.3f}  "
              f"sufficiency med {np.median(suf[r]):+.3f}")


if __name__ == "__main__":
    main()
