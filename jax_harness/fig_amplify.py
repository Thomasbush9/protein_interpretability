"""Amplification figure: the gain fix works, but almost entirely for the wrong reason.

Reads runs/amp_*.npz. The apparent headline -- rho(TM to WT, dG) rising 0.19 ->
0.51 as the mutant-minus-wild-type difference is scaled 8x -- does not survive
its own control. A difference vector borrowed from a DIFFERENT variant and
rescaled to the same norm reaches 0.40, i.e. ~78 % of the effect. What
amplification mainly does is convert the MAGNITUDE of the trunk's response into
structural displacement; the DIRECTION the trunk implies adds +0.11, whose 95 %
CI includes zero.

    A  rho(TM, dG) vs gamma, true vs norm-matched control
    B  what it costs: mean TM to wild type and pLDDT vs gamma
    C  the readouts ranked -- including ||dz|| straight off the trunk, which
       beats every decoded structure here and needs no sampling at all
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402
from analyze_amplify import gap_bootstrap  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_h = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_s = next((f for f in ("Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial") if f in _h), None)
plt.rcParams.update({**({"font.family": "sans-serif", "font.sans-serif": [_s]} if _s else {}),
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
                     "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
TRUE_C, PERM_C, TRUNK_C = "#eb6834", "#8a8782", "#2a78d6"


def title(ax, letter, text, sub=None):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=10.5, fontweight="bold",
                 color=INK, pad=20 if sub else 6)
    if sub:
        ax.text(0, 1.018, sub, transform=ax.transAxes, fontsize=8.2, color=INK2,
                va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/amp_*.npz")
    ap.add_argument("--out", default="figures/amplify.png")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    per = []
    for f in files:
        d = np.load(f)
        sc, ca, pl, caw = d["score"], d["ca"], d["plddt"], d["ca_wt"].astype(float)
        ki, gm = [str(x) for x in d["cond_kind"]], d["cond_gamma"]
        rec = {"name": str(d["assay"]).split("_Tsu")[0],
               "direct": abs(spearmanr(d["dz_norm"], sc).correlation), "t": {}, "p": {},
               "tm": {}, "pl": {}}
        for c in range(ca.shape[1]):
            tm = np.array([geom.tm_score(ca[i, c].astype(float), caw) for i in range(len(sc))])
            rec[("t" if ki[c] == "true" else "p")][float(gm[c])] = \
                spearmanr(tm, sc).correlation
            if ki[c] == "true":
                rec["tm"][float(gm[c])] = tm.mean()
                rec["pl"][float(gm[c])] = pl[:, c].mean()
        per.append(rec)

    gam = sorted(per[0]["t"])
    cg = sorted(per[0]["p"])
    tmean = [np.mean([r["t"][g] for r in per]) for g in gam]
    pmean = [np.mean([r["p"][g] for r in per]) for g in cg]
    gapb = gap_bootstrap(files)

    fig = plt.figure(figsize=(14.2, 5.0))
    gs = fig.add_gridspec(1, 3, wspace=.34, left=.055, right=.985, top=.66, bottom=.135)

    # ---- A: the headline and its control -----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for r in per:
        ax.plot(gam, [r["t"][g] for g in gam], color=TRUE_C, lw=.9, alpha=.35, zorder=2)
        ax.plot(cg, [r["p"][g] for g in cg], color=PERM_C, lw=.9, alpha=.35,
                ls="--", zorder=2)
    ax.plot(gam, tmean, color=TRUE_C, lw=2.4, marker="o", ms=6, zorder=4,
            label="true difference")
    ax.plot(cg, pmean, color=PERM_C, lw=2.4, marker="s", ms=6, ls="--", zorder=4,
            label="another variant's difference,\nrescaled to the same norm")
    ax.axhline(0, color=GRID, lw=1)
    ax.axhline(np.mean([r["direct"] for r in per]), color=TRUNK_C, lw=1.5, ls=":")
    ax.text(8, np.mean([r["direct"] for r in per]) - .052,
            "||dz|| off the trunk, no decoding", fontsize=7.8, color=TRUNK_C, ha="right")
    ax.annotate("", xy=(8, tmean[-1]), xytext=(8, pmean[-1]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text(7.6, (tmean[-1] + pmean[-1]) / 2, f"gap\n{gapb.mean():+.2f}", fontsize=8,
            color=INK, ha="right", va="center")
    ax.set_xlabel("gamma  (scaling of the mutant-minus-wild-type difference)")
    ax.set_ylabel("rho(TM to wild type, measured dG)")
    ax.set_xticks(gam)
    ax.set_ylim(-.22, .80)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    title(ax, "A", "amplification raises the correlation...",
          "...and so does a difference with the wrong direction")

    # ---- B: what it costs ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for r in per:
        ax.plot(gam, [r["tm"][g] for g in gam], color=TRUE_C, lw=1.4, marker="o", ms=4)
    ax.set_xlabel("gamma")
    ax.set_ylabel("mean TM to wild type")
    ax.set_xticks(gam)
    ax.set_ylim(.84, 1.0)
    ax2 = ax.twinx()
    for r in per:
        ax2.plot(gam, [r["pl"][g] for g in gam], color=TRUNK_C, lw=1.2, ls="--",
                 marker="^", ms=4)
    ax2.set_ylabel("mean pLDDT", color=TRUNK_C)
    ax2.tick_params(axis="y", colors=TRUNK_C)
    ax2.spines["top"].set_visible(False)
    ax.plot([], [], color=TRUE_C, marker="o", label="TM to WT (left)")
    ax.plot([], [], color=TRUNK_C, ls="--", marker="^", label="pLDDT (right)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    title(ax, "B", "the cost of getting there",
          "at gamma=8 the structure is 2 A off and confidence has dropped")

    # ---- C: the readouts ranked --------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    labs = ["TM to WT\n(ordinary, gamma=1)", "TM to WT\nperm control, gamma=8",
            "TM to WT\namplified, gamma=8", "||dz|| off the trunk\n(no decoding)"]
    vals = [tmean[gam.index(1.0)], pmean[-1], tmean[-1],
            np.mean([r["direct"] for r in per])]
    cols = [TRUE_C, PERM_C, TRUE_C, TRUNK_C]
    y = np.arange(len(labs))
    ax.barh(y, vals, color=cols, height=.40, zorder=3)
    for k, (v, lb) in enumerate(zip(vals, labs)):
        ax.text(0.006, k - .34, lb.replace("\n", "  "), va="bottom", ha="left",
                fontsize=8.2, color=INK)
        ax.text(v + .012, k, f"{v:+.3f}", va="center", fontsize=9,
                color=INK, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(len(labs) - .5, -.75)
    ax.set_xlim(0, .80)
    ax.set_xlabel("rho with measured dG")
    ax.axvline(0.548, color=INK2, lw=1.1, ls=":", zorder=1)
    ax.text(0.792, 1.5, "Pairformer probe 0.548 (held-out)", fontsize=7.4,
            color=INK2, ha="right", va="center", rotation=90)
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    title(ax, "C", "amplification is not the best readout",
          "the trunk's own perturbation norm beats every decoded structure")

    fig.text(.055, .950,
             "Scaling the mutation-specific difference does make the structure track "
             "stability -- but mostly by turning perturbation SIZE into displacement",
             fontsize=12.5, fontweight="bold", color=INK)
    fig.text(.055, .835,
             f"3 assays x 60 variants, diffusion key held fixed across gamma. The control gives "
             f"variant i the difference vector of variant j rescaled to ||dz_i||, so it carries the "
             f"right magnitude and the wrong direction; it reaches {pmean[-1]:+.3f} against the true "
             f"{tmean[-1]:+.3f}.\nThe direction-specific gap is {gapb.mean():+.3f}, 95 % CI "
             f"[{np.percentile(gapb, 2.5):+.3f}, {np.percentile(gapb, 97.5):+.3f}] -- it includes "
             f"zero, so a direction-specific effect is NOT established. gamma=0 (mutant atoms, "
             f"wild-type trunk state) gives {tmean[0]:+.3f}, as it should.",
             fontsize=8.4, color=INK2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=190, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")
    print(f"  true  {dict(zip(gam, np.round(tmean, 3)))}")
    print(f"  perm  {dict(zip(cg, np.round(pmean, 3)))}")
    print(f"  gap   {gapb.mean():+.3f} CI [{np.percentile(gapb,2.5):+.3f}, "
          f"{np.percentile(gapb,97.5):+.3f}]")


if __name__ == "__main__":
    main()
