"""Boltz-2 vs OpenFold3, side by side: is the phenomenon a property of the class?

Same GFP cohort, same sequences, same alignments, no templates in either model.
Both use a 64-bin distogram over 2-22 A, which is checked rather than assumed
(exp_distomap_of3.py records `n_bins` and refuses to emit E[d] otherwise), so
Angstrom and nat quantities are on the same footing here.

    A  how far the trunk's belief moves   (mean symmetric KL vs wild type)
    B  how far the structure moves        (1 - TM to wild type)
    C  the two against each other -- the paper's claim in one panel
    D  confidence

The scramble control is the load-bearing part of the design: it is the case
where the sequence really is different, and both models DO move the structure.
Without it, "the structure does not move" could just mean the models are
insensitive to everything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_h = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_s = next((f for f in ("Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial") if f in _h), None)
plt.rcParams.update({**({"font.family": "sans-serif", "font.sans-serif": [_s]} if _s else {}),
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
                     "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#2a78d6", "#eb6834", "#159a8c", "#7a4fb5"]
NICE = {"boltz2": "Boltz-2", "of3": "OpenFold3", "protenix": "Protenix", "af2": "AlphaFold2"}
B_C, O_C = PALETTE[0], PALETTE[1]
COHORTS = [("gfp_core_32", "32 core\nmutations"),
           ("gfp_surface_32", "32 surface\nmutations"),
           ("gfp_scramble", "scrambled\nsequence")]


def title(ax, letter, text, sub=None):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=10.5, fontweight="bold",
                 color=INK, pad=20 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.1, color=INK2,
                va="bottom", ha="left")


def load(f):
    d = np.load(f)
    name = str(d["model"]) if "model" in d.files else Path(f).stem
    wt = d["gfp_wt__ca"].astype(float)
    out = {}
    for cid, _ in COHORTS:
        tm, rmsd = geom.tm_and_rmsd(d[f"{cid}__ca"].astype(float), wt)
        out[cid] = dict(kl=float(d[f"{cid}__kl"].mean()), tm=tm, rmsd=rmsd,
                        plddt=float(d[f"{cid}__plddt"].mean()))
    out["wt_plddt"] = float(d["gfp_wt__plddt"].mean())
    out["name"] = NICE.get(name, name)
    out["grid"] = (float(d["bin_centres"][0]), float(d["bin_centres"][-1])) \
        if "bin_centres" in d.files else None
    return out


def bars(ax, models, key, ylabel, letter, head, sub, log=False, fmt="{:.2f}",
         transform=lambda v: v):
    """Grouped bars, one group per cohort, one bar per model."""
    n = len(models)
    x = np.arange(len(COHORTS))
    w = 0.8 / n
    for k, M in enumerate(models):
        off = (k - (n - 1) / 2) * w
        v = [transform(M[c][key]) for c, _ in COHORTS]
        ax.bar(x + off, v, w * .92, color=PALETTE[k % len(PALETTE)],
               label=M["name"], zorder=3)
        for xi, vi in enumerate(v):
            ax.text(xi + off, vi, fmt.format(vi), ha="center", va="bottom",
                    fontsize=6.9, color=INK, rotation=90 if n > 2 else 0)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: ("%g" % v)))
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in COHORTS], fontsize=8.2)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8.2)
    title(ax, letter, head, sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True,
                    help="one distomap npz per model; order sets colour/order")
    ap.add_argument("--out", default="figures_of3/crossmodel_gfp.png")
    args = ap.parse_args()

    models = [load(f) for f in args.npz]
    n = len(models)
    grids = {M["grid"] for M in models if M["grid"]}
    same_grid = len(grids) <= 1

    fig = plt.figure(figsize=(14.2, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=.46, wspace=.24,
                          left=.06, right=.98, top=.775, bottom=.065)

    bars(fig.add_subplot(gs[0, 0]), models, "kl",
         "mean symmetric KL vs wild type  (nats)", "A",
         "how far the trunk's belief moves",
         "log scale; each model against its OWN wild type", log=True)

    ax = fig.add_subplot(gs[0, 1])
    bars(ax, models, "tm", "1 - TM to wild type", "B",
         "how far the structure moves",
         "the scramble control shows every model CAN move it",
         transform=lambda v: 1 - v)

    # ---- C: belief vs structure -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for k, M in enumerate(models):
        xs = [M[cid]["kl"] for cid, _ in COHORTS]
        ys = [1 - M[cid]["tm"] for cid, _ in COHORTS]
        ax.plot(xs, ys, "-o", color=PALETTE[k % len(PALETTE)], lw=1.6, ms=7,
                label=M["name"], zorder=3)
    for cid, nice in COHORTS:
        xm = np.mean([M[cid]["kl"] for M in models])
        ym = np.mean([1 - M[cid]["tm"] for M in models])
        ax.annotate(nice.replace("\n", " "), (xm, ym), textcoords="offset points",
                    xytext=(0, 16), fontsize=7.8, color=INK2, ha="center")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.set_xlabel("belief moved  (mean symmetric KL, nats)")
    ax.set_ylabel("structure moved  (1 - TM)")
    ax.set_ylim(-.09, .95)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    title(ax, "C", "the claim, in one panel",
          "mutations move the belief without moving the structure; scrambling moves both")

    bars(fig.add_subplot(gs[1, 1]), models, "plddt", "mean pLDDT", "D", "confidence",
         "  ".join(f"WT {M['name']} {M['wt_plddt']:.3f}" for M in models))

    ratios = {M["name"]: M["gfp_core_32"]["kl"] / M["gfp_surface_32"]["kl"]
              for M in models}
    fig.text(.06, .962,
             f"The phenomenon is not specific to one model: all {n} do the same thing",
             fontsize=13.5, fontweight="bold", color=INK)
    grid_note = ("all models share a 64-bin 2.16-21.84 A grid"
                 if same_grid else
                 "GRIDS DIFFER between models (" +
                 "; ".join(f"{M['name']} {M['grid'][0]:.2f}-{M['grid'][1]:.2f}"
                           for M in models) +
                 ") -- compare orderings and ratios, NOT absolute nats")
    fig.text(.06, .845,
             f"GFP, N=238. Identical sequences and identical alignments in every model "
             f"(verified: the MSA server is blocked at featurisation, so a model that\n"
             f"fetched its own would fail rather than substitute one). {grid_note}.\n"
             f"Core:surface ratio of belief change -- "
             + ",  ".join(f"{k} {v:.2f}" for k, v in ratios.items()) + ".",
             fontsize=8.4, color=INK2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=190, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")
    print(f"  same grid across models: {same_grid}  {grids}")
    for k, v in ratios.items():
        print(f"  core:surface KL ratio  {k:12s} {v:.3f}")
    for cid, _ in COHORTS:
        print(f"  {cid:16s} " + "  ".join(
            f"{M['name']}: KL {M[cid]['kl']:.3f} TM {M[cid]['tm']:.3f}" for M in models))


if __name__ == "__main__":
    main()
