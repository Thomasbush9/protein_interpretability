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
B_C, O_C = "#2a78d6", "#eb6834"
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
    wt = d["gfp_wt__ca"].astype(float)
    out = {}
    for cid, _ in COHORTS:
        tm, rmsd = geom.tm_and_rmsd(d[f"{cid}__ca"].astype(float), wt)
        out[cid] = dict(kl=float(d[f"{cid}__kl"].mean()), tm=tm, rmsd=rmsd,
                        plddt=float(d[f"{cid}__plddt"].mean()))
    out["wt_plddt"] = float(d["gfp_wt__plddt"].mean())
    return out


def bars(ax, B, O, key, ylabel, letter, head, sub, log=False):
    x = np.arange(len(COHORTS))
    w = 0.36
    b = [B[c][key] for c, _ in COHORTS]
    o = [O[c][key] for c, _ in COHORTS]
    ax.bar(x - w / 2, b, w, color=B_C, label="Boltz-2", zorder=3)
    ax.bar(x + w / 2, o, w, color=O_C, label="OpenFold3", zorder=3)
    for xi, (bv, ov) in enumerate(zip(b, o)):
        ax.text(xi - w / 2, bv, f"{bv:.2f}", ha="center", va="bottom",
                fontsize=7.6, color=INK)
        ax.text(xi + w / 2, ov, f"{ov:.2f}", ha="center", va="bottom",
                fontsize=7.6, color=INK)
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
    ap.add_argument("--boltz", default="runs/distomap_gfp.npz")
    ap.add_argument("--of3", default="runs/distomap_of3_gfp.npz")
    ap.add_argument("--out", default="figures_of3/crossmodel_gfp.png")
    args = ap.parse_args()

    B, O = load(args.boltz), load(args.of3)

    fig = plt.figure(figsize=(14.2, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=.44, wspace=.24,
                          left=.06, right=.98, top=.795, bottom=.065)

    bars(fig.add_subplot(gs[0, 0]), B, O, "kl",
         "mean symmetric KL vs wild type  (nats)", "A",
         "how far the trunk's belief moves",
         "log scale; higher = the model's pairwise beliefs changed more", log=True)

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(COHORTS)); w = 0.36
    b = [1 - B[c]["tm"] for c, _ in COHORTS]
    o = [1 - O[c]["tm"] for c, _ in COHORTS]
    ax.bar(x - w / 2, b, w, color=B_C, label="Boltz-2", zorder=3)
    ax.bar(x + w / 2, o, w, color=O_C, label="OpenFold3", zorder=3)
    for xi, (bv, ov) in enumerate(zip(b, o)):
        ax.text(xi - w / 2, bv, f"TM {1-bv:.2f}", ha="center", va="bottom",
                fontsize=7.6, color=INK)
        ax.text(xi + w / 2, ov, f"TM {1-ov:.2f}", ha="center", va="bottom",
                fontsize=7.6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in COHORTS], fontsize=8.2)
    ax.set_ylabel("1 - TM to wild type")
    ax.legend(frameon=False, fontsize=8.2)
    title(ax, "B", "how far the structure moves",
          "the scramble control shows both models CAN move it")

    # ---- C: belief vs structure -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for D, c, nm, dy in ((B, B_C, "Boltz-2", 11), (O, O_C, "OpenFold3", -15)):
        xs = [D[cid]["kl"] for cid, _ in COHORTS]
        ys = [1 - D[cid]["tm"] for cid, _ in COHORTS]
        ax.plot(xs, ys, "-o", color=c, lw=1.6, ms=8, label=nm, zorder=3)
        for (cid, nice), xi, yi in zip(COHORTS, xs, ys):
            ax.annotate(nice.replace("\n", " "), (xi, yi), textcoords="offset points",
                        xytext=(-6, dy), fontsize=7.4, color=c, ha="right")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.set_xlabel("belief moved  (mean symmetric KL, nats)")
    ax.set_ylabel("structure moved  (1 - TM)")
    ax.set_ylim(-.09, .92)
    lo = min(min(B[c]["kl"] for c, _ in COHORTS), min(O[c]["kl"] for c, _ in COHORTS))
    hi = max(max(B[c]["kl"] for c, _ in COHORTS), max(O[c]["kl"] for c, _ in COHORTS))
    ax.set_xlim(lo * 0.45, hi * 1.6)   # room for the leftmost label
    ax.legend(frameon=False, fontsize=8.2)
    title(ax, "C", "the claim, in one panel",
          "mutations move the belief without moving the structure; scrambling moves both")

    bars(fig.add_subplot(gs[1, 1]), B, O, "plddt", "mean pLDDT", "D",
         "confidence",
         f"wild type: Boltz-2 {B['wt_plddt']:.3f}, OpenFold3 {O['wt_plddt']:.3f}")

    kb = B["gfp_core_32"]["kl"] / B["gfp_surface_32"]["kl"]
    ko = O["gfp_core_32"]["kl"] / O["gfp_surface_32"]["kl"]
    fig.text(.06, .962,
             "The phenomenon is not specific to Boltz-2: OpenFold3 does the same thing",
             fontsize=13.5, fontweight="bold", color=INK)
    fig.text(.06, .868,
             f"GFP, N=238, identical sequences and alignments, no templates in either model, "
             f"both with a 64-bin distogram over 2-22 A (checked, not assumed).\n"
             f"32 buried mutations move the trunk's beliefs by {B['gfp_core_32']['kl']:.2f} nats "
             f"(Boltz-2) and {O['gfp_core_32']['kl']:.2f} nats (OpenFold3) while the predicted "
             f"structures stay at TM {B['gfp_core_32']['tm']:.3f} and {O['gfp_core_32']['tm']:.3f}. "
             f"The core:surface ratio of belief change is {kb:.2f} vs {ko:.2f}.",
             fontsize=8.5, color=INK2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=190, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")
    print(f"  core:surface KL ratio  Boltz-2 {kb:.3f}   OpenFold3 {ko:.3f}")
    for cid, nice in COHORTS:
        print(f"  {cid:16s} KL {B[cid]['kl']:6.3f}/{O[cid]['kl']:6.3f}  "
              f"TM {B[cid]['tm']:.3f}/{O[cid]['tm']:.3f}  "
              f"pLDDT {B[cid]['plddt']:.3f}/{O[cid]['plddt']:.3f}   (Boltz-2/OF3)")


if __name__ == "__main__":
    main()
