"""Real per-variant alignments vs the grafted wild-type one.

Every experiment in this project grafts the wild type's alignment onto each
variant, which fixes the alignment as an exactly known variable. It also makes
`msa_prior` -- the homolog rows -- identical between the two runs being compared,
so injecting it CANNOT do anything and its 0.000 is true by construction rather
than by measurement.

This figure is the control for that. Same three assays, same protocol, both arms
capped to 512 rows so depth is not a second variable; the only difference is
whether each variant carries its own re-searched alignment.

    A  necessity and sufficiency per route, both arms
    B  the total internal effect -- grafting turns out to INFLATE it
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
GRAFT_C, REAL_C = "#8a8782", "#2a78d6"
ROUTES = ["z_direct", "msa_bcast", "msa_query", "msa_prior"]
NICE = {"z_direct": "z_direct\npair rep.", "msa_bcast": "msa_bcast\nMSA -> pair",
        "msa_query": "msa_query\nquery row", "msa_prior": "msa_prior\nhomolog rows"}


def load(pat):
    per = {r: [] for r in ROUTES}
    tot = []
    for f in sorted(glob.glob(pat)):
        for x in json.load(open(f)):
            tot.append(x["total_A"])
            for r in ROUTES:
                per[r].append((x[f"restore_{r}_necessity"],
                               x[f"inject_{r}_sufficiency"]))
    return per, np.array(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="runs/routes_realcap_*.json")
    ap.add_argument("--grafted", default="runs/routes_grafted512_*.json")
    ap.add_argument("--out", default="figures/realmsa_routes.png")
    args = ap.parse_args()

    real, t_real = load(args.real)
    graf, t_graf = load(args.grafted)

    fig = plt.figure(figsize=(13.2, 5.3))
    gs = fig.add_gridspec(1, 2, wspace=.26, left=.06, right=.985, top=.66,
                          bottom=.15, width_ratios=[2, 1])

    # ---- A: routes, both arms ---------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ROUTES))
    w = 0.20
    for k, (lab, d, c, off) in enumerate([
            ("grafted WT alignment", graf, GRAFT_C, -1.5),
            ("real per-variant alignment", real, REAL_C, 0.5)]):
        nec = [np.median([v[0] for v in d[r]]) for r in ROUTES]
        suf = [np.median([v[1] for v in d[r]]) for r in ROUTES]
        ax.bar(x + off * w, nec, w, color=c, alpha=.55, zorder=3,
               label=f"{lab} — necessity")
        ax.bar(x + (off + 1) * w, suf, w, color=c, zorder=3,
               label=f"{lab} — sufficiency")
        for xi, (a, b) in enumerate(zip(nec, suf)):
            ax.text(xi + off * w, a + .012, f"{a:.2f}", ha="center", va="bottom",
                    fontsize=6.8, color=INK)
            ax.text(xi + (off + 1) * w, b + .012, f"{b:.2f}", ha="center",
                    va="bottom", fontsize=6.8, color=INK)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[r] for r in ROUTES], fontsize=8.2)
    ax.set_ylabel("fraction of the mutation's effect")
    ax.set_ylim(-.12, 1.18)
    ax.legend(frameon=False, fontsize=7.6, ncol=2, loc="upper center")
    ax.annotate("", xy=(3 + 0.5 * w + w, np.median([v[1] for v in real["msa_prior"]])),
                xytext=(3 - 1.5 * w + w, 0.02),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))
    ax.text(3 - 0.1, .55, "0.000 by construction\n-> 0.44 when the\nalignment is real",
            fontsize=7.8, color="#c0392b", ha="right")
    ax.set_title("A   the homolog rows do carry the mutation", loc="left",
                 fontsize=10.5, fontweight="bold", color=INK, pad=20)
    ax.text(0, 1.02, "both arms capped at 512 rows, so depth is not the difference",
            transform=ax.transAxes, fontsize=8.1, color=INK2, va="bottom")

    # ---- B: total effect ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    parts = [t_graf, t_real]
    bp = ax.boxplot(parts, widths=.5, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.6), showfliers=False)
    for patch, c in zip(bp["boxes"], [GRAFT_C, REAL_C]):
        patch.set_facecolor(c); patch.set_alpha(.35)
    for i, (v, c) in enumerate(zip(parts, [GRAFT_C, REAL_C]), start=1):
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).normal(0, .05, len(v)),
                   v, s=14, color=c, alpha=.8, zorder=3, lw=0)
        ax.text(i + .30, np.median(v), f"{np.median(v):.4f}", va="center",
                ha="left", fontsize=8.6, color=INK, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"grafted\n(n={len(t_graf)})", f"real\n(n={len(t_real)})"],
                       fontsize=8.4)
    ax.set_ylabel("total internal effect  ||D(mut) - D(WT)||  (A)")
    ax.set_title("B   grafting inflates the effect", loc="left", fontsize=10.5,
                 fontweight="bold", color=INK, pad=20)
    ax.text(0, 1.02, f"median ratio real/grafted = "
                     f"{np.median(t_real)/np.median(t_graf):.2f}x",
            transform=ax.transAxes, fontsize=8.1, color=INK2, va="bottom")

    fig.text(.06, .945,
             "The grafted alignment is a control, and it has two costs worth stating",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(.06, .80,
             "Grafting makes every variant share one alignment body, so `msa_prior` is identical "
             "between the two runs and injecting it cannot do anything -- its 0.000 is arithmetic, "
             "not evidence.\nWith real per-variant alignments it reaches 0.440 sufficiency. "
             "z_direct still dominates (0.963), so the main route claim survives; \"the alignment "
             "routes carry nothing\" does not.\nSeparately, grafting sets a mutated query against "
             "wild-type homologs -- a conflict a re-searched alignment partly absorbs -- which "
             "inflates the measured internal effect about 2.6x.",
             fontsize=8.3, color=INK2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=190, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")
    for r in ROUTES:
        a = np.median([v[1] for v in real[r]]); b = np.median([v[1] for v in graf[r]])
        print(f"  {r:11s} sufficiency real {a:+.3f}  grafted {b:+.3f}")
    print(f"  total effect real {np.median(t_real):.4f} A  grafted {np.median(t_graf):.4f} A")


if __name__ == "__main__":
    main()
