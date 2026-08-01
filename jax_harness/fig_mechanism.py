"""The mechanism figure: the distogram moves, the structure does not.

Reads runs/distomap_gfp.npz (produced by exp_distomap.py) and draws, for the
wild type and the 32-core-mutation cohort of GFP:

    A  WT expected-distance map          E[d] over all residue pairs
    B  mutant expected-distance map      same, same colour scale
    C  difference map                    E[d](mut) - E[d](WT), diverging
    D  symmetric-KL map                  what the distribution did, not the mean
    E  structure overlay                 TM-aligned CA traces, WT vs mutant
    F  one pair's distogram              the histogram behind a single cell of D

Panels A-D are the trunk's belief. Panel E is what the sampler produced from
that belief. The point of the figure is that C/D are large and E is not.

Run on a login node -- no jax, no model, just the saved arrays.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

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
WT_C, MUT_C = "#2a78d6", "#eb6834"


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _site_ticks(ax, sites, N):
    """Mark mutated positions just outside both axes, without gridding the map."""
    off = N * .012
    for s in sites:
        ax.plot([s, s], [-off, -off * 2.6], color=INK, lw=.8, clip_on=False, zorder=6)
        ax.plot([-off, -off * 2.6], [s, s], color=INK, lw=.8, clip_on=False, zorder=6)


def title(ax, letter, text, sub=None):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=10.5, fontweight="bold",
                 color=INK, pad=21 if sub else 6)
    if sub:
        ax.text(0, 1.018, sub, transform=ax.transAxes, fontsize=8.2, color=INK2,
                va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="runs/distomap_gfp.npz")
    ap.add_argument("--mut", default="gfp_core_32")
    ap.add_argument("--out", default="figures/mechanism.png")
    args = ap.parse_args()

    d = np.load(args.npz)
    wt, mt = "gfp_wt", args.mut
    ed_w, ed_m = d[f"{wt}__ed"], d[f"{mt}__ed"]
    kl = d[f"{mt}__kl"]
    ca_w, ca_m = d[f"{wt}__ca"].astype(float), d[f"{mt}__ca"].astype(float)
    centres = d["bin_centres"]
    muts = {i: m for i, m in zip(d["ids"], d["mutations"])}
    sites = sorted(int(t[1:-1]) - 1 for t in str(muts[mt]).split(";") if t)
    N = ed_w.shape[0]

    tm, rmsd = geom.tm_and_rmsd(ca_m, ca_w)
    dif = ed_m - ed_w

    fig = plt.figure(figsize=(14.4, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=.40, wspace=.30,
                          left=.05, right=.965, top=.865, bottom=.065)

    # ---- A / B: the two belief maps, on one shared colour scale ------------
    vmax = float(max(ed_w.max(), ed_m.max()))
    for k, (ed, name, lab) in enumerate([(ed_w, "wild type", "A"),
                                         (ed_m, "32 core mutations", "B")]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(ed, cmap="viridis_r", vmin=2, vmax=vmax, origin="lower",
                       interpolation="nearest")
        title(ax, lab, f"predicted distance map, {name}",
              "E[d] per residue pair, Angstrom")
        ax.set_xlabel("residue j")
        ax.set_ylabel("residue i")
        cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
        cb.set_label("E[d]  (A)", fontsize=8)
        cb.outline.set_visible(False)

    # ---- C: difference ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    lim = float(np.abs(dif).max())
    im = ax.imshow(dif, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim),
                   origin="lower", interpolation="nearest")
    title(ax, "C", "difference,  mutant minus wild type",
          f"red = pair pushed apart;  max |change| {lim:.2f} A;  ticks = mutated sites")
    ax.set_xlabel("residue j")
    ax.set_ylabel("residue i")
    _site_ticks(ax, sites, N)
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    cb.set_label("delta E[d]  (A)", fontsize=8)
    cb.outline.set_visible(False)

    # ---- D: symmetric KL ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(kl, cmap="magma_r", vmin=0, vmax=float(np.percentile(kl, 99.5)),
                   origin="lower", interpolation="nearest")
    title(ax, "D", "symmetric KL, mutant vs wild type",
          f"mean {kl.mean():.3f} nats;  ticks = mutated sites")
    ax.set_xlabel("residue j")
    ax.set_ylabel("residue i")
    _site_ticks(ax, sites, N)
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    cb.set_label("sym. KL  (nats)", fontsize=8)
    cb.outline.set_visible(False)

    # ---- E: structure overlay ----------------------------------------------
    # mpl_toolkits is absent from the analysis venv, so the two chains are drawn
    # as orthogonal projections onto the wild type's own principal axes: view 1
    # is PC1 x PC2, view 2 is PC1 x PC3, i.e. the same structure rotated 90 deg.
    r = geom.tm_align_result(ca_m, ca_w)
    ca_m_al = ca_m @ np.asarray(r.u).T + np.asarray(r.t)
    ctr = ca_w.mean(0)
    axes3 = np.linalg.svd(ca_w - ctr, full_matrices=False)[2]
    pw3, pm3 = (ca_w - ctr) @ axes3.T, (ca_m_al - ctr) @ axes3.T
    span = np.abs(np.concatenate([pw3, pm3])).max() * 1.06

    sub = gs[1, 1].subgridspec(1, 2, wspace=.06)
    for v, (u, w, vname) in enumerate([(0, 1, "view 1"), (0, 2, "view 2, rotated 90 deg")]):
        ax = fig.add_subplot(sub[0, v])
        ax.plot(pw3[:, u], pw3[:, w], color=WT_C, lw=1.4, zorder=3,
                label="wild type" if v == 0 else None)
        ax.plot(pm3[:, u], pm3[:, w], color=MUT_C, lw=1.4, alpha=.85, zorder=4,
                label="32 core mutations" if v == 0 else None)
        ax.scatter(pw3[sites, u], pw3[sites, w], color=INK, s=7, zorder=5,
                   label="mutated sites" if v == 0 else None)
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(.5, -.03, vname, transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color=INK2)
        if v == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=8.2,
                      bbox_to_anchor=(-.04, 1.02), handlelength=1.2)
            title(ax, "E", "predicted structures, TM-aligned",
                  f"TM {tm:.3f}, RMSD {rmsd:.2f} A;  black dots = the 32 mutated sites")

    # ---- F: the histogram behind one cell ----------------------------------
    ax = fig.add_subplot(gs[1, 2])
    lw = d[f"{wt}__logits"]
    lm = d[f"{mt}__logits"]
    off = kl.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(np.argmax(off), off.shape)
    pw, pm = softmax(lw[i, j]), softmax(lm[i, j])
    w = float(centres[1] - centres[0])
    ax.bar(centres, pw, width=w * .92, color=WT_C, alpha=.80, label="wild type", zorder=3)
    ax.bar(centres, pm, width=w * .92, color=MUT_C, alpha=.72, label="mutant", zorder=3)
    ax.axvline((pw * centres).sum(), color=WT_C, ls="--", lw=1.1, zorder=4)
    ax.axvline((pm * centres).sum(), color=MUT_C, ls="--", lw=1.1, zorder=4)
    ax.set_xlabel("distance bin centre  (A)")
    ax.set_ylabel("probability")
    ax.legend(frameon=False, fontsize=8.4)
    title(ax, "F", f"one cell of D:  pair ({i + 1}, {j + 1})",
          f"sym. KL {off[i, j]:.2f} nats;  E[d] {(pw * centres).sum():.2f} -> "
          f"{(pm * centres).sum():.2f} A (dashed)")

    fig.text(.05, .968,
             "The trunk's belief changes; the sampled structure does not",
             fontsize=13.5, fontweight="bold", color=INK)
    fig.text(.05, .922,
             f"GFP, N={N} residues.  32 buried positions mutated to charged residues. "
             f"Mean pair distance moves by up to {lim:.1f} A (C) and the pair distributions "
             f"by {kl.mean():.2f} nats on average (D), yet the two predicted structures "
             f"superimpose at TM {tm:.3f} (E).",
             fontsize=8.8, color=INK2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=190, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")
    print(f"  TM {tm:.4f}  RMSD {rmsd:.3f}  max|dE[d]| {lim:.3f}  meanKL {kl.mean():.4f}")
    print(f"  worst pair ({i + 1},{j + 1}) KL {off[i, j]:.3f}")


if __name__ == "__main__":
    main()
