"""Figure: does internal-beats-output replicate on held-out proteins?

The exploratory result was four assays, all 61-68 residue Tsuboyama domains,
collected while the method was being developed. This is the confirmatory run:
sixteen HELD-OUT assays, disjoint from the twelve the basis was fitted on and
from those four, 40 to 118 residues, four of them from other labs measuring
other phenotypes. Same mutations in all three models by construction.

  A  per-assay Spearman, internal against the model's own pLDDT. Every dot is
     one assay -- the independent unit the interval is over -- so the reader can
     see the sixteen values rather than their average. A dot above the diagonal
     is an assay where the trunk beat the confidence head.
  B  the paired within-model gap with its assay-level bootstrap interval,
     exploratory beside confirmatory. Protenix is the panel that matters: its
     exploratory interval reached to +0.002, one assay from crossing zero.

WHY ALL 128 CHANNELS, STATED IN THE TITLE. The archived cross-model probe tuned
its channel count over a grid topping out at 64 and recorded `kept: 128,
truncated: false`. It was a <=64-channel probe wearing a 128 label. This figure
is drawn from `--ks 128`, where the probe really does use the whole pair row,
and the protocol block records the grid it searched.

NOT A RANKING. The three models differ in depth, distogram grid and alignment
handling; only the within-model comparison is meaningful. The bars are placed
side by side because they answer the same question separately, not because they
are on a common scale.

    uv run python jax_harness/fig_heldout16.py \\
        --full128 runs/xmodel_io_heldout16_full128.json \\
        --exploratory runs/xmodel_io_vec.json \\
        --out figures/heldout16.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402

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
SLOT = {"boltz2": "#2a78d6", "of3": "#eb6834", "protenix": "#1baf7a"}
C_REF = "#8a8885"
NICE = {"boltz2": "Boltz-2", "of3": "OpenFold3", "protenix": "Protenix"}


def internal_key(d):
    """The internal predictor's name, which encodes the channel count."""
    return next(k for k in d["order"] if k.startswith("internal"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full128", required=True)
    ap.add_argument("--exploratory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = json.load(open(a.full128))
    E = json.load(open(a.exploratory)) if a.exploratory else None
    MODELS = D["models"]
    KEY = internal_key(D)
    if "per_assay" not in D:
        raise SystemExit(
            f"{a.full128} carries no per_assay block, so panel A would have to "
            f"invent the points it draws. Rerun analyze_xmodel_io.py.")

    fig = plt.figure(figsize=(11.4, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.30,
                          left=0.065, right=0.93, top=0.79, bottom=0.13)

    # ---- A: per-assay, internal vs pLDDT -----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    lim = [-0.15, 0.95]
    ax.plot(lim, lim, color=C_REF, lw=1.0, ls="--", zorder=1)
    ax.annotate("equal", (0.80, 0.80), fontsize=7.8, color=C_REF,
                rotation=45, ha="center", va="bottom")
    for m in MODELS:
        pa = D["per_assay"][m]
        xs = [pa["pLDDT"][k] for k in pa[KEY]]
        ys = [pa[KEY][k] for k in pa[KEY]]
        ax.scatter(xs, ys, s=34, color=SLOT[m], alpha=0.85, lw=0.6,
                   edgecolor=SURF, zorder=3,
                   label=f"{NICE[m]}  ({sum(1 for p, q in zip(xs, ys) if q > p)}"
                         f"/{len(xs)} above)")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("Spearman of the model's own pLDDT")
    ax.set_ylabel("Spearman of the internal probe")
    ax.grid(True, color=GRID, lw=0.7), ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("A   every held-out assay, internal against pLDDT",
                 loc="left", fontsize=10.5, color=INK, pad=20)
    ax.annotate(f"one dot per assay; {len(D['assays'])} held-out proteins, "
                f"40-118 aa", (0, 1), xytext=(0, 6), xycoords="axes fraction",
                textcoords="offset points", fontsize=8.4, color=INK2,
                va="bottom", ha="left")

    # ---- B: the gap, exploratory beside confirmatory ------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    ypos, labels = [], []
    for i, m in enumerate(MODELS):
        base = i * 2.6
        for j, (src, tag, alpha) in enumerate(
                ((D, f"held-out, {len(D['assays'])}", 1.0),
                 (E, f"exploratory, {len(E['assays'])}" if E else None, 0.42))):
            if src is None:
                continue
            g = src["internal_minus"]["pLDDT"][m]
            y = base - j * 0.85
            ax.plot([g["ci_lo"], g["ci_hi"]], [y, y], color=SLOT[m], lw=3.4,
                    alpha=alpha, solid_capstyle="round", zorder=3)
            ax.scatter([g["gap"]], [y], s=42, color=SLOT[m], alpha=alpha,
                       zorder=4, edgecolor=SURF, lw=0.7)
            ax.annotate(f"{g['gap']:+.3f}", (g["ci_hi"], y), xytext=(7, 0),
                        textcoords="offset points", fontsize=8.2, va="center",
                        color=INK if alpha == 1.0 else INK2,
                        annotation_clip=False)
            ypos.append(y), labels.append(f"{NICE[m]}\n{tag} assays")
    ax.set_yticks(ypos), ax.set_yticklabels(labels, fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlabel("internal minus pLDDT  (Spearman, assay-level bootstrap)")
    lo = min(src["internal_minus"]["pLDDT"][m]["ci_lo"]
             for src in (D, E) if src for m in MODELS)
    hi = max(src["internal_minus"]["pLDDT"][m]["ci_hi"]
             for src in (D, E) if src for m in MODELS)
    ax.set_xlim(min(lo, 0) - 0.02, hi + 0.09)   # room for the value labels
    ax.grid(True, axis="x", color=GRID, lw=0.7), ax.set_axisbelow(True)
    ax.set_title("B   the gap, and what sixteen assays did to it",
                 loc="left", fontsize=10.5, color=INK, pad=20)
    ax.annotate("faded = the 4 exploratory assays; solid = 16 DIFFERENT "
                "held-out proteins",
                (0, 1), xytext=(0, 6), xycoords="axes fraction",
                textcoords="offset points", fontsize=8.4, color=INK2,
                va="bottom", ha="left")

    feat = D["protocol"]["features"]
    depths = ", ".join(f"{NICE[m]} {D['layers'][m]}" for m in MODELS)
    fig.suptitle(
        "Internal state beats emitted confidence in all three architectures, "
        "on proteins none of them was tuned on",
        x=0.07, y=0.955, ha="left", fontsize=12.5, color=INK)
    fig.text(0.07, 0.885,
             f"{len(D['assays'])} held-out assays  ·  {feat['kept']} of "
             f"{feat['width']} pair channels at the final trunk layer  ·  "
             f"paired within model, never across  ·  trunk depth {depths}",
             ha="left", fontsize=8.6, color=INK2)

    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out}")
    for m in MODELS:
        g = D["internal_minus"]["pLDDT"][m]
        pa = D["per_assay"][m]
        above = sum(1 for k in pa[KEY] if pa[KEY][k] > pa["pLDDT"][k])
        print(f"  {NICE[m]:10s} gap {g['gap']:+.3f} "
              f"[{g['ci_lo']:+.3f}, {g['ci_hi']:+.3f}]  "
              f"{above}/{len(pa[KEY])} assays above")


if __name__ == "__main__":
    main()
