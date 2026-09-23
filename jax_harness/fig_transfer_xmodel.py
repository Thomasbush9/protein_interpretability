"""Figure: the +0.758 comparison, run in three architectures on held-out proteins.

This is the LEAVE-ONE-ASSAY-OUT design -- train a probe on 15 proteins, predict
a 16th it has never seen -- which is the design behind the headline. It is not
the within-assay position-split comparison; those numbers are not comparable and
should not be drawn on one axis.

  A  the predictors, per model, beside the archived 12-assay result. The
     question the paper asks: does the trunk beat the richest description of the
     structure the model actually emits, and does it beat substitution
     chemistry?
  B  the paired gap with its assay-level bootstrap interval.
  C  why Boltz-2's transferred number falls from +0.758 to +0.552, and it is not
     the method weakening. The twelve development assays are all Tsuboyama
     cDNA-display stability; the held-out sixteen add four assays from other
     labs measuring growth, toxicity and photoreceptor function. Those four are
     the four lowest, with no overlap between the groups.

NOT A RANKING ACROSS MODELS. Depths, distogram grids and alignment handling all
differ, and the `internal` scalar block is not even the same width (4 quantities
x trunk depth = 256 / 192 / 64). Only the within-model gap is meaningful.

    uv run python jax_harness/fig_transfer_xmodel.py \\
        --transfer runs/transfer_heldout16_{model}.json \\
        --archive runs/transfer_full.json --out figures/transfer_xmodel.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
SLOT = {"archive": "#6f6c66", "boltz2": "#2a78d6",
        "of3": "#eb6834", "protenix": "#1baf7a"}
NICE = {"archive": "Boltz-2, 12 basis\n(archived headline)", "boltz2": "Boltz-2",
        "of3": "OpenFold3", "protenix": "Protenix"}
C_REF = "#8a8885"
GAP = "internal 128-dim - output-rich"
PREDS = [("internal_vec", "internal\n128 ch"), ("chemistry", "substitution\nchemistry"),
         ("output_rich", "emitted structure\n10 features"), ("TM_to_WT", "TM to\nwild type")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", nargs="+", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--cohort", default="heldout_assays")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = {}
    for f in a.transfer:
        stem = Path(f).stem
        m = next(k for k in ("boltz2", "of3", "protenix") if k in stem)
        D[m] = json.load(open(f))
    D["archive"] = json.load(open(a.archive))
    KEYS = ["archive", "boltz2", "of3", "protenix"]

    from protein_interpretability.collection import Cohort
    stability = {a_.id.split("_")[0]: ("Tsuboyama_2023" in a_.id)
                 for a_ in Cohort.load(a.cohort)}

    fig = plt.figure(figsize=(14.2, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 0.9], wspace=0.32,
                          left=0.055, right=0.985, top=0.78, bottom=0.15)

    # ---- A: the predictors --------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    w, n = 0.19, len(PREDS)
    for i, k in enumerate(KEYS):
        vals = [D[k]["predictors"][p]["mean"] for p, _ in PREDS]
        xs = np.arange(n) + (i - 1.5) * w
        ax.bar(xs, vals, width=w * 0.92, color=SLOT[k],
               alpha=1.0 if k != "archive" else 0.55,
               edgecolor=SURF, lw=0.6, zorder=3,
               label=NICE[k].replace("\n", " "))
        if k == "archive":
            ax.annotate(f"{vals[0]:+.3f}", (xs[0], vals[0]), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.6,
                        color=INK2)
    ax.set_xticks(np.arange(n)), ax.set_xticklabels([lab for _, lab in PREDS],
                                                    fontsize=8.2)
    ax.set_ylabel("Spearman on a protein never seen in training")
    ax.axhline(0, color=INK, lw=0.8)
    ax.grid(True, axis="y", color=GRID, lw=0.7), ax.set_axisbelow(True)
    ax.legend(fontsize=7.6, frameon=False, loc="upper right", ncol=1)
    ax.set_title("A   leave-one-assay-out transfer", loc="left",
                 fontsize=10.5, color=INK, pad=20)
    ax.annotate("train on 15 proteins, predict the 16th; chemistry is "
                "model-independent, hence identical",
                (0, 1), xytext=(0, 6), xycoords="axes fraction",
                textcoords="offset points", fontsize=8.4, color=INK2,
                va="bottom", ha="left")

    # ---- B: the gap ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    for i, k in enumerate(KEYS):
        g = D[k]["gaps"][GAP]
        y = -i
        ax.plot([g["ci_lo"], g["ci_hi"]], [y, y], color=SLOT[k], lw=3.6,
                alpha=1.0 if k != "archive" else 0.55,
                solid_capstyle="round", zorder=3)
        ax.scatter([g["gap"]], [y], s=44, color=SLOT[k], zorder=4,
                   edgecolor=SURF, lw=0.7,
                   alpha=1.0 if k != "archive" else 0.55)
        ax.annotate(f"{g['gap']:+.3f}   {g['wins']}/{g['n_assays']}",
                    (g["ci_hi"], y), xytext=(7, 0), textcoords="offset points",
                    fontsize=8.2, va="center", color=INK, annotation_clip=False)
    ax.set_yticks([-i for i in range(len(KEYS))])
    ax.set_yticklabels([NICE[k] for k in KEYS], fontsize=8.2)
    lo = min(D[k]["gaps"][GAP]["ci_lo"] for k in KEYS)
    hi = max(D[k]["gaps"][GAP]["ci_hi"] for k in KEYS)
    ax.set_xlim(min(lo, 0) - 0.02, hi + 0.16)
    ax.set_xlabel("internal minus emitted structure")
    ax.grid(True, axis="x", color=GRID, lw=0.7), ax.set_axisbelow(True)
    ax.set_title("B   the gap, with the assay as the unit", loc="left",
                 fontsize=10.5, color=INK, pad=20)
    ax.annotate("paired within model; never across", (0, 1), xytext=(0, 6),
                xycoords="axes fraction", textcoords="offset points",
                fontsize=8.4, color=INK2, va="bottom", ha="left")

    # ---- C: phenotype, not length -------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(0)
    for i, m in enumerate(("boltz2", "of3", "protenix")):
        pa = D[m]["predictors"]["internal_vec"]["per_assay"]
        for j, (grp, mark) in enumerate(((True, "o"), (False, "D"))):
            vals = [v for k, v in pa.items() if stability.get(k, True) is grp]
            xs = i + (j - 0.5) * 0.42 + rng.normal(0, 0.035, len(vals))
            ax.scatter(xs, vals, s=30 if grp else 46, marker=mark,
                       color=SLOT[m], alpha=0.9 if grp else 1.0,
                       edgecolor=SURF if grp else INK, lw=0.6, zorder=3)
            ax.plot([i + (j - 0.5) * 0.42 - 0.14, i + (j - 0.5) * 0.42 + 0.14],
                    [np.mean(vals)] * 2, color=INK, lw=1.6, zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Boltz-2", "OpenFold3", "Protenix"], fontsize=8.6)
    ax.set_ylabel("transferred Spearman, per assay")
    ax.grid(True, axis="y", color=GRID, lw=0.7), ax.set_axisbelow(True)
    ax.scatter([], [], s=30, marker="o", color=C_REF, label="stability (12)")
    ax.scatter([], [], s=46, marker="D", color=C_REF, edgecolor=INK, lw=0.6,
               label="other phenotype (4)")
    ax.set_ylim(-0.02, 0.97)          # room for the legend under the data
    ax.legend(fontsize=7.8, frameon=False, loc="lower left",
              handletextpad=0.4, borderpad=0.2)
    for i, m in enumerate(("boltz2", "of3", "protenix")):
        pa = D[m]["predictors"]["internal_vec"]["per_assay"]
        s = np.mean([v for k, v in pa.items() if stability.get(k, True)])
        o = np.mean([v for k, v in pa.items() if not stability.get(k, True)])
        # anchored to the ENDS of the mean bars, not their centres, or the
        # label sits on top of the line it is describing
        ax.annotate(f"{s:.2f}", (i - 0.21 - 0.14, s), xytext=(-3, -2.5),
                    textcoords="offset points", fontsize=7.6, color=INK,
                    ha="right", va="center")
        ax.annotate(f"{o:.2f}", (i + 0.21 + 0.14, o), xytext=(3, -2.5),
                    textcoords="offset points", fontsize=7.6, color=INK,
                    ha="left", va="center")
    ax.set_title("C   what the drop actually is", loc="left",
                 fontsize=10.5, color=INK, pad=20)
    ax.annotate("bars are group means", (0, 1), xytext=(0, 6),
                xycoords="axes fraction", textcoords="offset points",
                fontsize=8.4, color=INK2, va="bottom", ha="left")

    n_assays = D["boltz2"]["protocol"]["n_assays"]
    fig.suptitle(
        "The internal-over-emitted gap transfers to unseen proteins in all "
        "three architectures",
        x=0.055, y=0.95, ha="left", fontsize=12.8, color=INK)
    fig.text(0.055, 0.875,
             f"leave-one-assay-out  ·  {n_assays} held-out proteins, disjoint "
             f"from the 12 the method was built on  ·  all 128 pair channels at "
             f"the final trunk layer  ·  ridge, lambda=10",
             ha="left", fontsize=8.6, color=INK2)
    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out}")
    for m in ("boltz2", "of3", "protenix"):
        pa = D[m]["predictors"]["internal_vec"]["per_assay"]
        s = [v for k, v in pa.items() if stability.get(k, True)]
        o = [v for k, v in pa.items() if not stability.get(k, True)]
        g = D[m]["gaps"][GAP]
        print(f"  {m:9s} internal {D[m]['predictors']['internal_vec']['mean']:+.3f}"
              f"  gap {g['gap']:+.3f}  stability {np.mean(s):+.3f} (n={len(s)})"
              f"  other {np.mean(o):+.3f} (n={len(o)})")


if __name__ == "__main__":
    main()
