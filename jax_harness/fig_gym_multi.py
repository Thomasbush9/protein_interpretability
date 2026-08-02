"""Internal vs output across models -- the central claim, replicated.

One figure for the aggregate, one per model, all from the same data so the
per-model folders hold like-for-like plots.
"""
from __future__ import annotations
import argparse, glob, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from analyze_gym_multi import INTERNAL, grouped_split, fit_internal  # noqa
import geom  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_h = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_s = next((f for f in ("Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial") if f in _h), None)
plt.rcParams.update({**({"font.family": "sans-serif", "font.sans-serif": [_s]} if _s else {}),
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
                     "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#2a78d6", "#eb6834", "#159a8c"]
NICE = {"boltz2": "Boltz-2", "of3": "OpenFold3", "protenix": "Protenix"}
ORDER = ["internal", "TM to WT", "pLDDT", "pLDDT@site", "position-only"]


def collect(files, splits=5):
    per = defaultdict(lambda: defaultdict(list))
    gaps = defaultdict(list)
    for f in sorted(files):
        d = np.load(f)
        model = str(d["model"])
        y, pos = d["score"], d["pos"]
        caw = d["ca_wt"].astype(float)
        tm = np.array([geom.tm_score(c.astype(float), caw) for c in d["ca"]])
        X = np.column_stack([d[k] for k in INTERNAL])
        rng = np.random.default_rng(0)
        for s in range(splits):
            tr, te = grouped_split(pos, rng)
            if te.sum() < 8 or tr.sum() < 20:
                continue
            ri = fit_internal(X, y, pos, tr, te, s)   # same estimator as the table
            rt = spearmanr(tm[te], y[te]).correlation
            per[model]["internal"].append(ri)
            per[model]["TM to WT"].append(rt)
            per[model]["pLDDT"].append(spearmanr(d["plddt_mean"][te], y[te]).correlation)
            per[model]["pLDDT@site"].append(spearmanr(d["plddt_site"][te], y[te]).correlation)
            tp, tv = pos[tr], y[tr]
            pred = np.array([tv[np.argmin(np.abs(tp - p))] for p in pos[te]])
            per[model]["position-only"].append(spearmanr(pred, y[te]).correlation)
            gaps[model].append(ri - rt)
    return per, gaps


def bar_panel(ax, per, models, title_txt, sub):
    n = len(models)
    x = np.arange(len(ORDER)); w = 0.8 / n
    for k, m in enumerate(models):
        off = (k - (n - 1) / 2) * w
        v = [np.nanmean(per[m][o]) for o in ORDER]
        e = [np.nanstd(per[m][o]) / np.sqrt(max(len(per[m][o]), 1)) for o in ORDER]
        ax.bar(x + off, v, w * .9, yerr=e, capsize=2,
               color=PALETTE[k % 3], label=NICE.get(m, m), zorder=3,
               error_kw=dict(lw=.8, ecolor=INK2))
        for xi, vi in enumerate(v):
            ax.text(xi + off, vi + .012, f"{vi:.2f}", ha="center", va="bottom",
                    fontsize=6.8, color=INK, rotation=90 if n > 1 else 0)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=8.2)
    ax.set_ylabel("Spearman rho with measured dG")
    ax.set_ylim(0, .72)
    if n > 1:
        ax.legend(frameon=False, fontsize=8.4)
    ax.set_title(title_txt, loc="left", fontsize=11, fontweight="bold",
                 color=INK, pad=20)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.1, color=INK2,
            va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/gymm_*.npz")
    ap.add_argument("--outdir", default="figures_models")
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    per, gaps = collect(files)
    models = [m for m in ("boltz2", "of3", "protenix") if m in per]
    rng = np.random.default_rng(0)

    # ---- aggregate --------------------------------------------------------
    fig = plt.figure(figsize=(13.4, 5.9))
    gs = fig.add_gridspec(1, 2, wspace=.24, left=.06, right=.985, top=.655, bottom=.13,
                          width_ratios=[1.7, 1])
    bar_panel(fig.add_subplot(gs[0, 0]), per, models,
              "A   every predictor, every model",
              "4 assays x 100 variants, position-grouped splits, identical rows per predictor")
    ax = fig.add_subplot(gs[0, 1])
    for k, m in enumerate(models):
        g = np.array(gaps[m])
        bs = np.array([np.nanmean(g[i]) for i in
                       (rng.integers(0, len(g), len(g)) for _ in range(4000))])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.errorbar([k], [g.mean()], yerr=[[g.mean() - lo], [hi - g.mean()]],
                    fmt="o", ms=9, color=PALETTE[k % 3], capsize=4, lw=1.6)
        ax.text(k, hi + .012, f"{g.mean():+.3f}", ha="center", va="bottom",
                fontsize=8.6, color=INK)
    ax.axhline(0, color=INK, lw=1.1, ls="--")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([NICE.get(m, m) for m in models], fontsize=8.6)
    ax.set_ylabel("internal minus TM-to-WT")
    ax.set_title("B   the gap, with 95 % CI", loc="left", fontsize=11,
                 fontweight="bold", color=INK, pad=20)
    ax.text(0, 1.02, "above the dashed line = the trunk beats the model's own structure",
            transform=ax.transAxes, fontsize=8.1, color=INK2, va="bottom")

    sig = [m for m in models
           if np.percentile([np.nanmean(np.array(gaps[m])[i]) for i in
                             (np.random.default_rng(1).integers(0, len(gaps[m]), len(gaps[m]))
                              for _ in range(2000))], 2.5) > 0]
    fig.text(.06, .945,
             f"The trunk out-ranks the emitted structure in all three models; "
             f"the gap clears zero in {len(sig)} of {len(models)}",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(.06, .795,
             "Internal here is FIVE final-trunk distogram features, not the 256 per-layer features "
             "of the Boltz-2 headline (0.548) -- these numbers are NOT comparable to it.\n"
             "Boltz-2's gap includes zero HERE while its full protocol (12 assays, 250 variants, "
             "256 features) gives +0.335 [+0.288, +0.380]. On these same 4 assays that protocol\n"
             "gives internal 0.480 vs TM 0.191; the weaker feature set and the mosaic wrapper's "
             "sampler settings both narrow the gap. Treat this as a like-for-like CROSS-MODEL\n"
             "comparison under one protocol, not as a restatement of the headline.",
             fontsize=8.2, color=INK2)
    out = Path(args.outdir) / "aggregate" / "internal_vs_output.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, facecolor=fig.get_facecolor())
    print(f"wrote {out}")

    # ---- one per model ----------------------------------------------------
    for m in models:
        fig = plt.figure(figsize=(7.6, 5.2))
        ax = fig.add_axes([.12, .13, .84, .60])
        bar_panel(ax, per, [m], f"{NICE.get(m, m)}: internal vs output",
                  "4 assays x 100 variants, position-grouped splits")
        g = np.array(gaps[m])
        bs = np.array([np.nanmean(g[i]) for i in
                       (rng.integers(0, len(g), len(g)) for _ in range(4000))])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        fig.text(.12, .90, f"{NICE.get(m, m)}", fontsize=13, fontweight="bold", color=INK)
        fig.text(.12, .845,
                 f"internal minus TM-to-WT = {g.mean():+.3f}, 95 % CI [{lo:+.3f}, {hi:+.3f}]"
                 f"{'  (excludes zero)' if lo > 0 else '  (includes zero)'}\n"
                 f"Internal = 5 final-trunk distogram features; not comparable to the "
                 f"Boltz-2 256-feature headline.",
                 fontsize=8.3, color=INK2)
        o = Path(args.outdir) / m / "internal_vs_output.png"
        o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(o, dpi=190, facecolor=fig.get_facecolor())
        print(f"wrote {o}")


if __name__ == "__main__":
    main()
