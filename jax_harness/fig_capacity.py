"""Figure: the trunk-over-emitted gap is not richness and it is not capacity.

  A,B  rho against the number of principal components the probe is allowed,
       one panel per cohort. The basis is refitted inside every leave-one-assay-
       out fold on the TRAINING assays only. Horizontal rules mark the emitted
       blocks at their own full width, so the reading is direct: where the
       internal curve crosses them is how many trunk dimensions it takes to beat
       everything the structure module emitted.
  C    the paired gaps with assay-level bootstrap intervals, including the
       capacity-matched ones -- the trunk read at 10 components against
       output_rich at its own full width of 10.

NOT A RANKING ACROSS MODELS. Depths and alignment handling differ; only the
within-model comparison is meaningful.

    uv run python jax_harness/fig_capacity.py \\
        --heldout runs/geometry_heldout16.json \\
        --panel runs/geometry_panel5.json --out figures/capacity.png
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
SLOT = {"boltz2": "#2a78d6", "of3": "#eb6834", "protenix": "#1baf7a"}
LABEL = {"boltz2": "Boltz-2", "of3": "OpenFold3", "protenix": "Protenix"}
BASE = {"geometry": ("#7a5ea8", "emitted geometry, 37"),
        "rich": ("#8f3a3a", "output_rich, 10"),
        "chem": ("#5f5f58", "chemistry, 17")}


def curve_panel(ax, res, title, show_legend=False):
    for m, r in res["models"].items():
        c = r["curve"]["internal"]
        d = sorted(int(k) for k in c)
        y = [c[str(k)]["mean"] for k in d]
        lo = [c[str(k)]["lo"] for k in d]
        hi = [c[str(k)]["hi"] for k in d]
        ax.fill_between(d, lo, hi, color=SLOT[m], alpha=.11, linewidth=0)
        ax.plot(d, y, "-o", color=SLOT[m], ms=3.2, lw=1.6, label=LABEL[m])
    # The emitted blocks at their own full width, averaged over models: the
    # spread between models is drawn as a band so one rule does not imply the
    # three architectures agree more than they do.
    for b, (col, lab) in BASE.items():
        v = [r["full"][b]["mean"] for r in res["models"].values()]
        ax.axhspan(min(v), max(v), color=col, alpha=.10, linewidth=0)
        ax.axhline(float(np.mean(v)), color=col, lw=1.15, ls="--", label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xlabel("principal components of the trunk the probe may use")
    ax.set_ylabel("Spearman, held-out assay")
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(frameon=False, fontsize=7.6, loc="lower right", ncol=2)


def gap_panel(ax, heldout, panel):
    keys = [("gaps", "internal_minus_rich", "internal 128\n− rich 10"),
            ("gaps", "internal_minus_geometry", "internal 128\n− geometry 37"),
            ("matched", "at_10_internal_minus_rich", "internal @10\n− rich 10"),
            ("matched", "at_37_internal_minus_geometry",
             "internal @37\n− geometry 37"),
            ("gaps", "geometry_minus_rich", "geometry 37\n− rich 10")]
    models = list(SLOT)
    w, x0 = .13, np.arange(len(keys))
    for ci, (res, hatch, cohort) in enumerate(
            [(heldout, "", "held-out 16"), (panel, "///", "panel5 25")]):
        for mi, m in enumerate(models):
            off = (ci * 3 + mi - 2.5) * w
            v = [res["models"][m][sec][k] for sec, k, _ in keys]
            ax.bar(x0 + off, [q["mean"] for q in v], width=w * .92,
                   color=SLOT[m], alpha=.95 if ci == 0 else .55,
                   hatch=hatch, edgecolor="white", linewidth=.4,
                   label=f"{LABEL[m]}, {cohort}")
            ax.errorbar(x0 + off, [q["mean"] for q in v],
                        yerr=[[q["mean"] - q["lo"] for q in v],
                              [q["hi"] - q["mean"] for q in v]],
                        fmt="none", ecolor=INK2, elinewidth=.75, capsize=1.6)
    ax.axhline(0, color=INK, lw=.9)
    ax.set_xticks(x0)
    ax.set_xticklabels([k[2] for k in keys], fontsize=7.8)
    ax.set_ylabel("paired gap in Spearman, 95% CI over assays")
    ax.set_title("C   the gap survives a richer emitted block and a matched width",
                 fontsize=10.5, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.9, ncol=2, loc="upper right")


def main() -> int:
    ap = argparse.ArgumentParser()
    W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
    ap.add_argument("--heldout", default=str(W / "runs/geometry_heldout16.json"))
    ap.add_argument("--panel", default=str(W / "runs/geometry_panel5.json"))
    ap.add_argument("--out", default=str(W / "figures/capacity.png"))
    a = ap.parse_args()
    heldout = json.loads(Path(a.heldout).read_text())
    panel = json.loads(Path(a.panel).read_text())

    fig = plt.figure(figsize=(13.6, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, .92], hspace=.42, wspace=.2,
                          left=.06, right=.985, top=.93, bottom=.09)
    curve_panel(fig.add_subplot(gs[0, 0]),
                heldout, f"A   held-out 16 assays  (n={heldout['n_assays']})",
                show_legend=True)
    curve_panel(fig.add_subplot(gs[0, 1]),
                panel, f"B   panel5, no stability assay  (n={panel['n_assays']})")
    gap_panel(fig.add_subplot(gs[1, :]), heldout, panel)
    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
