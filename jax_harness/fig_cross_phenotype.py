"""Figure: how much of the mutation direction is shared across phenotypes.

  A  fit on one cohort, predict a disjoint one. The filled bar is what
     transferred; the open outline behind it is the within-cohort
     leave-one-assay-out ceiling on the SAME test assays -- same phenotype,
     unseen protein. The black tick is substitution chemistry, which is
     model-independent and therefore transfers for free: the trunk has to beat
     it or the number says nothing about the model.
  B  principal angles between the two cohorts' top-4 trunk subspaces. The first
     two directions are shared in every architecture; the fourth mostly is not.
     Chemistry, a direction that genuinely IS shared, saturates near 1.0 on all
     four and shows what agreement looks like.
  C  cosine between the two cohorts' fitted probes inside a shared basis, as the
     basis is widened, against the split-half null -- two probes fitted on
     disjoint halves of the SAME cohort. The null is the scale: at d=2 the
     cross-phenotype probes agree as well as a cohort agrees with itself.
  D  blind transfer with the basis fitted on the training cohort alone, against
     the full 128-channel number. Fewer dimensions transfer BETTER.

    uv run python jax_harness/fig_cross_phenotype.py \\
        --cross runs/cross_phenotype.json \\
        --agree runs/direction_agreement.json \\
        --out figures/cross_phenotype.png
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
DIRS = [("stability_to_panel5", "stability 12\n→ panel5 25"),
        ("panel5_to_stability", "panel5 25\n→ stability 12"),
        ("stability_to_heldout_other", "stability 12\n→ other 4")]


def panel_transfer(ax, cross):
    x0, w = np.arange(len(DIRS)), .26
    for mi, m in enumerate(SLOT):
        off = (mi - 1) * w
        r = cross["models"][m]
        tr = [r[k]["internal"]["transfer"]["mean"] if k in r else np.nan
              for k, _ in DIRS]
        ce = [r[k]["internal"]["within_cohort_ceiling"]["mean"] if k in r
              else np.nan for k, _ in DIRS]
        ch = [r[k]["chem"]["transfer"]["mean"] if k in r else np.nan
              for k, _ in DIRS]
        lo = [r[k]["internal"]["transfer"]["lo"] if k in r else np.nan
              for k, _ in DIRS]
        hi = [r[k]["internal"]["transfer"]["hi"] if k in r else np.nan
              for k, _ in DIRS]
        ax.bar(x0 + off, ce, width=w * .9, facecolor="none",
               edgecolor=SLOT[m], linewidth=1.0, linestyle=":",
               label="within-cohort ceiling" if mi == 0 else None)
        ax.bar(x0 + off, tr, width=w * .9, color=SLOT[m], alpha=.9,
               label=LABEL[m])
        ax.errorbar(x0 + off, tr, yerr=[np.array(tr) - lo, np.array(hi) - tr],
                    fmt="none", ecolor=INK2, elinewidth=.8, capsize=1.8)
        ax.plot(x0 + off, ch, "_", color=INK, ms=11, mew=1.5,
                label="chemistry (transfers free)" if mi == 0 else None)
    ax.axhline(0, color=INK, lw=.9)
    ax.set_xticks(x0)
    ax.set_xticklabels([d[1] for d in DIRS], fontsize=8.2)
    ax.set_ylabel("Spearman on the test cohort")
    ax.set_title("A   the trunk transfers across the phenotype switch",
                 fontsize=10.5, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper left")


def panel_angles(ax, agree, d="4"):
    n = 4
    x0, w = np.arange(n), .2
    for mi, m in enumerate(SLOT):
        c = agree["models"][m]["internal"][d]["feature_subspace"]["cosines"]
        ax.bar(x0 + (mi - 1.5) * w, c[:n], width=w * .9, color=SLOT[m],
               alpha=.9, label=LABEL[m])
    c = agree["models"]["boltz2"]["chem"][d]["feature_subspace"]["cosines"]
    ax.bar(x0 + 1.5 * w, c[:n], width=w * .9, color=INK2, alpha=.65,
           label="chemistry (shared)")
    floor = np.sqrt(agree["models"]["boltz2"]["internal"][d]["random_floor"]["mean"])
    # sqrt of the mean squared cosine, i.e. the RMS cosine two random
    # 4-dimensional subspaces of R^128 produce. Labelled as RMS because the
    # stored floor is an overlap (mean cos^2) and calling it "the cosine" would
    # be off by a square root.
    ax.axhline(floor, color="#8f3a3a", lw=1.1, ls="--",
               label="random subspaces (RMS)")
    ax.set_xticks(x0)
    # Plain labels, not mathtext: the STIX fonts mathtext falls back to are
    # not installed on this cluster and their absence is a hard error, not a
    # substitution.
    ax.set_xticklabels(["1st", "2nd", "3rd", "4th"])
    ax.set_xlabel("principal angle between the two cohorts' subspaces")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("principal-angle cosine")
    ax.set_title("B   the first two trunk directions are shared, the fourth is not",
                 fontsize=10.5, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="lower left")


def panel_cosine(ax, agree):
    for m in SLOT:
        blk = agree["models"][m]["internal"]
        d = sorted(int(k) for k in blk)
        ax.plot(d, [blk[str(k)]["shared_basis"]["cosine_across"] for k in d],
                "-o", color=SLOT[m], ms=3.6, lw=1.7, label=LABEL[m])
        nul = [(blk[str(k)]["shared_basis"]["null"]["stability"]["mean"]
                + blk[str(k)]["shared_basis"]["null"]["panel5"]["mean"]) / 2
               for k in d]
        ax.plot(d, nul, ":", color=SLOT[m], lw=1.3)
    ax.plot([], [], ":", color=INK2, label="split-half null (same cohort)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8])
    ax.set_xticklabels([2, 4, 8])
    ax.set_xlabel("dimension of the shared basis")
    ax.set_ylabel("cosine between the two cohorts' probes")
    ax.set_title("C   agreement is near-total at d=2 and decays as d grows",
                 fontsize=10.5, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.4, loc="lower left")


def panel_blind(ax, agree, cross):
    for m in SLOT:
        blk = agree["models"][m]["internal"]
        d = sorted(int(k) for k in blk)
        y = [blk[str(k)]["blind_transfer_train_only_basis"]
             ["panel5_to_stability"]["mean"] for k in d]
        ax.plot(d, y, "-o", color=SLOT[m], ms=3.6, lw=1.7, label=LABEL[m])
        full = cross["models"][m]["panel5_to_stability"]["internal"]["transfer"]["mean"]
        ax.axhline(full, color=SLOT[m], lw=1.1, ls="--", alpha=.75)
    ax.plot([], [], "--", color=INK2, label="all 128 channels")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8])
    ax.set_xticklabels([2, 4, 8])
    ax.set_xlabel("dimension of the train-only basis")
    ax.set_ylabel("Spearman, panel5 → stability")
    ax.set_title("D   a low-dimensional probe transfers BETTER than 128 channels",
                 fontsize=10.5, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left", ncol=2)


def main() -> int:
    W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", default=str(W / "runs/cross_phenotype.json"))
    ap.add_argument("--agree", default=str(W / "runs/direction_agreement.json"))
    ap.add_argument("--out", default=str(W / "figures/cross_phenotype.png"))
    a = ap.parse_args()
    cross = json.loads(Path(a.cross).read_text())
    agree = json.loads(Path(a.agree).read_text())

    fig = plt.figure(figsize=(13.6, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=.40, wspace=.22,
                          left=.06, right=.985, top=.93, bottom=.08)
    panel_transfer(fig.add_subplot(gs[0, 0]), cross)
    panel_angles(fig.add_subplot(gs[0, 1]), agree)
    panel_cosine(fig.add_subplot(gs[1, 0]), agree)
    panel_blind(fig.add_subplot(gs[1, 1]), agree, cross)
    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
