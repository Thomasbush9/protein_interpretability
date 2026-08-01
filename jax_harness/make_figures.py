"""Figures for the Boltz-2 pairformer interpretability experiments.

Reads the JSON emitted by exp_depth / exp_paths / exp_layers and writes one
PNG per experiment. Run on a login node -- pure matplotlib, no model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed categorical order, assigned by slot and never cycled.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# This image has no DejaVu Sans (matplotlib's default), and the default-font
# fallback is disabled, so an unset family raises rather than degrading. Pick
# whichever sans face is actually installed.
_have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_sans = next(
    (f for f in ("Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial", "Liberation Sans") if f in _have),
    None,
)

plt.rcParams.update({
    **({"font.family": "sans-serif", "font.sans-serif": [_sans]} if _sans else {}),
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
})


def log_axis(ax, which="x"):
    """Set a log scale with plain ASCII ticks.

    matplotlib's default log formatter emits mathtext ($10^{2}$), which is
    rendered with DejaVu Sans regardless of font.family -- and DejaVu is not
    installed in this image, so it raises instead of falling back. A plain
    formatter sidesteps mathtext entirely.
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter

    def fmt(v, _):
        if v >= 1000:
            return f"{v/1000:g}k"
        return f"{v:g}"

    for a in which:
        axis = ax.xaxis if a == "x" else ax.yaxis
        (ax.set_xscale if a == "x" else ax.set_yscale)("log")
        axis.set_major_formatter(FuncFormatter(fmt))
        axis.set_minor_formatter(NullFormatter())


def style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)


def fig_depth(path: Path, out: Path):
    rows = json.loads(path.read_text())
    S = np.array([r["depth"] for r in rows], float)
    sens = np.array([r["sens_A"] for r in rows], float)
    flip = np.array([r["contact_disagree"] for r in rows], float)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    ax.plot(S, sens, "-o", color=C[0], lw=2, ms=5, zorder=3)
    # 1/S reference anchored at the shallowest point -- what pure OuterProductMean
    # dilution of the query row would predict.
    ax.plot(S, sens[0] / S, "--", color=INK2, lw=1.2, zorder=2, label="pure 1/S dilution")
    log_axis(ax, "xy")
    ax.legend(frameon=False, fontsize=8)
    style(ax, "Mutation sensitivity vs MSA depth", "MSA depth S (rows)",
          "mean |dE[d]| vs WT  (A)")
    ax.annotate(f"{sens[0]:.2f} A\nsingle sequence", (S[0], sens[0]),
                textcoords="offset points", xytext=(10, -4), fontsize=8, color=INK2)
    ax.annotate(f"{sens[-1]:.2f} A\nfull MSA", (S[-1], sens[-1]),
                textcoords="offset points", xytext=(-16, 12), fontsize=8, color=INK2,
                ha="right")

    ax = axes[1]
    ax.plot(S, flip * 100, "-o", color=C[1], lw=2, ms=5, zorder=3)
    log_axis(ax, "x")
    style(ax, "Contact-map disagreement vs WT", "MSA depth S (rows)",
          "residue pairs flipping contact call (%)")

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


def fig_paths(path: Path, out: Path):
    rows = json.loads(path.read_text())
    routes = ["z_direct", "s_direct", "msa_bcast", "msa_query", "msa_prior"]
    label = {
        "z_direct": "z_direct\ns_inputs -> z_init",
        "s_direct": "s_direct\ns_inputs -> s_init",
        "msa_bcast": "msa_bcast\ns_inputs -> every MSA row",
        "msa_query": "msa_query\nMSA row 0",
        "msa_prior": "msa_prior\nMSA rows 1..S",
    }
    fig, axes = plt.subplots(1, len(rows), figsize=(4.6 * len(rows), 4.0), squeeze=False)
    y = np.arange(len(routes))

    for ax, r in zip(axes[0], rows):
        nec = [r[f"restore_{k}_necessity"] for k in routes]
        suf = [r[f"inject_{k}_sufficiency"] for k in routes]
        ax.barh(y - 0.2, nec, 0.36, color=C[0], zorder=3, label="necessity (restore->WT)")
        ax.barh(y + 0.2, suf, 0.36, color=C[1], zorder=3, label="sufficiency (inject->WT)")
        ax.set_yticks(y)
        ax.set_yticklabels([label[k] for k in routes], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=0.8)
        ax.grid(True, axis="x", zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"{r['mutant']}   (total {r['total_A']:.3f} A)",
                     color=INK, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("fraction of the mutation's effect")
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


def fig_layers(path: Path, out: Path):
    d = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    for i, (mid, rec) in enumerate(d["mutants"].items()):
        ed = np.array(rec["pf_ed_div"])
        zd = np.array(rec["pf_z_div"])
        L = np.arange(len(ed))
        axes[0].plot(L, ed, color=C[i % len(C)], lw=2, label=mid, zorder=3)
        axes[1].plot(L, zd, color=C[i % len(C)], lw=2, label=mid, zorder=3)

    style(axes[0], "Structure-space divergence by Pairformer layer",
          "Pairformer layer (0-63)", "mean |dE[d]| vs WT  (A)")
    style(axes[1], "Representation-space divergence by layer",
          "Pairformer layer (0-63)", "mean |d|z|| vs WT")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


def fig_sublayers(path: Path, out: Path):
    """Per-op contribution to divergence. transition_z vs the geometric ops."""
    d = json.loads(path.read_text())
    ops, lo, hi = d["ops"], *d["band"]
    muts = list(d["mutants"])
    fig, axes = plt.subplots(1, len(muts) + 1, figsize=(4.6 * (len(muts) + 1), 3.8))

    for ax, mid in zip(axes[:-1], muts):
        rec = d["mutants"][mid]
        for i, op in enumerate(ops):
            ax.plot(np.cumsum(rec["delta_per_op"][op]), color=C[i % len(C)] if op != "transition_z"
                    else "#d03b3b", lw=2.2 if op == "transition_z" else 1.4,
                    label=op, zorder=4 if op == "transition_z" else 3)
        ax.axvspan(lo, hi, color="#f0efec", zorder=0)
        style(ax, f"{mid}: cumulative contribution by op",
              "Pairformer layer (0-63)", "cumulative change in |dE[d]| (A)")
        ax.axhline(0, color=INK2, lw=0.8)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")

    ax = axes[-1]
    y = np.arange(len(ops))
    w = 0.36
    for j, mid in enumerate(muts):
        vals = [d["mutants"][mid]["total_sum"][o] for o in ops]
        ax.barh(y + (j - 0.5) * w, vals, w, color=C[j % len(C)], zorder=3, label=mid)
    ax.set_yticks(y); ax.set_yticklabels(ops, fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(True, axis="x", zorder=0); ax.set_axisbelow(True)
    ax.set_title("net over all 64 layers (negative = suppresses)", color=INK,
                 fontsize=10, loc="left", pad=8)
    ax.set_xlabel("change in |dE[d]|  (A)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=170); print(f"wrote {out}")


def fig_subspace(path: Path, out: Path):
    """Is dz shrinking, or moving out of the readout? Neither."""
    d = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for i, (mid, rec) in enumerate(d["mutants"].items()):
        L = np.arange(len(rec["dz_norm"]))
        axes[0].plot(L, rec["dz_norm"], color=C[i % len(C)], lw=2, label=f"{mid}  |dz|", zorder=3)
        axes[0].plot(L, rec["readable_norm"], color=C[i % len(C)], lw=1.6, ls="--",
                     label=f"{mid}  readable part", zorder=3)
        axes[1].plot(L, rec["frac_readable"], color=C[i % len(C)], lw=2, label=mid, zorder=3)
    style(axes[0], "Mutation footprint in z grows across the stack",
          "Pairformer layer (0-63)", "norm of dz over sampled pairs")
    style(axes[1], "Fraction of dz the distogram head can see",
          "Pairformer layer (0-63)", "readable fraction of |dz|^2")
    axes[1].set_ylim(0, 0.5)
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=170); print(f"wrote {out}")


def fig_kl(path: Path, out: Path):
    """The correction: KL rises to the end, E[d] falls, entropy halves."""
    d = json.loads(path.read_text())
    ent = np.array(d["wt_entropy_nats"])
    muts = list(d["mutants"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    for i, mid in enumerate(muts):
        rec = d["mutants"][mid]
        L = np.arange(len(rec["sym_kl"]))
        axes[0].plot(L, rec["sym_kl"], color=C[i % len(C)], lw=2, label=mid, zorder=3)
        axes[1].plot(L, rec["d_ed_A"], color=C[i % len(C)], lw=2, label=mid, zorder=3)
    style(axes[0], "Scale-free: KL(mutant || WT) rises to the output",
          "Pairformer layer (0-63)", "symmetric KL (nats)")
    style(axes[1], "In Angstrom: the same signal appears to fall",
          "Pairformer layer (0-63)", "mean |dE[d]| vs WT  (A)")
    axes[2].plot(np.arange(len(ent)), ent, color="#d03b3b", lw=2, zorder=3)
    style(axes[2], "...because the distogram sharpens",
          "Pairformer layer (0-63)", "WT distogram entropy (nats)")
    for ax in axes[:2]:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=170); print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--figures", required=True)
    a = ap.parse_args()
    runs, figs = Path(a.runs), Path(a.figures)
    figs.mkdir(parents=True, exist_ok=True)

    for name, fn in (
        ("depth_gfp_core32.json", fig_depth),
        ("paths_gfp.json", fig_paths),
        ("layers_gfp.json", fig_layers),
        ("sublayers_gfp.json", fig_sublayers),
        ("subspace_gfp.json", fig_subspace),
        ("kl_gfp.json", fig_kl),
    ):
        p = runs / name
        if p.exists():
            fn(p, figs / p.with_suffix(".png").name)
        else:
            print(f"skip (missing): {p}")
