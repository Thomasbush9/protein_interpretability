"""The circuit as matrices over layers: residue x layer, separation x layer, op x layer.

One figure per mutant, three panels sharing the layer axis, so the whole
Pairformer story can be read left-to-right in depth:

  top     residue x layer   where in the chain the mutation acts, and when.
                            Mutated positions ticked on the left.
  middle  separation x layer  local vs long-range contacts, over depth.
  bottom  operation x layer   which of the five writes into z moves the
                              divergence, per layer (from exp_sublayers, KL).

Sequential magnitude data -> one hue, light to dark, never a rainbow. The op
panel is the one signed quantity, so it gets a diverging map with a neutral
zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_sans = next((f for f in ("Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial") if f in _have), None)
plt.rcParams.update({
    **({"font.family": "sans-serif", "font.sans-serif": [_sans]} if _sans else {}),
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.unicode_minus": False,
})

# sequential: single hue, light -> dark (blue ramp)
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# diverging: two poles, neutral grey midpoint
DIV = LinearSegmentedColormap.from_list(
    "div", ["#1c5cab", "#86b6ef", "#f0efec", "#eda100", "#c0392b"])


def panel(ax, M, *, cmap, norm=None, vmax=None, title, ylabel, yticks=None,
          yticklabels=None, cbar_label=""):
    kw = dict(aspect="auto", origin="lower", cmap=cmap, interpolation="nearest")
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = 0, vmax
    im = ax.imshow(M, **kw)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
    ax.set_ylabel(ylabel)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=7)
    cb = plt.colorbar(im, ax=ax, pad=0.012, fraction=0.025)
    cb.ax.tick_params(labelsize=7, color=INK2)
    cb.outline.set_visible(False)
    cb.set_label(cbar_label, fontsize=7, color=INK2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--sublayers", default=None)
    ap.add_argument("--figures", required=True)
    a = ap.parse_args()

    d = json.loads(Path(a.matrix).read_text())
    sub = json.loads(Path(a.sublayers).read_text()) if a.sublayers and Path(a.sublayers).exists() else None
    figs = Path(a.figures); figs.mkdir(parents=True, exist_ok=True)
    sep_labels = d["sep_labels"]

    for mid, rec in d["mutants"].items():
        res = np.array(rec["residue_by_layer"])       # [N, L]
        sep = np.array(rec["separation_by_layer"])    # [B, L]
        L = res.shape[1]
        has_ops = bool(sub and mid in sub.get("mutants", {}))
        has_enr = "enrichment_at_mutated" in rec
        heights = [3.0, 2.2] + ([2.2] if has_ops else []) + ([1.5] if has_enr else [])
        fig, axes = plt.subplots(len(heights), 1, figsize=(11, sum(heights)),
                                 sharex=True,
                                 gridspec_kw={"height_ratios": heights})

        # cap the colour scale at a high percentile so a few extreme pairs do
        # not wash the map out
        panel(axes[0], res, cmap=SEQ, vmax=float(np.percentile(res, 99)),
              title=f"{mid}: where the mutation acts, by residue and layer",
              ylabel="residue index", cbar_label="symmetric KL")
        for r in rec.get("mutated_rows", []):
            axes[0].plot(-0.8, r, marker="_", color="#c0392b", ms=6, mew=1.6,
                         clip_on=False, zorder=5)
        if rec.get("mutated_rows"):
            # annotate inside the axes so it cannot collide with the title
            axes[0].text(0.995, 0.03, "red ticks (left margin) = mutated positions",
                         transform=axes[0].transAxes, fontsize=7, color="#c0392b",
                         ha="right", va="bottom")

        panel(axes[1], sep, cmap=SEQ, vmax=float(sep.max()),
              title="the same signal, binned by sequence separation |i-j|",
              ylabel="separation", yticks=np.arange(len(sep_labels)),
              yticklabels=sep_labels, cbar_label="symmetric KL")

        if has_ops:
            ops = sub["ops"]
            D = np.array([sub["mutants"][mid]["delta_per_op"][o] for o in ops])
            v = float(np.abs(D).max())
            panel(axes[2], D, cmap=DIV, norm=TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v),
                  title="which write into z moves the divergence (blue = reduces)",
                  ylabel="operation", yticks=np.arange(len(ops)), yticklabels=ops,
                  cbar_label="change in KL")

        if has_enr:
            ax = axes[-1]
            e = np.array(rec["enrichment_at_mutated"])
            ax.plot(np.arange(len(e)), e, color="#c0392b", lw=2, zorder=3)
            ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
            ax.set_ylim(0, max(e.max() * 1.12, 1.3))
            ax.set_title("how concentrated the signal is on mutated residues "
                         "(1.0 = no enrichment)", color=INK, fontsize=10, loc="left", pad=6)
            ax.set_ylabel("KL enrichment")
            ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            # keep the x-extent aligned with the imshow panels above
            ax.set_xlim(-0.5, len(e) - 0.5)

        axes[-1].set_xlabel("Pairformer layer (0-63)")
        axes[-1].set_xlim(-0.5, L - 0.5)
        fig.tight_layout()
        out = figs / f"matrix_{mid}.png"
        fig.savefig(out, dpi=170)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
