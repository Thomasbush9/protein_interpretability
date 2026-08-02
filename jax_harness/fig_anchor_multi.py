"""The behavioural anchor across models: structure vs confidence vs internal state.

The GFP dose series -- 1 to 32 buried mutations, matched surface controls,
random-substitution loads, and a scrambled sequence -- read three ways for each
model:

    structure   TM to wild type          (what a user sees)
    confidence  mean pLDDT               (what a user is told)
    internal    mean symmetric KL        (what the trunk actually did)

Plotted against mutation load on one axis per readout, one line per model. The
point of the figure is the SHAPE: the internal curve rises steadily while the
structure curve stays flat until the sequence stops being a mutant of the wild
type at all.

Two modes, kept apart on purpose:

  --mode msa      Boltz-2 / OpenFold3 / Protenix, full alignment
  --mode single   the same three PLUS AlphaFold2, single-sequence

They are not interchangeable. mosaic's AF2 wrapper is single-sequence only
(`assert not use_msa`, max_msa_clusters=1), so AF2 can only be compared to the
others with their alignments removed too. Single-sequence is a genuinely
different operating point -- we measured it ~4.4x more mutation-sensitive than
full depth -- so mixing the two on one axis would manufacture a difference
between AF2 and the rest that is really a difference in input.
"""
from __future__ import annotations

import argparse
import csv
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
PALETTE = {"boltz2": "#2a78d6", "of3": "#eb6834", "protenix": "#159a8c", "af2": "#7a4fb5"}
NICE = {"boltz2": "Boltz-2", "of3": "OpenFold3", "protenix": "Protenix", "af2": "AlphaFold2"}
MODES = {"core": ("core mutations", "o", "-"),
         "random": ("random substitutions", "s", "--")}


def load(f, manifest):
    d = np.load(f)
    name = str(d["model"])
    wt = d["gfp_wt__ca"].astype(float)
    out = {}
    for cid in [str(i) for i in d["ids"]]:
        if cid == "gfp_wt":
            continue
        tm, _ = geom.tm_and_rmsd(d[f"{cid}__ca"].astype(float), wt)
        out[cid] = dict(tm=tm, plddt=float(d[f"{cid}__plddt"].mean()),
                        kl=float(d[f"{cid}__kl"].mean()),
                        n_mut=int(manifest[cid]["n_mut"]),
                        mode=manifest[cid]["mode"])
    return name, out, float(d["gfp_wt__plddt"].mean())


def panel(ax, models, key, ylabel, letter, head, sub, log=False, wt_ref=None):
    for name, rows, wtp in models:
        c = PALETTE.get(name, INK2)
        for mode, (_lab, mk, ls) in MODES.items():
            pts = sorted([r for r in rows.values() if r["mode"] == mode],
                         key=lambda r: r["n_mut"])
            if not pts:
                continue
            x = [p["n_mut"] for p in pts]
            y = [p[key] for p in pts]
            ax.plot(x, y, ls, marker=mk, ms=4.5, lw=1.5, color=c, alpha=.95,
                    zorder=3)
    if wt_ref is not None:
        for name, rows, wtp in models:
            ax.axhline(wtp, color=PALETTE.get(name, INK2), lw=.8, ls=":", alpha=.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("mutations introduced")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{letter}   {head}", loc="left", fontsize=10.5,
                 fontweight="bold", color=INK, pad=20)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.1, color=INK2,
            va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--mode", choices=["msa", "single"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="data/gfp_physics/manifest.csv")
    args = ap.parse_args()

    man = {r["id"]: r for r in csv.DictReader(open(args.manifest))}
    models = [load(f, man) for f in args.npz]
    models.sort(key=lambda m: list(NICE).index(m[0]) if m[0] in NICE else 9)

    fig = plt.figure(figsize=(14.4, 5.3))
    gs = fig.add_gridspec(1, 3, wspace=.27, left=.055, right=.985, top=.665, bottom=.145)

    panel(fig.add_subplot(gs[0, 0]), models, "tm", "TM to wild type", "A",
          "structure  -- what a user sees",
          "flat until the sequence is no longer a mutant")
    panel(fig.add_subplot(gs[0, 1]), models, "plddt", "mean pLDDT", "B",
          "confidence  -- what a user is told",
          "dotted line = each model's own wild-type value", wt_ref=True)
    panel(fig.add_subplot(gs[0, 2]), models, "kl",
          "mean symmetric KL vs wild type  (nats)", "C",
          "internal  -- what the trunk did",
          "log scale; rises steadily from the first mutation", log=True)

    handles = [plt.Line2D([], [], color=PALETTE.get(n, INK2), lw=2,
                          label=NICE.get(n, n)) for n, _, _ in models]
    handles += [plt.Line2D([], [], color=INK2, lw=1.4, ls=ls, marker=mk, ms=4.5,
                           label=lab) for lab, mk, ls in MODES.values()]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False,
               fontsize=8.4, bbox_to_anchor=(.985, .90))

    single = args.mode == "single"
    # In single-sequence mode most models do not fold GFP at all, so the anchor
    # reading does not apply and the title must not claim it does.
    wt_plddts = {n: w for n, _, w in models}
    failed = [NICE.get(n, n) for n, w in wt_plddts.items() if w < 0.55]
    if single and failed:
        head = ("Without an alignment, " + ", ".join(failed) +
                " do not fold GFP -- so this is NOT a usable behavioural comparison")
    elif single:
        head = ("The behavioural anchor, single-sequence: structure barely moves, "
                "the trunk moves a lot")
    else:
        head = ("The behavioural anchor: the structure barely moves, confidence moves "
                "a little, the trunk moves a lot")
    fig.text(.055, .945, head, fontsize=13, fontweight="bold", color=INK)
    fig.text(.055, .795,
             ("GFP, N=238. ALL MODELS SINGLE-SEQUENCE -- the only mode mosaic's AlphaFold2 "
              "wrapper supports (it asserts not use_msa), so the other three are run the same "
              "way for a matched comparison.\n"
              "WILD-TYPE pLDDT: " +
              ",  ".join(f"{NICE.get(n, n)} {w:.2f}" for n, _, w in models) +
              ". Three of four never fold the WILD TYPE, so their TM curves in A are "
              "noise between unfolded predictions,\nnot a mutation effect. Only Boltz-2 "
              "retains single-sequence capability here. AF2 cannot be behaviourally compared "
              "on GFP through this wrapper; a small domain, or MSA support, is needed."
              if single else
              "GFP, N=238, full alignment, identical sequences and alignments in every "
              "model. AlphaFold2 is absent because mosaic's wrapper is single-sequence "
              "only -- see the matched single-sequence figure.\nDistogram grids differ "
              "between models, so compare the SHAPE of each curve, not absolute nats "
              "across models."),
             fontsize=8.3, color=INK2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=190, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")
    for name, rows, wtp in models:
        core = sorted([r for r in rows.values() if r["mode"] == "core"],
                      key=lambda r: r["n_mut"])
        if core:
            lo, hi = core[0], core[-1]
            print(f"  {NICE.get(name, name):11s} WT pLDDT {wtp:.3f} | "
                  f"{lo['n_mut']:2d} mut: TM {lo['tm']:.3f} KL {lo['kl']:.3f} | "
                  f"{hi['n_mut']:2d} mut: TM {hi['tm']:.3f} KL {hi['kl']:.3f}")


if __name__ == "__main__":
    main()
