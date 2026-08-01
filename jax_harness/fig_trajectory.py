"""Trajectory figure: the mutant's denoising path stays inside the WT's own noise.

Reads runs/traj2_*.npz (exp_trajectory.py, corrected version with --wt-keys).

The comparison that matters is a ratio, not a distance. Boltz-2's sampler draws
fresh noise per call, so a mutant trajectory and a wild-type trajectory are two
independent samples; their separation is partly the mutation and partly the
sampler. The denominator removes the second part: the wild type is sampled with
4 independent keys and the 6 pairwise separations among *those* runs are the
noise floor. A ratio above 1 means the mutation moved the path further than
resampling the same sequence does. Below 1 means it did not.

    A  divergence and floor vs sigma, one protein
    B  the ratio vs sigma, all proteins, with the ratio = 1 line
    C  rho(divergence, measured dG) vs sigma, against the Pairformer's rho
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_h = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_s = next((f for f in ("Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial") if f in _h), None)
plt.rcParams.update({**({"font.family": "sans-serif", "font.sans-serif": [_s]} if _s else {}),
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
                     "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
WT_C, MUT_C, ACC = "#2a78d6", "#eb6834", "#7a4fb5"
PF_RHO = 0.548  # Pairformer probe, pooled over 12 assays (exp_gym2 / probe_gym)


def log_axis(ax, which="x"):
    """Log ticks written as plain decimals -- mathtext needs a font we don't have."""
    a = ax.xaxis if which == "x" else ax.yaxis
    ax.set_xscale("log") if which == "x" else ax.set_yscale("log")
    a.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: ("%g" % v) if v >= .01 else ("%.3f" % v)))
    a.set_minor_formatter(matplotlib.ticker.NullFormatter())


def title(ax, letter, text, sub=None):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=10.5, fontweight="bold",
                 color=INK, pad=20 if sub else 6)
    if sub:
        ax.text(0, 1.018, sub, transform=ax.transAxes, fontsize=8.2, color=INK2,
                va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/traj2_*.npz")
    ap.add_argument("--out", default="figures/trajectory.png")
    args = ap.parse_args()

    runs = []
    for f in sorted(glob.glob(args.glob)):
        d = np.load(f)
        name = Path(f).stem[6:].split("_Tsuboyama")[0]
        sig = d["sigmas"][:-1]
        div, floor, sc = d["divergence"], d["floor"], d["score"]
        rho = np.array([spearmanr(div[:, s], sc).correlation for s in range(div.shape[1])])
        runs.append(dict(name=name, sig=sig, div=div.mean(0), div_sd=div.std(0),
                         fl=floor.mean(0), fl_sd=floor.std(0), rho=rho,
                         n=div.shape[0], npairs=floor.shape[0]))
    print(f"{len(runs)} proteins")

    fig = plt.figure(figsize=(14.2, 4.5))
    gs = fig.add_gridspec(1, 3, wspace=.27, left=.05, right=.985, top=.74, bottom=.155)

    # ---- A: the two curves, for one protein --------------------------------
    r = runs[0]
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(r["sig"], r["fl"] - r["fl_sd"], r["fl"] + r["fl_sd"],
                    color=WT_C, alpha=.18, lw=0)
    ax.plot(r["sig"], r["fl"], color=WT_C, lw=1.9,
            label=f"wild type vs wild type\n({r['npairs']} key pairs) = the floor")
    ax.fill_between(r["sig"], r["div"] - r["div_sd"], r["div"] + r["div_sd"],
                    color=MUT_C, alpha=.18, lw=0)
    ax.plot(r["sig"], r["div"], color=MUT_C, lw=1.9,
            label=f"mutant vs wild type\n({r['n']} variants)")
    log_axis(ax, "x")
    log_axis(ax, "y")
    ax.invert_xaxis()
    ax.set_xlabel("noise level sigma  (A), denoising runs left to right")
    ax.set_ylabel("RMSD between trajectories  (A)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    title(ax, "A", f"{r['name']}: mutant path vs the sampler's own spread",
          "the orange curve never leaves the blue band")

    # ---- B: the ratio -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for k, r in enumerate(runs):
        ax.plot(r["sig"], r["div"] / np.maximum(r["fl"], 1e-9),
                lw=1.7, color=[MUT_C, ACC, "#159a8c"][k % 3], label=r["name"])
    ax.axhline(1, color=INK, lw=1.1, ls="--")
    ax.text(.985, 1.02, "mutation moves the path more than resampling does",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=7.8, color=INK2)
    log_axis(ax, "x")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("noise level sigma  (A)")
    ax.set_ylabel("divergence / noise floor")
    ax.legend(frameon=False, fontsize=8.2, loc="lower left")
    title(ax, "B", "the ratio stays below 1 at every step",
          "3 proteins, 100 variants each")

    # ---- C: does the divergence track the measured effect? ------------------
    ax = fig.add_subplot(gs[0, 2])
    for k, r in enumerate(runs):
        ax.plot(r["sig"], r["rho"], lw=1.7,
                color=[MUT_C, ACC, "#159a8c"][k % 3], label=r["name"])
    ax.axhline(0, color=GRID, lw=1)
    ax.axhline(PF_RHO, color=WT_C, lw=1.4, ls="--")
    ax.text(.985, PF_RHO + .02, f"Pairformer probe, {PF_RHO:.2f}",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color=WT_C)
    log_axis(ax, "x")
    ax.invert_xaxis()
    ax.set_ylim(-.45, .65)
    ax.set_xlabel("noise level sigma  (A)")
    ax.set_ylabel("Spearman rho with measured dG")
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")
    title(ax, "C", "no step of the path predicts stability",
          "rho between per-variant divergence and measured dG")

    fig.text(.05, .93,
             "The sampler does not carry the mutation: its path stays inside its own noise",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(.05, .865,
             "Divergence is the Kabsch RMSD between a mutant trajectory and a wild-type "
             "trajectory at the same denoising step. Because Boltz-2 draws fresh noise per "
             "call the two are independent samples, so the number only means something "
             "against a same-sequence floor: the wild type sampled with 4 keys, all 6 "
             "pairwise separations. The ratio never exceeds 1.",
             fontsize=8.6, color=INK2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=190, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")
    for r in runs:
        rt = r["div"] / np.maximum(r["fl"], 1e-9)
        print(f"  {r['name']:14s} ratio min {rt.min():.2f} max {rt.max():.2f} "
              f"final {rt[-1]:.2f} | |rho| max {np.abs(r['rho']).max():.3f}")


if __name__ == "__main__":
    main()
