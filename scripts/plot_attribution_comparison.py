#!/usr/bin/env python3
"""Compare block-restricted vs whole-structure gradient attribution.

Loads two attribution_<stem>_<region>.pt files (block + whole) and produces
a 6-panel figure that addresses two questions in one figure:

  - Which channel (query vs MSA) carries the per-residue attribution?
  - Is the chromophore-block attribution map specific, or does the same
    pattern emerge when the loss spans the whole distogram?

Usage::

    python scripts/plot_attribution_comparison.py \\
        --block /path/to/attribution_seq_00132_block.pt \\
        --whole /path/to/attribution_seq_00132_whole.pt \\
        --wt_yaml  /path/to/original.yaml \\
        --mut_yaml /path/to/seq_00132.yaml \\
        --out_png  /path/to/comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", type=Path, required=True)
    p.add_argument("--whole", type=Path, required=True)
    p.add_argument("--wt_yaml", type=Path, required=True)
    p.add_argument("--mut_yaml", type=Path, required=True)
    p.add_argument("--out_png", type=Path, required=True)
    return p.parse_args()


def load_seq(yaml_path: Path) -> str:
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg["sequences"][0]["protein"]["sequence"]


def per_layer_profiles(state):
    pg = state["pair_grad_norm"].float()  # (n_layers, N, N)
    B = sorted(int(p) for p in state["block_positions_0idx"])
    B_t = torch.tensor(B)
    per_layer_all = pg.mean(dim=(1, 2)).numpy()
    pair_BB = pg[:, B_t][:, :, B_t]
    per_layer_BB = pair_BB.mean(dim=(1, 2)).numpy()
    return per_layer_all, per_layer_BB


def per_residue_channels(state):
    """Returns (query_grad_norm, msa_grad_norm) as numpy arrays of shape (N,)."""
    iq = state["input_grad_norm"].float().numpy()
    im_t = state.get("msa_grad_norm")
    im = im_t.float().numpy() if im_t is not None else None
    return iq, im


def main() -> None:
    args = parse_args()
    sb = torch.load(args.block, map_location="cpu", weights_only=False)
    sw = torch.load(args.whole, map_location="cpu", weights_only=False)

    if sb.get("region") != "block" or sw.get("region") != "whole":
        print(
            f"[warn] region tags don't match expectation: "
            f"block.region={sb.get('region')!r} whole.region={sw.get('region')!r}"
        )

    wt_seq = load_seq(args.wt_yaml)
    mut_seq = load_seq(args.mut_yaml)
    N = len(wt_seq)
    assert len(mut_seq) == N
    mutations = [i for i in range(N) if wt_seq[i] != mut_seq[i]]

    B = sorted(int(p) for p in sb["block_positions_0idx"])
    B_set = set(B)
    n_layers = sb["pair_grad_norm"].shape[0]

    pl_all_b, pl_BB_b = per_layer_profiles(sb)
    pl_all_w, pl_BB_w = per_layer_profiles(sw)
    iq_b, im_b = per_residue_channels(sb)
    iq_w, im_w = per_residue_channels(sw)

    if im_b is None or im_w is None:
        raise SystemExit(
            "msa_grad_norm missing in one of the files — re-run with the updated "
            "run_chromophore_attribution.py that captures msa_module.msa_proj."
        )

    positions = np.arange(N) + 1  # 1-indexed for display
    B_arr = np.array([p + 1 for p in B])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Block vs whole-structure attribution — {sb.get('record_id', '?')} "
        f"(N={N}, |B|={len(B)}, mutations={len(mutations)}, n_layers={n_layers})\n"
        f"loss(block)={float(sb['loss']):.3f}  "
        f"loss(whole)={float(sw['loss']):.3f}  "
        f"n_contacts(block)={sb['n_contacts']}  n_contacts(whole)={sw['n_contacts']}",
        fontsize=11,
    )

    # Row 1 = block loss; Row 2 = whole loss
    rows = [("block loss", sb, pl_all_b, pl_BB_b, iq_b, im_b, "C3"),
            ("whole loss", sw, pl_all_w, pl_BB_w, iq_w, im_w, "C0")]

    for r, (label, state, pl_all, pl_BB, iq, im, color) in enumerate(rows):
        ax_a, ax_b, ax_c = axes[r, 0], axes[r, 1], axes[r, 2]

        # (a) per-layer profile, log scale
        layers = np.arange(n_layers)
        ax_a.semilogy(layers, pl_all, "-o", ms=3, lw=1.0, color="C0", label="whole grid")
        ax_a.semilogy(layers, pl_BB, "-s", ms=3, lw=1.0, color="C3", label="B × B")
        ax_a.set_xlabel("Pairformer layer")
        ax_a.set_ylabel("mean ‖∇z‖")
        ax_a.set_title(f"({chr(97 + r * 3)}) Per-layer profile — {label}")
        ax_a.legend(fontsize=9)
        ax_a.grid(alpha=0.3, which="both")

        # (b) per-residue query + MSA, overlaid line plot
        ax_b.plot(positions, iq, color="C0", lw=0.8, label="query (input)")
        ax_b.plot(positions, im, color="C3", lw=0.8, label="MSA (msa_proj)")
        # B residues marked at top
        ymax_b = max(iq.max(), im.max())
        for p in B_arr:
            ax_b.axvline(p, color="grey", alpha=0.15, lw=0.6)
        ax_b.scatter(B_arr, [ymax_b * 1.04] * len(B_arr), marker="v",
                     color="grey", s=12, label="B residues")
        ax_b.set_xlabel("residue position (1-indexed)")
        ax_b.set_ylabel("‖∇·‖")
        ax_b.set_title(f"({chr(97 + r * 3 + 1)}) Per-residue query vs MSA — {label}")
        ax_b.legend(fontsize=9, loc="best")
        ax_b.set_xlim(0, N + 1)
        ax_b.set_ylim(top=ymax_b * 1.10)
        ax_b.grid(alpha=0.3, axis="y")

        # (c) query vs MSA scatter, B residues highlighted
        not_B = [i for i in range(N) if i not in B_set]
        not_B_idx = np.array(not_B)
        B_idx = np.array(B)
        ax_c.scatter(
            iq[not_B_idx], im[not_B_idx],
            s=12, color="lightgrey", alpha=0.7, label="not in B",
        )
        ax_c.scatter(
            iq[B_idx], im[B_idx],
            s=22, color="C3", alpha=0.85, label="B residues",
        )
        # Reference line: query = MSA
        m = max(iq.max(), im.max())
        ax_c.plot([0, m], [0, m], "--", color="grey", lw=0.6, alpha=0.5)
        ax_c.set_xlabel("query ‖∇‖")
        ax_c.set_ylabel("MSA ‖∇‖")
        ax_c.set_title(f"({chr(97 + r * 3 + 2)}) Query vs MSA per residue — {label}")
        ax_c.legend(fontsize=9, loc="best")
        ax_c.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=140)
    plt.close()
    print(f"[plot] wrote -> {args.out_png}")

    # ---- Console summary
    print("\n[summary] per-layer argmax / max / min")
    for label, pl_all, pl_BB in [
        ("block loss", pl_all_b, pl_BB_b),
        ("whole loss", pl_all_w, pl_BB_w),
    ]:
        print(
            f"  {label:>11}: whole-grid argmax={int(pl_all.argmax()):>2}  "
            f"range={pl_all.min():.2e} ↔ {pl_all.max():.2e}  "
            f"({pl_all.max()/pl_all.min():.1f}x);"
        )
        print(
            f"  {' ':>11}  B×B       argmax={int(pl_BB.argmax()):>2}  "
            f"range={pl_BB.min():.2e} ↔ {pl_BB.max():.2e}  "
            f"({pl_BB.max()/pl_BB.min():.1f}x)"
        )

    print("\n[summary] query vs MSA channel totals (sum over residues)")
    for label, iq, im in [
        ("block loss", iq_b, im_b),
        ("whole loss", iq_w, im_w),
    ]:
        q_total, m_total = float(iq.sum()), float(im.sum())
        ratio = q_total / m_total if m_total > 0 else float("inf")
        print(
            f"  {label:>11}: sum(query)={q_total:.3e}  sum(MSA)={m_total:.3e}  "
            f"query/MSA={ratio:.2f}"
        )

    print("\n[summary] correlation between block and whole maps")
    # Spearman would be more robust but requires scipy; pearson is fine for first cut.
    def _corr(a, b):
        a = np.asarray(a) - np.asarray(a).mean()
        b = np.asarray(b) - np.asarray(b).mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float((a * b).sum() / denom) if denom > 0 else float("nan")
    print(f"  query attribution: pearson(block, whole) = {_corr(iq_b, iq_w):.3f}")
    print(f"  MSA attribution:   pearson(block, whole) = {_corr(im_b, im_w):.3f}")
    print(f"  per-layer (whole grid): pearson = {_corr(pl_all_b, pl_all_w):.3f}")
    print(f"  per-layer (B × B):      pearson = {_corr(pl_BB_b, pl_BB_w):.3f}")

    print("\n[top 10 query-attributed residues, block loss]")
    print(f"  {'rank':>4}  {'pos':>4}  {'wt':>3}  {'mut':>3}  {'in_B':>5}  query  msa")
    top_idx = np.argsort(-iq_b)[:10]
    for r, i in enumerate(top_idx):
        flag_B = "yes" if i in B_set else ""
        print(
            f"  {r+1:>4}  {i+1:>4}  {wt_seq[i]:>3}  {mut_seq[i]:>3}  "
            f"{flag_B:>5}  {iq_b[i]:.2e}  {im_b[i]:.2e}"
        )

    print("\n[top 10 MSA-attributed residues, block loss]")
    top_idx = np.argsort(-im_b)[:10]
    for r, i in enumerate(top_idx):
        flag_B = "yes" if i in B_set else ""
        print(
            f"  {r+1:>4}  {i+1:>4}  {wt_seq[i]:>3}  {mut_seq[i]:>3}  "
            f"{flag_B:>5}  {iq_b[i]:.2e}  {im_b[i]:.2e}"
        )


if __name__ == "__main__":
    main()
