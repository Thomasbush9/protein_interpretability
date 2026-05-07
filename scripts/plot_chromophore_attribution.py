#!/usr/bin/env python3
"""Plot chromophore-block gradient-attribution results.

Reads an attribution_<stem>.pt produced by run_chromophore_attribution.py and
produces a 4-panel figure:

  (a) Per-Pairformer-layer mean pair-grad magnitude (whole grid)
  (b) Per-Pairformer-layer mean pair-grad magnitude restricted to B × B
  (c) Per-residue input-embedding attribution, with B and mutations marked
  (d) Layer × position heatmap (pair-grad summed over j)

Usage::

    python scripts/plot_chromophore_attribution.py \\
        /path/to/attribution_seq_00132.pt \\
        --wt_yaml  /path/to/original.yaml \\
        --mut_yaml /path/to/seq_00132.yaml \\
        --out_png  /path/to/attribution_seq_00132.png
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
    p.add_argument("attribution_pt", type=Path)
    p.add_argument("--wt_yaml", type=Path, required=True)
    p.add_argument("--mut_yaml", type=Path, required=True)
    p.add_argument("--out_png", type=Path, required=True)
    return p.parse_args()


def load_seq(yaml_path: Path) -> str:
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg["sequences"][0]["protein"]["sequence"]


def main() -> None:
    args = parse_args()
    state = torch.load(args.attribution_pt, map_location="cpu", weights_only=False)

    wt_seq = load_seq(args.wt_yaml)
    mut_seq = load_seq(args.mut_yaml)
    if len(wt_seq) != len(mut_seq):
        raise SystemExit(f"sequence length mismatch wt={len(wt_seq)} mut={len(mut_seq)}")
    N = len(wt_seq)

    pair_grad = state["pair_grad_norm"].float()  # (n_layers, N, N)
    input_grad = state["input_grad_norm"].float()  # (N,)
    B = sorted(int(p) for p in state["block_positions_0idx"])  # 0-indexed
    n_layers = pair_grad.shape[0]
    loss = float(state["loss"])
    record_id = state.get("record_id", "?")

    if pair_grad.shape[1] != N or input_grad.shape[0] != N:
        raise SystemExit(
            f"shape mismatch: pair_grad N={pair_grad.shape[1]} input_grad "
            f"N={input_grad.shape[0]} sequences N={N}"
        )

    # --- Mutation set
    mutations = [i for i in range(N) if wt_seq[i] != mut_seq[i]]
    print(
        f"sequence length={N}  mutations={len(mutations)} "
        f"({100*len(mutations)/N:.1f}%)"
    )
    print(f"|B|={len(B)}  loss={loss:.4f}  layers={n_layers}")

    # --- Aggregations
    B_t = torch.tensor(B)
    pair_BB = pair_grad[:, B_t][:, :, B_t]  # (n_layers, |B|, |B|)

    per_layer_all = pair_grad.mean(dim=(1, 2)).numpy()  # (n_layers,)
    per_layer_BB = pair_BB.mean(dim=(1, 2)).numpy()  # (n_layers,)
    layer_pos_heat = pair_grad.sum(dim=2).numpy()  # (n_layers, N)

    # --- Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Chromophore-block gradient attribution — {record_id}\n"
        f"loss={loss:.4f}  |B|={len(B)}  n_layers={n_layers}  "
        f"mutations={len(mutations)}/{N}",
        fontsize=12,
    )

    # (a) per-layer profile (whole grid + B×B overlay)
    ax = axes[0, 0]
    layers = np.arange(n_layers)
    ax.plot(layers, per_layer_all, "-o", ms=3, lw=1.0, color="C0", label="whole grid")
    ax2 = ax.twinx()
    ax2.plot(layers, per_layer_BB, "-s", ms=3, lw=1.0, color="C3", label="B × B only")
    ax.set_xlabel("Pairformer layer")
    ax.set_ylabel("mean ‖∇z‖ (whole grid)", color="C0")
    ax2.set_ylabel("mean ‖∇z‖ (B × B)", color="C3")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax.set_title("(a) Per-layer mean pair-rep gradient magnitude")
    ax.grid(alpha=0.3)
    # Combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)

    # (b) per-layer log scale (both axes) — easier to see distribution
    ax = axes[0, 1]
    ax.semilogy(layers, per_layer_all, "-o", ms=3, lw=1.0, color="C0", label="whole grid")
    ax.semilogy(layers, per_layer_BB, "-s", ms=3, lw=1.0, color="C3", label="B × B only")
    ax.set_xlabel("Pairformer layer")
    ax.set_ylabel("mean ‖∇z‖ (log scale)")
    ax.set_title("(b) Same, log scale")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # (c) per-residue input attribution, B + mutations marked
    ax = axes[1, 0]
    positions = np.arange(N) + 1  # 1-indexed for display
    ax.bar(positions, input_grad.numpy(), color="lightgrey", width=1.0, label="all residues")
    # B residues — overlay
    B_arr = np.array([p + 1 for p in B])  # 1-indexed
    ax.bar(B_arr, input_grad[B_t].numpy(), color="C0", width=1.0, label="B residues")
    # mutations among B — overlay in red
    B_set = set(B)
    mut_set = set(mutations)
    B_and_mut = sorted(B_set & mut_set)
    if B_and_mut:
        B_and_mut_arr = np.array([p + 1 for p in B_and_mut])
        ax.bar(
            B_and_mut_arr,
            input_grad[torch.tensor(B_and_mut)].numpy(),
            color="C3",
            width=1.0,
            label="B ∩ mutations",
        )
    # mutations outside B — light red ticks at the bottom
    mut_outside_B = sorted(mut_set - B_set)
    if mut_outside_B:
        ax.scatter(
            [p + 1 for p in mut_outside_B],
            [-input_grad.max().item() * 0.02] * len(mut_outside_B),
            color="C3",
            marker="|",
            s=20,
            alpha=0.4,
            label="mutations outside B",
        )
    ax.set_xlabel("residue position (1-indexed)")
    ax.set_ylabel("‖∇input embedding‖")
    ax.set_title("(c) Per-residue input-embedding attribution")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3, axis="y")
    ax.set_xlim(0, N + 1)

    # (d) layer × position heatmap
    ax = axes[1, 1]
    # Use log scale on values for visibility (add small epsilon)
    eps = 1e-12
    heat = np.log10(layer_pos_heat + eps)
    im = ax.imshow(
        heat,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=(0.5, N + 0.5, -0.5, n_layers - 0.5),
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="log10 sum_j ‖∇z[i,j]‖")
    # B residues — vertical ticks at top
    for p in B:
        ax.axvline(p + 1, color="white", lw=0.3, alpha=0.4)
    ax.set_xlabel("residue position i (1-indexed)")
    ax.set_ylabel("Pairformer layer")
    ax.set_title("(d) Layer × position attribution (sum over j; B residues marked)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=140)
    plt.close()
    print(f"[plot] wrote -> {args.out_png}")

    # --- Console summary
    print("\n[summary]")
    print(f"  argmax layer (whole grid): {int(per_layer_all.argmax())}")
    print(f"  argmax layer (B × B):      {int(per_layer_BB.argmax())}")
    print(
        f"  per-layer (whole grid) range: "
        f"min={per_layer_all.min():.3e}  max={per_layer_all.max():.3e}  "
        f"ratio={per_layer_all.max()/per_layer_all.min():.1f}x"
    )
    print(
        f"  per-layer (B × B) range:      "
        f"min={per_layer_BB.min():.3e}  max={per_layer_BB.max():.3e}  "
        f"ratio={per_layer_BB.max()/per_layer_BB.min():.1f}x"
    )

    # Top input-attributed residues
    top_idx = input_grad.argsort(descending=True)[:15].tolist()
    print("\n[top 15 input-attributed residues]")
    print(f"  {'rank':>4}  {'pos':>4}  {'wt':>3}  {'mut':>3}  {'in_B':>5}  ‖∇‖")
    for r, i in enumerate(top_idx):
        flag_B = "yes" if i in set(B) else ""
        print(
            f"  {r+1:>4}  {i+1:>4}  {wt_seq[i]:>3}  {mut_seq[i]:>3}  "
            f"{flag_B:>5}  {float(input_grad[i]):.3e}"
        )


if __name__ == "__main__":
    main()
