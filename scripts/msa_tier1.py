#!/usr/bin/env python3
"""Tier 1 MSA characterisation for the spurious-correction analysis.

Compares one or more "mutant" MSAs to a WT MSA on:

  - depth (# sequences)
  - effective depth Neff at 80% identity threshold (HHsuite-style)
  - shared-sequence fraction (Jaccard on hit IDs)
  - per-position Shannon entropy (over 20 aa + gap)
  - conservation profile correlation (Pearson on entropy)

Refutes/supports the MSA-leak hypothesis at the data level before any
forward pass. See docs/diffusion_boltz.md §7.4 (Tier 1).

Usage:
    python scripts/msa_tier1.py \\
        --wt /path/to/wt.a3m \\
        --mutant /path/to/mut.a3m [--mutant /path/to/other.a3m ...] \\
        [--out DIR] [--id-threshold 0.8]
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# A-Z plus '-'. Anything else -> gap. Lowercase = insertion relative to query
# (a3m convention) and is stripped before alignment-matrix construction.
AA = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_IDX = {c: i for i, c in enumerate(AA)}
GAP_IDX = AA_TO_IDX["-"]
NUM_TOKENS = len(AA)  # 21


@dataclass
class MSA:
    path: Path
    ids: list[str]            # header IDs (first whitespace-separated token, '>' stripped)
    seqs: list[str]           # query-aligned sequences (uppercase + '-'), all length L
    matrix: np.ndarray        # (N, L) int8, indices into AA

    @property
    def depth(self) -> int:
        return self.matrix.shape[0]

    @property
    def length(self) -> int:
        return self.matrix.shape[1]

    @property
    def query(self) -> str:
        return self.seqs[0]


def parse_a3m(path: Path) -> MSA:
    ids: list[str] = []
    seqs: list[str] = []
    cur_id: str | None = None
    cur_seq_parts: list[str] = []

    def flush():
        if cur_id is None:
            return
        # a3m: lowercase letters are insertions relative to query — strip them.
        seq = re.sub(r"[a-z]", "", "".join(cur_seq_parts))
        # Anything outside the 21-token alphabet (e.g. 'X', 'B', 'Z', '*') -> gap.
        seq = "".join(c if c in AA_TO_IDX else "-" for c in seq)
        ids.append(cur_id)
        seqs.append(seq)

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                flush()
                # ID = first whitespace-separated token after '>'
                cur_id = line[1:].split()[0] if len(line) > 1 else ""
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line)
        flush()

    if not seqs:
        raise ValueError(f"No sequences parsed from {path}")

    L = len(seqs[0])
    bad = [(i, ids[i], len(s)) for i, s in enumerate(seqs) if len(s) != L]
    if bad:
        # Drop misaligned rows but warn.
        keep = [i for i, s in enumerate(seqs) if len(s) == L]
        print(f"[warn] {path.name}: dropping {len(bad)} rows with length != {L} "
              f"(first bad: {bad[:3]})")
        ids = [ids[i] for i in keep]
        seqs = [seqs[i] for i in keep]

    matrix = np.array(
        [[AA_TO_IDX[c] for c in s] for s in seqs],
        dtype=np.int8,
    )
    return MSA(path=path, ids=ids, seqs=seqs, matrix=matrix)


def compute_neff(matrix: np.ndarray, id_threshold: float = 0.8) -> float:
    """HHsuite-style Neff: sum_i 1 / (number of seqs within id_threshold of seq i).

    Identity = fraction of *non-gap-in-either* aligned columns that match.
    Symmetric, includes self.
    """
    N, L = matrix.shape
    weights = np.zeros(N, dtype=np.float32)
    # Chunk to avoid an N x N matrix when N is large.
    chunk = 256
    not_gap = matrix != GAP_IDX  # (N, L) bool
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        block = matrix[start:end]                              # (B, L)
        block_ng = not_gap[start:end]                          # (B, L)
        # match[b, j, l] = block[b,l] == matrix[j,l]
        # Avoid the full broadcast: do it column-by-column wouldn't help; use the
        # fact that matrix is int8 and L is small (~few hundred).
        eq = block[:, None, :] == matrix[None, :, :]           # (B, N, L)
        both_ng = block_ng[:, None, :] & not_gap[None, :, :]   # (B, N, L)
        n_compared = both_ng.sum(axis=-1).astype(np.float32)   # (B, N)
        n_matches = (eq & both_ng).sum(axis=-1).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            ident = np.where(n_compared > 0, n_matches / n_compared, 0.0)
        neighbors = (ident >= id_threshold).sum(axis=1)        # includes self
        weights[start:end] = 1.0 / np.maximum(neighbors, 1)
    return float(weights.sum())


def per_position_entropy(matrix: np.ndarray, include_gap: bool = False) -> np.ndarray:
    """Shannon entropy per column, in nats. Shape (L,)."""
    N, L = matrix.shape
    out = np.zeros(L, dtype=np.float64)
    for col in range(L):
        c = matrix[:, col]
        if not include_gap:
            c = c[c != GAP_IDX]
        if c.size == 0:
            out[col] = 0.0
            continue
        counts = np.bincount(c, minlength=NUM_TOKENS).astype(np.float64)
        if not include_gap:
            counts[GAP_IDX] = 0
        p = counts / counts.sum()
        nz = p > 0
        out[col] = -np.sum(p[nz] * np.log(p[nz]))
    return out


def shared_id_jaccard(a: MSA, b: MSA) -> tuple[float, int, int, int]:
    sa, sb = set(a.ids[1:]), set(b.ids[1:])  # exclude query
    inter = sa & sb
    union = sa | sb
    j = len(inter) / len(union) if union else 0.0
    return j, len(inter), len(sa), len(sb)


def query_identity(a: MSA, b: MSA) -> float:
    qa = a.matrix[0]
    qb = b.matrix[0]
    if qa.shape != qb.shape:
        return float("nan")
    both_ng = (qa != GAP_IDX) & (qb != GAP_IDX)
    if not both_ng.any():
        return float("nan")
    return float((qa[both_ng] == qb[both_ng]).mean())


def report(wt: MSA, mut: MSA, id_threshold: float, out_dir: Path | None) -> dict:
    print(f"\n=== {mut.path.name} vs {wt.path.name} ===")

    # Basic shape
    print(f"  WT  : depth={wt.depth:>5d}  L={wt.length}")
    print(f"  mut : depth={mut.depth:>5d}  L={mut.length}")

    # Query-vs-query identity (verify perturbation %)
    qid = query_identity(wt, mut)
    print(f"  query-vs-query identity         : {qid:.4f}  "
          f"(=> ~{(1 - qid) * 100:.1f}% mutated)")

    # Shared-sequence fraction (Jaccard on UniRef IDs)
    j, inter, na, nb = shared_id_jaccard(wt, mut)
    print(f"  shared-hit-ID Jaccard           : {j:.4f}  "
          f"(|∩|={inter}, |WT|={na}, |mut|={nb})")
    overlap_in_mut = inter / nb if nb else 0.0
    overlap_in_wt = inter / na if na else 0.0
    print(f"  fraction of mut hits seen in WT : {overlap_in_mut:.4f}")
    print(f"  fraction of WT  hits seen in mut: {overlap_in_wt:.4f}")

    # Neff
    neff_wt = compute_neff(wt.matrix, id_threshold)
    neff_mut = compute_neff(mut.matrix, id_threshold)
    print(f"  Neff @ id>={id_threshold}            : "
          f"WT={neff_wt:.1f}  mut={neff_mut:.1f}  "
          f"ratio={neff_mut / neff_wt:.3f}")

    # Per-position entropy + correlation
    e_wt = per_position_entropy(wt.matrix, include_gap=False)
    e_mut = per_position_entropy(mut.matrix, include_gap=False)
    if e_wt.shape == e_mut.shape:
        # Pearson on column-entropies
        r = float(np.corrcoef(e_wt, e_mut)[0, 1])
        # Per-column entropy delta stats
        delta = e_mut - e_wt
        print(f"  conservation-profile Pearson r  : {r:.4f}")
        print(f"  mean Δentropy (mut - wt)        : {delta.mean():+.4f}  "
              f"(median {np.median(delta):+.4f}, std {delta.std():.4f})")
    else:
        r = float("nan")
        print("  [warn] entropy profiles have different L; skipping correlation")

    # Stratify shared-ID overlap by sequence identity bucket of WT hits
    # (cheap signal of "does the mutant retain only the close homologs, or all?")
    if inter > 0 and nb > 0 and na > 0:
        wt_ids_set = set(wt.ids[1:])
        mut_ids_set = set(mut.ids[1:])
        # Bucket WT hits by their identity to WT query.
        wt_qid_to_query = (wt.matrix[1:] == wt.matrix[0][None, :]).sum(axis=1) / wt.length
        buckets = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        print("  WT-hit retention in mutant MSA, by WT-identity bucket:")
        for lo, hi in buckets:
            mask = (wt_qid_to_query >= lo) & (wt_qid_to_query < hi)
            bucket_ids = [wt.ids[1 + i] for i in np.where(mask)[0]]
            if not bucket_ids:
                continue
            kept = sum(1 for x in bucket_ids if x in mut_ids_set)
            print(f"    [{lo:.2f}, {hi:.2f}) : "
                  f"{kept:>4d}/{len(bucket_ids):>4d}  "
                  f"({kept / len(bucket_ids):.3f})")

    out = {
        "wt_depth": wt.depth, "mut_depth": mut.depth,
        "query_identity": qid,
        "shared_jaccard": j, "shared_inter": inter,
        "neff_wt": neff_wt, "neff_mut": neff_mut, "neff_ratio": neff_mut / neff_wt,
        "entropy_pearson_r": r,
        "entropy_wt": e_wt, "entropy_mut": e_mut,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Use parent dir name in the filename so cohort layouts (p20/, p40/, p70/)
        # with identical .a3m basenames don't overwrite each other.
        tag = f"{mut.path.parent.name}__{mut.path.stem}"
        plot_path = out_dir / f"entropy__{tag}__vs__{wt.path.stem}.png"
        fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
        ax0 = axes[0]
        ax0.plot(e_wt, label="WT", lw=1.0, color="tab:blue")
        ax0.plot(e_mut, label="mut", lw=1.0, color="tab:orange", alpha=0.8)
        ax0.set_ylabel("Shannon entropy (nats)")
        ax0.set_title(f"Per-position entropy — {mut.path.stem} vs {wt.path.stem}  "
                      f"(Pearson r = {r:.3f}, Neff ratio = {neff_mut / neff_wt:.2f}, "
                      f"shared Jaccard = {j:.2f})")
        ax0.legend(loc="upper right")
        ax0.grid(alpha=0.3)
        ax1 = axes[1]
        ax1.plot(e_mut - e_wt, lw=0.9, color="tab:red")
        ax1.axhline(0, color="k", lw=0.6)
        ax1.set_ylabel("Δentropy (mut − WT)")
        ax1.set_xlabel("position")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        print(f"  plot saved -> {plot_path}")

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wt", type=Path, required=True)
    p.add_argument("--mutant", type=Path, action="append", required=True,
                   help="One or more mutant MSAs (.a3m). Repeat the flag.")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional output directory for plots/CSVs.")
    p.add_argument("--id-threshold", type=float, default=0.8,
                   help="Identity threshold for Neff (default 0.8).")
    args = p.parse_args()

    print(f"loading WT MSA: {args.wt}")
    wt = parse_a3m(args.wt)
    print(f"  -> depth={wt.depth}, L={wt.length}")
    for mpath in args.mutant:
        print(f"\nloading mutant MSA: {mpath}")
        mut = parse_a3m(mpath)
        print(f"  -> depth={mut.depth}, L={mut.length}")
        report(wt, mut, args.id_threshold, args.out)


if __name__ == "__main__":
    main()
