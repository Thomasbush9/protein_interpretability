from pathlib import Path

import torch
import numpy as np


def parse_a3m(path: str | Path) -> list[str]:
    """Parse an A3M file and return a list of aligned sequences (no headers)."""
    sequences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def conservation_score(msa_path: str | Path) -> np.ndarray:
    """Compute per-residue conservation from an A3M MSA file.

    For each column aligned to the reference (first) sequence, computes the
    fraction of sequences that match the reference amino acid at that position.
    Gaps in the reference are skipped; lowercase insertions in other sequences
    are ignored.

    Parameters
    ----------
    msa_path : path to .a3m file

    Returns
    -------
    np.ndarray of shape (L,) with conservation scores in [0, 1] for each
    position in the reference sequence.
    """
    sequences = parse_a3m(msa_path)
    ref = sequences[0]

    # Reference columns are the uppercase/non-gap positions of the first sequence
    ref_cols = [i for i, c in enumerate(ref) if c.isupper() and c != "-"]
    L = len(ref_cols)

    # For each other sequence, extract only reference-aligned columns
    # (skip lowercase insertion characters in a3m format)
    aligned = []
    for seq in sequences[1:]:
        cols = []
        col_idx = 0
        for c in seq:
            if c == "-" or c.isupper():
                # This is an aligned column
                if col_idx in ref_cols:
                    cols.append(c.upper())
                col_idx += 1
            # lowercase = insertion relative to reference, skip
        if len(cols) == L:
            aligned.append(cols)

    if not aligned:
        return np.ones(L)

    ref_residues = [ref[i] for i in ref_cols]
    conservation = np.zeros(L)
    for j in range(L):
        matches = sum(1 for seq in aligned if seq[j] == ref_residues[j])
        conservation[j] = matches / len(aligned)

    return conservation


def sort_layer_activations(layer_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    For one layer activation tensor (B, L, D), compute mean |act| per node and sort by most active.
    Returns:
        act_sorted: (D,) activity values in descending order
        top_indices: (D,) node indices that achieve that order
    """
    node_activity = layer_tensor.abs().squeeze().mean(dim=0).float().cpu().numpy()  # (D,)
    top_indices = np.argsort(-node_activity)
    act_sorted = node_activity[top_indices]
    return act_sorted, top_indices


