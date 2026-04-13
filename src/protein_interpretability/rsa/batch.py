"""Batch divergence pipeline: WT vs mutant distance across layers and steps.

Loops over mutant directories, loads multi-step hidden representations,
computes per-layer divergence from wildtype at multiple spatial scales
(mutation site, local window, max residue, full sequence), and returns
a flat DataFrame.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import torch
import pandas as pd
from torch import Tensor

from protein_interpretability.rsa.loader import load_all_steps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mutation-position parsing (from selected.tsv)
# ---------------------------------------------------------------------------

_MUT_RE = re.compile(r"^([A-Z])([A-Z*])(\d+)([A-Z*])$")
_SEQ_IDX_RE = re.compile(r"seq_(\d+)")


def parse_mutation_positions(mutations_str: str) -> list[int]:
    """Extract 1-indexed mutation positions from a colon-separated token string.

    Example: ``"SA108D:SG46E"`` → ``[108, 46]``
    """
    positions = []
    for tok in mutations_str.split(":"):
        tok = tok.strip()
        if not tok:
            continue
        m = _MUT_RE.match(tok)
        if m is None:
            raise ValueError(f"Unparseable mutation token: {tok!r}")
        positions.append(int(m.group(3)))
    return positions


def load_mutation_map(tsv_path: str | Path) -> dict[str, list[int]]:
    """Load a ``selected.tsv`` and return ``{seq_id: [positions]}```.

    The TSV has columns ``idx``, ``mutations``, ``brightness``.
    Positions are converted from 1-indexed (TSV) to 0-indexed (tensor).

    Returns
    -------
    dict[str, list[int]]
        ``{"seq_00132": [107, 45], ...}`` (0-indexed positions).
    """
    tsv_path = Path(tsv_path)
    mut_map: dict[str, list[int]] = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = int(row["idx"])
            seq_id = f"seq_{idx:05d}"
            positions_1idx = parse_mutation_positions(row["mutations"])
            # Convert to 0-indexed
            mut_map[seq_id] = [p - 1 for p in positions_1idx]
    return mut_map


# ---------------------------------------------------------------------------
# Per-residue divergence helpers
# ---------------------------------------------------------------------------

def _cosine_per_residue(wt: Tensor, mut: Tensor) -> Tensor:
    """1 - cosine_sim per residue, shape (N,)."""
    return 1.0 - torch.nn.functional.cosine_similarity(wt, mut, dim=-1)


def _l2_per_residue(wt: Tensor, mut: Tensor) -> Tensor:
    """L2 norm of difference per residue, shape (N,)."""
    return (wt - mut).norm(dim=-1)


_RESIDUE_METRICS = {
    "cosine": _cosine_per_residue,
    "l2": _l2_per_residue,
}


def _build_window_mask(
    positions: list[int], seq_len: int, window: int,
) -> list[int]:
    """Expand mutation positions by ±window, clipped to [0, seq_len)."""
    indices = set()
    for p in positions:
        for offset in range(-window, window + 1):
            idx = p + offset
            if 0 <= idx < seq_len:
                indices.add(idx)
    return sorted(indices)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_mutant_dirs(base_dir: str | Path) -> list[Path]:
    """Find all subdirectories containing ``hidden_reps.pt``.

    Parameters
    ----------
    base_dir : str | Path
        Root directory to search recursively.

    Returns
    -------
    list[Path]
        Sorted list of directories (parents of ``hidden_reps.pt`` files).
    """
    base_dir = Path(base_dir)
    return sorted(d.parent for d in base_dir.rglob("hidden_reps.pt"))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_divergence(
    wt_reps_path: str | Path,
    mutant_dirs: dict[str, list[Path]],
    mutation_map: dict[str, list[int]] | None = None,
    model_type: str = "boltz2",
    site: str = "layer_s",
    metric: str = "cosine",
    windows: list[int] | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Compute per-layer, per-step divergence from WT for every mutant.

    When ``mutation_map`` is provided, computes spatially resolved metrics:
    mutation-site, windowed, max-residue, and full-sequence divergence.
    Without it, only full-sequence mean and max are computed.

    Parameters
    ----------
    wt_reps_path : str | Path
        Path to the wildtype ``hidden_reps.pt`` file.
    mutant_dirs : dict[str, list[Path]]
        Mapping from class label (e.g. ``"high_effect"``, ``"neutral"``)
        to lists of mutant directories, each containing ``hidden_reps.pt``.
    mutation_map : dict[str, list[int]] | None
        ``{seq_id: [0-indexed positions]}``.  From :func:`load_mutation_map`.
    model_type : str
        Model type for the loader (default ``"boltz2"``).
    site : str
        Activation site to extract (default ``"layer_s"``).
    metric : str
        Per-residue metric: ``"cosine"`` (1 - cos_sim) or ``"l2"``.
    windows : list[int] | None
        Window half-widths for windowed divergence.
        Default ``[0, 5, 10]`` (0 = mutation site only).
    device : str
        Torch device for loading tensors.

    Returns
    -------
    pd.DataFrame
        Columns: ``mutant_id``, ``class``, ``step``, ``layer``,
        ``mean_div``, ``max_div``, and if mutation_map provided:
        ``mut_site_div``, ``window_k_div`` for each k in windows.
    """
    if windows is None:
        windows = [0, 5, 10]

    residue_fn = _RESIDUE_METRICS.get(metric)
    if residue_fn is None:
        raise ValueError(
            f"Unknown metric '{metric}'. Supported: {list(_RESIDUE_METRICS)}"
        )

    # Load WT once (all steps)
    logger.info("Loading wildtype representations from %s", wt_reps_path)
    wt_steps = load_all_steps(wt_reps_path, model_type, site, device)
    logger.info(
        "WT loaded: %d steps, %d layers per step",
        len(wt_steps), len(next(iter(wt_steps.values()))),
    )

    records: list[dict] = []
    total = sum(len(dirs) for dirs in mutant_dirs.values())
    processed = 0
    skipped_no_map = 0

    for cls_label, dirs in mutant_dirs.items():
        for mut_dir in dirs:
            mutant_id = mut_dir.name
            reps_path = mut_dir / "hidden_reps.pt"

            if not reps_path.exists():
                logger.warning("Missing hidden_reps.pt in %s, skipping", mut_dir)
                continue

            # Look up mutation positions
            mut_positions = None
            if mutation_map is not None:
                mut_positions = mutation_map.get(mutant_id)
                if mut_positions is None:
                    skipped_no_map += 1
                    logger.debug(
                        "No mutation map entry for %s, site metrics will be NaN",
                        mutant_id,
                    )

            mut_steps = load_all_steps(reps_path, model_type, site, device)

            common_steps = sorted(set(wt_steps) & set(mut_steps))
            for step_idx in common_steps:
                wt_layer_reps = wt_steps[step_idx]
                mut_layer_reps = mut_steps[step_idx]
                layers = sorted(set(wt_layer_reps) & set(mut_layer_reps))

                for layer_idx in layers:
                    wt_t = wt_layer_reps[layer_idx]
                    mut_t = mut_layer_reps[layer_idx]
                    per_res = residue_fn(wt_t, mut_t)  # (N,)
                    seq_len = per_res.shape[0]

                    row: dict = {
                        "mutant_id": mutant_id,
                        "class": cls_label,
                        "step": step_idx,
                        "layer": layer_idx,
                        "mean_div": per_res.mean().item(),
                        "max_div": per_res.max().item(),
                    }

                    if mut_positions is not None:
                        # Mutation-site divergence (mean over mutated residues)
                        valid_pos = [p for p in mut_positions if p < seq_len]
                        if valid_pos:
                            row["mut_site_div"] = per_res[valid_pos].mean().item()
                        else:
                            row["mut_site_div"] = float("nan")

                        # Windowed divergence for each window size
                        for k in windows:
                            if k == 0:
                                # Same as mut_site_div
                                row["window_0_div"] = row["mut_site_div"]
                            else:
                                win_idx = _build_window_mask(
                                    valid_pos, seq_len, k,
                                )
                                row[f"window_{k}_div"] = (
                                    per_res[win_idx].mean().item()
                                    if win_idx else float("nan")
                                )
                    else:
                        row["mut_site_div"] = float("nan")
                        for k in windows:
                            row[f"window_{k}_div"] = float("nan")

                    records.append(row)

            del mut_steps
            processed += 1
            if processed % 20 == 0 or processed == total:
                logger.info("Processed %d/%d mutants", processed, total)

    if skipped_no_map > 0:
        logger.warning(
            "%d mutants had no entry in mutation_map (site metrics are NaN)",
            skipped_no_map,
        )

    df = pd.DataFrame(records)
    logger.info("Done. DataFrame shape: %s", df.shape)
    return df
