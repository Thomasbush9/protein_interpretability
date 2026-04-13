"""Batch divergence pipeline: WT vs mutant distance across layers and steps.

Loops over mutant directories, loads multi-step hidden representations,
computes per-layer divergence from wildtype using existing
:func:`.comparison.mutation_divergence`, and returns a flat DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from protein_interpretability.rsa.comparison import mutation_divergence
from protein_interpretability.rsa.loader import load_all_steps

logger = logging.getLogger(__name__)


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


def compute_divergence(
    wt_reps_path: str | Path,
    mutant_dirs: dict[str, list[Path]],
    model_type: str = "boltz2",
    site: str = "layer_s",
    methods: list[str] | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Compute per-layer, per-step divergence from WT for every mutant.

    Parameters
    ----------
    wt_reps_path : str | Path
        Path to the wildtype ``hidden_reps.pt`` file.
    mutant_dirs : dict[str, list[Path]]
        Mapping from class label (e.g. ``"high_effect"``, ``"neutral"``)
        to lists of mutant directories, each containing ``hidden_reps.pt``.
    model_type : str
        Model type for the loader (default ``"boltz2"``).
    site : str
        Activation site to extract (default ``"layer_s"``).
    methods : list[str] | None
        Divergence methods to compute.  Default ``["cosine", "frobenius"]``.
        Supported: ``"cosine"``, ``"frobenius"``, ``"cka"``.
    device : str
        Torch device for loading tensors.

    Returns
    -------
    pd.DataFrame
        Flat table with columns: ``mutant_id``, ``class``, ``step``,
        ``layer``, and one column per divergence method (e.g.
        ``cosine_div``, ``frobenius_div``).
    """
    if methods is None:
        methods = ["cosine", "frobenius"]

    # Load WT once (all steps)
    logger.info("Loading wildtype representations from %s", wt_reps_path)
    wt_steps = load_all_steps(wt_reps_path, model_type, site, device)
    logger.info("WT loaded: %d steps, %d layers per step",
                len(wt_steps), len(next(iter(wt_steps.values()))))

    records: list[dict] = []
    total = sum(len(dirs) for dirs in mutant_dirs.values())
    processed = 0

    for cls_label, dirs in mutant_dirs.items():
        for mut_dir in dirs:
            mutant_id = mut_dir.name
            reps_path = mut_dir / "hidden_reps.pt"

            if not reps_path.exists():
                logger.warning("Missing hidden_reps.pt in %s, skipping", mut_dir)
                continue

            mut_steps = load_all_steps(reps_path, model_type, site, device)

            # Iterate over steps present in both WT and mutant
            common_steps = sorted(set(wt_steps) & set(mut_steps))
            for step_idx in common_steps:
                # Compute divergence for each method
                div_by_method: dict[str, dict[int, float]] = {}
                for method in methods:
                    result = mutation_divergence(
                        wt_steps[step_idx], mut_steps[step_idx], method=method,
                    )
                    div_by_method[method] = dict(
                        zip(result["layers"], result["divergence"])
                    )

                # Build rows (one per layer)
                layers = sorted(div_by_method[methods[0]].keys())
                for layer_idx in layers:
                    row = {
                        "mutant_id": mutant_id,
                        "class": cls_label,
                        "step": step_idx,
                        "layer": layer_idx,
                    }
                    for method in methods:
                        row[f"{method}_div"] = div_by_method[method][layer_idx]
                    records.append(row)

            # Free mutant tensors
            del mut_steps

            processed += 1
            if processed % 20 == 0 or processed == total:
                logger.info("Processed %d/%d mutants", processed, total)

    df = pd.DataFrame(records)
    logger.info("Done. DataFrame shape: %s", df.shape)
    return df
