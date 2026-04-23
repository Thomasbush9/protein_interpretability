"""Layer-wise comparison of pairformer hidden reps against a reference.

Works on the raw ``hidden_reps.pt`` dict produced by
``protein_interpretability.extract_hidden_reps``. Multi-step files use
top-level keys ``step_0``, ``step_1``, ... each mapping to a dict of
``pairformer_{L}_{site}`` tensors. Flat (single-step) files are also
supported transparently.

Pairwise sites (``layer_z``, ``tri_att_start``, ``tri_att_end``) carry
``(N, N, D)`` tensors; per-token sites (``layer_s``) carry ``(N, D)``.
All metric functions reduce over the feature dim (``dim=-1``) so they
work for both without branching.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch


PAIR_SITES: tuple[str, ...] = ("layer_z", "tri_att_start", "tri_att_end")
TOKEN_SITES: tuple[str, ...] = ("layer_s",)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _key(layer_n: int, layer_type: str, prefix: str = "pairformer") -> str:
    return f"{prefix}_{layer_n}_{layer_type}"


def _step_key(step: int | str) -> str:
    if isinstance(step, int):
        return f"step_{step}"
    if isinstance(step, str):
        return step if step.startswith("step_") else f"step_{step}"
    raise TypeError(f"step must be int or str, got {type(step).__name__}")


def _step_sub(reps: dict, step: int | str) -> dict:
    """Return the inner layer dict for ``step``, falling back to ``reps`` for flat files."""
    key = _step_key(step)
    return reps[key] if key in reps else reps


def load_reps(path: str | Path, device: str = "cpu") -> dict:
    """Load a raw ``hidden_reps.pt`` file as a nested dict."""
    return torch.load(path, map_location=device, weights_only=True)


def get_layer_ids(
    reps: dict,
    step: int | str,
    layer_type: str = "layer_z",
    prefix: str = "pairformer",
) -> list[int]:
    """Sorted layer indices present for ``{prefix}_{L}_{layer_type}`` keys."""
    sub = _step_sub(reps, step)
    pat = re.compile(rf"{re.escape(prefix)}_(\d+)_{re.escape(layer_type)}$")
    return sorted({int(m.group(1)) for k in sub if (m := pat.match(str(k)))})


def get_tensor(
    reps: dict,
    step: int | str,
    layer_n: int,
    layer_type: str = "layer_z",
    prefix: str = "pairformer",
) -> torch.Tensor:
    """Return the tensor for ``(step, layer_n, layer_type)`` with the batch dim squeezed."""
    sub = _step_sub(reps, step)
    return sub[_key(layer_n, layer_type, prefix)].squeeze(0)


# ---------------------------------------------------------------------------
# Metric functions: (ref, mut) -> np.ndarray over positions
# ---------------------------------------------------------------------------

def cosine_map(ref: torch.Tensor, mut: torch.Tensor) -> np.ndarray:
    """Per-position cosine similarity over the feature dim. Range ``[-1, 1]``."""
    return torch.nn.functional.cosine_similarity(ref, mut, dim=-1).cpu().numpy()


def l2_map(ref: torch.Tensor, mut: torch.Tensor) -> np.ndarray:
    """Per-position L2 distance ``||ref - mut||_2`` over the feature dim."""
    return (ref - mut).norm(dim=-1).cpu().numpy()


def norm_diff_map(ref: torch.Tensor, mut: torch.Tensor) -> np.ndarray:
    """Per-position absolute difference of feature-vector norms ``| ||ref|| - ||mut|| |``."""
    return (ref.norm(dim=-1) - mut.norm(dim=-1)).abs().cpu().numpy()


MetricFn = Callable[[torch.Tensor, torch.Tensor], np.ndarray]
METRICS: dict[str, MetricFn] = {
    "cosine": cosine_map,
    "l2": l2_map,
    "norm_diff": norm_diff_map,
}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize(arr: np.ndarray) -> dict[str, float]:
    """Reduce a per-position metric map to scalar stats."""
    flat = np.asarray(arr).reshape(-1)
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "std": float(flat.std()),
        "n_positions": int(flat.size),
    }


def compare_reps(
    ref: dict,
    mut: dict,
    step: int | str,
    layer_type: str = "layer_z",
    metric: str = "cosine",
    prefix: str = "pairformer",
) -> pd.DataFrame:
    """Per-layer summary stats of ``metric(ref, mut)`` at a chosen step/site.

    Keys are matched as ``{prefix}_{L}_{layer_type}`` — use
    ``prefix="pairformer"`` for Boltz2 and ``prefix="trunk"`` for ESMFold
    (or any other prefix used by a new extractor).

    Returns a DataFrame with columns
    ``layer, mean, median, min, max, std, n_positions``.
    Only layers present in both ``ref`` and ``mut`` are reported.
    """
    fn = METRICS.get(metric)
    if fn is None:
        raise ValueError(f"Unknown metric {metric!r}. Supported: {sorted(METRICS)}")

    ref_layers = set(get_layer_ids(ref, step, layer_type, prefix))
    mut_layers = set(get_layer_ids(mut, step, layer_type, prefix))
    layers = sorted(ref_layers & mut_layers)
    if not layers:
        raise ValueError(
            f"No overlapping layers for step={step} prefix={prefix!r} "
            f"layer_type={layer_type!r}. "
            f"ref={sorted(ref_layers)} mut={sorted(mut_layers)}"
        )

    rows: list[dict] = []
    for lid in layers:
        r = get_tensor(ref, step, lid, layer_type, prefix)
        m = get_tensor(mut, step, lid, layer_type, prefix)
        if r.shape != m.shape:
            raise ValueError(
                f"Shape mismatch at layer {lid} ({prefix}_*_{layer_type}): "
                f"ref {tuple(r.shape)} vs mut {tuple(m.shape)}"
            )
        rows.append({"layer": lid, **summarize(fn(r, m))})
    return pd.DataFrame(rows)
