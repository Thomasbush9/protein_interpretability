"""Model-agnostic loading of per-layer representations.

This is the **only** module with model-specific knowledge.  It converts
outputs from Boltz2, ESM3, or any future model into a standardised dict
format that the rest of the RSA toolkit consumes.

Standard output format
----------------------
``load_representations`` returns a dict with:

* ``"layer_reps"`` — ``dict[int, Tensor]`` mapping layer index to a
  ``(N, D)`` per-token representation.
* ``"n_tokens"`` — ``int``, number of tokens (residues).
* ``"model_type"`` — ``str``, e.g. ``"boltz2"`` or ``"esm3"``.

``load_distance_matrix`` returns a ``(N, N)`` Cα distance matrix as a
torch Tensor.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


# ======================================================================
# Representation loading
# ======================================================================

def load_representations(
    path: str | Path,
    model_type: str = "boltz2",
    site: str = "layer_s",
    device: torch.device | str = "cpu",
) -> dict:
    """Load per-layer representations into a standardised format.

    Parameters
    ----------
    path : str | Path
        Path to the hidden-reps file (e.g. ``hidden_reps.pt``).
    model_type : ``"boltz2"`` | ``"esm3"``
        Which model produced the file.  Determines how keys are parsed.
    site : str
        Which activation site to extract.  For Boltz2 this is typically
        ``"layer_s"`` or ``"attention"``.  For ESM3 it is ``"block"``
        or ``"attn"``.
    device : torch.device | str
        Target device for loaded tensors.

    Returns
    -------
    dict
        ``{"layer_reps": {int: Tensor}, "n_tokens": int, "model_type": str}``
    """
    data = torch.load(path, map_location=device, weights_only=True)

    parser = _PARSERS.get(model_type)
    if parser is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported: {list(_PARSERS)}"
        )

    return parser(data, site, model_type)


def _parse_boltz2(data: dict, site: str, model_type: str) -> dict:
    """Parse Boltz2 hidden_reps.pt into standardised format.

    Boltz2 keys look like ``"pairformer_{layer_idx}_{site}"``.
    """
    layer_reps: dict[int, Tensor] = {}
    pattern = re.compile(rf"pairformer_(\d+)_{re.escape(site)}")

    for key, tensor in data.items():
        m = pattern.fullmatch(key)
        if m is not None:
            layer_idx = int(m.group(1))
            # Remove batch dimension if present: (1, N, D) -> (N, D)
            t = tensor.squeeze(0) if tensor.dim() == 3 else tensor
            layer_reps[layer_idx] = t

    if not layer_reps:
        available = sorted(data.keys())
        raise KeyError(
            f"No keys matching site '{site}' found. Available keys: {available[:10]}..."
        )

    n_tokens = next(iter(layer_reps.values())).shape[0]
    return {"layer_reps": layer_reps, "n_tokens": n_tokens, "model_type": model_type}


def _parse_esm3(data: dict, site: str, model_type: str) -> dict:
    """Parse ESM3 activations into standardised format.

    ESM3 keys look like ``"layer_{idx}_{site}"`` (e.g. ``"layer_5_block"``).
    """
    layer_reps: dict[int, Tensor] = {}
    pattern = re.compile(rf"layer_(\d+)_{re.escape(site)}")

    # data may be a list of step dicts (ESM3ActivationExtractor stores
    # a list); take the last step if so.
    if isinstance(data, list):
        data = data[-1] if data else {}

    for key, tensor in data.items():
        m = pattern.fullmatch(key)
        if m is not None:
            layer_idx = int(m.group(1))
            t = tensor.squeeze(0) if tensor.dim() == 3 else tensor
            layer_reps[layer_idx] = t

    if not layer_reps:
        available = sorted(data.keys())
        raise KeyError(
            f"No keys matching site '{site}' found. Available keys: {available[:10]}..."
        )

    n_tokens = next(iter(layer_reps.values())).shape[0]
    return {"layer_reps": layer_reps, "n_tokens": n_tokens, "model_type": model_type}


_PARSERS = {
    "boltz2": _parse_boltz2,
    "esm3": _parse_esm3,
}


# ======================================================================
# Structural distance matrix
# ======================================================================

def load_distance_matrix(
    structure_path: str | Path,
    chain_id: str | None = None,
    atom_name: str = "CA",
    device: torch.device | str = "cpu",
) -> Tensor:
    """Compute a Cα distance matrix from a PDB/mmCIF file.

    Reuses :func:`protein_interpretability.scoring.utils.load_structure`
    and :func:`~.extract_residue_coordinates` so that there is a single
    source of truth for structure parsing.

    Parameters
    ----------
    structure_path : str | Path
        Path to ``.pdb`` or ``.cif`` file.
    chain_id : str, optional
        Restrict to one chain.
    atom_name : str
        Atom to use (default ``"CA"``).
    device : torch.device | str

    Returns
    -------
    Tensor, shape ``(N, N)`` — pairwise Euclidean distances.
    """
    from protein_interpretability.scoring.utils import (
        extract_residue_coordinates,
        load_structure,
    )

    structure = load_structure(structure_path)
    coords_np, _ = extract_residue_coordinates(
        structure, chain_id=chain_id, atom_name=atom_name,
    )
    coords = torch.from_numpy(coords_np).float().to(device)
    return torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)


def load_model_outputs(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Load model_outputs.pt and return the dict on the given device.

    Parameters
    ----------
    path : str | Path
        Path to ``model_outputs.pt``.
    device : torch.device | str

    Returns
    -------
    dict[str, Tensor]
    """
    data = torch.load(path, map_location=device, weights_only=True)
    return data


def seq_sep_matrix(N: int, device: torch.device | str = "cpu") -> Tensor:
    """Absolute sequence-separation matrix ``|i - j|``.

    Useful as a confound RDM for :func:`~.metrics.partial_rsa`.

    Parameters
    ----------
    N : int
        Number of tokens.

    Returns
    -------
    Tensor, shape ``(N, N)``, dtype float32
    """
    idx = torch.arange(N, device=device, dtype=torch.float32)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
