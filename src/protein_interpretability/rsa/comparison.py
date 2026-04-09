"""Higher-level comparison workflows built on :mod:`.metrics`.

These functions take standardised representation dicts (from
:mod:`.loader`) and produce analysis results — RSA curves, CKA
matrices, mutation divergence profiles.

All heavy computation stays on GPU when inputs are on GPU.
Results are returned as dicts of tensors or plain Python structures
so that the caller chooses how to visualise or persist them.
"""

from __future__ import annotations

import torch
from torch import Tensor

from protein_interpretability.rsa.metrics import (
    cosine_rdm,
    linear_cka,
    partial_rsa,
    rsa,
    seq_sep_mask,
)


# ======================================================================
# Layerwise RSA: how structural is each layer?
# ======================================================================

def layerwise_rsa(
    layer_reps: dict[int, Tensor],
    reference_rdm: Tensor,
    min_seq_sep: int = 12,
    method: str = "spearman",
    rdm_fn=cosine_rdm,
) -> dict[str, list]:
    """RSA between each layer's RDM and a structural reference.

    Computes, for every layer, the correlation between the
    representation RDM and a reference RDM (typically Cα distances).
    Restricts to long-range pairs (``|i-j| >= min_seq_sep``) to remove
    the trivial chain-proximity signal.

    Parameters
    ----------
    layer_reps : dict[int, Tensor]
        ``{layer_idx: (N, D)}`` from :func:`.loader.load_representations`.
    reference_rdm : Tensor, shape ``(N, N)``
        Ground-truth dissimilarity (e.g. Cα distance matrix).
    min_seq_sep : int
        Minimum ``|i - j|`` for included pairs.  Set to 0 to include
        everything.
    method : ``"spearman"`` | ``"pearson"``
    rdm_fn : callable
        Function that maps ``(N, D) -> (N, N)`` RDM.
        Default :func:`.metrics.cosine_rdm`.

    Returns
    -------
    dict with keys:
        ``"layers"`` — sorted list of layer indices.
        ``"rsa"``    — list of RSA values (float), one per layer.
    """
    N = reference_rdm.shape[0]
    mask = seq_sep_mask(N, min_sep=min_seq_sep, device=reference_rdm.device)

    layers_sorted = sorted(layer_reps)
    rsa_values: list[float] = []

    for idx in layers_sorted:
        rep = layer_reps[idx]
        rep_rdm = rdm_fn(rep)
        r = rsa(rep_rdm, reference_rdm, mask=mask, method=method)
        rsa_values.append(r.item())

    return {"layers": layers_sorted, "rsa": rsa_values}


def layerwise_partial_rsa(
    layer_reps: dict[int, Tensor],
    reference_rdm: Tensor,
    confound_rdm: Tensor,
    min_seq_sep: int = 0,
    rdm_fn=cosine_rdm,
) -> dict[str, list]:
    """Partial RSA controlling for a confound (e.g. sequence separation).

    Same interface as :func:`layerwise_rsa` but uses
    :func:`.metrics.partial_rsa`.

    Parameters
    ----------
    layer_reps : dict[int, Tensor]
    reference_rdm : Tensor, shape ``(N, N)``
    confound_rdm : Tensor, shape ``(N, N)``
        Typically ``seq_sep_matrix(N)`` from :mod:`.loader`.
    min_seq_sep : int
        Additional mask on ``|i - j|``.  0 means no mask.
    rdm_fn : callable

    Returns
    -------
    dict with ``"layers"`` and ``"partial_rsa"`` lists.
    """
    N = reference_rdm.shape[0]
    mask = seq_sep_mask(N, min_sep=min_seq_sep, device=reference_rdm.device) if min_seq_sep > 0 else None

    layers_sorted = sorted(layer_reps)
    values: list[float] = []

    for idx in layers_sorted:
        rep_rdm = rdm_fn(layer_reps[idx])
        r = partial_rsa(rep_rdm, reference_rdm, confound_rdm, mask=mask)
        values.append(r.item())

    return {"layers": layers_sorted, "partial_rsa": values}


# ======================================================================
# Cross-layer CKA: which layers are similar?
# ======================================================================

def cross_layer_cka(layer_reps: dict[int, Tensor]) -> dict[str, object]:
    """L × L CKA similarity matrix across layers.

    Parameters
    ----------
    layer_reps : dict[int, Tensor]
        ``{layer_idx: (N, D)}``.

    Returns
    -------
    dict with keys:
        ``"layers"`` — sorted list of layer indices.
        ``"cka"``    — Tensor of shape ``(L, L)`` with CKA values.
    """
    layers_sorted = sorted(layer_reps)
    L = len(layers_sorted)
    reps = [layer_reps[i] for i in layers_sorted]

    cka_matrix = torch.zeros(L, L, device=reps[0].device)

    for i in range(L):
        cka_matrix[i, i] = 1.0
        for j in range(i + 1, L):
            c = linear_cka(reps[i], reps[j])
            cka_matrix[i, j] = c
            cka_matrix[j, i] = c

    return {"layers": layers_sorted, "cka": cka_matrix}


# ======================================================================
# Mutation analysis: WT vs mutant divergence
# ======================================================================

def mutation_divergence(
    wt_reps: dict[int, Tensor],
    mut_reps: dict[int, Tensor],
    method: str = "cosine",
) -> dict[str, list]:
    """Per-layer global divergence between WT and mutant representations.

    For each layer, measures how different the two representation matrices
    are overall.  This shows *at which layers* a mutation causes the
    model to diverge.

    Parameters
    ----------
    wt_reps, mut_reps : dict[int, Tensor]
        ``{layer_idx: (N, D)}``.  Must share the same layer indices
        and sequence length.
    method : ``"cosine"`` | ``"cka"`` | ``"frobenius"``
        - ``"cosine"``: 1 - mean per-residue cosine similarity.
        - ``"cka"``: 1 - CKA between the two matrices.
        - ``"frobenius"``: Frobenius norm of the difference, normalised
          by the mean of the two norms.

    Returns
    -------
    dict with keys:
        ``"layers"``     — sorted list of layer indices.
        ``"divergence"`` — list of divergence values (0 = identical).
    """
    common = sorted(set(wt_reps) & set(mut_reps))
    if not common:
        raise ValueError("No overlapping layer indices between WT and mutant.")

    values: list[float] = []
    for idx in common:
        w, m = wt_reps[idx], mut_reps[idx]
        if method == "cosine":
            cos_sim = torch.nn.functional.cosine_similarity(w, m, dim=-1)  # (N,)
            values.append((1.0 - cos_sim.mean()).item())
        elif method == "cka":
            values.append((1.0 - linear_cka(w, m)).item())
        elif method == "frobenius":
            diff_norm = (w - m).norm()
            avg_norm = (w.norm() + m.norm()) / 2 + 1e-12
            values.append((diff_norm / avg_norm).item())
        else:
            raise ValueError(f"Unknown method '{method}'")

    return {"layers": common, "divergence": values}


def perturbation_profile(
    wt_reps: dict[int, Tensor],
    mut_reps: dict[int, Tensor],
) -> dict[str, object]:
    """Per-residue, per-layer perturbation magnitude.

    For each layer and each residue position, computes the L2 norm
    of the representation difference.  This shows *which residues*
    are most affected by the mutation at each layer.

    Parameters
    ----------
    wt_reps, mut_reps : dict[int, Tensor]
        ``{layer_idx: (N, D)}``.

    Returns
    -------
    dict with keys:
        ``"layers"``  — sorted list of layer indices.
        ``"profile"`` — Tensor of shape ``(L, N)`` with per-position
                        L2 perturbation norms.
    """
    common = sorted(set(wt_reps) & set(mut_reps))
    if not common:
        raise ValueError("No overlapping layer indices between WT and mutant.")

    profiles = []
    for idx in common:
        delta = wt_reps[idx] - mut_reps[idx]  # (N, D)
        profiles.append(delta.norm(dim=-1))    # (N,)

    return {"layers": common, "profile": torch.stack(profiles)}  # (L, N)


def rsa_divergence(
    wt_reps: dict[int, Tensor],
    mut_reps: dict[int, Tensor],
    reference_rdm: Tensor,
    min_seq_sep: int = 12,
    rdm_fn=cosine_rdm,
) -> dict[str, list]:
    """Compare RSA curves: ΔRSA = RSA_wt - RSA_mut at each layer.

    A positive ΔRSA at a layer means the WT representation is *more*
    structurally aligned than the mutant at that layer — the mutation
    disrupts the model's structural encoding.

    Parameters
    ----------
    wt_reps, mut_reps : dict[int, Tensor]
    reference_rdm : Tensor, shape ``(N, N)``
    min_seq_sep : int
    rdm_fn : callable

    Returns
    -------
    dict with ``"layers"``, ``"rsa_wt"``, ``"rsa_mut"``, ``"delta_rsa"``.
    """
    wt_curve = layerwise_rsa(wt_reps, reference_rdm, min_seq_sep, rdm_fn=rdm_fn)
    mut_curve = layerwise_rsa(mut_reps, reference_rdm, min_seq_sep, rdm_fn=rdm_fn)

    # Align on common layers
    wt_dict = dict(zip(wt_curve["layers"], wt_curve["rsa"]))
    mut_dict = dict(zip(mut_curve["layers"], mut_curve["rsa"]))
    common = sorted(set(wt_dict) & set(mut_dict))

    return {
        "layers": common,
        "rsa_wt": [wt_dict[i] for i in common],
        "rsa_mut": [mut_dict[i] for i in common],
        "delta_rsa": [wt_dict[i] - mut_dict[i] for i in common],
    }
