"""Representational Similarity Analysis toolkit.

Architecture
------------
* :mod:`.metrics`    — pure-tensor ops (RDM, RSA, CKA). GPU-native.
* :mod:`.comparison` — analysis workflows (layerwise RSA, mutation divergence).
* :mod:`.loader`     — model-agnostic loading. Only module with model knowledge.

Quick start
-----------
::

    from protein_interpretability.rsa import (
        load_representations,
        load_distance_matrix,
        layerwise_rsa,
        cross_layer_cka,
        mutation_divergence,
        perturbation_profile,
    )

    # Load two mutants
    wt = load_representations("wt/hidden_reps.pt", model_type="boltz2")
    mut = load_representations("mut/hidden_reps.pt", model_type="boltz2")

    # Structural reference
    dist = load_distance_matrix("wt/structure.cif")

    # Analysis
    rsa_curve = layerwise_rsa(wt["layer_reps"], dist)
    cka_mat   = cross_layer_cka(wt["layer_reps"])
    div       = mutation_divergence(wt["layer_reps"], mut["layer_reps"])
    perturb   = perturbation_profile(wt["layer_reps"], mut["layer_reps"])
"""

from protein_interpretability.rsa.batch import (
    compute_divergence,
    compute_spatial_divergence,
    discover_mutant_dirs,
    load_mutation_map,
)
from protein_interpretability.rsa.comparison import (
    cross_layer_cka,
    layerwise_partial_rsa,
    layerwise_rsa,
    mutation_divergence,
    perturbation_profile,
    rsa_divergence,
)
from protein_interpretability.rsa.loader import (
    load_all_steps,
    load_distance_matrix,
    load_model_outputs,
    load_representations,
    seq_sep_matrix,
)
from protein_interpretability.rsa.metrics import (
    correlation_rdm,
    cosine_rdm,
    euclidean_rdm,
    linear_cka,
    partial_rsa,
    partial_spearman,
    pearson,
    rsa,
    seq_sep_mask,
    spearman,
    upper_tri,
    upper_tri_masked,
)

__all__ = [
    # loader
    "load_representations",
    "load_all_steps",
    "load_distance_matrix",
    "load_model_outputs",
    "seq_sep_matrix",
    # metrics
    "cosine_rdm",
    "euclidean_rdm",
    "correlation_rdm",
    "upper_tri",
    "upper_tri_masked",
    "seq_sep_mask",
    "pearson",
    "spearman",
    "partial_spearman",
    "rsa",
    "partial_rsa",
    "linear_cka",
    # batch
    "compute_divergence",
    "compute_spatial_divergence",
    "discover_mutant_dirs",
    "load_mutation_map",
    # comparison
    "layerwise_rsa",
    "layerwise_partial_rsa",
    "cross_layer_cka",
    "mutation_divergence",
    "perturbation_profile",
    "rsa_divergence",
]
