"""Gradient attribution on Boltz2 distogram logits.

V1 surfaces: query embedding (output of ``model.input_embedder``) and MSA
module output (last call). Pair-representation gradients are deferred — see
``Lab/protein-interp/log/2026-05-05-gradient-attribution-design.md``.
"""

from .capture import GradientCapture
from .io import (
    SCHEMA_VERSION,
    AttributionResult,
    collect_provenance,
    load_result,
    save_result,
)
from .runner import run_per_step
from .targets import (
    DEFAULT_CONTACT_BIN_HI,
    DEFAULT_NUM_BINS,
    AttributionTarget,
    ContactBinNLL,
    DistogramKL,
    MeanContactNLL,
    PairLogProb,
)

__all__ = [
    "AttributionResult",
    "AttributionTarget",
    "ContactBinNLL",
    "DEFAULT_CONTACT_BIN_HI",
    "DEFAULT_NUM_BINS",
    "DistogramKL",
    "GradientCapture",
    "MeanContactNLL",
    "PairLogProb",
    "SCHEMA_VERSION",
    "collect_provenance",
    "load_result",
    "run_per_step",
    "save_result",
]
