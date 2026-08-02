"""One adapter per model, one output schema for every analysis script.

The design constraint: **extractors may be model-specific, analysis must not be.**
Writing three copies of every analysis function is how the three models end up
being compared on quantities that are not actually the same quantity.

mosaic already provides most of this. `StructurePredictionModel` gives every
model the same two calls --

    features, writer = model.target_only_features([TargetChain(...)])
    out = model.model_output(features=features, recycling_steps=..., key=...)

-- and `out` is a `StructureModelOutput` with normalised fields:
`distogram_logits [N,N,B]`, `distogram_bins [B]`, `plddt [N]`,
`backbone_coordinates [N,4,3]`, `atom37_coords`, `residue_idx`.

Using it removes three things that had to be hand-handled when the OpenFold3
extractor was written directly against `jopenfold3`:
  * OF3's a3m basename filter and its unconditional template stage,
  * pLDDT being per-ATOM in OF3 and per-TOKEN in Protenix/Boltz-2,
  * the distogram bin grid, which is now *reported by the model* instead of
    being assumed to match Boltz-2's 2-22 A / 64.

That last one matters most: a cross-model KL is only meaningful if both models
bin distances the same way, and this is the difference between checking that and
hoping for it.

`run_one()` returns a plain dict of numpy arrays -- the schema every
`exp_distomap_*` writes and every analysis script reads. See `pi_schema.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _boltz2():
    from mosaic.models.boltz2 import Boltz2
    return Boltz2()


def _of3():
    from mosaic.models.of3 import OF3
    return OF3()


def _protenix():
    # load_model() returns the low-level protenij module, NOT the mosaic
    # wrapper -- both are called `Protenix`, which is an easy hour to lose.
    from mosaic.models.protenix import Protenix, load_model
    return Protenix(protenix=load_model(), default_sample_steps=20)


def _af2():
    from mosaic.models.af2 import load_af2
    return load_af2(multimer=False)


BUILDERS = {"boltz2": _boltz2, "of3": _of3, "protenix": _protenix, "af2": _af2}

# Grids we trust more than the wrapper's own `distogram_bins`.
#
# The mosaic Boltz-2 wrapper reports a grid spanning exactly 2.00-22.00 A over
# 64 entries, i.e. `linspace(2, 22, 64)`. Boltz-2's actual bin CENTRES are
# 2.15625 ... 21.84375 (width 0.3125) -- what `pi_core.BIN_CENTRES` holds and
# what every Boltz-2 number in this project was computed with. The two differ by
# up to ~0.16 A, which is small but systematic, and using the wrapper's version
# here would silently make new E[d] values disagree with the old ones.
# KL and entropy are unaffected either way: they are sums over bins.
_BOLTZ_MIN, _BOLTZ_MAX, _BOLTZ_B = 2.0, 22.0, 64
_w = (_BOLTZ_MAX - _BOLTZ_MIN) / _BOLTZ_B
BIN_OVERRIDE = {
    "boltz2": _BOLTZ_MIN + _w / 2 + _w * np.arange(_BOLTZ_B),
}


def available() -> list[str]:
    return sorted(BUILDERS)


def load(name: str):
    """Load a model by short name."""
    if name not in BUILDERS:
        raise KeyError(f"unknown model {name!r}; have {available()}")
    return BUILDERS[name]()


def bin_centres(bins: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin centres in Angstrom, from whatever convention the model reports.

    Models disagree about whether `distogram_bins` is centres (length B) or
    breaks (length B-1, AlphaFold's convention). Both are handled and the
    result is always length B, so downstream E[d] is comparable.
    """
    bins = np.asarray(bins, dtype=float)
    if len(bins) == n_bins:
        return bins
    if len(bins) == n_bins - 1:
        # AF-style breaks: first and last bins are open-ended, so extend by the
        # median spacing rather than inventing a width for them
        step = float(np.median(np.diff(bins)))
        lo, hi = bins[0] - step / 2, bins[-1] + step / 2
        edges = np.concatenate([[lo], bins, [hi]])
        return (edges[:-1] + edges[1:]) / 2
    raise ValueError(
        f"distogram_bins has length {len(bins)} for {n_bins} bins -- neither "
        "centres nor breaks; refusing to guess the grid")


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


@dataclass
class Extraction:
    """Everything the analysis layer needs, identical across models."""
    logits: np.ndarray      # [N, N, B]
    p: np.ndarray           # [N, N, B]
    ed: np.ndarray          # [N, N]   expected distance, Angstrom
    entropy: np.ndarray     # [N, N]   nats
    ca: np.ndarray          # [N, 3]
    plddt: np.ndarray       # [N]
    centres: np.ndarray     # [B]
    n_bins: int


def run_one(model, seq: str, a3m: str | None, *, recycles=3, sampling_steps=None,
            key=None, name: str | None = None) -> Extraction:
    """Featurise, run, and normalise. Model-specific quirks stop here."""
    import jax
    from mosaic.structure_prediction import TargetChain

    if key is None:
        key = jax.random.key(0)
    chain = TargetChain(sequence=seq, use_msa=a3m is not None, msa_path=a3m)
    features, _writer = model.target_only_features([chain])
    out = model.model_output(features=features, recycling_steps=recycles,
                             sampling_steps=sampling_steps, key=key)

    logits = np.asarray(out.distogram_logits)
    while logits.ndim > 3:              # drop batch / sample dims, keep [N,N,B]
        logits = logits[0]
    n_bins = logits.shape[-1]
    if name in BIN_OVERRIDE and len(BIN_OVERRIDE[name]) == n_bins:
        centres = np.asarray(BIN_OVERRIDE[name], dtype=float)
    else:
        centres = bin_centres(np.asarray(out.distogram_bins), n_bins)
    p = softmax(logits)

    # backbone_coordinates is [N, 4, 3] in N, CA, C, O order for every wrapper
    bb = np.asarray(out.backbone_coordinates)
    while bb.ndim > 3:
        bb = bb[0]
    ca = bb[:, 1]

    plddt = np.asarray(out.plddt)
    while plddt.ndim > 1:
        plddt = plddt[0]

    return Extraction(
        logits=logits.astype(np.float32), p=p,
        ed=(p * centres).sum(-1).astype(np.float32),
        entropy=(-(p * np.log(p + 1e-12)).sum(-1)).astype(np.float32),
        ca=ca.astype(np.float32), plddt=plddt.astype(np.float32),
        centres=centres.astype(np.float32), n_bins=n_bins,
    )


def sym_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jeffreys divergence per residue pair. Valid for any shared binning."""
    return ((p - q) * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1)
