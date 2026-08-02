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
from pathlib import Path

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
            key=None, name: str | None = None, work=None) -> Extraction:
    """Featurise, run, and normalise. Model-specific quirks stop here."""
    import jax
    from mosaic.structure_prediction import TargetChain

    if key is None:
        key = jax.random.key(0)
    if name is not None and a3m is not None:
        features, _depth = features_for(name, model, seq, a3m, work=Path(work or "."))
    else:
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


# ---------------------------------------------------------------------------
# Alignment control
#
# Only the Boltz-2 wrapper honours TargetChain.msa_path. OF3 and Protenix ignore
# it and query api.colabfold.com, so the uniform path would silently run three
# models on three DIFFERENT alignments -- destroying the one variable the
# grafted-MSA design exists to hold fixed, and making runs network-dependent.
#
# Two defences, because "we passed the right path" is not the same as "the model
# used it":
#   1. per-model featurisation that injects our a3m the way each pipeline wants,
#   2. block_network(), which makes the MSA server unreachable, so any fallback
#      is a hard failure instead of a quiet substitution.
# ---------------------------------------------------------------------------

class MSAServerBlocked(RuntimeError):
    """Raised when a model tries to fetch an alignment instead of using ours."""


def block_network():
    """Make every known MSA-server entry point raise.

    Call before featurising. If a wrapper falls back to the server the run dies
    with this error rather than returning a plausible result computed from an
    alignment we did not choose.
    """
    def boom(*a, **k):
        raise MSAServerBlocked(
            "a model tried to query the ColabFold MSA server. The alignment must "
            "come from the a3m we supply -- see pi_models.features_for()."
        )

    blocked = []
    try:
        import protenix.web_service.colab_request_utils as cru
        cru.run_mmseqs2_service = boom
        blocked.append("protenix.run_mmseqs2_service")
    except Exception:
        pass
    for mod_path, attr in (("mosaic.msa", "preprocess_colabfold_msas"),
                           ("mosaic.models.of3", "preprocess_colabfold_msas")):
        try:
            import importlib
            m = importlib.import_module(mod_path)
            if hasattr(m, attr):
                setattr(m, attr, boom)
                blocked.append(f"{mod_path}.{attr}")
        except Exception:
            pass
    return blocked


def _a3m_depth(path) -> int:
    """Number of sequences in an a3m (lines starting '>')."""
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n


def msa_array(name: str, features):
    """The raw MSA array, wherever each pipeline keeps it."""
    if name == "of3":
        return np.asarray(features.msa)                    # [B, Nm, Nt, 32]
    if isinstance(features, dict) and "msa" in features:
        return np.asarray(features["msa"])
    return None


def msa_depth(name: str, features) -> int:
    """Number of alignment ROWS the model actually holds.

    The axis differs per pipeline, so it is identified by elimination against
    the token count rather than assumed -- reading the wrong axis silently
    returns the sequence length, which looks like a plausible depth.
    """
    m = msa_array(name, features)
    if m is None:
        return -1
    sh = [d for d in m.shape if d > 1]
    if name == "of3":
        return int(m.shape[1])
    # boltz-2 / protenix: [..., Nm, Nt] or [..., Nm, Nt, C]; the token axis is
    # the one matching N_token, so the depth is the other large axis
    return int(max(sh[:2])) if len(sh) >= 2 else -1


def features_for(name: str, model, seq: str, a3m: str, work: Path):
    """Featurise `seq` with OUR alignment, whatever the wrapper prefers.

    Returns (features, msa_depth). `msa_depth` is read back from the features,
    not from what we passed in, so it reflects what the model actually saw.
    """
    from mosaic.structure_prediction import TargetChain
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    a3m = str(Path(a3m).resolve())

    if name == "boltz2":
        # the only wrapper that honours msa_path
        feats, _ = model.target_only_features(
            [TargetChain(sequence=seq, use_msa=True, msa_path=a3m)])
        return feats, msa_depth("boltz2", feats)

    if name == "of3":
        # OF3's parser SKIPS any alignment whose basename is not a key of
        # msa.max_seq_counts -- it must be presented as colabfold_main.a3m.
        import shutil
        from mosaic.models.of3 import _build_query_set, _featurize
        d = work / "of3_msa"
        d.mkdir(parents=True, exist_ok=True)
        dst = d / "colabfold_main.a3m"
        shutil.copyfile(a3m, dst)
        qs = _build_query_set([TargetChain(sequence=seq, use_msa=True)])
        for q in qs.queries.values():
            for c in q.chains:
                c.main_msa_file_paths = [dst]
        feats, _atom = _featurize(qs)
        from jopenfold3.batch import Batch
        batch = feats if isinstance(feats, Batch) else Batch.from_torch_dict(feats)
        return batch, msa_depth("of3", batch)

    if name == "protenix":
        # Protenix wants a DIRECTORY holding pairing.a3m / non_pairing.a3m
        from protenix.data.template import ChainInput, featurize
        d = work / "protenix_msa"
        d.mkdir(parents=True, exist_ok=True)
        text = Path(a3m).read_text()
        (d / "non_pairing.a3m").write_text(text)
        # pairing.a3m is for multimer pairing; a single chain needs only the
        # query row, and an empty file makes the parser unhappy
        first = text.split(">", 2)
        (d / "pairing.a3m").write_text(">" + first[1] if len(first) > 1 else f">q\n{seq}\n")
        feats, _atom, _ = featurize(
            [ChainInput(sequence=seq, compute_msa=True, precomputed_msa_dir=str(d))])
        return feats, msa_depth("protenix", feats)

    raise KeyError(f"no alignment-controlled featuriser for {name!r}")


def extraction_from(out, name: str | None = None) -> Extraction:
    """Normalise a StructureModelOutput. Split out of run_one so callers that
    build features themselves (the alignment-controlled path) share it."""
    logits = np.asarray(out.distogram_logits)
    while logits.ndim > 3:
        logits = logits[0]
    n_bins = logits.shape[-1]
    if name in BIN_OVERRIDE and len(BIN_OVERRIDE[name]) == n_bins:
        centres = np.asarray(BIN_OVERRIDE[name], dtype=float)
    else:
        centres = bin_centres(np.asarray(out.distogram_bins), n_bins)
    p = softmax(logits)
    bb = np.asarray(out.backbone_coordinates)
    while bb.ndim > 3:
        bb = bb[0]
    plddt = np.asarray(out.plddt)
    while plddt.ndim > 1:
        plddt = plddt[0]
    return Extraction(
        logits=logits.astype(np.float32), p=p,
        ed=(p * centres).sum(-1).astype(np.float32),
        entropy=(-(p * np.log(p + 1e-12)).sum(-1)).astype(np.float32),
        ca=bb[:, 1].astype(np.float32), plddt=plddt.astype(np.float32),
        centres=centres.astype(np.float32), n_bins=n_bins,
    )


def sym_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jeffreys divergence per residue pair. Valid for any shared binning."""
    return ((p - q) * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1)
