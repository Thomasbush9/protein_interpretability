"""The normalised record every model must produce, and the checks that enforce it.

`pi_models` has cited this module -- as `pi_schema.py` -- since it was written:
"`run_one()` returns a plain dict of numpy arrays, the schema every
`exp_distomap_*` writes and every analysis script reads. See `pi_schema.py`."
That file has never existed in any commit. The schema was real but unwritten,
which means it has been enforced by three models happening to agree rather than
by anything that would notice when they stopped.

WHAT THIS CHECKS, AND WHY EACH CHECK IS HERE. Every rule below corresponds to a
way these three models actually differ, documented in `pi_models`' own docstring
as a quirk that had to be hand-handled:

  bins        A cross-model KL is only meaningful if both models bin distance
              the same way. Boltz-2 uses 2-22 A over 64 bins; the others report
              their own grid. Comparing a distogram against a mismatched centre
              vector produces a number, silently, and that number is wrong.
  plddt       pLDDT is per-ATOM in OpenFold3 and per-TOKEN in Protenix and
              Boltz-2. The two disagree in LENGTH, so the check is that plddt
              and the coordinates describe the same N.
  shapes      Sample and batch dimensions are squeezed by `run_one` with a
              `while ndim > 3` loop. If a wrapper ever returns a shape that loop
              does not reduce, the array stays [1,N,N,B] and every downstream
              reduction silently averages over a length-1 axis.
  p           Probabilities must sum to 1 along the bin axis. A softmax applied
              twice, or applied to already-normalised probabilities, still
              produces a plausible-looking array.

The module is numpy-only on purpose: the analysis layer must be able to
validate an artifact it reads without the environment that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Field -> (axes, meaning). Axes are symbolic: N tokens, B distogram bins.
EXTRACTION_SCHEMA: dict[str, tuple[tuple[str, ...], str]] = {
    "logits":  (("N", "N", "B"), "distogram logits"),
    "p":       (("N", "N", "B"), "distogram probabilities, sum 1 over B"),
    "ed":      (("N", "N"),      "expected distance, Angstrom"),
    "entropy": (("N", "N"),      "distogram entropy, nats"),
    "ca":      (("N", 3),        "CA coordinates, Angstrom"),
    "plddt":   (("N",),          "predicted lDDT, per token"),
    "centres": (("B",),          "distogram bin centres, Angstrom"),
}

# Tolerances. `p` is stored float32, so a row of 64 bins accumulates a few
# ulp of error; 1e-4 is loose enough for that and far tighter than any real
# normalisation bug, which shows up as a factor, not a rounding difference.
P_SUM_TOL = 1e-4
CENTRE_TOL = 1e-3


class SchemaError(ValueError):
    """A record does not satisfy the cross-model contract."""


@dataclass(frozen=True)
class RecordShape:
    """The two free dimensions of a record, recovered from the record itself."""
    n_tokens: int
    n_bins: int


def _get(record, field):
    if isinstance(record, dict):
        if field not in record:
            raise SchemaError(f"missing field {field!r}")
        return record[field]
    if not hasattr(record, field):
        raise SchemaError(f"missing field {field!r}")
    return getattr(record, field)


def shape_of(record) -> RecordShape:
    """N and B, taken from `logits`, which is the only field that carries both."""
    logits = np.asarray(_get(record, "logits"))
    if logits.ndim != 3:
        raise SchemaError(
            f"logits has {logits.ndim} dimensions {logits.shape}, expected 3 "
            "[N,N,B] -- an unsqueezed batch or sample axis reaches every "
            "downstream reduction as a length-1 average")
    if logits.shape[0] != logits.shape[1]:
        raise SchemaError(f"logits is not square in its token axes: {logits.shape}")
    return RecordShape(n_tokens=logits.shape[0], n_bins=logits.shape[2])


def validate(record, *, check_probabilities=True) -> RecordShape:
    """Raise `SchemaError` unless `record` satisfies the contract. Returns its shape.

    Accepts anything with the fields as attributes or keys, so it works on a
    `pi_models.Extraction`, a plain dict, or an `npz` handle.
    """
    shape = shape_of(record)
    dims = {"N": shape.n_tokens, "B": shape.n_bins}

    for field, (axes, meaning) in EXTRACTION_SCHEMA.items():
        arr = np.asarray(_get(record, field))
        want = tuple(dims[a] if isinstance(a, str) else a for a in axes)
        if arr.shape != want:
            raise SchemaError(
                f"{field} has shape {arr.shape}, expected {want} ({meaning}). "
                + ("pLDDT is per-ATOM in OpenFold3 and per-TOKEN in Protenix and "
                   "Boltz-2; a length mismatch here is usually that."
                   if field == "plddt" else ""))
        if not np.all(np.isfinite(arr)):
            bad = int((~np.isfinite(arr)).sum())
            raise SchemaError(f"{field} holds {bad} non-finite values")

    centres = np.asarray(_get(record, "centres"), dtype=float)
    if np.any(np.diff(centres) <= 0):
        raise SchemaError(
            "distogram bin centres are not strictly increasing; a cross-model "
            "comparison against this grid would be meaningless")

    if check_probabilities:
        p = np.asarray(_get(record, "p"), dtype=np.float64)
        sums = p.sum(-1)
        worst = float(np.abs(sums - 1.0).max())
        if worst > P_SUM_TOL:
            raise SchemaError(
                f"distogram probabilities deviate from 1 by up to {worst:.2e} "
                f"(tolerance {P_SUM_TOL:.0e}) -- a double softmax leaves an "
                "array that still looks like a distribution")

    return shape


def assert_comparable(a, b, *, what="records") -> None:
    """Raise unless two records may be compared bin-for-bin.

    This is the check a cross-model KL needs and never had. Two models can each
    be internally valid and still be incomparable, because a KL computed across
    two different distance grids is a well-formed number that means nothing.
    """
    sa, sb = shape_of(a), shape_of(b)
    if sa.n_bins != sb.n_bins:
        raise SchemaError(
            f"{what} bin the distance axis differently: {sa.n_bins} vs "
            f"{sb.n_bins} bins -- not comparable")
    ca = np.asarray(_get(a, "centres"), dtype=float)
    cb = np.asarray(_get(b, "centres"), dtype=float)
    worst = float(np.abs(ca - cb).max())
    if worst > CENTRE_TOL:
        raise SchemaError(
            f"{what} share a bin COUNT but not a bin GRID: centres differ by up "
            f"to {worst:.3f} A. Boltz-2 uses 2-22 A over 64 bins; a model that "
            "reports a different grid must not be compared against it directly")
    if sa.n_tokens != sb.n_tokens:
        raise SchemaError(
            f"{what} describe different token counts: {sa.n_tokens} vs "
            f"{sb.n_tokens}")


def describe(record) -> dict:
    """Tensor descriptors for an artifact's metadata block."""
    shape = shape_of(record)
    fields = {}
    for field in EXTRACTION_SCHEMA:
        arr = np.asarray(_get(record, field))
        fields[field] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return {"n_tokens": shape.n_tokens, "n_bins": shape.n_bins, "fields": fields}
