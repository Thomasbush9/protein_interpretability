"""Does the schema refuse what it claims to refuse?

Every test asserts a REFUSAL or a round trip, never that the happy path runs --
a guard is worth exactly what it does on the bad path. The bad paths here are
not invented: each one is a documented way these three models differ, taken
from `pi_models`' own list of quirks it had to hand-handle.

    uv run pytest tests/test_records.py
"""

from __future__ import annotations

import numpy as np
import pytest

from protein_interpretability.collection.records import (
    SchemaError,
    assert_comparable,
    describe,
    validate,
)

N, B = 12, 64


def a_record(n=N, b=B, centres=None):
    """A well-formed record: Boltz-2's 2-22 A grid over b bins."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(n, n, b)).astype(np.float32)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    p = (e / e.sum(-1, keepdims=True)).astype(np.float32)
    if centres is None:
        centres = np.linspace(2.0, 22.0, b)
    centres = np.asarray(centres, dtype=np.float32)
    return {
        "logits": logits,
        "p": p,
        "ed": (p * centres).sum(-1).astype(np.float32),
        "entropy": (-(p * np.log(p + 1e-12)).sum(-1)).astype(np.float32),
        "ca": rng.normal(size=(n, 3)).astype(np.float32),
        "plddt": rng.uniform(50, 95, size=n).astype(np.float32),
        "centres": centres,
    }


def test_valid_record_round_trips():
    shape = validate(a_record())
    assert (shape.n_tokens, shape.n_bins) == (N, B)
    assert describe(a_record())["fields"]["plddt"]["shape"] == [N]


def test_refuses_unsqueezed_batch_axis():
    """[1,N,N,B] is the shape `while ndim > 3` fails to reduce."""
    r = a_record()
    r["logits"] = r["logits"][None]
    with pytest.raises(SchemaError, match="expected 3"):
        validate(r)


def test_refuses_per_atom_plddt():
    """OpenFold3 reports pLDDT per atom, Boltz-2 per token: a length mismatch."""
    r = a_record()
    r["plddt"] = np.repeat(r["plddt"], 4)
    with pytest.raises(SchemaError, match="plddt"):
        validate(r)


def test_refuses_double_softmax():
    r = a_record()
    e = np.exp(r["p"])
    r["p"] = (e / e.sum(-1, keepdims=True) * 1.5).astype(np.float32)
    with pytest.raises(SchemaError, match="deviate from 1"):
        validate(r)


def test_refuses_non_finite():
    r = a_record()
    r["ed"][3, 4] = np.nan
    with pytest.raises(SchemaError, match="non-finite"):
        validate(r)


def test_refuses_unsorted_bin_centres():
    r = a_record()
    r["centres"] = r["centres"][::-1].copy()
    with pytest.raises(SchemaError, match="increasing"):
        validate(r)


def test_refuses_missing_field():
    r = a_record()
    del r["entropy"]
    with pytest.raises(SchemaError, match="entropy"):
        validate(r)


# ---- comparability: the check a cross-model KL needs and never had ---------

def test_comparable_records_pass():
    assert_comparable(a_record(), a_record()) is None


def test_refuses_different_bin_counts():
    with pytest.raises(SchemaError, match="bin the distance axis differently"):
        assert_comparable(a_record(b=64), a_record(b=39))


def test_refuses_same_count_different_grid():
    """The dangerous case: 64 bins either way, but not the same 64 bins.

    This produces a perfectly well-formed KL that means nothing, and nothing
    else in the pipeline would notice.
    """
    other = a_record(centres=np.linspace(3.25, 52.0, B))
    with pytest.raises(SchemaError, match="not a bin GRID"):
        assert_comparable(a_record(), other)


def test_refuses_different_token_counts():
    with pytest.raises(SchemaError, match="different token counts"):
        assert_comparable(a_record(n=12), a_record(n=13))
