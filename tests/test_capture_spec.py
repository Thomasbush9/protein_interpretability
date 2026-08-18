"""What a CaptureSpec refuses, and the one mismatch it exists to catch.

The anchoring case is real and still live in `runs/`: `dz_site` is a 128-channel
VECTOR in the `gym2s_*` captures and a per-layer NORM in the `xm_*` ones. A spec
that promises one and receives the other must fail loudly, because the shape it
receives is perfectly legal for a different quantity — that is how `deep2_*`
handed a norm to a probe expecting a direction and got +0.468 instead of an error.

    uv run pytest tests/test_capture_spec.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from protein_interpretability.collection import CaptureSpec, CaptureSpecError
from protein_interpretability.collection.capture_spec import (
    MODEL_LAYERS,
    PAIR_WIDTH,
)


def spec(**kw) -> CaptureSpec:
    base = dict(model="boltz2", fields=("dz_site", "kl_site"), layers="all",
                reduction="vector", recycles=3)
    base.update(kw)
    return CaptureSpec(**base)


class FakeCapture:
    """Just enough of `Capture` to be validated against a spec."""

    def __init__(self, arrays):
        self.path = "fake.npz"
        self.files = tuple(arrays)
        self._a = arrays

    def _arr(self, k):
        return self._a[k]


# ---- refusals --------------------------------------------------------------

def test_refuses_an_unknown_model():
    with pytest.raises(CaptureSpecError, match="unknown model"):
        spec(model="esmfold").validate()


def test_refuses_a_model_with_no_recorded_depth():
    """af2 is in the registry with depth None because nothing here records it.
    A guessed depth silently changes which layer 'final' means."""
    assert MODEL_LAYERS["af2"] is None
    with pytest.raises(CaptureSpecError, match="no recorded trunk depth"):
        spec(model="af2").validate()


def test_refuses_capturing_nothing():
    with pytest.raises(CaptureSpecError, match="write nothing"):
        spec(fields=()).validate()


def test_refuses_an_unknown_field():
    with pytest.raises(CaptureSpecError, match="unknown field"):
        spec(fields=("dz_site", "not_a_field")).validate()


def test_refuses_a_layer_outside_the_models_depth():
    """protenix has 16 blocks; layer 40 is valid for boltz2 and not for it."""
    spec(model="boltz2", layers=(40,)).validate()
    with pytest.raises(CaptureSpecError, match="outside protenix's 16 blocks"):
        spec(model="protenix", layers=(40,)).validate()


def test_refuses_an_empty_layer_list():
    with pytest.raises(CaptureSpecError, match="captures nothing"):
        spec(layers=()).validate()


def test_refuses_an_unknown_layer_keyword():
    with pytest.raises(CaptureSpecError, match="must be 'all', 'final'"):
        spec(layers="middle").validate()


def test_refuses_a_distogram_without_a_pair_sample():
    with pytest.raises(CaptureSpecError, match="needs n_pairs"):
        spec(fields=("disto",)).validate()
    spec(fields=("disto",), n_pairs=1477).validate()


def test_refuses_a_reduction_that_would_do_nothing():
    with pytest.raises(CaptureSpecError, match="would silently do nothing"):
        spec(fields=("kl_site",), reduction="norm").validate()


def test_refuses_an_unsupported_dtype():
    with pytest.raises(CaptureSpecError, match="unsupported dtype"):
        spec(dtype="bfloat16").validate()


# ---- shapes ----------------------------------------------------------------

def test_vector_and_norm_differ_by_the_channel_axis():
    v = spec(reduction="vector").expected_shapes(n_variants=250, n_tokens=69)
    n = spec(reduction="norm").expected_shapes(n_variants=250, n_tokens=69)
    assert v["dz_site"] == (250, 64, PAIR_WIDTH)
    assert n["dz_site"] == (250, 64)
    assert v["kl_site"] == n["kl_site"] == (250, 64)


def test_shapes_match_a_real_archive():
    """Checked against gym2s_ARGR: 250 variants, 69 tokens, 64 layers."""
    got = spec(fields=("dz_site", "ds_site", "kl_site", "disto", "ca"),
               n_pairs=1477).expected_shapes(n_variants=250, n_tokens=69)
    assert got["dz_site"] == (250, 64, 128)
    assert got["ds_site"] == (250, 64, 384)
    assert got["kl_site"] == (250, 64)
    assert got["disto"] == (250, 1477, 64)
    assert got["ca"] == (250, 69, 3)


def test_final_captures_one_layer():
    assert spec(layers="final").n_layers == 1
    assert spec(layers="all", model="of3").n_layers == 48


# ---- memory ----------------------------------------------------------------

def test_estimate_matches_the_real_archive_within_a_few_percent():
    """gym2s_ARGR is 128.5 MB on disk for these fields."""
    s = spec(fields=("dz_site", "ds_site", "kl_site", "kl_glob", "disto", "ca"),
             n_pairs=1477)
    mb = s.estimate_bytes(n_variants=250, n_tokens=69) / 1e6
    assert 125 < mb < 132, mb


def test_the_reduced_capture_is_the_reason_the_full_tensor_is_not_stored():
    """The plan prefers mutation-row captures over the full N x N x C x L
    tensor; this makes that a number instead of an opinion."""
    s = spec(fields=("dz_site",))
    row = s.estimate_bytes(n_variants=250, n_tokens=69)
    full = s.full_pair_tensor_bytes(n_variants=250, n_tokens=69)
    assert full / row > 4000, f"{full / row:.0f}x"


# ---- validating a written archive -----------------------------------------

def test_accepts_an_archive_matching_its_spec():
    s = spec(fields=("dz_site", "kl_site"))
    cap = FakeCapture({"dz_site": np.zeros((250, 64, 128), np.float32),
                       "kl_site": np.zeros((250, 64), np.float32)})
    s.validate_capture(cap, n_variants=250, n_tokens=69)


def test_catches_a_norm_written_where_a_vector_was_promised():
    """The +0.468 failure, as a shape check."""
    s = spec(fields=("dz_site",), reduction="vector")
    cap = FakeCapture({"dz_site": np.zeros((250, 64), np.float32)})
    with pytest.raises(CaptureSpecError, match=r"NORM where the spec promised"):
        s.validate_capture(cap, n_variants=250, n_tokens=69)


def test_catches_a_vector_written_where_a_norm_was_promised():
    s = spec(fields=("dz_site",), reduction="norm")
    cap = FakeCapture({"dz_site": np.zeros((250, 64, 128), np.float32)})
    with pytest.raises(CaptureSpecError, match="vector where the spec promised"):
        s.validate_capture(cap, n_variants=250, n_tokens=69)


def test_catches_a_promised_field_that_is_absent():
    s = spec(fields=("dz_site", "kl_site"))
    cap = FakeCapture({"dz_site": np.zeros((250, 64, 128), np.float32)})
    with pytest.raises(CaptureSpecError, match="promised but absent"):
        s.validate_capture(cap, n_variants=250, n_tokens=69)


def test_catches_the_wrong_number_of_layers():
    """An of3 capture handed to a boltz2 spec: 48 blocks against 64."""
    s = spec(model="boltz2", fields=("dz_site",))
    cap = FakeCapture({"dz_site": np.zeros((100, 48, 128), np.float32)})
    with pytest.raises(CaptureSpecError, match=r"\(100, 48, 128\)"):
        s.validate_capture(cap, n_variants=100, n_tokens=63)


# ---- provenance ------------------------------------------------------------

def test_protocol_states_the_reduction():
    p = spec(reduction="norm", fields=("dz_site",)).protocol()
    assert p["reduction"] == "norm"
    assert p["model"] == "boltz2" and p["n_layers_captured"] == 64
    assert p["capture_fields"] == ["dz_site"]
