"""The declaration must match the arrays, checked before anything is written.

The cross-model task declared `reduction="vector"` and listed `dz_site` among
its fields, while `exp_gym_deep` writes `dz_site` as a per-layer NORM. The
protocol block therefore promised a (V, L, 128) direction beside a (V, L) array.
The arrays were correct -- `dz_vec` held the real vectors -- and only the
declaration lied, which is the deep2_* failure with the sides swapped.

Two things let it through. `CaptureSpec` had no way to say "vectors under `_vec`
and norms under `_site`", which is exactly what that capture family does, so
neither available reduction was true of it. And `validate_capture` runs on an
artifact that already exists, while the adapter never called it at all.

The shapes below are the ones `exp_gym_deep` really returns, not idealised ones,
which is the point of the test.

    uv run pytest tests/test_capture_contract.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from protein_interpretability.collection import CaptureSpec
from protein_interpretability.collection.capture_spec import CaptureSpecError

# What exp_gym_deep.collect_assay actually produces, verified against
# xm_protenix_r1_SQSTM (100 variants, 16 layers, 40 tokens).
N, L, T = 100, 16, 40
FIELDS = ("dz_vec", "ds_vec", "dz_site", "ds_site", "kl_site", "kl_glob",
          "ca", "plddt_mean", "plddt_site", "score", "pos")


def kernel_arrays():
    """The real shapes: `_vec` are directions, `_site` are per-layer norms."""
    return {
        "dz_vec": np.zeros((N, L, 128), np.float32),
        "ds_vec": np.zeros((N, L, 384), np.float32),
        "dz_site": np.zeros((N, L), np.float32),       # NORM, not a vector
        "ds_site": np.zeros((N, L), np.float32),       # NORM, not a vector
        "kl_site": np.zeros((N, L), np.float32),
        "kl_glob": np.zeros((N, L), np.float32),
        "ca": np.zeros((N, T, 3), np.float32),
        "plddt_mean": np.zeros(N),
        "plddt_site": np.zeros(N),
        "score": np.zeros(N),
        "pos": np.zeros(N, np.int64),
        # bookkeeping the spec has no opinion about
        "ca_wt": np.zeros((T, 3), np.float32),
        "capture_drift": np.array(4.1e-4),
        "mutant": np.array(["A1B"] * N),
        "n_layers": np.array(L),
    }


def spec(reduction="both", fields=FIELDS):
    return CaptureSpec(model="protenix", fields=fields, layers="all",
                       reduction=reduction, recycles=3, dtype="float32")


# ---- the bug, as a test ----------------------------------------------------

def test_declaring_vector_over_the_real_arrays_is_refused():
    """This is what shipped. It must never pass again."""
    with pytest.raises(CaptureSpecError, match="the NORM where the spec promised"):
        spec("vector").validate_arrays(kernel_arrays(), n_variants=N, n_tokens=T)


def test_the_honest_declaration_passes():
    spec("both").validate_arrays(kernel_arrays(), n_variants=N, n_tokens=T)


def test_both_gives_each_field_its_true_shape():
    want = spec("both").expected_shapes(n_variants=N, n_tokens=T)
    assert want["dz_vec"] == (N, L, 128), "the direction keeps its channels"
    assert want["dz_site"] == (N, L), "the norm does not"
    assert want["ds_vec"] == (N, L, 384)
    assert want["ds_site"] == (N, L)


# ---- `both` has to mean what it says ---------------------------------------

def test_both_without_a_vector_field_is_refused():
    with pytest.raises(CaptureSpecError, match="names no `_vec` field"):
        spec("both", ("dz_site", "kl_glob")).validate()


def test_both_without_a_norm_field_is_refused():
    with pytest.raises(CaptureSpecError, match="promises a norm"):
        spec("both", ("dz_vec", "kl_glob")).validate()


def test_norm_still_refuses_a_vec_field():
    """The older rule stays: a `_vec` field is a vector by name."""
    with pytest.raises(CaptureSpecError, match="a vector by name"):
        spec("norm", ("dz_vec", "dz_site")).validate()


# ---- what the pre-write check catches --------------------------------------

def test_a_missing_promised_field_is_refused():
    arrays = kernel_arrays()
    del arrays["kl_site"]
    with pytest.raises(CaptureSpecError, match="kl_site: promised but absent"):
        spec().validate_arrays(arrays, n_variants=N, n_tokens=T)


def test_a_wrong_layer_count_is_refused():
    """A selection applied to some fields and not others -- the shape of a
    half-finished layer slice."""
    arrays = kernel_arrays()
    arrays["kl_glob"] = np.zeros((N, 3), np.float32)
    with pytest.raises(CaptureSpecError, match=r"kl_glob: \(100, 3\)"):
        spec().validate_arrays(arrays, n_variants=N, n_tokens=T)


def test_a_wrong_dtype_is_refused():
    arrays = kernel_arrays()
    arrays["dz_vec"] = arrays["dz_vec"].astype(np.float16)
    with pytest.raises(CaptureSpecError, match="dtype float16"):
        spec().validate_arrays(arrays, n_variants=N, n_tokens=T)


def test_per_variant_columns_are_not_dtype_checked():
    """`score` is float64 and `pos` is int64 by nature; the spec's dtype governs
    the representation arrays, not the identity columns."""
    arrays = kernel_arrays()
    assert arrays["score"].dtype == np.float64
    assert arrays["pos"].dtype == np.int64
    spec().validate_arrays(arrays, n_variants=N, n_tokens=T)


def test_a_vector_where_a_norm_was_promised_is_also_refused():
    arrays = kernel_arrays()
    arrays["dz_site"] = np.zeros((N, L, 128), np.float32)
    with pytest.raises(CaptureSpecError,
                       match="the vector where the spec promised a norm"):
        spec().validate_arrays(arrays, n_variants=N, n_tokens=T)


# ---- bookkeeping is listed, not ignored ------------------------------------

def test_undeclared_arrays_are_reported():
    extra = spec().undeclared(kernel_arrays())
    assert "capture_drift" in extra and "ca_wt" in extra
    for declared in FIELDS:
        assert declared not in extra
