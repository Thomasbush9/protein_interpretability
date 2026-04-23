import numpy as np
import pandas as pd
import pytest
import torch

from protein_interpretability.rsa.pair_metrics import (
    METRICS,
    _key,
    _step_key,
    compare_reps,
    cosine_map,
    get_layer_ids,
    get_tensor,
    l2_map,
    norm_diff_map,
    summarize,
)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def test_key_format():
    assert _key(7, "layer_z") == "pairformer_7_layer_z"
    assert _key(0, "tri_att_start") == "pairformer_0_tri_att_start"


def test_key_format_custom_prefix():
    assert _key(3, "block", prefix="trunk") == "trunk_3_block"


def test_step_key_accepts_int_and_str():
    assert _step_key(3) == "step_3"
    assert _step_key("step_2") == "step_2"
    assert _step_key("4") == "step_4"


def test_step_key_rejects_other_types():
    with pytest.raises(TypeError):
        _step_key(1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Layer discovery / tensor access (multi-step + flat)
# ---------------------------------------------------------------------------

def _mk_multistep_reps(n_steps: int = 2, n_layers: int = 3, N: int = 4, D: int = 5):
    """Build a synthetic multi-step reps dict with pairwise + per-token sites."""
    reps: dict = {}
    for s in range(n_steps):
        inner: dict = {}
        for L in range(n_layers):
            inner[f"pairformer_{L}_layer_z"] = torch.randn(1, N, N, D)
            inner[f"pairformer_{L}_tri_att_start"] = torch.randn(1, N, N, D)
            inner[f"pairformer_{L}_layer_s"] = torch.randn(1, N, D)
        reps[f"step_{s}"] = inner
    return reps


def test_get_layer_ids_filters_by_layer_type():
    reps = _mk_multistep_reps(n_steps=1, n_layers=4)
    assert get_layer_ids(reps, 0, "layer_z") == [0, 1, 2, 3]
    assert get_layer_ids(reps, 0, "tri_att_start") == [0, 1, 2, 3]
    assert get_layer_ids(reps, 0, "tri_att_end") == []  # not present


def test_get_layer_ids_supports_flat_file():
    """A flat (non-multistep) file should be treated as the inner dict."""
    reps = _mk_multistep_reps(n_steps=1, n_layers=2)["step_0"]
    assert get_layer_ids(reps, 0, "layer_z") == [0, 1]


def test_get_tensor_squeezes_batch():
    reps = _mk_multistep_reps(n_steps=1, n_layers=2, N=4, D=5)
    t = get_tensor(reps, 0, 1, "layer_z")
    assert t.shape == (4, 4, 5)
    t_s = get_tensor(reps, 0, 0, "layer_s")
    assert t_s.shape == (4, 5)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def test_cosine_identical_is_one():
    x = torch.randn(3, 3, 8)
    m = cosine_map(x, x)
    assert m.shape == (3, 3)
    assert np.allclose(m, 1.0, atol=1e-6)


def test_cosine_orthogonal_is_zero():
    D = 4
    a = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # (1, 1, D)
    b = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    m = cosine_map(a, b)
    assert m.shape == (1, 1)
    assert np.allclose(m, 0.0, atol=1e-6)


def test_cosine_handles_per_token_shape():
    x = torch.randn(5, 8)
    m = cosine_map(x, x)
    assert m.shape == (5,)
    assert np.allclose(m, 1.0, atol=1e-6)


def test_l2_identical_is_zero():
    x = torch.randn(3, 3, 6)
    assert np.allclose(l2_map(x, x), 0.0)


def test_l2_known_value():
    a = torch.zeros(1, 1, 3)
    b = torch.tensor([[[3.0, 4.0, 0.0]]])
    assert np.allclose(l2_map(a, b), 5.0)


def test_norm_diff_identical_is_zero():
    x = torch.randn(2, 2, 4)
    assert np.allclose(norm_diff_map(x, x), 0.0)


def test_norm_diff_known_value():
    a = torch.tensor([[[3.0, 4.0]]])   # ||a|| = 5
    b = torch.tensor([[[0.0, 12.0]]])  # ||b|| = 12
    assert np.allclose(norm_diff_map(a, b), 7.0)


def test_metrics_registry_covers_all_callables():
    assert set(METRICS) == {"cosine", "l2", "norm_diff"}
    for fn in METRICS.values():
        assert callable(fn)


# ---------------------------------------------------------------------------
# Summaries + end-to-end
# ---------------------------------------------------------------------------

def test_summarize_basic():
    s = summarize(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s["mean"] == 2.5
    assert s["median"] == 2.5
    assert s["min"] == 1.0
    assert s["max"] == 4.0
    assert s["n_positions"] == 4
    assert s["std"] == pytest.approx(np.std([1, 2, 3, 4]))


def test_compare_reps_identical_cosine_is_one():
    reps = _mk_multistep_reps(n_steps=1, n_layers=3)
    df = compare_reps(reps, reps, step=0, layer_type="layer_z", metric="cosine")
    assert isinstance(df, pd.DataFrame)
    assert list(df["layer"]) == [0, 1, 2]
    assert np.allclose(df["mean"], 1.0, atol=1e-6)
    assert np.allclose(df["std"], 0.0, atol=1e-6)


def test_compare_reps_identical_l2_is_zero():
    reps = _mk_multistep_reps(n_steps=1, n_layers=2)
    df = compare_reps(reps, reps, step=0, layer_type="layer_z", metric="l2")
    assert np.allclose(df["mean"], 0.0)
    assert np.allclose(df["max"], 0.0)


def test_compare_reps_unknown_metric_raises():
    reps = _mk_multistep_reps(n_steps=1, n_layers=1)
    with pytest.raises(ValueError, match="Unknown metric"):
        compare_reps(reps, reps, step=0, layer_type="layer_z", metric="bogus")


def test_compare_reps_shape_mismatch_raises():
    reps_a = _mk_multistep_reps(n_steps=1, n_layers=1, N=4, D=5)
    reps_b = _mk_multistep_reps(n_steps=1, n_layers=1, N=4, D=6)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_reps(reps_a, reps_b, step=0, layer_type="layer_z", metric="cosine")


def test_get_layer_ids_with_custom_prefix():
    """ESMFold-style ``trunk_{L}_{site}`` keys should be discoverable."""
    inner = {
        "trunk_0_block": torch.randn(1, 4, 6),
        "trunk_1_block": torch.randn(1, 4, 6),
        "pairformer_0_layer_z": torch.randn(1, 4, 4, 6),  # wrong prefix, ignored
    }
    reps = {"step_0": inner}
    assert get_layer_ids(reps, 0, "block", prefix="trunk") == [0, 1]
    assert get_layer_ids(reps, 0, "block", prefix="pairformer") == []


def test_compare_reps_with_custom_prefix():
    """End-to-end with ``prefix='trunk'`` on synthetic ESMFold-shaped reps."""
    inner = {
        f"trunk_{L}_block": torch.randn(1, 5, 8) for L in range(3)
    }
    reps = {"step_0": inner}
    df = compare_reps(reps, reps, step=0, layer_type="block",
                      metric="cosine", prefix="trunk")
    assert list(df["layer"]) == [0, 1, 2]
    assert np.allclose(df["mean"], 1.0, atol=1e-6)


def test_compare_reps_intersects_layers():
    """Only layers present in both ref and mut are reported."""
    ref = _mk_multistep_reps(n_steps=1, n_layers=3)
    # Drop layer 2 from mut
    mut = {"step_0": {k: v for k, v in ref["step_0"].items()
                      if not k.startswith("pairformer_2_")}}
    df = compare_reps(ref, mut, step=0, layer_type="layer_z", metric="cosine")
    assert list(df["layer"]) == [0, 1]
