"""kl_glob and kl_site must be two measurements, not one written twice.

The bug these guard against left no trace: `collect_pairformer_layers.py`
computed the global reduction twice and stored it under both names, so the
archive had both fields, at the promised shapes, with plausible values. Only the
numbers were the same.

Synthetic logits, no model: the whole point of putting the reduction in the
package is that it can be checked here.

    uv run pytest tests/test_reductions.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from protein_interpretability.collection.reductions import (
    ReductionError,
    kl_reductions,
    site_mask,
    symmetric_kl,
    uncovered_sites,
)


# ---- the divergence itself -------------------------------------------------

def test_identical_logits_give_zero():
    x = np.array([[0.0, 1.0, 2.0], [3.0, -1.0, 0.5]])
    assert np.abs(symmetric_kl(x, x)).max() < 1e-12


def test_the_divergence_is_symmetric():
    """It is Jeffreys, not KL, and the field name says nothing about direction
    -- so if this ever became one-sided, every archived kl_glob would silently
    mean something else."""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(4, 7)), rng.normal(size=(4, 7))
    assert np.allclose(symmetric_kl(a, b), symmetric_kl(b, a))


def test_it_matches_the_archived_expression():
    """Expression for expression against `exp_gym.skl`, restated here so the
    harness does not have to be importable (it needs jax) for this to be
    checked."""
    def skl(la, lb):
        def sm(x):
            x = x - x.max(-1, keepdims=True)
            e = np.exp(x)
            return e / e.sum(-1, keepdims=True)
        pa, pb = sm(la), sm(lb)
        return ((pa - pb) * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum(-1)

    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 5, 64)).astype(np.float32)
    b = rng.normal(size=(3, 5, 64)).astype(np.float32)
    assert np.array_equal(symmetric_kl(a, b), skl(a, b))


def test_float32_stays_float32():
    """The archives were made at the dtype the model returned. Promoting here
    would move every number a regression check holds fixed."""
    a = np.zeros((2, 3, 8), np.float32)
    assert symmetric_kl(a, a + 1).dtype == np.float32


# ---- the two reductions ----------------------------------------------------

def _logits(n_layers, n_pairs, n_bins=8):
    return np.zeros((n_layers, n_pairs, n_bins), np.float64)


def test_site_and_global_are_different_numbers():
    """Four pairs, one of which touches token 5, and only that pair moves.

    kl_glob is then the site divergence divided by four; kl_site is the site
    divergence itself. Both are computable by hand, which is what makes this a
    test of the values and not just of their difference.
    """
    ii = np.array([5, 1, 2, 3])
    jj = np.array([9, 4, 6, 7])
    lw = _logits(2, 4)
    lm = _logits(2, 4)
    lm[:, 0, 0] = 2.0                      # only the site pair diverges

    out = kl_reductions(lm, lw, ii, jj, token=5)
    d = symmetric_kl(lm[:, 0], lw[:, 0])   # the one moved pair, per layer

    assert np.allclose(out["kl_site"], d)
    assert np.allclose(out["kl_glob"], d / 4)
    assert not np.allclose(out["kl_site"], out["kl_glob"])


def test_the_site_reduction_ignores_pairs_that_do_not_touch_the_site():
    ii = np.array([5, 1, 2])
    jj = np.array([0, 4, 6])
    lw, lm = _logits(1, 3), _logits(1, 3)
    lm[:, 1:, 0] = 5.0                     # everything BUT the site pair moves

    out = kl_reductions(lm, lw, ii, jj, token=5)
    assert np.allclose(out["kl_site"], 0.0)
    assert out["kl_glob"] > 0.0


def test_either_end_of_a_pair_counts():
    ii = np.array([0, 1])
    jj = np.array([7, 3])
    assert site_mask(ii, jj, 7).tolist() == [True, False]
    assert site_mask(ii, jj, 1).tolist() == [False, True]


def test_shapes_are_per_layer():
    ii, jj = np.array([1, 2, 3]), np.array([4, 1, 6])
    out = kl_reductions(_logits(6, 3), _logits(6, 3), ii, jj, token=1)
    assert out["kl_glob"].shape == (6,)
    assert out["kl_site"].shape == (6,)


# ---- the refusals ----------------------------------------------------------

def test_an_uncovered_site_refuses_rather_than_writing_zero():
    """The archived producer wrote np.zeros(L) here. A zero in kl_site reads as
    'this mutation moved the distogram not at all' -- the strongest claim the
    field can make, for the one variant that was never measured."""
    ii, jj = np.array([1, 2]), np.array([3, 4])
    with pytest.raises(ReductionError, match="no sampled pair touches"):
        kl_reductions(_logits(2, 2), _logits(2, 2), ii, jj, token=99)


def test_a_mismatched_pair_sample_is_refused():
    ii, jj = np.array([1, 2, 3]), np.array([4, 5, 6])
    with pytest.raises(ReductionError, match="not from the same capture"):
        kl_reductions(_logits(2, 5), _logits(2, 5), ii, jj, token=1)


def test_unpaired_indices_are_refused():
    with pytest.raises(ReductionError, match="not paired"):
        site_mask(np.array([1, 2, 3]), np.array([1, 2]), 1)


def test_uncovered_sites_is_answerable_before_the_gpu_starts():
    """The check the collection script runs before its first trunk pass."""
    ii, jj = np.array([0, 1, 2]), np.array([5, 6, 7])
    assert uncovered_sites(ii, jj, [0, 5, 6]) == []
    assert uncovered_sites(ii, jj, [3, 4]) == [3, 4]


# ---- the wiring the bug lived in -------------------------------------------

def test_the_collection_script_takes_both_fields_from_one_call():
    """The regression is a WIRING one: both fields computed from the same
    expression. Returning them from one call is what makes that a deliberate
    edit rather than a copy-paste, so the script is checked for the call.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "experiments" / "collection"
           / "collect_pairformer_layers.py").read_text()
    assert "kl_reductions(" in src, (
        "the collection script no longer sources its KL fields from "
        "collection.reductions; if it computes them itself again, nothing "
        "stops the two reductions being the same one twice")
    assert 'kl["kl_site"]' in src and 'kl["kl_glob"]' in src
