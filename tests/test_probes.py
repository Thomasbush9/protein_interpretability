"""The three details of the probe that change the number if you drop them.

Each of these was carried by hand at several call sites before the recipe moved
into the library, and each is the kind of thing that survives a code review
because it looks like a formatting choice.

    uv run pytest tests/test_probes.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from protein_interpretability.analysis.probes import (
    EPS,
    leave_one_group_out,
    ridge_fit,
    ridge_pred,
    select_k,
    zscore,
)


def test_zscore_survives_a_constant_column():
    """Some channels are flat on some assays. Without the epsilon this is a
    divide-by-zero that propagates nan through the entire fit."""
    X = np.column_stack([np.arange(10.0), np.full(10, 3.0)])
    Z = zscore(X)
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z[:, 1], 0.0)


def test_zscore_epsilon_is_in_the_denominator_not_added_after():
    col = np.array([1.0, 1.0, 1.0, 1.0])
    assert zscore(col)[0] == pytest.approx(0.0, abs=1e-12)
    assert EPS > 0


def _centred(rng, n=200, d=3):
    """Exactly zero-mean columns, so the intercept decouples from the slopes and
    the invariant below is sharp rather than approximate."""
    X = rng.normal(size=(n, d))
    return X - X.mean(0)


def test_ridge_does_not_penalise_the_intercept():
    """The invariant: mean(prediction) == mean(y) at EVERY lambda.

    That is what an unpenalised intercept buys. Penalising it instead shrinks
    predictions toward the origin of a target that was only centred within
    assay, and leave-one-assay-out reads that bias as poor transfer.
    """
    rng = np.random.default_rng(0)
    X = _centred(rng)
    y = 50.0 + X @ np.array([1.0, -2.0, 0.5])
    for lam in (1.0, 1e3, 1e6):
        pred = ridge_pred(ridge_fit(X, y, lam), X)
        assert pred.mean() == pytest.approx(y.mean(), abs=1e-9), (
            f"intercept shrank at lambda={lam:g}")


def test_penalising_the_intercept_would_visibly_shrink_it():
    """The bug this guards against, written out, so the test above is not just
    asserting whatever the code happens to do."""
    rng = np.random.default_rng(0)
    X = _centred(rng)
    y = 50.0 + X @ np.array([1.0, -2.0, 0.5])
    lam = 1e4

    Xb = np.column_stack([X, np.ones(len(X))])
    naive = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)

    assert ridge_fit(X, y, lam)[-1] == pytest.approx(y.mean(), abs=1e-9)
    assert naive[-1] < 0.6 * y.mean(), (
        "with the intercept penalised it collapses toward zero; that is the "
        "difference the single `A[-1, -1] -= lam` line makes")


def test_ridge_shrinks_slopes_but_not_the_offset():
    rng = np.random.default_rng(1)
    X = _centred(rng)
    y = 10.0 + X @ np.array([5.0, 5.0, 5.0])
    weak, strong = ridge_fit(X, y, 1.0), ridge_fit(X, y, 1e5)
    assert np.abs(strong[:-1]).sum() < np.abs(weak[:-1]).sum()
    assert strong[-1] == pytest.approx(y.mean(), abs=1e-9)


def test_select_k_returns_the_informative_columns():
    rng = np.random.default_rng(2)
    noise = rng.normal(size=(300, 8))
    signal = rng.normal(size=300)
    X = np.column_stack([noise[:, :4], signal, noise[:, 4:]])
    idx = select_k(X, signal, 1)
    assert idx[0] == 4


def test_leave_one_group_out_holds_each_group_out_in_turn():
    rng = np.random.default_rng(3)
    w = rng.normal(size=6)
    blocks = {}
    for name in ("A", "B", "C"):
        X = rng.normal(size=(120, 6))
        blocks[name] = {"X": X, "y": X @ w + 0.05 * rng.normal(size=120)}
    out = leave_one_group_out(blocks, lam=1.0)
    assert set(out) == {"A", "B", "C"}
    assert all(v > 0.8 for v in out.values()), (
        "a shared linear signal must transfer across groups")


def test_leave_one_group_out_refuses_a_single_group():
    with pytest.raises(ValueError, match="at least two groups"):
        leave_one_group_out({"only": {"X": np.zeros((4, 2)), "y": np.zeros(4)}})


def test_leave_one_group_out_refuses_ragged_feature_widths():
    """Pooling features across groups is only meaningful if the columns mean the
    same thing; differing widths mean they cannot."""
    blocks = {"A": {"X": np.zeros((5, 3)), "y": np.zeros(5)},
              "B": {"X": np.zeros((5, 4)), "y": np.zeros(5)}}
    with pytest.raises(ValueError, match="feature widths differ"):
        leave_one_group_out(blocks)


def test_group_scaling_is_within_group_not_pooled():
    """Two groups on wildly different scales must still transfer.

    Pooled standardisation would let one group's spread dominate the other's,
    which is exactly what per-assay z-scoring exists to prevent.
    """
    rng = np.random.default_rng(4)
    w = rng.normal(size=4)
    small = rng.normal(size=(150, 4))
    big = rng.normal(size=(150, 4)) * 1000.0
    blocks = {
        "small": {"X": small, "y": small @ w},
        "big": {"X": big, "y": big @ w},
    }
    out = leave_one_group_out(blocks, lam=1.0)
    assert all(v > 0.9 for v in out.values()), out
