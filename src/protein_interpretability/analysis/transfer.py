"""Two probe protocols the frozen leave-one-assay-out probe cannot express.

`probes.leave_one_group_out` answers "does this work on a protein the probe has
never seen". It is deliberately not touched here -- the archived numbers are
reproduced through it to 0.00e+00, and it should stay that way. What it cannot
express is:

    a capacity-matched comparison  The internal block is 128 channels and the
        emitted block is 10 or 37. If the gap is reported without matching
        dimension, "the trunk knows more" and "128 > 10" are the same number.
        `leave_one_group_out_reduced` projects a block onto its first d
        principal components -- fitted on the TRAINING assays only -- so both
        sides can be read at the same width, and the whole rho-versus-d curve
        can be drawn instead of one point.

    a cross-cohort probe  Every archived number trains and tests inside one
        cohort. `fit_groups_predict_groups` trains on one set of assays and
        predicts a disjoint set, which is what asks whether the direction found
        in folding stability is the SAME direction that matters for fitness,
        abundance and activity, or merely an analogous one.

Both keep the three details that `probes` exists to stop being dropped: the
unpenalised intercept, the `sd + 1e-9` z-score, and fitting nothing on the test
rows. The PCA basis is a fitted object and obeys the third: it never sees a
test assay.

Numpy only: this is analysis, so it must import no backend.
"""

from __future__ import annotations

import numpy as np

from protein_interpretability.analysis import statistics as st
from protein_interpretability.analysis.probes import (
    ridge_fit, ridge_pred, zscore,
)


def pca_basis(X, d: int):
    """(mean, components) of the first d principal directions of X.

    Pass TRAINING rows only. Like `select_k`, this function cannot check that,
    which is why it is said here. Returned components are (d, n_features), so
    projection is `(X - mean) @ comp.T`.
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(0)
    # full_matrices=False: we only ever want the top d of min(n, p) directions.
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:d]


def pca_apply(X, mu, comp):
    return (np.asarray(X, dtype=float) - mu) @ comp.T


def _standardise(blocks):
    """Within-group z-scoring, the frozen convention, applied group by group.

    The target is kept twice: `yz` is what the pooled fit trains on, `y` is what
    the held-out Spearman is taken against. Spearman is rank-based so the two
    give the same correlation; both are kept so this reads identically to
    `leave_one_group_out` rather than looking like a different protocol.
    """
    return {n: {"X": zscore(blocks[n]["X"]),
                "y": np.asarray(blocks[n]["y"], dtype=float),
                "yz": zscore(blocks[n]["y"])} for n in sorted(blocks)}


def _one_width(z, what):
    widths = {z[n]["X"].shape[1] for n in z}
    if len(widths) > 1:
        raise ValueError(f"feature widths differ across {what}: {widths}")
    return widths.pop()


def leave_one_group_out_reduced(blocks: dict, *, d: int, lam: float = 10.0):
    """Leave-one-assay-out on the first d principal components of the block.

    The basis is refitted inside every fold from the training assays alone, so
    the held-out assay contributes nothing to the choice of directions. With
    d >= the block width this is the unreduced probe up to a rotation, and
    ridge is not rotation invariant in general -- but it is invariant under an
    ORTHOGONAL change of basis with an unpenalised intercept, which is what a
    full-rank PCA is. The tests assert that equivalence rather than assuming it.
    """
    z = _standardise(blocks)
    names = sorted(z)
    if len(names) < 2:
        raise ValueError("leave-one-group-out needs at least two groups")
    width = _one_width(z, "groups")
    if d > width:
        raise ValueError(f"d={d} exceeds the block width {width}")

    out = {}
    for held in names:
        train = [n for n in names if n != held]
        Xtr = np.concatenate([z[n]["X"] for n in train], 0)
        ytr = np.concatenate([z[n]["yz"] for n in train], 0)
        mu, comp = pca_basis(Xtr, d)
        w = ridge_fit(pca_apply(Xtr, mu, comp), ytr, lam)
        pred = ridge_pred(w, pca_apply(z[held]["X"], mu, comp))
        out[held] = float(st.spearman(pred, z[held]["y"]))
    return out


def fit_groups_predict_groups(train: dict, test: dict, *, lam: float = 10.0,
                              d: int | None = None):
    """Fit on every training assay pooled, predict each test assay separately.

    `train` and `test` map assay name to `{"X", "y"}` and must be disjoint --
    that is checked, because the whole point of the protocol is that no test
    assay influenced the fit, and an overlap would silently turn this into a
    within-cohort number that looks like a transfer one.

    Returns {test group: spearman}. With `d` set, the same train-only PCA as
    `leave_one_group_out_reduced` is applied first.
    """
    shared = set(train) & set(test)
    if shared:
        raise ValueError(
            f"train and test share {len(shared)} assay(s): {sorted(shared)}. "
            f"A cross-cohort number computed over overlapping sets is not a "
            f"transfer number.")
    if not train or not test:
        raise ValueError("both a training and a test set are required")

    ztr, zte = _standardise(train), _standardise(test)
    w_tr, w_te = _one_width(ztr, "training groups"), _one_width(zte, "test groups")
    if w_tr != w_te:
        raise ValueError(
            f"training block is {w_tr} wide and test block {w_te}; the columns "
            f"must mean the same thing on both sides")

    Xtr = np.concatenate([ztr[n]["X"] for n in sorted(ztr)], 0)
    ytr = np.concatenate([ztr[n]["yz"] for n in sorted(ztr)], 0)
    if d is not None:
        if d > w_tr:
            raise ValueError(f"d={d} exceeds the block width {w_tr}")
        mu, comp = pca_basis(Xtr, d)
        Xtr = pca_apply(Xtr, mu, comp)

    w = ridge_fit(Xtr, ytr, lam)
    out = {}
    for n in sorted(zte):
        Xte = zte[n]["X"]
        if d is not None:
            Xte = pca_apply(Xte, mu, comp)
        out[n] = float(st.spearman(ridge_pred(w, Xte), zte[n]["y"]))
    return out

def orthonormal(X, d: int):
    """An orthonormal basis for the top-d principal subspace of X, (n_features, d).

    Returned as columns, which is what `principal_angles` wants. The centring
    matches `pca_basis`; the two are the same decomposition read two ways.
    """
    mu, comp = pca_basis(X, d)
    # `comp` rows are already orthonormal (right singular vectors), so the
    # transpose is an orthonormal column basis and no re-orthogonalisation is
    # needed. Asserted rather than assumed, because a silently non-orthonormal
    # basis makes every principal angle wrong in a way that still prints.
    Q = comp.T
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8)
    return Q


def principal_angles(Q1, Q2):
    """Cosines of the principal angles between two subspaces, descending.

    Q1, Q2 are orthonormal column bases. The singular values of Q1.T @ Q2 are
    the cosines: 1.0 means a shared direction, 0.0 means orthogonal. Unlike a
    cosine between two fitted weight vectors this is invariant to how each
    subspace happens to be parameterised, which is the whole point -- ridge
    coefficients on 128 correlated channels are only loosely identified, and
    two runs can name the same subspace with very different vectors.
    """
    Q1, Q2 = np.asarray(Q1, dtype=float), np.asarray(Q2, dtype=float)
    if Q1.shape[0] != Q2.shape[0]:
        raise ValueError(f"subspaces live in different spaces: "
                         f"{Q1.shape[0]} and {Q2.shape[0]}")
    return np.clip(np.linalg.svd(Q1.T @ Q2, compute_uv=False), 0.0, 1.0)


def subspace_overlap(Q1, Q2) -> float:
    """Mean squared cosine of the principal angles, in [0, 1].

    The natural scalar summary: 1.0 for identical subspaces, and for two
    RANDOM d-dimensional subspaces of R^p it has expectation d/p -- 0.031 for
    d=4 in 128 channels. That floor is what makes a number like 0.4 readable.
    """
    c = principal_angles(Q1, Q2)
    return float((c ** 2).mean())


def random_subspace(p: int, d: int, rng) -> np.ndarray:
    """A uniformly random d-dimensional subspace of R^p, as an orthonormal basis."""
    return np.linalg.qr(rng.normal(size=(p, d)))[0]
