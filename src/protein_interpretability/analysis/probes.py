"""The linear probe this project reports its results with.

Small enough to be re-typed at each call site, which is exactly why it should
not be: `analyze_transfer`, `compare_internal_output` and every ad-hoc check
have carried their own copy, and the three details below are the ones that get
dropped in transcription. They are not incidental — each changes the number.

    intercept       ridge does not penalise it. Shrinking the intercept pulls
                    every prediction toward the origin of a target that was only
                    centred WITHIN assay, and a leave-one-assay-out design then
                    reads that bias as poor transfer.
    z-score epsilon `sd + 1e-9`, not `sd`. A constant feature column is not
                    hypothetical here — some channels are flat on some assays —
                    and dividing by zero poisons the whole fit, silently, via nan.
    selection       top-k by |Spearman| on the TRAINING rows only. Selecting on
                    all rows leaks the test assay into feature choice, which is
                    the classic way a transfer number comes out too good.

Numpy only: this is analysis, so it must import no backend.
"""

from __future__ import annotations

import numpy as np

from protein_interpretability.analysis import statistics as st

EPS = 1e-9


def zscore(a) -> np.ndarray:
    """Per-column standardisation, with the epsilon that makes it total."""
    a = np.asarray(a, dtype=float)
    return (a - a.mean(0)) / (a.std(0) + EPS)


def zstats(a) -> tuple[np.ndarray, np.ndarray]:
    """(mean, sd) for applying one assay's scaling to another — the inductive
    mode, where the scale must be learned on training assays only."""
    a = np.asarray(a, dtype=float)
    return a.mean(0), a.std(0) + EPS


def zapply(a, mu, sd) -> np.ndarray:
    return (np.asarray(a, dtype=float) - mu) / sd


def ridge_fit(X, y, lam: float) -> np.ndarray:
    """Ridge coefficients with an UNPENALISED intercept, appended last."""
    X = np.asarray(X, dtype=float)
    Xb = np.column_stack([X, np.ones(len(X))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ np.asarray(y, dtype=float))


def ridge_pred(w, X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.column_stack([X, np.ones(len(X))]) @ w


def select_k(X, y, k: int) -> np.ndarray:
    """Indices of the top-k columns by |Spearman| against y.

    Pass TRAINING rows only. There is no way for this function to check that,
    which is why it is said here rather than assumed.
    """
    return np.argsort(-st.rank_corr_columns(X, y))[:k]


def leave_one_group_out(blocks: dict, *, lam: float = 10.0, k: int | None = None):
    """Fit on every group but one, test on it, for each group in turn.

    `blocks` maps a group name (an assay) to `{"X": features, "y": target}`.
    Features and target are standardised WITHIN each group before pooling —
    which is well defined only because the columns mean the same thing in every
    group, and that is a claim about the data, not a convenience.

    Returns {group: spearman on that held-out group}.
    """
    names = sorted(blocks)
    if len(names) < 2:
        raise ValueError("leave-one-group-out needs at least two groups")

    z = {n: {"X": zscore(blocks[n]["X"]),
             "y": np.asarray(blocks[n]["y"], dtype=float),
             "yz": zscore(blocks[n]["y"])} for n in names}

    widths = {z[n]["X"].shape[1] for n in names}
    if len(widths) > 1:
        raise ValueError(f"feature widths differ across groups: {widths}")

    out = {}
    for held in names:
        train = [n for n in names if n != held]
        Xtr = np.concatenate([z[n]["X"] for n in train], 0)
        ytr = np.concatenate([z[n]["yz"] for n in train], 0)
        Xte = z[held]["X"]
        if k is not None and k < Xtr.shape[1]:
            idx = select_k(Xtr, ytr, k)
            Xtr, Xte = Xtr[:, idx], Xte[:, idx]
        w = ridge_fit(Xtr, ytr, lam)
        out[held] = float(st.spearman(ridge_pred(w, Xte), z[held]["y"]))
    return out
