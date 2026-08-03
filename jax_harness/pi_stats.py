"""Tie-aware rank statistics and cluster-correct uncertainty.

Every analysis in this harness that reports a rank correlation, a partial rank
correlation, or an interval around one should get it from here. Three bugs
motivated centralising it, all of them found by the August 2026 publication
audit, and all of them the kind that changes a number rather than crashing:

1. **`np.argsort(np.argsort(x))` is not a rank.** It is a tie-BREAKING
   permutation: equal values get consecutive integers in whatever order argsort
   happened to visit them. On continuous features the difference is invisible;
   on a binary or heavily tied variable it is not. The position-only ProteinGym
   baseline predicts a single constant on the held-out rows, so its true
   Spearman is undefined -- the old code scored it +0.069, purely from the
   order argsort assigned to 250 identical values. Same failure in the RSA
   same-position control, which is binary by construction.

2. **Partial Spearman is not "residualise the ranks, then rank again".**
   analyze_rsa.py computed `spearman(resid(rx), resid(ry))`, which re-ranks
   residuals and throws away the linear geometry that made them residuals. The
   standard procedure residualises the ranks and takes the PEARSON correlation
   of what is left; that is what `partial_spearman` does here.

3. **Repeated splits within an assay are not independent observations.**
   Bootstrapping 60 overlapping train/test splits gives an interval around the
   splitting noise, not around the population of assays. `cluster_bootstrap`
   resamples assays (optionally resampling splits within a resampled assay,
   which is the hierarchical version) so the interval answers "would another
   set of assays show this?".

A fourth, `mantel_permutation`, exists because RDM entries are not independent
either: with n variants there are n(n-1)/2 pair entries but only n underlying
things that can be permuted.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


def _clean(*arrays):
    """Drop rows where any input is non-finite. Returns the filtered arrays."""
    arrays = [np.asarray(a, dtype=float).ravel() for a in arrays]
    keep = np.ones(len(arrays[0]), bool)
    for a in arrays:
        keep &= np.isfinite(a)
    return [a[keep] for a in arrays]


def is_degenerate(x, tol=1e-12) -> bool:
    """True when x carries no rank information at all (constant, or all tied).

    A predictor that emits one value for every test row has no defined Spearman
    against anything. Reporting 0.0 for it would be a claim; reporting NaN is
    the honest answer, and this is the test that distinguishes the two.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    return x.size < 2 or float(np.nanstd(x)) <= tol


def spearman(x, y):
    """Tie-aware Spearman. NaN when either side carries no rank information."""
    x, y = _clean(x, y)
    if x.size < 3 or is_degenerate(x) or is_degenerate(y):
        return float("nan")
    return float(spearmanr(x, y).correlation)


def spearman_p(x, y):
    """Tie-aware Spearman with its p-value. NaN pair when undefined."""
    x, y = _clean(x, y)
    if x.size < 3 or is_degenerate(x) or is_degenerate(y):
        return float("nan"), float("nan")
    r = spearmanr(x, y)
    return float(r.correlation), float(r.pvalue)


def partial_spearman(x, y, covars):
    """Spearman(x, y) controlling for one or more covariates.

    The standard procedure: rank every variable with ties averaged, least-
    squares residualise rank(x) and rank(y) on the ranked covariates plus an
    intercept, then take the PEARSON correlation of the two residual vectors.

    Do not re-rank the residuals. Residualisation is a linear operation whose
    output is meaningful on the rank scale; ranking it again discards the
    adjustment it just made and quietly reintroduces the covariate.
    """
    covars = [covars] if np.ndim(covars[0]) == 0 else list(covars)
    cleaned = _clean(x, y, *covars)
    x, y, covars = cleaned[0], cleaned[1], cleaned[2:]
    if x.size < 4 or is_degenerate(x) or is_degenerate(y):
        return float("nan")

    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([rankdata(c) for c in covars] + [np.ones(x.size)])

    def resid(v):
        coef, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ coef

    ex, ey = resid(rx), resid(ry)
    if is_degenerate(ex) or is_degenerate(ey):
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def rank_corr_columns(X, y):
    """|Spearman| of every COLUMN of X against y, tie-aware, in one pass.

    Equivalent to looping `spearman(X[:, j], y)` but ranks the whole matrix at
    once, which matters because feature selection re-runs this for every
    candidate k on every inner fold of every split of every assay. Constant
    columns get 0 rather than NaN so they simply sort last.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[keep], y[keep]
    if X.shape[0] < 3:
        return np.zeros(X.shape[1])
    rx = np.apply_along_axis(rankdata, 0, X)
    ry = rankdata(y)
    rx = rx - rx.mean(0)
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rx * ry[:, None]).sum(0) / denom
    return np.abs(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))


def cluster_bootstrap(groups, stat=np.mean, n_boot=10000, seed=0,
                      hierarchical=True, ci=(2.5, 97.5)):
    """Resample CLUSTERS, not observations. Returns (point, lo, hi, n_clusters).

    `groups` maps a cluster label (an assay, a protein) to that cluster's
    observations -- typically the repeated train/test splits run inside it.
    Clusters are drawn with replacement; when `hierarchical`, the observations
    inside each drawn cluster are resampled too, which propagates within-assay
    splitting noise instead of pretending the cluster mean is exact.

    The point estimate is the mean of the per-cluster statistics, so an assay
    contributes once regardless of how many splits it happens to have -- an
    assay with 10 splits should not outvote one with 5.
    """
    if isinstance(groups, dict):
        keys = sorted(groups)
        obs = [np.asarray(groups[k], dtype=float).ravel() for k in keys]
    else:
        keys = list(range(len(groups)))
        obs = [np.asarray(g, dtype=float).ravel() for g in groups]
    obs = [o[np.isfinite(o)] for o in obs]
    obs = [o for o in obs if o.size]
    k = len(obs)
    if k == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(stat([stat(o) for o in obs]))
    if k == 1:
        return point, float("nan"), float("nan"), 1

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        vals = []
        for i in pick:
            o = obs[i]
            vals.append(stat(rng.choice(o, o.size, replace=True))
                        if hierarchical else stat(o))
        draws[b] = stat(vals)
    lo, hi = np.percentile(draws, ci)
    return point, float(lo), float(hi), k


def paired_cluster_bootstrap(groups_a, groups_b, n_boot=10000, seed=0,
                             hierarchical=True, ci=(2.5, 97.5)):
    """Cluster bootstrap of a PAIRED difference (predictor A minus B).

    The two predictors are scored on identical rows in identical splits, so the
    difference is taken within a split before anything is aggregated. Resampling
    A and B independently would discard that pairing and inflate the interval.
    """
    keys = sorted(set(groups_a) & set(groups_b))
    diffs = {}
    for kk in keys:
        a = np.asarray(groups_a[kk], dtype=float).ravel()
        b = np.asarray(groups_b[kk], dtype=float).ravel()
        n = min(a.size, b.size)
        d = a[:n] - b[:n]
        d = d[np.isfinite(d)]
        if d.size:
            diffs[kk] = d
    return cluster_bootstrap(diffs, n_boot=n_boot, seed=seed,
                             hierarchical=hierarchical, ci=ci)


def mantel_permutation(rdm_x, rdm_y, n_items, iu=None, covars=(), n_perm=10000,
                       seed=0):
    """Mantel test: permute the n ITEMS, not the n(n-1)/2 pair entries.

    An RDM built from n variants has far more entries than independent things,
    because every variant appears in n-1 of them. Treating the entries as a
    sample gives p-values that are wrong by orders of magnitude. The Mantel
    procedure permutes variant labels and rebuilds one RDM under each
    permutation, which respects that dependence exactly.

    Returns (observed, p_two_sided, null_sd). `rdm_x` and `rdm_y` are the
    upper-triangle vectors produced by `iu = np.triu_indices(n_items, 1)`;
    the same `iu` must be passed so the permuted matrix is read back the same way.
    """
    iu = np.triu_indices(n_items, 1) if iu is None else iu
    x = np.asarray(rdm_x, dtype=float).ravel()
    y = np.asarray(rdm_y, dtype=float).ravel()
    covars = [np.asarray(c, dtype=float).ravel() for c in covars]

    def stat(a, b):
        return (partial_spearman(a, b, covars) if covars else spearman(a, b))

    obs = stat(x, y)
    if not np.isfinite(obs):
        return obs, float("nan"), float("nan")

    # rebuild the square form once so each permutation is a cheap re-index
    sq = np.zeros((n_items, n_items))
    sq[iu] = x
    sq = sq + sq.T

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for p in range(n_perm):
        o = rng.permutation(n_items)
        null[p] = stat(sq[np.ix_(o, o)][iu], y)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return obs, float("nan"), float("nan")
    # +1 in numerator and denominator: the observed arrangement is itself one
    # of the permutations, so a p of exactly 0 is not attainable
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + null.size)
    return obs, float(p), float(np.std(null))


def circular_shift_test(x, y, covars=(), n_perm=None, min_shift=5):
    """Permutation test for a quantity indexed along a chain (a residue number).

    Residues near each other in sequence have similar values, so an ordinary
    shuffle produces a null with none of that smoothness and calls almost
    anything significant. A circular shift slides the whole response vector
    along the chain: the autocorrelation travels with it, but its alignment to
    the predictor is destroyed. That is the right null for "does the response
    track distance from the mutation, beyond what any smooth profile would give?"

    Exhaustive over all valid shifts unless `n_perm` caps it. Shifts smaller
    than `min_shift` (and their mirror at the far end) are skipped because a
    one-residue shift is essentially the observed arrangement.

    Returns (observed, p_two_sided, null_sd).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    covars = [np.asarray(c, dtype=float).ravel() for c in covars]
    n = x.size

    def stat(a):
        return (partial_spearman(a, y, covars) if covars else spearman(a, y))

    obs = stat(x)
    if not np.isfinite(obs):
        return obs, float("nan"), float("nan")

    shifts = [s for s in range(1, n) if min_shift <= s <= n - min_shift]
    if n_perm is not None and len(shifts) > n_perm:
        shifts = list(np.linspace(min_shift, n - min_shift, n_perm).astype(int))
    null = np.array([stat(np.roll(x, s)) for s in shifts])
    null = null[np.isfinite(null)]
    if null.size == 0:
        return obs, float("nan"), float("nan")
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + null.size)
    return obs, float(p), float(np.std(null))


def block_permutation(x, y, blocks, covars=(), n_perm=10000, seed=0):
    """Permutation test for values that are autocorrelated within a block.

    Residues inside one protein are spatially autocorrelated, so shuffling them
    freely destroys the dependence and produces an optimistic null. Permuting
    whole blocks -- or permuting only within a block, which is what this does --
    keeps the local structure the null needs to be fair.

    Returns (observed, p_two_sided, null_sd).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    blocks = np.asarray(blocks).ravel()
    covars = [np.asarray(c, dtype=float).ravel() for c in covars]

    def stat(a):
        return (partial_spearman(a, y, covars) if covars else spearman(a, y))

    obs = stat(x)
    if not np.isfinite(obs):
        return obs, float("nan"), float("nan")

    idx_by_block = [np.where(blocks == b)[0] for b in np.unique(blocks)]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for p in range(n_perm):
        xp = x.copy()
        for idx in idx_by_block:
            xp[idx] = x[rng.permutation(idx)]
        null[p] = stat(xp)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return obs, float("nan"), float("nan")
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + null.size)
    return obs, float(p), float(np.std(null))
