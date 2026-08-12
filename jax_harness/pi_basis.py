"""One definition of the shared PC basis.

Eight scripts built this object by hand: per-assay z-score, pool across assays,
subtract the pooled mean, SVD, fix the sign. The helper `zc` was duplicated
verbatim seven times. Nothing anywhere said "this is PC2" -- the definition
lived in eight docstrings and was reconstructed by whoever wrote the next
analysis.

The three decisions that determine what PC2 *is* had drifted apart accordingly.
Sign orientation, which the causal steering result depends on, was three
different rules:

  analyze_heldout:169   cluster bootstrap of the sign against kl_glob
  analyze_chem:208      mean sign against DMS, TRAINING assays only
  analyze_transfer:217  none at all -- ridge is sign-invariant, so it never showed

The comment at analyze_heldout:165 reads "exactly as analyze_svd/analyze_chem
do". That sentence was the only thing keeping three implementations in
agreement, and it was not true of analyze_chem's leave-one-assay-out block
twenty lines further down.

This module owns the policy. It does not own the SVD -- `analyze_svd.basis_of`
is a fine primitive and seven call sites there are right to use it directly.
What was missing was the layer above: which rows are standardised, on what, and
which way the components point.

WHAT IS DELIBERATELY NOT HERE. Per-assay per-layer bases fitted without pooling
(`analyze_svd.basis_of` under vmap) are a different object: no shared
orientation, so their components cannot be compared between proteins or named.
They produce the truncation curves and they stay where they are.

THREE THINGS THIS INTERFACE MAKES HARD TO GET WRONG.

  1. Standardising on all rows and fitting on training rows is a half-leak that
     moves the answer by about 0.003 and is invisible on inspection. `rows`
     governs the scale, the pooled mean and the decomposition together, so
     "did this touch held-out data" is one readable fact.

  2. Orienting on kl_glob while fitting on a training subset orients on
     held-out data. That combination raises.

  3. A last-layer basis applied at every depth was nearly orthogonal at
     mid-depth -- 0.09 against 0.031 chance -- and reversed a published
     conclusion before anyone measured it. Shape cannot catch this: a depth
     profile and a final-layer row are both 128 wide. `project` therefore takes
     the layer as an argument and refuses a mismatch.

THE BRIDGE. Components live in the standardised basis, so the raw-space vector
for component c is `e_c = s * v_c` and the readout covector is `w_c = v_c / s`,
with `w_c . e_c = 1`. Five files derived that by hand and inverting it is
silent -- you get a plausible number. `to_raw` and `readout` are the only
supported way to cross.

Usage:

    b = pi_basis.fit(blocks, layer=-1, orient_on="kl_glob", orient_ref=kl)
    P = b.project(X, layer=-1)          # (n, k) component scores
    e = b.to_raw(1, assay="1CEI")       # PC2 as a raw-channel vector
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_stats  # noqa: E402

# The z-score denominator is `sd + EPS`, and the eight sites did not agree on
# EPS either: 1e-9 in six of them, 1e-8 in analyze_basis. On channels with a
# spread near 1e-2 that is a relative shift of ~1e-7 -- far above the 1e-10 bar
# this module is held to, so it cannot be a constant. 1e-9 is the majority and
# the default; analyze_basis passes eps=1e-8 and keeps reproducing itself.
EPS = 1e-9
ORIENT_RULES = ("kl_glob", "dms_train", None)


def _svd(M):
    """SVD on the accelerator, returned as numpy.

    jnp because analyze_transfer refits inside a leave-one-assay-out loop --
    twelve folds for every k in the sweep -- and because a job that holds a GPU
    should use it. numpy on the way out so the twenty-odd existing call sites
    need no changes. This boundary is why the equivalence bar is 1e-10 rather
    than bit-for-bit.

    JAX DEFAULTS TO FLOAT32 and downcasts a float64 array on the way in without
    saying so. The first run of this module's own test caught it: orthonormality
    was off by 1.05e-06 on a 40x8 matrix, where float64 gives ~1e-15. A silently
    single-precision basis is exactly the failure this project keeps paying for
    -- a plausible number, no error -- so x64 is requested explicitly, verified,
    and numpy takes over if it is unavailable rather than returning a degraded
    basis. x64 is scoped to this call: setting it globally would change the
    numerics of every other analysis that imports this module.
    """
    import contextlib

    import jax.numpy as jnp
    M = np.asarray(M, np.float64)
    try:
        from jax.experimental import enable_x64
        ctx = enable_x64()
    except ImportError:
        ctx = contextlib.nullcontext()
    with ctx:
        if jnp.asarray(M).dtype != jnp.float64:
            s, vt = np.linalg.svd(M, full_matrices=False)[1:]
        else:
            s, vt = jnp.linalg.svd(jnp.asarray(M), full_matrices=False)[1:]
            s, vt = np.asarray(s, np.float64), np.asarray(vt, np.float64)
    return vt, s ** 2 / max(float((s ** 2).sum()), EPS)


def _standardise(X, center, mu=None, sd=None, eps=EPS, zscore=True):
    """The former `zc`, with its one variant.

    `center=False` is analyze_scrutiny's wild-type-anchoring control: divide by
    the spread but do not remove the mean, so whatever the anchor contributes
    survives into the decomposition. It is a flag rather than a fork precisely
    because that check only has force if the control and the thing it controls
    for are provably the same code.
    """
    # zscore=False is analyze_transfer's construction (:216-219): no per-assay
    # operation at all -- rows pass through untouched and the pooled mean does
    # whatever centring happens. That is a DIFFERENT OBJECT from PC2. Without
    # standardising, a protein whose channels happen to have a larger scale
    # pulls the components toward itself, so the result partly encodes which
    # protein a row came from. The k-sweep's "rotated" curve is built this way
    # and nothing said so; it is a flag here so the difference is visible at
    # the call site instead of buried in a loop.
    if not zscore:
        return X
    # eps is added HERE, never baked into a stored `sd`. exp_jac keeps the raw
    # spread and adds it only in the covector (`W = V / (sd + EPS)`, `E = V *
    # sd`), so a stored `sd + eps` would count it twice on the readout side.
    sd = (X.std(0) if sd is None else sd) + eps
    if not center:
        return X / sd
    mu = X.mean(0) if mu is None else mu
    return (X - mu) / sd


def _slice(A, layer):
    """(n, L, D) -> (n, D), or pass a 2-D block through.

    Returns the block and whether the layer was measured or merely asserted.
    Callers that already sliced (analyze_heldout's load_assay does) can only
    assert it, and the protocol block records which of the two it was.
    """
    A = np.asarray(A, np.float64)
    if A.ndim == 3:
        return A[:, layer, :], A.shape[1], False
    if A.ndim == 2:
        return A, None, True
    raise ValueError(f"expected (n, L, D) or (n, D), got {A.shape}")


@dataclass
class Basis:
    """A fitted shared basis, and everything needed to use it correctly.

    `V` is stored UNORIENTED with the signs in `orient`, matching pc2_v2.npz,
    so a saved basis round-trips through the format exp_jac and analyze_rotate
    already read. Use `.components` rather than `.V` unless you specifically
    want the raw singular vectors.
    """

    V: np.ndarray                      # (k, D) right singular vectors, rows
    orient: np.ndarray                 # (k,) +-1
    gm: np.ndarray                     # (D,) pooled mean, standardised space
    ev: np.ndarray                     # (k,) explained variance ratio
    mu: dict = field(repr=False)       # assay -> (D,) raw channel mean
    sd: dict = field(repr=False)       # assay -> (D,) raw channel spread
    mu_pooled: np.ndarray = field(repr=False)
    sd_pooled: np.ndarray = field(repr=False)
    layer: int = -1
    n_layers: int | None = None
    layer_asserted: bool = False
    centered: bool = True
    orient_on: str | None = "kl_glob"
    orient_k: int = 2
    rows_restricted: bool = False
    assays: tuple = ()
    n_rows: int = 0
    eps: float = EPS
    zscore: bool = True

    # ---- using it ---------------------------------------------------------
    @property
    def components(self):
        """(k, D) with the sign convention applied. This is PC1, PC2, ..."""
        return self.V * self.orient[:, None]

    def project(self, X, layer, assay=None, standardise="own"):
        """Component scores for a block of variants.

        `layer` is required and checked. It is the only defence against the
        depth mistake, since a depth profile and a final-layer row have the
        same width and the same dtype.

        `standardise="own"` uses the projected block's own channel statistics
        -- the transductive convention every existing call site uses, where a
        held-out protein contributes its own scale. `"train"` uses the pooled
        statistics from the fit, which is the inductive variant: strictly
        weaker, and the only version in which a held-out protein contributes
        nothing at all.
        """
        return self.features(X, layer, standardise) @ self.components.T

    def features(self, X, layer, standardise="own"):
        """The standardised, centred rows -- what `project` decomposes.

        A probe fitted on the model's own channels rather than on components
        needs exactly this, and callers were re-deriving it with a local `zc`.
        Exposing it is also what lets the transductive/inductive pair be two
        arguments instead of two hand-written expressions:

            Zt = b.features(X, -1)                      # own statistics
            Zi = b.features(X, -1, standardise="train") # training statistics

        which is the difference between a held-out protein contributing its
        own scale and contributing nothing at all.
        """
        if layer != self.layer:
            raise ValueError(
                f"this basis was fitted at layer {self.layer}; you asked for "
                f"layer {layer}. The PC basis rotates with depth (0.09 overlap "
                f"at mid-depth against 0.031 chance), so a last-layer basis "
                f"does not describe another layer. Refit at {layer}.")
        Xb, _, _ = _slice(X, layer)
        if standardise == "own":
            Z = _standardise(Xb, self.centered, eps=self.eps,
                             zscore=self.zscore)
        elif standardise == "train":
            Z = _standardise(Xb, self.centered, self.mu_pooled, self.sd_pooled,
                             eps=self.eps, zscore=self.zscore)
        else:
            raise ValueError(f"standardise must be 'own' or 'train', "
                             f"got {standardise!r}")
        return Z - self.gm

    def to_raw(self, c, assay):
        """Component c as a raw-channel VECTOR: e_c = s * v_c."""
        return self.components[c] * self.sd[assay]

    def readout(self, c, assay):
        """Component c as a raw-channel COVECTOR: w_c = v_c / s.

        Not the same vector as `to_raw`. `readout(c) . to_raw(c) == 1`; using
        one where the other belongs rescales by s^2 per channel and returns a
        number that looks fine.
        """
        return self.components[c] / (self.sd[assay] + self.eps)

    # ---- what it knows about itself ---------------------------------------
    @property
    def protocol(self):
        """The basis-specific half of a pi_protocol block.

        This object is the only thing that knows the layer, the centring, the
        orientation rule, the width and whether rows were restricted all at
        once -- which is why hand-written blocks got them wrong. `design` and
        `source` still belong to the analysis; merge this into its call.
        """
        return {
            "basis": {
                "module": "pi_basis",
                "layer": int(self.layer),
                "n_layers": self.n_layers,
                "layer_measured": not self.layer_asserted,
                "centered": bool(self.centered),
                "orient_on": self.orient_on,
                "orient_k": int(self.orient_k),
                "orient_signs": [int(s) for s in self.orient],
                "n_components": int(self.V.shape[0]),
                "dim": int(self.V.shape[1]),
                "fit_assays": list(self.assays),
                "fit_rows": int(self.n_rows),
                "rows_restricted": bool(self.rows_restricted),
                "eps": float(self.eps),
                "zscore": bool(self.zscore),
                "explained_variance": [float(x) for x in self.ev[:8]],
            }
        }

    # ---- on disk ----------------------------------------------------------
    def save(self, path):
        """Write the keys pc2_v2.npz already uses, plus the rest.

        exp_jac reads V and orient; analyze_rotate and analyze_ops read sd and
        gm. Those consumers sit downstream of published figures, so the format
        stays as it is and gains fields rather than changing them.
        """
        assays = list(self.assays)
        np.savez_compressed(
            path, V=self.V, orient=self.orient, gm=self.gm, ev=self.ev,
            sd=np.stack([self.sd[n] for n in assays]) if assays else np.zeros((0, 0)),
            mu=np.stack([self.mu[n] for n in assays]) if assays else np.zeros((0, 0)),
            mu_pooled=self.mu_pooled, sd_pooled=self.sd_pooled,
            assays=np.array(assays), layer=self.layer,
            n_layers=-1 if self.n_layers is None else self.n_layers,
            layer_asserted=self.layer_asserted, centered=self.centered,
            orient_on="" if self.orient_on is None else self.orient_on,
            orient_k=self.orient_k, rows_restricted=self.rows_restricted,
            n_rows=self.n_rows, eps=self.eps, zscore=self.zscore)


def load(path):
    """Read a basis written by `Basis.save`."""
    d = np.load(path, allow_pickle=True)
    assays = [str(x) for x in d["assays"]]
    nl = int(d["n_layers"])
    return Basis(
        V=np.asarray(d["V"], np.float64), orient=np.asarray(d["orient"], np.float64),
        gm=np.asarray(d["gm"], np.float64), ev=np.asarray(d["ev"], np.float64),
        mu={n: np.asarray(d["mu"][i], np.float64) for i, n in enumerate(assays)},
        sd={n: np.asarray(d["sd"][i], np.float64) for i, n in enumerate(assays)},
        mu_pooled=np.asarray(d["mu_pooled"], np.float64),
        sd_pooled=np.asarray(d["sd_pooled"], np.float64),
        layer=int(d["layer"]), n_layers=None if nl < 0 else nl,
        layer_asserted=bool(d["layer_asserted"]), centered=bool(d["centered"]),
        orient_on=str(d["orient_on"]) or None, orient_k=int(d["orient_k"]),
        rows_restricted=bool(d["rows_restricted"]), assays=tuple(assays),
        n_rows=int(d["n_rows"]), eps=float(d["eps"]),
        zscore=bool(d["zscore"]))


def save_stack(path, bases, assays):
    """Write a depth stack in basis_depth.npz's layout.

    analyze_basis produces one basis per layer and analyze_rotate/analyze_ops
    read the stack through `--basis`. V is stored ORIENTED here, as that file
    always has; the per-layer objects keep their own signs.
    """
    np.savez_compressed(
        path,
        V=np.stack([b.components for b in bases]),                  # (L, k, D)
        sd=np.stack([np.stack([b.sd[n] for b in bases]) for n in assays]),
        gm=np.stack([b.gm for b in bases]),
        assays=np.array(list(assays)))


# --------------------------------------------------------------------------
def fit(blocks, *, layer, center=True, orient_on="kl_glob", orient_ref=None,
        orient_k=2, rows=None, n_pc=None, seed=0, n_boot=2000, eps=EPS,
        zscore=True):
    """Fit the shared basis across assays.

    blocks      {assay: (n, L, D) or (n, D)}. A 3-D block is sliced at `layer`
                and the layer is then a measured fact; a 2-D block means the
                caller asserted it, which the protocol records.
    rows        {assay: bool mask} restricting the fit. Governs the channel
                scale, the pooled mean AND the decomposition together -- they
                are not independently meaningful, and separating them is the
                half-leak that is hard to see.
    orient_on   "kl_glob" (the canonical rule: cluster bootstrap of the sign
                against the KL column), "dms_train" (mean sign against the DMS
                score over fit assays only), or None for unoriented.
    orient_ref  {assay: (n,)} the quantity to orient against. Required unless
                orient_on is None.
    orient_k    how many leading components get a fixed sign. 2 matches
                analyze_heldout and analyze_pc2; analyze_chem, analyze_attrib
                and analyze_scrutiny use 4.
    """
    if orient_on not in ORIENT_RULES:
        raise ValueError(f"orient_on must be one of {ORIENT_RULES}, "
                         f"got {orient_on!r}")
    if orient_on == "kl_glob" and rows is not None:
        raise ValueError(
            "orient_on='kl_glob' with a restricted `rows` orients the sign on "
            "held-out rows: the bootstrap would read the very data the "
            "restriction exists to keep out. Use orient_on='dms_train', which "
            "takes its sign from the fit rows only, or drop the restriction.")
    if orient_on is not None and not orient_ref:
        raise ValueError(f"orient_on={orient_on!r} needs orient_ref, a mapping "
                         f"from assay to the quantity to orient against.")

    names = sorted(blocks)
    if not names:
        raise ValueError("no assays given")

    Z, n_layers, asserted, keep = {}, None, False, {}
    for n in names:
        Xb, nl, asrt = _slice(blocks[n], layer)
        n_layers = nl if nl is not None else n_layers
        asserted = asserted or asrt
        m = np.ones(len(Xb), bool) if rows is None else np.asarray(rows[n], bool)
        if m.sum() < 2:
            raise ValueError(f"{n}: {int(m.sum())} fit rows is not enough to "
                             f"estimate a channel spread")
        keep[n] = (Xb, m)
        Z[n] = _standardise(Xb[m], center, eps=eps, zscore=zscore)

    Xg = np.concatenate([Z[n] for n in names], 0)
    gm = Xg.mean(0) if center else np.zeros(Xg.shape[1])
    V, ev = _svd(Xg - gm)
    if n_pc is not None:
        V, ev = V[:n_pc], ev[:n_pc]

    # Orientation. Singular-vector signs are arbitrary; without a fixed
    # convention held-out correlations carry a meaningless sign and averaging
    # them cancels. The sign is chosen ONCE, on the basis, and never re-chosen
    # downstream.
    orient = np.ones(V.shape[0])
    ok = min(orient_k, V.shape[0])
    if orient_on is not None:
        P = {n: (Z[n] - gm) @ V.T for n in names}
        for c in range(ok):
            ref = {n: np.asarray(orient_ref[n], float)[keep[n][1]] for n in names}
            if orient_on == "kl_glob":
                g = {n: [pi_stats.spearman(P[n][:, c], ref[n])] for n in names}
                s = pi_stats.cluster_bootstrap(g, n_boot=n_boot, seed=seed,
                                               hierarchical=False)[0]
            else:  # dms_train -- mean sign over the fit assays only
                s = np.mean([pi_stats.spearman(P[n][:, c], ref[n]) for n in names])
            if s < 0:
                orient[c] = -1.0

    raw = np.concatenate([keep[n][0][keep[n][1]] for n in names], 0)
    return Basis(
        V=V, orient=orient, gm=gm, ev=ev,
        mu={n: keep[n][0][keep[n][1]].mean(0) for n in names},
        sd={n: keep[n][0][keep[n][1]].std(0) for n in names},
        mu_pooled=raw.mean(0), sd_pooled=raw.std(0),
        layer=layer, n_layers=n_layers, layer_asserted=asserted,
        centered=center, orient_on=orient_on, orient_k=ok,
        rows_restricted=rows is not None, assays=tuple(names),
        n_rows=int(len(Xg)), eps=eps, zscore=zscore)
