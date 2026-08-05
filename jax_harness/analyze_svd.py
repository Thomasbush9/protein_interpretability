"""Phase A of the SVD study: which directions does the mutation response use?

The transfer result (`transfer_v1.json`) already establishes that a probe
trained on eleven proteins predicts stability on a twelfth it has never seen,
at +0.572 -- at or above the within-assay ceiling. So a protein-general
mutation direction demonstrably exists. This analysis is not a search for one.
Its job is to say WHAT it is: how many directions carry the signal, whether
they are the high-variance ones, and whether twelve unrelated folds are using
the same ones.

Everything here runs on `dz_site` (or `ds_site`) from the archives, which
`exp_gym2` already stores as a DIFFERENCE, mutant minus wild type at the same
residue -- so no re-derivation is needed and no GPU forward pass is involved.

Four things are computed, in increasing order of what they can support.

  variance        Ordinary scree, per layer. Descriptive only. The top
                  component of an UNCENTRED decomposition is essentially the
                  mean mutation direction -- "something was substituted here"
                  -- and it would dominate every plot if it were not separated
                  out, so both centrings are run and reported apart.

  prediction      Held-out Spearman using the top k components, under the same
                  position-grouped protocol as the published probe, so the
                  numbers sit on the same scale as the +0.542 it reports. The
                  basis and the scaler are fitted on TRAINING positions only
                  and frozen before the held-out rows are projected; a basis
                  learned on all rows would leak the test positions into the
                  coordinate system itself, which is the easiest way to get a
                  flattering and meaningless curve here.

                  Components are ordered two ways -- by variance and by
                  training-set association with DMS -- because there is no
                  reason for those to agree, and if they disagree the
                  interesting direction is not the big one.

  agreement       Principal angles between the top-k subspaces of DIFFERENT
                  assays. The 128 coordinates are the model's own pair channels
                  and mean the same thing in every protein, so these subspaces
                  are directly comparable. This is the measurement that turns
                  "the probe transfers" into a statement about a specific
                  shared subspace. Note that per-dimension standardisation is
                  deliberately NOT applied for this part: rescaling each channel
                  by its own within-assay spread would rotate each basis by a
                  different amount and make the comparison meaningless.

  noise floor     The same principal angles between `gym2_*` and `gym2s_*`,
                  which are two independent runs of the IDENTICAL variants --
                  same seed, same sampling, verified by comparing the `mutant`
                  lists. That gives an empirical answer to "how many components
                  survive run-to-run drift" without spending a single GPU hour
                  on repeat inference, and it upper-bounds how many components
                  are worth interpreting.

Why this is a GPU job and not a numpy loop: a single SVD here is trivial, but
the analysis is 12 assays x 64 layers x 2 centrings x 5 splits of them, and the
permutation null multiplies the ridge solves by another few hundred. That is
millions of small dense solves, which is exactly the shape batched linear
algebra is good at and exactly the shape a Python loop is bad at. Everything
below is written as batched `jnp` over (layer, lambda, component-count,
permutation).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import pi_chem  # noqa: E402
import pi_stats  # noqa: E402
from compare_internal_output import grouped_split  # noqa: E402

LAMS = jnp.array([0.1, 1.0, 10.0, 100.0, 1000.0])
KS = (1, 2, 4, 8, 16, 32, 64, 128)
EPS = 1e-9


# --------------------------------------------------------------------------
# batched primitives
# --------------------------------------------------------------------------
def _rank(x, axis=-1):
    """Ordinal ranks along `axis`.

    Ties are broken by position rather than averaged, which differs from
    `pi_stats.spearman`. For continuous DMS scores and continuous ridge
    predictions ties are vanishingly rare, and `main` checks the resulting
    discrepancy against the tie-aware implementation before anything is
    reported -- see the `spearman agreement` line in the output.
    """
    return jnp.argsort(jnp.argsort(x, axis=axis), axis=axis).astype(jnp.float64)


def spearman_batch(pred, y):
    """Spearman of every row of `pred[..., m]` against `y[m]`. NaN if constant."""
    rp = _rank(pred, -1)
    ry = _rank(y, -1)
    rp = rp - rp.mean(-1, keepdims=True)
    ry = ry - ry.mean()
    den = jnp.sqrt((rp ** 2).sum(-1) * (ry ** 2).sum())
    return jnp.where(den > 0, (rp * ry).sum(-1) / jnp.where(den > 0, den, 1.0),
                     jnp.nan)


def ridge_grid(Ptr, ytr, Pte, lams=LAMS):
    """Ridge over a grid of lambdas, batched across the leading axis.

    Ptr (L, n, k), ytr (n,), Pte (L, m, k) -> predictions (L, n_lam, m).
    Both sides are assumed already centred on training means, so there is no
    intercept to leave unpenalised.
    """
    k = Ptr.shape[-1]
    G = jnp.einsum("lnk,lnj->lkj", Ptr, Ptr)
    b = jnp.einsum("lnk,n->lk", Ptr, ytr)
    A = G[:, None] + lams[None, :, None, None] * jnp.eye(k)
    w = jnp.linalg.solve(A, b[:, None, :, None])[..., 0]     # (L, n_lam, k)
    return jnp.einsum("lmk,lqk->lqm", Pte, w)


def fit_select(Ptr, ytr, Pin, yin, Pva, yva, Pte):
    """Choose lambda on an inner grouped split, refit on all training rows.

    Returns held-out predictions (L, m) and the chosen lambda index (L,).
    """
    rho_in = spearman_batch(ridge_grid(Pin, yin, Pva), yva)      # (L, n_lam)
    # abs FIRST, then replace NaN. The other order turns a degenerate fit
    # (NaN -> -1) into an apparently perfect one (|-1| = 1) and selects it.
    li = jnp.argmax(jnp.nan_to_num(jnp.abs(rho_in), nan=-1.0), axis=-1)
    pred = ridge_grid(Ptr, ytr, Pte)                             # (L, n_lam, m)
    return jnp.take_along_axis(pred, li[:, None, None], 1)[:, 0], li


def principal_cos2(Ba, Bb):
    """Mean squared cosine of the principal angles between two subspaces.

    Ba, Bb are (L, k, D) with orthonormal ROWS. The singular values of
    Ba Bb^T are the cosines of the principal angles, so the mean of their
    squares is a rotation-invariant alignment score in [0, 1] that never asks
    two individual components to correspond -- which they cannot be made to do,
    since PC signs are arbitrary and near-degenerate PCs rotate freely.
    Two random k-dimensional subspaces of R^D score k/D.
    """
    s = jnp.linalg.svd(jnp.einsum("lkd,ljd->lkj", Ba, Bb), compute_uv=False)
    return (s ** 2).mean(-1)


# --------------------------------------------------------------------------
def basis_of(X, center):
    """Per-layer SVD basis of X (L, n, D). Returns (Vt, explained_variance)."""
    if center:
        X = X - X.mean(1, keepdims=True)
    s, vt = jnp.linalg.svd(X, full_matrices=False)[1:]
    return vt, s ** 2 / jnp.maximum((s ** 2).sum(-1, keepdims=True), EPS)


def load(files, key, want_scalars=True):
    out = {}
    for f in sorted(files):
        d = np.load(f, allow_pickle=True)
        name = Path(f).stem.split("_")[1] if Path(f).stem.startswith("gym2") \
            else Path(f).stem
        rec = {"X": np.asarray(d[key], np.float64), "y": np.asarray(d["score"], float),
               "pos": np.asarray(d["pos"]), "mutant": [str(m) for m in d["mutant"]]}
        if want_scalars:
            for s in ("kl_glob", "kl_site", "dmu_glob", "dsd_glob",
                      "shift_glob", "spread_glob"):
                if s in d.files:
                    rec[s] = np.asarray(d[s], np.float64)
        out[name] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/runs/"
    ap.add_argument("--glob", default=R + "gym2s_*.npz")
    ap.add_argument("--replicate-glob", default=R + "gym2_*.npz",
                    help="independent repeat of the same variants; '' to skip")
    ap.add_argument("--block", default="dz_site", choices=["dz_site", "ds_site"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--angle-k", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default="")
    a = ap.parse_args()

    print(f"jax devices: {jax.devices()}\n")
    A = load(glob.glob(a.glob), a.block)
    names = sorted(A)
    L, D = A[names[0]]["X"].shape[1:]
    ks = [k for k in KS if k <= D] + ([D] if D not in KS else [])
    print(f"{len(names)} assays, {L} layers, {D} dims, block={a.block}")
    print(f"component counts: {ks}\n")

    # ---- sanity: does the batched Spearman agree with the tie-aware one? ---
    rng0 = np.random.default_rng(0)
    d0 = A[names[0]]
    probe = rng0.normal(size=len(d0["y"]))
    gpu_rho = float(spearman_batch(jnp.asarray(probe)[None], jnp.asarray(d0["y"]))[0])
    cpu_rho = pi_stats.spearman(probe, d0["y"])
    print(f"spearman agreement (ordinal vs tie-aware): {gpu_rho:+.6f} vs "
          f"{cpu_rho:+.6f}   delta {abs(gpu_rho - cpu_rho):.2e}\n")

    # ---- sanity: does the batched ridge match a plain numpy one? -----------
    # The whole analysis rests on `ridge_grid`, which does the solve batched
    # over (layer, lambda) in one call. If that einsum/solve is wrong, every
    # number below is wrong in a way no amount of statistics would reveal, so
    # it is checked against the obvious loop on real data before use.
    _pos = d0["pos"]
    _tr, _te = grouped_split(_pos, a.frac, np.random.default_rng(0))
    _ti, _ei = np.where(_tr)[0], np.where(_te)[0]
    _X = jnp.asarray(d0["X"]).transpose(1, 0, 2)[-1:]                # (1, N, D)
    _Z = (_X - _X[:, _ti].mean(1, keepdims=True)) / (
        _X[:, _ti].std(1, keepdims=True) + EPS)
    _vt = basis_of(_Z[:, _ti], center=False)[0][:, :8]
    _P = jnp.einsum("lnd,lkd->lnk", _Z, _vt)
    _y = jnp.asarray(d0["y"])
    _yc = _y - _y[_ti].mean()
    _gpu = np.asarray(ridge_grid(_P[:, _ti], _yc[_ti], _P[:, _ei])[0, 2])
    _A = np.asarray(_P[0, _ti]), np.asarray(_yc[_ti]), np.asarray(_P[0, _ei])
    _w = np.linalg.solve(_A[0].T @ _A[0] + 10.0 * np.eye(8), _A[0].T @ _A[1])
    _ref = _A[2] @ _w
    print(f"batched ridge vs numpy reference: max abs difference "
          f"{np.abs(_gpu - _ref).max():.3e}\n")

    res = {"protocol": {"block": a.block, "n_layers": L, "dim": D, "ks": ks,
                        "seeds": a.seeds, "frac": a.frac, "n_perm": a.n_perm,
                        "angle_k": a.angle_k, "assays": names,
                        "spearman_delta": abs(gpu_rho - cpu_rho)}}
    store = {}

    # ======================================================================
    # 1. held-out prediction as a function of how many components are kept
    # ======================================================================
    for center in (True, False):
        tag = "centered" if center else "uncentered"
        print(f"=== {tag} basis: held-out Spearman vs number of components ===\n")
        VIEWS = ("pc_var", "pc_pred", "raw_sel")
        curves = {v: {n: np.full((a.seeds, len(ks), L), np.nan) for n in names}
                  for v in VIEWS}
        evr = {n: np.zeros((L, min(D, 32))) for n in names}

        for n in names:
            X = jnp.asarray(A[n]["X"]).transpose(1, 0, 2)        # (L, N, D)
            y = jnp.asarray(A[n]["y"])
            pos = A[n]["pos"]
            for s in range(a.seeds):
                rng = np.random.default_rng(s)
                tr, te = grouped_split(pos, a.frac, rng)
                itr, iva = grouped_split(pos[tr], a.frac, rng)
                ti, ei = np.where(tr)[0], np.where(te)[0]
                Xtr = X[:, ti]
                mu = Xtr.mean(1, keepdims=True) if center else 0.0
                sd = Xtr.std(1, keepdims=True) + EPS
                Z = (X - mu) / sd
                vt, ev = basis_of(Z[:, ti], center=False)        # already centred
                if s == 0:
                    evr[n] = np.asarray(ev[:, : evr[n].shape[1]])
                # Train-centre once, then project. The component scores are
                # deliberately NOT re-standardised afterwards: V is orthogonal,
                # so ridge on ALL D unscaled components is exactly ridge on Zc
                # itself, and the k=D endpoint of the curve therefore coincides
                # with the raw control by construction. Whitening the scores
                # would break that identity and make the curve's shape partly a
                # story about rescaling low-variance directions rather than
                # about dimensionality, which is what it is supposed to measure.
                Zc = Z - Z[:, ti].mean(1, keepdims=True)
                P = jnp.einsum("lnd,lkd->lnk", Zc, vt)           # (L, N, R)
                yc = y - y[ti].mean()

                # Ranked by association with DMS on TRAIN rows only -- for the
                # components, and separately for the raw coordinates. The raw
                # view is the control that decides what the PCA is worth: it is
                # the published probe's own recipe (standardise on train, take
                # the top-k features by training |rho|, ridge) applied to the
                # 128 pair channels instead of to four scalar summaries. If it
                # lands where the component curve lands, the informative thing
                # is dz itself and the decomposition is only describing it.
                rho_c = spearman_batch(P[:, ti].transpose(0, 2, 1), y[ti])
                order_p = jnp.argsort(-jnp.abs(jnp.nan_to_num(rho_c)), axis=-1)
                rho_z = spearman_batch(Zc[:, ti].transpose(0, 2, 1), y[ti])
                order_z = jnp.argsort(-jnp.abs(jnp.nan_to_num(rho_z)), axis=-1)

                for ki, k in enumerate(ks):
                    for v in VIEWS:
                        if v == "pc_var":
                            Q = P[..., :k]
                        elif v == "pc_pred":
                            Q = jnp.take_along_axis(P, order_p[:, None, :k], axis=2)
                        else:
                            Q = jnp.take_along_axis(Zc, order_z[:, None, :k], axis=2)
                        pred, _ = fit_select(
                            Q[:, ti], yc[ti], Q[:, ti][:, itr], yc[ti][itr],
                            Q[:, ti][:, iva], yc[ti][iva], Q[:, ei])
                        curves[v][n][s, ki] = np.asarray(
                            spearman_batch(pred, y[ei]))

        # Pooled over assays as the mean of the LAST EIGHT layers -- the depth
        # range the published probe's feature selection concentrates in, and
        # the same window `analyze_channels` reads. Deliberately not the max
        # over layers: choosing the best of 64 layers by held-out performance
        # is a selection on the test set and would inflate every point on this
        # curve. The full per-layer surface goes to the npz instead.
        pooled = {}
        LABS = {"pc_var": "components, variance-ordered",
                "pc_pred": "components, prediction-ordered",
                "raw_sel": "RAW channels, prediction-selected (control)"}
        for v in VIEWS:
            lab, C = LABS[v], curves[v]
            row = {}
            for ki, k in enumerate(ks):
                g = {n: [float(np.nanmean(C[n][:, ki, -8:]))] for n in names}
                pt, lo, hi, _ = pi_stats.cluster_bootstrap(
                    g, n_boot=10000, seed=0, hierarchical=False)
                row[k] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                          "per_assay": {n: g[n][0] for n in names}}
            pooled[lab] = row
            print(f"  {lab}")
            print("     " + "".join(f"{('k=' + str(k)):>12s}" for k in ks))
            print("     " + "".join(f"{row[k]['mean']:>+12.3f}" for k in ks))
        # At k = D the rotation is complete and the selection has kept every
        # channel, so the two views are the same fit written in two bases and
        # must agree exactly. Any drift here means the projection or the
        # selection indexing is wrong.
        kD = ks.index(D)
        gap = max(float(np.nanmax(np.abs(curves["pc_var"][n][:, kD]
                                         - curves["raw_sel"][n][:, kD])))
                  for n in names)
        print(f"\n     identity check at k={D} (rotation vs selection): "
              f"max |difference| {gap:.2e}")
        print()
        res[tag] = {"curves": pooled,
                    "explained_variance_top": {n: evr[n].tolist() for n in names}}
        for v in VIEWS:
            store[f"curve_{v}_{tag}"] = np.stack([curves[v][n] for n in names])
        store[f"evr_{tag}"] = np.stack([evr[n] for n in names])

    # ======================================================================
    # 2. permutation null at a small, interpretable k
    # ======================================================================
    K0 = 8
    print(f"=== permutation null at k={K0} (labels shuffled, scored against "
          f"the shuffled labels) ===\n")
    print("   Shuffling only the TRAINING labels and scoring against the true")
    print("   held-out ones is NOT a null here: the fitted direction still lies")
    print("   inside a subspace whose axes are individually predictive, so the")
    print("   'null' inherits their association with DMS and comes out huge.")
    print("   The whole label vector is permuted and the permuted values are")
    print("   what the prediction is scored against.\n")
    null = {}
    for n in names:
        X = jnp.asarray(A[n]["X"]).transpose(1, 0, 2)
        y = jnp.asarray(A[n]["y"])
        pos = A[n]["pos"]
        rng = np.random.default_rng(0)
        tr, te = grouped_split(pos, a.frac, rng)
        ti, ei = np.where(tr)[0], np.where(te)[0]
        Xtr = X[:, ti]
        Z = (X - Xtr.mean(1, keepdims=True)) / (Xtr.std(1, keepdims=True) + EPS)
        vt, _ = basis_of(Z[:, ti], center=False)
        Zc = Z - Z[:, ti].mean(1, keepdims=True)
        P = jnp.einsum("lnd,lkd->lnk", Zc, vt)[..., :K0]
        def held_out_rho(yy):
            """Fit on the training rows of `yy`, score on its held-out rows."""
            return spearman_batch(
                ridge_grid(P[:, ti], yy[ti] - yy[ti].mean(), P[:, ei])[:, 2],
                yy[ei])                                              # lam = 10

        real = np.asarray(held_out_rho(y))
        perm_y = jnp.stack([jnp.asarray(rng.permutation(np.asarray(y)))
                            for _ in range(a.n_perm)])
        pr = np.asarray(jax.vmap(held_out_rho)(perm_y))              # (n_perm, L)
        # Max-statistic correction: the real value is the best of 64 layers, so
        # the null it is judged against must also be the best of 64. Comparing
        # against the null at one pre-chosen layer would count the layer search
        # as free and inflate significance by roughly the number of layers.
        best = int(np.nanargmax(np.abs(real)))
        real_stat = float(np.nanmax(np.abs(real)))
        null_stat = np.nanmax(np.abs(pr), axis=1)                    # (n_perm,)
        p = float((null_stat >= real_stat).mean())
        null[n] = {"best_layer": best, "rho": float(real[best]),
                   "null_max_mean": float(np.nanmean(null_stat)),
                   "null_max_p95": float(np.nanpercentile(null_stat, 95)),
                   "p_perm_maxstat": p}
        print(f"   {n:8s} layer {best:2d}  rho {real[best]:+.3f}   "
              f"null max|rho| p95 {null[n]['null_max_p95']:.3f}   p={p:.3f}")
    res["permutation_null"] = {
        "k": K0, "per_assay": null,
        "caveat": "free row permutation of the label vector; it does not "
                  "preserve the within-position autocorrelation of DMS, so it "
                  "is a null for 'no association' rather than for 'no "
                  "association beyond position structure'. The position-"
                  "grouped split is what handles the latter."}
    print()

    # ======================================================================
    # 3. do different proteins use the same subspace?
    # ======================================================================
    k = a.angle_k
    print(f"=== subspace agreement, top-{k} components (raw, mean-removed) ===\n")
    B = {}
    for n in names:
        X = jnp.asarray(A[n]["X"]).transpose(1, 0, 2)
        vt, _ = basis_of(X, center=True)
        B[n] = vt[:, :k]
    chance = k / D
    M = np.full((len(names), len(names), L), np.nan)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j <= i:
                continue
            M[i, j] = M[j, i] = np.asarray(principal_cos2(B[ni], B[nj]))
    iu = np.triu_indices(len(names), 1)
    per_layer = np.nanmean(M[iu[0], iu[1]], axis=0)
    print(f"   chance for random {k}-dim subspaces of R^{D}: {chance:.3f}")
    print(f"   mean cos^2 across the {len(iu[0])} assay pairs, by layer:")
    for lo in range(0, L, 8):
        seg = per_layer[lo:lo + 8]
        print(f"     layers {lo:2d}-{lo + len(seg) - 1:2d}  " +
              " ".join(f"{v:.3f}" for v in seg))
    g = {f"{names[i]}|{names[j]}": [float(np.nanmean(M[i, j, -8:]))]
         for i, j in zip(*iu)}
    pt, lo_, hi_ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                              hierarchical=False)[:3]
    print(f"\n   last 8 layers, pooled over assay pairs: {pt:.3f} "
          f"[{lo_:.3f}, {hi_:.3f}]  vs chance {chance:.3f}")
    res["subspace_agreement"] = {"k": k, "chance": chance,
                                 "per_layer_mean": per_layer.tolist(),
                                 "last8_pooled": {"mean": pt, "ci_lo": lo_,
                                                  "ci_hi": hi_},
                                 "pairs_last8": {kk: v[0] for kk, v in g.items()}}
    store["subspace_matrix"] = M

    # ======================================================================
    # 4. how much of that subspace survives an independent repeat?
    # ======================================================================
    if a.replicate_glob:
        Rp = load(glob.glob(a.replicate_glob), a.block, want_scalars=False)
        shared = [n for n in names if n in Rp]
        print(f"\n=== run-to-run stability, {len(shared)} assays with a repeat ===\n")
        rep = {}
        for n in shared:
            if list(Rp[n]["mutant"]) != list(A[n]["mutant"]):
                print(f"   {n:8s} SKIPPED: variant lists differ between runs")
                continue
            Xr = jnp.asarray(Rp[n]["X"]).transpose(1, 0, 2)
            vt, _ = basis_of(Xr, center=True)
            c2 = np.asarray(principal_cos2(B[n], vt[:, :k]))
            rep[n] = float(np.nanmean(c2[-8:]))
            print(f"   {n:8s} mean cos^2(run1, run2) over last 8 layers: "
                  f"{rep[n]:.4f}")
        if rep:
            g = {n: [v] for n, v in rep.items()}
            pt, lo_, hi_ = pi_stats.cluster_bootstrap(
                g, n_boot=10000, seed=0, hierarchical=False)[:3]
            print(f"\n   pooled {pt:.4f} [{lo_:.4f}, {hi_:.4f}]   "
                  f"(chance {chance:.3f})")
            print("   This is the ceiling: cross-assay agreement cannot be")
            print("   interpreted as higher than the repeat of one assay.")
            res["replicate_stability"] = {"per_assay": rep,
                                          "pooled": {"mean": pt, "ci_lo": lo_,
                                                     "ci_hi": hi_}}

    # ======================================================================
    # 5. what are the leading components made of?
    # ======================================================================
    # The basis is learned ONCE on all twelve assays pooled, not per assay.
    # That is not a convenience: the sign of a singular vector is arbitrary, so
    # averaging a signed correlation across twelve independently-computed bases
    # cancels to nothing no matter how strong the association is in each one.
    # An earlier version of this section did exactly that and reported ~0.02
    # against DMS for every component while the same quantity was -0.62 in a
    # two-assay run -- the difference was sign cancellation, not effect size.
    # A shared basis gives every assay the same coordinate system and the same
    # sign convention, which is also the object the transfer result implies
    # exists. Rows are z-scored WITHIN assay first so that no protein dominates
    # the decomposition through its own representation scale.
    print("\n=== component annotation on a SHARED basis, last layer, top 8 ===\n")
    SCAL = ["kl_glob", "kl_site", "dmu_glob", "dsd_glob", "shift_glob", "spread_glob"]

    def zc(M):
        return (M - M.mean(0)) / (M.std(0) + EPS)

    Xg = np.concatenate([zc(np.asarray(A[n]["X"])[:, -1, :]) for n in names], 0)
    Vg = np.asarray(basis_of(jnp.asarray(Xg)[None], center=True)[0][0][:8])  # (8, D)
    ann = {}
    for n in names:
        P = (zc(np.asarray(A[n]["X"])[:, -1, :]) - Xg.mean(0)) @ Vg.T       # (N, 8)
        cols = {"DMS": A[n]["y"]}
        for s in SCAL:
            if s in A[n]:
                cols[s] = A[n][s][:, -1]
        C = pi_chem.chem_matrix(A[n]["mutant"])
        for j, nm in enumerate(pi_chem.CHEM_FEATURES):
            cols["chem:" + nm] = C[:, j]
        ann[n] = {lab: [pi_stats.spearman(P[:, c], v) for c in range(8)]
                  for lab, v in cols.items()}
    def pooled_rho(lab, c):
        """Assay-level bootstrap of a signed correlation. Signs are shared now."""
        g = {n: [ann[n][lab][c]] for n in names if np.isfinite(ann[n][lab][c])}
        if not g:
            return np.nan, np.nan, np.nan
        return pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                          hierarchical=False)[:3]

    keys = ["DMS"] + [s for s in SCAL if s in A[names[0]]]
    print(f"   {'quantity':14s}" + "".join(f"{('PC' + str(c + 1)):>9s}"
                                           for c in range(8)))
    ann_pooled = {}
    for lab in keys:
        v = [pooled_rho(lab, c) for c in range(8)]
        ann_pooled[lab] = [{"mean": m, "ci_lo": lo, "ci_hi": hi}
                           for m, lo, hi in v]
        star = ["*" if np.isfinite(lo) and (lo > 0 or hi < 0) else " "
                for _, lo, hi in v]
        print(f"   {lab:14s}" + "".join(f"{m:>+8.3f}{s}"
                                        for (m, _, _), s in zip(v, star)))
    print("   * = assay-level 95% interval excludes zero")

    chem_pooled = {}
    for nm in pi_chem.CHEM_FEATURES:
        chem_pooled[nm] = [pooled_rho("chem:" + nm, c)[0] for c in range(8)]
    top = sorted(((abs(v), nm, c) for nm, vs in chem_pooled.items()
                  for c, v in enumerate(vs) if np.isfinite(v)), reverse=True)[:8]
    print("\n   strongest chemistry associations (pooled signed rho):")
    for _, nm, c in top:
        m, lo, hi = pooled_rho("chem:" + nm, c)
        print(f"     PC{c + 1}  {nm:22s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    res["annotation_last_layer"] = {"per_assay": ann, "pooled": ann_pooled,
                                    "basis": "shared across all assays",
                                    "chem_pooled": chem_pooled}

    # ======================================================================
    # 6. is the shared subspace the thing that transfers?
    # ======================================================================
    # `transfer_v1.json` shows a probe trained on eleven proteins scoring
    # +0.572 on the twelfth. Section 3 shows the twelve mutation subspaces
    # largely coincide. This closes the loop between them: learn the basis on
    # eleven assays, keep k directions, and ask how much of that transfer
    # survives. If a handful of directions reproduce it, the general signal has
    # been reduced to something small enough to inspect, patch and steer.
    #
    # Restricted to the last eight layers to keep the pooled decompositions
    # (2750 x D per layer per held-out assay) affordable; that is the depth
    # window every other block in this project reads.
    print("\n=== leave-one-assay-out on a shared basis (last 8 layers) ===\n")
    LK = [1, 2, 4, 8, 16, 32, min(64, D), D]
    LK = sorted({k for k in LK if k <= D})
    Zg = {n: zc(np.asarray(A[n]["X"])[:, -8:, :]) for n in names}      # (N,8,D)
    yz = {n: (A[n]["y"] - A[n]["y"].mean()) / (A[n]["y"].std() + EPS) for n in names}
    loao = {k: {} for k in LK}
    for h in names:
        tr_n = [n for n in names if n != h]
        Xtr = jnp.asarray(np.concatenate([Zg[n] for n in tr_n], 0)).transpose(1, 0, 2)
        ytr = jnp.asarray(np.concatenate([yz[n] for n in tr_n]))
        Xte = jnp.asarray(Zg[h]).transpose(1, 0, 2)
        vt, _ = basis_of(Xtr, center=True)
        mu = Xtr.mean(1, keepdims=True)
        Ptr = jnp.einsum("lnd,lkd->lnk", Xtr - mu, vt)
        Pte = jnp.einsum("lnd,lkd->lnk", Xte - mu, vt)
        for k in LK:
            pred = ridge_grid(Ptr[..., :k], ytr - ytr.mean(), Pte[..., :k])[:, 2]
            loao[k][h] = float(np.nanmean(
                np.asarray(spearman_batch(pred, jnp.asarray(A[h]["y"])))))
    print("   " + "".join(f"{('k=' + str(k)):>12s}" for k in LK))
    lo_sum = {}
    for k in LK:
        pt, lo_, hi_, _ = pi_stats.cluster_bootstrap(
            {n: [v] for n, v in loao[k].items()}, n_boot=10000, seed=0,
            hierarchical=False)
        lo_sum[k] = {"mean": pt, "ci_lo": lo_, "ci_hi": hi_,
                     "per_assay": loao[k]}
    print("   " + "".join(f"{lo_sum[k]['mean']:>+12.3f}" for k in LK))
    print("\n   published leave-one-assay-out probe (transfer_v1): +0.572")
    print("   Reference only -- that probe uses four scalar summaries over all")
    print("   64 layers, this uses the raw pair channels over the last 8, so")
    print("   the two are not the same feature space.")
    res["loao_shared_basis"] = {"layers": "last 8", "ks": LK, "results": lo_sum}

    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {a.out}")
    if a.npz:
        np.savez_compressed(a.npz, assays=np.array(names), ks=np.array(ks), **store)
        print(f"wrote {a.npz}")


if __name__ == "__main__":
    main()
