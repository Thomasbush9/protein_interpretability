"""Do three structure predictors represent mutations the same way?

All three happen to use a 128-dimensional pair representation. That is a
coincidence of architecture, not a shared coordinate system: Boltz-2, OpenFold3
and Protenix were trained independently, so channel 5 of one has no relation to
channel 5 of another. Equal dimensionality makes it *possible* to compute
principal angles between their subspaces and *wrong* to interpret the result, so
that quantity is computed here only as a NEGATIVE control -- it should sit at
the chance level k/128, and if it does not, something is wrong with the premise
rather than interesting about the models.

Every real comparison therefore goes through the variant axis, which is the only
thing the three share: identical proteins, identical mutations, identical
alignments, run through the uniform extractor.

  complementarity  The question the paper needs. Each model's own probe,
                   identical protocol, out-of-fold predictions on the same
                   held-out rows; then does combining beat the best single
                   model? Reported two ways -- a ridge on the concatenated
                   features, and an unweighted average of the standardised
                   predictions, which adds no fitted capacity at all and is
                   therefore the harder test to argue with.

  CKA              Linear centred kernel alignment between the two Dz matrices.
                   Invariant to orthogonal transforms and isotropic scaling,
                   which is exactly the freedom an arbitrary basis has, so it
                   asks whether the models organise mutation space similarly
                   without ever asking a channel to correspond.

  RSA              The same idea as a rank statistic: build the variant x
                   variant distance matrix inside each model and correlate the
                   two. Coordinate-free by construction.

  CCA              How many directions are shared. Regularised, on the top PCs
                   rather than raw channels, because with 100 variants against
                   128 dimensions an unregularised canonical correlation is
                   near 1 for pure noise. Only the count of directions clearing
                   a permuted null is reported, never the raw correlations.

The permutation null shuffles WHOLE RESIDUE POSITIONS, not variants. Variants at
one site share a residue environment and are not exchangeable; a free shuffle
builds an easier null and inflates every agreement statistic.

The r1/r2 repeat of each model gives the ceiling. Cross-model agreement means
nothing without it: a low number could always be one model being noisy, and the
only way to exclude that is to measure how well a model agrees with itself.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402
import pi_stats  # noqa: E402
from compare_internal_output import fit_ridge_block, grouped_split  # noqa: E402

MODELS = ("boltz2", "of3", "protenix")
EPS = 1e-9
K_SUB = 8


def cka(X, Y):
    """Linear CKA between two column-centred matrices with matching rows."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro"))
    return float(num / (den + EPS))


def rdm(X):
    """Variant x variant euclidean distance matrix."""
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    return d


def block_perm(pos, rng):
    """Permute whole residue positions among positions of equal variant count."""
    idx = np.arange(len(pos))
    out = idx.copy()
    blocks = [np.where(pos == p)[0] for p in np.unique(pos)]
    by_size = {}
    for b in blocks:
        by_size.setdefault(len(b), []).append(b)
    for sz, grp in by_size.items():
        order = rng.permutation(len(grp))
        for dst, src in enumerate(order):
            out[grp[dst]] = idx[grp[src]]
    return out


def mantel(Ra, Rb, pos, n_perm=2000, seed=0):
    iu = np.triu_indices(len(Ra), 1)
    obs = pi_stats.spearman(Ra[iu], Rb[iu])
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        p = block_perm(pos, rng)
        null.append(pi_stats.spearman(Ra[iu], Rb[np.ix_(p, p)][iu]))
    null = np.asarray([x for x in null if np.isfinite(x)])
    return obs, float((np.abs(null) >= abs(obs)).mean()), float(null.std())


def reg_cca(X, Y, k=K_SUB, lam=1e-1):
    """Canonical correlations of the top-k PCs of each side, ridge-regularised."""
    def top(M):
        M = M - M.mean(0)
        V = np.linalg.svd(M, full_matrices=False)[2][:k]
        P = M @ V.T
        return P / (P.std(0) + EPS)
    A, B = top(X), top(Y)
    Saa = A.T @ A / len(A) + lam * np.eye(k)
    Sbb = B.T @ B / len(B) + lam * np.eye(k)
    Sab = A.T @ B / len(A)
    ia = np.linalg.inv(np.linalg.cholesky(Saa))
    ib = np.linalg.inv(np.linalg.cholesky(Sbb))
    return np.clip(np.linalg.svd(ia @ Sab @ ib.T, compute_uv=False), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/runs/"
    ap.add_argument("--dir", default=R)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assays = sorted({Path(f).stem.split("_", 3)[3]
                     for f in glob.glob(a.dir + "xm_boltz2_r1_*.npz")})
    print(f"{len(assays)} assays x {len(MODELS)} models x 2 runs\n")

    D = {}
    for asy in assays:
        cell = {}
        for m in MODELS:
            for run in ("r1", "r2"):
                f = Path(a.dir) / f"xm_{m}_{run}_{asy}.npz"
                if not f.exists():
                    continue
                d = np.load(f, allow_pickle=True)
                cell[(m, run)] = {"z": np.asarray(d["dz_vec"], float)[:, -1, :],
                                  "y": np.asarray(d["score"], float),
                                  "pos": np.asarray(d["pos"]),
                                  "mut": [str(x) for x in d["mutant"]]}
        ref = cell[(MODELS[0], "r1")]["mut"]
        for k_, v in cell.items():
            if v["mut"] != ref:
                raise SystemExit(f"{asy} {k_}: variant lists differ across models; "
                                 f"the whole comparison assumes shared rows")
        D[asy.split("_")[0]] = cell
    names = sorted(D)

    res = {"assays": names}

    # ---- negative control: are the channel spaces related at all? ---------
    print("Negative control -- principal angles between channel subspaces\n")
    print("   Equal dimensionality does not imply a shared basis. These should")
    print(f"   sit at chance ({K_SUB}/128 = {K_SUB/128:.3f}).\n")
    ang = {}
    for asy in names:
        for i, ma in enumerate(MODELS):
            for mb in MODELS[i + 1:]:
                Xa = D[asy][(ma, "r1")]["z"]
                Xb = D[asy][(mb, "r1")]["z"]
                Va = np.linalg.svd(Xa - Xa.mean(0), full_matrices=False)[2][:K_SUB]
                Vb = np.linalg.svd(Xb - Xb.mean(0), full_matrices=False)[2][:K_SUB]
                s = np.linalg.svd(Va @ Vb.T, compute_uv=False)
                ang.setdefault(f"{ma}|{mb}", {})[asy] = float((s ** 2).mean())
    for k_, v in ang.items():
        pt, lo, hi, _ = pi_stats.cluster_bootstrap({n: [v[n]] for n in v},
                                                   n_boot=10000, seed=0,
                                                   hierarchical=False)
        print(f"   {k_:20s} {pt:.3f} [{lo:.3f}, {hi:.3f}]")
    res["principal_angles_negative_control"] = {"chance": K_SUB / 128, "pairs": ang}

    # ---- CKA and RSA, with the same-model repeat as the ceiling -----------
    print("\nRepresentational agreement (ceiling = the model against its own repeat)\n")
    print(f"   {'pair':22s} {'CKA':>8s} {'RSA rho':>9s} {'RSA p':>8s}")
    cka_r, rsa_r = {}, {}
    pairs = [(m, m, "r1", "r2") for m in MODELS] + \
            [(MODELS[i], MODELS[j], "r1", "r1")
             for i in range(len(MODELS)) for j in range(i + 1, len(MODELS))]
    for ma, mb, ra, rb in pairs:
        lab = f"{ma}|{mb}" + (" (repeat)" if ma == mb else "")
        cs, rs, ps = {}, {}, {}
        for asy in names:
            Xa, Xb = D[asy][(ma, ra)]["z"], D[asy][(mb, rb)]["z"]
            pos = D[asy][(ma, ra)]["pos"]
            cs[asy] = [cka(Xa, Xb)]
            o, p, _ = mantel(rdm(Xa), rdm(Xb), pos, n_perm=a.n_perm)
            rs[asy], ps[asy] = [o], p
        c = pi_stats.cluster_bootstrap(cs, n_boot=10000, seed=0, hierarchical=False)
        r = pi_stats.cluster_bootstrap(rs, n_boot=10000, seed=0, hierarchical=False)
        cka_r[lab] = {"mean": c[0], "ci_lo": c[1], "ci_hi": c[2]}
        rsa_r[lab] = {"mean": r[0], "ci_lo": r[1], "ci_hi": r[2],
                      "p_max": max(ps.values())}
        print(f"   {lab:22s} {c[0]:8.3f} {r[0]:9.3f} {max(ps.values()):8.3f}")
    res["cka"], res["rsa"] = cka_r, rsa_r

    # ---- how many shared directions? --------------------------------------
    print("\nRegularised CCA on the top-8 PCs (count above a permuted null)\n")
    cc = {}
    for i, ma in enumerate(MODELS):
        for mb in MODELS[i + 1:]:
            n_sig = {}
            for asy in names:
                Xa, Xb = D[asy][(ma, "r1")]["z"], D[asy][(mb, "r1")]["z"]
                pos = D[asy][(ma, "r1")]["pos"]
                obs = reg_cca(Xa, Xb)
                rng = np.random.default_rng(0)
                null = np.stack([reg_cca(Xa, Xb[block_perm(pos, rng)])
                                 for _ in range(50)])
                thr = np.percentile(null, 95, axis=0)
                n_sig[asy] = [int((obs > thr).sum())]
            pt = pi_stats.cluster_bootstrap(n_sig, n_boot=10000, seed=0,
                                            hierarchical=False)
            cc[f"{ma}|{mb}"] = {"mean_n_directions": pt[0], "ci_lo": pt[1],
                                "ci_hi": pt[2], "per_assay": {k: v[0] for k, v in n_sig.items()}}
            print(f"   {ma}|{mb:10s} {pt[0]:.2f} of 8 directions "
                  f"[{pt[1]:.2f}, {pt[2]:.2f}]")
    res["cca"] = cc

    # ---- do the models carry complementary phenotype information? ---------
    print("\nPredictive complementarity\n")
    single, comb, avg = {m: {} for m in MODELS}, {}, {}
    for asy in names:
        pos = D[asy][(MODELS[0], "r1")]["pos"]
        y = D[asy][(MODELS[0], "r1")]["y"]
        sv = {m: [] for m in MODELS}
        cv, av = [], []
        for s in range(a.seeds):
            rng = np.random.default_rng(s)
            tr, te = grouped_split(pos, a.frac, rng)
            preds = {}
            for m in MODELS:
                X = D[asy][(m, "r1")]["z"]
                rho, _, _, idx = fit_ridge_block(X, y, pos, tr, te,
                                                 np.random.default_rng(s))
                sv[m].append(rho)
                mu, sd = X[tr].mean(0), X[tr].std(0) + EPS
                Xs = (X - mu) / sd
                from compare_internal_output import ridge_fit, ridge_pred
                w = ridge_fit(Xs[tr][:, idx], (y[tr] - y[tr].mean()) / (y[tr].std() + EPS), 1.0)
                preds[m] = ridge_pred(w, Xs[te][:, idx])
            Xc = np.column_stack([D[asy][(m, "r1")]["z"] for m in MODELS])
            cv.append(fit_ridge_block(Xc, y, pos, tr, te, np.random.default_rng(s))[0])
            P = np.column_stack([(preds[m] - preds[m].mean()) / (preds[m].std() + EPS)
                                 for m in MODELS])
            av.append(pi_stats.spearman(P.mean(1), y[te]))
        for m in MODELS:
            single[m][asy] = [float(np.nanmean(sv[m]))]
        comb[asy] = [float(np.nanmean(cv))]
        avg[asy] = [float(np.nanmean(av))]
    out_s = {}
    for m in MODELS:
        pt = pi_stats.cluster_bootstrap(single[m], n_boot=10000, seed=0,
                                        hierarchical=False)
        out_s[m] = {"mean": pt[0], "ci_lo": pt[1], "ci_hi": pt[2]}
        print(f"   {m:22s} {pt[0]:+.3f} [{pt[1]:+.3f}, {pt[2]:+.3f}]")
    for lab, g in (("concatenated features", comb), ("averaged predictions", avg)):
        pt = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0, hierarchical=False)
        out_s[lab] = {"mean": pt[0], "ci_lo": pt[1], "ci_hi": pt[2]}
        print(f"   {lab:22s} {pt[0]:+.3f} [{pt[1]:+.3f}, {pt[2]:+.3f}]")
    best = {asy: [max(single[m][asy][0] for m in MODELS)] for asy in names}
    print()
    for lab, g in (("concatenated", comb), ("averaged", avg)):
        pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(
            g, best, n_boot=10000, seed=0, hierarchical=False)
        wins = sum(1 for asy in names if g[asy][0] > best[asy][0])
        out_s[f"{lab} minus best single"] = {"gap": pt, "ci_lo": lo, "ci_hi": hi,
                                             "wins": wins, "n": len(names)}
        flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
        print(f"   {lab} minus best single model   {pt:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]  {wins}/{len(names)}{flag}")
    res["complementarity"] = out_s

    pi_archive.write_result(a.out, res, protocol=pi_protocol.protocol(
        script="analyze_xmodel.py",
        design="within-assay, position-grouped splits; each model compared "
               "against itself across two independent capture runs before any "
               "cross-model claim, so replicate noise is not read as agreement",
        layer=pi_protocol.layers("final"),
        features=pi_protocol.features("per-model pair representation", 128),
        source=a.dir, n_assays=len(assays), seeds=a.seeds, frac=a.frac,
        n_perm=a.n_perm,
        note="layer counts, distogram grids and alignment handling differ "
             "between the three trunks, so this is not a ranking"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
