"""Is the mutation subspace anything more than substitution chemistry?

This is the control the publication audit already named as deciding, and the
SVD results made it sharper rather than softer. PC1 correlates with volume
change at -0.80. PC2, the stability-and-certainty axis, still carries -0.53 with
volume. So the obvious deflationary reading of the whole study is available: the
Pairformer has learned what amino acid was substituted, chemistry predicts
stability, and everything else is decoration.

Three ways of asking it, all under the project's canonical estimator
(`fit_ridge_block`: standardise on train, select top-k features by training
|rho|, tune k and lambda on an inner grouped split) so the numbers sit on the
same scale as the published ones.

  incremental      Does PC2 add anything ON TOP of chemistry, and does
                   chemistry add anything on top of PC2? Both directions
                   matter: the first asks whether the model knows something
                   chemistry does not, the second whether it has thrown
                   chemistry away. Paired per assay.

  residual SVD     Regress every one of the 128 pair channels on the 17
                   chemistry descriptors -- fitted on TRAINING rows only -- and
                   decompose what is left. If a predictive direction survives,
                   it cannot be a re-encoding of the substitution. This is the
                   strong form of the question, and unlike a partial
                   correlation it also says what the surviving direction looks
                   like.

  PC2 alone        One number per variant, from a basis learned on eleven other
                   proteins, tested on the twelfth. No ridge, no feature
                   selection, no per-assay fitting -- the sign is taken from the
                   training assays and the score is correlated with DMS
                   directly. It is the least flattering way to present the
                   result and the easiest to state.

The chemistry block is `pi_chem.chem_matrix`: BLOSUM62, hydropathy, volume and
charge contrasts in signed and absolute form, plus proline/glycine indicators.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_chem  # noqa: E402
import pi_stats  # noqa: E402
from compare_internal_output import fit_ridge_block, grouped_split  # noqa: E402

EPS = 1e-9
N_PC = 4


def zc(M):
    return (M - M.mean(0)) / (M.std(0) + EPS)


def residualise(X, C, tr):
    """Remove the chemistry-predictable part of every channel of X.

    The regression is fitted on TRAINING rows only and then applied to all
    rows. Fitting it on everything would let the held-out positions influence
    what counts as "chemistry", which is the same leak the frozen SVD basis
    exists to avoid.
    """
    A = np.column_stack([C[tr], np.ones(tr.sum())])
    W = np.linalg.lstsq(A, X[tr], rcond=None)[0]
    return X - np.column_stack([C, np.ones(len(C))]) @ W


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/"
    ap.add_argument("--glob", default=R + "runs/gym2s_*.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = {}
    for f in sorted(glob.glob(a.glob)):
        d = np.load(f, allow_pickle=True)
        n = Path(f).stem.split("_")[1]
        A[n] = {"X": np.asarray(d["dz_site"], float)[:, -1, :],
                "C": pi_chem.chem_matrix([str(m) for m in d["mutant"]]),
                "y": np.asarray(d["score"], float),
                "pos": np.asarray(d["pos"]),
                "kl": np.asarray(d["kl_glob"], float)[:, -1]}
    names = sorted(A)
    print(f"{len(names)} assays\n")

    # ---- shared basis, oriented, exactly as the report defines it ---------
    Xg = np.concatenate([zc(A[n]["X"]) for n in names], 0)
    gm = Xg.mean(0)
    V = np.linalg.svd(Xg - gm, full_matrices=False)[2][:N_PC]
    for c in range(N_PC):
        g = {n: [pi_stats.spearman((zc(A[n]["X"]) - gm) @ V[c], A[n]["kl"])]
             for n in names}
        if pi_stats.cluster_bootstrap(g, n_boot=2000, seed=0,
                                      hierarchical=False)[0] < 0:
            V[c] = -V[c]
    for n in names:
        A[n]["P"] = (zc(A[n]["X"]) - gm) @ V.T          # (n, N_PC)

    # ======================================================================
    # 1. incremental value, both directions
    # ======================================================================
    BLOCKS = {
        "chemistry (17)": lambda r, tr: r["C"],
        "PC2 alone (1)": lambda r, tr: r["P"][:, 1:2],
        "PC1-4 (4)": lambda r, tr: r["P"],
        "chemistry + PC2 (18)": lambda r, tr: np.column_stack([r["C"], r["P"][:, 1]]),
        "chemistry + PC1-4 (21)": lambda r, tr: np.column_stack([r["C"], r["P"]]),
        "full dz (128)": lambda r, tr: r["X"],
        "chemistry + full dz (145)": lambda r, tr: np.column_stack([r["C"], r["X"]]),
        "dz residualised on chemistry (128)":
            lambda r, tr: residualise(r["X"], r["C"], tr),
    }
    res = {k: {} for k in BLOCKS}
    for n in names:
        r = A[n]
        for bn, mk in BLOCKS.items():
            vals = []
            for s in range(a.seeds):
                rng = np.random.default_rng(s)
                tr, te = grouped_split(r["pos"], a.frac, rng)
                X = mk(r, tr)
                vals.append(fit_ridge_block(X, r["y"], r["pos"], tr, te, rng)[0])
            res[bn][n] = float(np.nanmean(vals))
    print("Held-out Spearman, position-grouped, canonical estimator\n")
    summary = {}
    for bn in BLOCKS:
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(
            {n: [res[bn][n]] for n in names}, n_boot=10000, seed=0,
            hierarchical=False)
        summary[bn] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                       "per_assay": res[bn]}
        print(f"   {bn:36s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\nPaired increments\n")
    gaps = {}
    for lab, ka, kb in (
            ("PC2 adds to chemistry", "chemistry + PC2 (18)", "chemistry (17)"),
            ("PC1-4 add to chemistry", "chemistry + PC1-4 (21)", "chemistry (17)"),
            ("chemistry adds to PC2", "chemistry + PC2 (18)", "PC2 alone (1)"),
            ("full dz adds to chemistry", "chemistry + full dz (145)", "chemistry (17)"),
            ("full dz beats chemistry", "full dz (128)", "chemistry (17)"),
            ("chemistry-residual dz beats chemistry",
             "dz residualised on chemistry (128)", "chemistry (17)")):
        pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(
            {n: [res[ka][n]] for n in names}, {n: [res[kb][n]] for n in names},
            n_boot=10000, seed=0, hierarchical=False)
        wins = sum(1 for n in names if res[ka][n] > res[kb][n])
        gaps[lab] = {"gap": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins,
                     "n": len(names)}
        flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
        print(f"   {lab:40s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"{wins}/{len(names)}{flag}")

    # ======================================================================
    # 2. what does the chemistry-residual subspace look like?
    # ======================================================================
    print("\nChemistry-residual subspace\n")
    Rg = []
    for n in names:
        tr = np.ones(len(A[n]["y"]), bool)
        Rg.append(zc(residualise(A[n]["X"], A[n]["C"], tr)))
    Rg = np.concatenate(Rg, 0)
    Vr = np.linalg.svd(Rg - Rg.mean(0), full_matrices=False)[2][:N_PC]
    ang = np.linalg.svd(V @ Vr.T, compute_uv=False)
    print(f"   principal angles between the original and residual top-{N_PC} "
          f"subspaces:")
    print(f"      mean cos^2 {float((ang ** 2).mean()):.3f}   "
          f"(chance {N_PC / V.shape[1]:.3f})")
    off = 0
    ann = {}
    for n in names:
        m = len(A[n]["y"])
        Pr = (zc(residualise(A[n]["X"], A[n]["C"], np.ones(m, bool)))
              - Rg.mean(0)) @ Vr.T
        ann[n] = [pi_stats.spearman(Pr[:, c], A[n]["y"]) for c in range(N_PC)]
        off += m
    print(f"\n   residual component vs DMS (assay-level):")
    for c in range(N_PC):
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(
            {n: [ann[n][c]] for n in names}, n_boot=10000, seed=0,
            hierarchical=False)
        flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
        print(f"      rPC{c+1}  {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]{flag}")

    # ======================================================================
    # 3. PC2 alone, transferred, no fitting on the held-out assay
    # ======================================================================
    print("\nPC2 alone, leave-one-assay-out (sign from the training assays)\n")
    loao = {}
    for h in names:
        tr_n = [n for n in names if n != h]
        Xt = np.concatenate([zc(A[n]["X"]) for n in tr_n], 0)
        gmt = Xt.mean(0)
        Vt = np.linalg.svd(Xt - gmt, full_matrices=False)[2][:N_PC]
        # orient on the TRAINING assays only, against DMS, then apply unchanged
        sgn = np.sign(np.mean([
            pi_stats.spearman((zc(A[n]["X"]) - gmt) @ Vt[1], A[n]["y"])
            for n in tr_n]))
        sc = ((zc(A[h]["X"]) - gmt) @ Vt[1]) * sgn
        loao[h] = pi_stats.spearman(sc, A[h]["y"])
        print(f"   {h:8s} {loao[h]:+.3f}")
    pt, lo, hi, _ = pi_stats.cluster_bootstrap(
        {n: [loao[n]] for n in names}, n_boot=10000, seed=0, hierarchical=False)
    print(f"\n   pooled {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    Path(a.out).write_text(json.dumps(
        {"blocks": summary, "increments": gaps,
         "residual_subspace": {
             "cos2_vs_original": float((ang ** 2).mean()),
             "chance": N_PC / V.shape[1],
             "component_vs_dms": {f"rPC{c+1}": {
                 "per_assay": {n: ann[n][c] for n in names}} for c in range(N_PC)}},
         "pc2_alone_loao": {"per_assay": loao, "mean": pt, "ci_lo": lo,
                            "ci_hi": hi}}, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
