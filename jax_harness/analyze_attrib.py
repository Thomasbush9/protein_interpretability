"""What about the mutant residue drives its PC2 score?

PC2 is a projection of dz = z_mut - z_wt, so its value could in principle be
determined by any of three things: which residue was REMOVED, which residue was
INTRODUCED, or the structural context of the site. Distinguishing them says
whether the direction encodes a substitution lookup or something about the
protein.

This is a stronger form of the chemistry control. The seventeen chemistry
descriptors are a hand-built summary of a substitution -- BLOSUM, volume,
hydropathy, charge. Amino-acid IDENTITY is the non-parametric version: twenty
indicators for the removed residue and twenty for the introduced one, which can
express any function of the substitution whatsoever, including everything the
descriptors capture and everything they miss. If PC2's association with DMS
survives removing that, it cannot be a substitution lookup in any form.

Four questions, in order:

  profile     Mean PC2 by introduced residue, and by removed residue. Purely
              descriptive, but it is what "the features of the mutant" means
              most directly, and the two profiles turn out not to be mirror
              images of each other.

  attribution How much of PC2's variance each block explains, under
              position-grouped cross-validation so a block cannot win by
              memorising sites: removed identity, introduced identity, both,
              and site identity on its own.

  residual    PC2 against DMS after residualising on amino-acid identity. This
              is the question. Chemistry left 98% of the association standing;
              identity is the harder test.

  context     PC2 against DMS after residualising on the SITE instead. A score
              that survives losing the substitution but not the site is a
              statement about where the mutation is, not what it is.
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
import pi_basis  # noqa: E402
import pi_stats  # noqa: E402
from compare_internal_output import (grouped_split, ridge_fit,  # noqa: E402
                                     ridge_pred)

AA = "ACDEFGHIKLMNPQRSTVWY"
EPS = 1e-9
N_PC = 4


def zc(M):
    return (M - M.mean(0)) / (M.std(0) + EPS)


def onehot(letters):
    M = np.zeros((len(letters), len(AA)))
    for i, c in enumerate(letters):
        j = AA.find(c)
        if j >= 0:
            M[i, j] = 1.0
    return M


def onehot_site(pos):
    u = np.unique(pos)
    M = np.zeros((len(pos), len(u)))
    for i, p in enumerate(pos):
        M[i, int(np.where(u == p)[0][0])] = 1.0
    return M


def resid(x, Z):
    """Residual of x after least-squares removal of the columns of Z."""
    Z = np.column_stack([Z, np.ones(len(Z))])
    return x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/runs/"
    ap.add_argument("--glob", default=R + "gym2s_*.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = {}
    for f in sorted(glob.glob(a.glob)):
        d = np.load(f, allow_pickle=True)
        muts = [str(m) for m in d["mutant"]]
        wt_aa, mu_aa = zip(*[(pi_chem.parse(m)[0], pi_chem.parse(m)[2])
                             for m in muts])
        A[Path(f).stem.split("_", 1)[1].split("_")[0]] = {
            "X": np.asarray(d["dz_site"], float)[:, -1, :],
            "y": np.asarray(d["score"], float),
            "pos": np.asarray(d["pos"]),
            "kl": np.asarray(d["kl_glob"], float)[:, -1],
            "wt": [c or "X" for c in wt_aa], "mu": [c or "X" for c in mu_aa]}
    names = sorted(A)
    print(f"{len(names)} assays\n")

    # shared, oriented basis. The comment this replaces said "identical
    # construction to the report", which was the only thing keeping the two in
    # agreement -- and was not true of analyze_chem's LOAO block. It is now
    # the same call, so agreement is a fact rather than a claim.
    B = pi_basis.fit({n: A[n]["X"] for n in names}, layer=-1,
                     orient_on="kl_glob",
                     orient_ref={n: A[n]["kl"] for n in names},
                     orient_k=N_PC, n_pc=N_PC, eps=EPS)
    for n in names:
        P = B.project(A[n]["X"], layer=-1)
        A[n]["p2"] = (P[:, 1] - P[:, 1].mean()) / (P[:, 1].std() + EPS)

    res = {"assays": names, "protocol": dict(B.protocol)}

    # ---------------------------------------------------------------- 1
    print("1. Mean PC2 by residue (z-scored within assay, pooled)\n")
    prof = {}
    for side, key in (("introduced", "mu"), ("removed", "wt")):
        rows = {}
        for c in AA:
            g = {n: [float(np.mean(A[n]["p2"][np.array(A[n][key]) == c]))]
                 for n in names if (np.array(A[n][key]) == c).sum() >= 3}
            if len(g) >= 6:
                pt, lo, hi, _ = pi_stats.cluster_bootstrap(
                    g, n_boot=4000, seed=0, hierarchical=False)
                rows[c] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                           "n_assays": len(g)}
        prof[side] = rows
        top = sorted(rows.items(), key=lambda kv: -kv[1]["mean"])
        print(f"   {side:11s} highest: " +
              ", ".join(f"{c} {v['mean']:+.2f}" for c, v in top[:5]))
        print(f"   {'':11s} lowest : " +
              ", ".join(f"{c} {v['mean']:+.2f}" for c, v in top[-5:]))
    res["profile"] = prof

    # ---------------------------------------------------------------- 2
    print("\n2. How much of PC2 is explained by each block "
          "(position-grouped CV)\n")
    BLOCKS = {
        "removed residue identity (20)": lambda r: onehot(r["wt"]),
        "introduced residue identity (20)": lambda r: onehot(r["mu"]),
        "both identities (40)": lambda r: np.column_stack(
            [onehot(r["wt"]), onehot(r["mu"])]),
        "chemistry descriptors (17)": lambda r: pi_chem.chem_matrix(
            [f"{w}1{m}" for w, m in zip(r["wt"], r["mu"])]),
        "site identity (one-hot)": lambda r: onehot_site(r["pos"]),
    }
    expl = {}
    for bn, mk in BLOCKS.items():
        g = {}
        for n in names:
            r, vals = A[n], []
            X = mk(r)
            for s in range(a.seeds):
                rng = np.random.default_rng(s)
                tr, te = grouped_split(r["pos"], a.frac, rng)
                # a site block cannot generalise across held-out sites by
                # construction; reported anyway so that fact is visible
                w = ridge_fit(X[tr], r["p2"][tr], 1.0)
                vals.append(pi_stats.spearman(ridge_pred(w, X[te]), r["p2"][te]))
            g[n] = [float(np.nanmean(vals))]
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                   hierarchical=False)
        expl[bn] = {"mean": pt, "ci_lo": lo, "ci_hi": hi}
        print(f"   {bn:34s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    res["explains_pc2"] = expl

    # ---------------------------------------------------------------- 3, 4
    print("\n3. Does PC2 still track DMS once the substitution is removed?\n")
    tests = {
        "raw": lambda r: r["p2"],
        "minus introduced identity": lambda r: resid(r["p2"], onehot(r["mu"])),
        "minus both identities": lambda r: resid(
            r["p2"], np.column_stack([onehot(r["wt"]), onehot(r["mu"])])),
        "minus chemistry": lambda r: resid(r["p2"], pi_chem.chem_matrix(
            [f"{w}1{m}" for w, m in zip(r["wt"], r["mu"])])),
        "minus site identity": lambda r: resid(r["p2"], onehot_site(r["pos"])),
        "minus both identities AND site": lambda r: resid(
            r["p2"], np.column_stack([onehot(r["wt"]), onehot(r["mu"]),
                                      onehot_site(r["pos"])])),
    }
    out = {}
    for lab, fn in tests.items():
        g = {n: [pi_stats.spearman(fn(A[n]), A[n]["y"])] for n in names}
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                   hierarchical=False)
        out[lab] = {"mean": pt, "ci_lo": lo, "ci_hi": hi}
        keep = 100 * abs(pt / out["raw"]["mean"]) if out.get("raw") else 100
        print(f"   {lab:32s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"   {keep:5.0f}% of raw")
    res["pc2_vs_dms_residualised"] = out

    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
