"""Read out the trunk-structure consistency experiment. Login node -- no jax.

Reads runs/cons_*.npz. The headline quantity is

    c_own = mean_ij log P_mut(d_ij | i,j)

the mutant's own distogram evaluated at the distances the decoder actually
produced. Low = the decoded structure does not satisfy what the trunk believes.

**The trap this script exists to avoid.** A destabilising mutation broadens the
distogram, and a broader distribution assigns lower log-probability to EVERY
structure. So `c_own` would correlate with dG through entropy alone, with no
disagreement involved -- the same shape of error as the norm-matched control in
exp_amplify, where the quantity held fixed turned out to be the signal.

Three entropy-aware readings, reported side by side:

    c_own            raw; confounded by entropy, shown only for reference
    c_own + H        "surprise excess" -- how much worse the observed structure
                     is than a typical draw FROM that same distogram. A broad
                     distogram raises H and lowers c_own by the same amount, so
                     this is first-order entropy-free.
    c_own - c_wtdist the same structure scored under the mutant's distogram
                     versus under the wild type's. Differences in structure
                     quality cancel; what remains is which distogram the
                     structure fits better.
    partial rho      rho(c_own, dG) with entropy partialled out, computed on
                     ranks.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402


def partial_spearman(x, y, *covars):
    """Spearman of x and y with covariates removed, on ranks (Fisher's method)."""
    R = lambda v: rankdata(v).astype(float)
    X, Y = R(x), R(y)
    if covars:
        C = np.column_stack([R(c) for c in covars])
        C = np.column_stack([np.ones(len(X)), C])
        X = X - C @ np.linalg.lstsq(C, X, rcond=None)[0]
        Y = Y - C @ np.linalg.lstsq(C, Y, rcond=None)[0]
    return float(np.corrcoef(X, Y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/cons_*.npz")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(args.glob)):
        d = np.load(f)
        nm = str(d["assay"]).split("_Tsu")[0]
        sc = d["score"]
        co = d["c_own"].mean(1)          # mean over the K samples
        cw = d["c_wtdist"].mean(1)
        H = d["entropy"]
        pl = d["plddt"].mean(1)
        ca, caw = d["ca"], d["ca_wt"]

        # reranking: best-of-K by consistency vs the mean over K
        best = d["c_own"].argmax(1)
        tm_all = np.array([[geom.tm_score(ca[i, k].astype(float), caw[0].astype(float))
                            for k in range(ca.shape[1])] for i in range(len(sc))])
        tm_best = tm_all[np.arange(len(sc)), best]
        tm_mean = tm_all.mean(1)

        r = dict(
            assay=nm, n=len(sc),
            rho_raw=spearmanr(co, sc).correlation,
            rho_surprise=spearmanr(co + H, sc).correlation,
            rho_diff=spearmanr(co - cw, sc).correlation,
            rho_partial=partial_spearman(co, sc, H),
            rho_entropy=spearmanr(H, sc).correlation,
            rho_tm_mean=spearmanr(tm_mean, sc).correlation,
            rho_tm_best=spearmanr(tm_best, sc).correlation,
            rho_plddt=spearmanr(pl, sc).correlation,
            wt_c=float(d["c_wt"].mean()), mut_c=float(co.mean()),
        )
        rows.append(r)

    print(f"{'assay':12s}{'n':>4s}{'c_own raw':>11s}{'+entropy':>10s}"
          f"{'own-vs-WT':>11s}{'partial':>9s}{'|':>2s}{'rho(H,dG)':>11s}"
          f"{'TM mean':>9s}{'TM best-K':>11s}{'pLDDT':>8s}")
    for r in rows:
        print(f"{r['assay']:12s}{r['n']:>4d}{r['rho_raw']:>+11.3f}"
              f"{r['rho_surprise']:>+10.3f}{r['rho_diff']:>+11.3f}"
              f"{r['rho_partial']:>+9.3f}{'|':>2s}{r['rho_entropy']:>+11.3f}"
              f"{r['rho_tm_mean']:>+9.3f}{r['rho_tm_best']:>+11.3f}"
              f"{r['rho_plddt']:>+8.3f}")
    m = lambda k: np.mean([r[k] for r in rows])
    print(f"{'MEAN':12s}{'':>4s}{m('rho_raw'):>+11.3f}{m('rho_surprise'):>+10.3f}"
          f"{m('rho_diff'):>+11.3f}{m('rho_partial'):>+9.3f}{'|':>2s}"
          f"{m('rho_entropy'):>+11.3f}{m('rho_tm_mean'):>+9.3f}"
          f"{m('rho_tm_best'):>+11.3f}{m('rho_plddt'):>+8.3f}")

    print(f"\n  reranking benefit (best-of-K minus mean): "
          f"{m('rho_tm_best') - m('rho_tm_mean'):+.3f}")
    print(f"  wild-type consistency {np.mean([r['wt_c'] for r in rows]):.4f}  "
          f"vs mutants {np.mean([r['mut_c'] for r in rows]):.4f}")
    print("\n  references: TM-to-WT 0.214 | pLDDT-at-site 0.037 | "
          "||dz|| off the trunk 0.637 | Pairformer probe 0.548 (held-out)")
    print("  NOTE: c_own raw is entropy-confounded. Judge on '+entropy',")
    print("        'own-vs-WT' and 'partial', which are not.")


if __name__ == "__main__":
    main()
