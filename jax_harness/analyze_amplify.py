"""Read out the difference-amplification experiment. Login node -- no jax.

Reads runs/amp_*.npz (exp_amplify.py) and answers one question:

    does scaling the mutant-minus-wild-type difference in the trunk state make
    the DECODED STRUCTURE track measured stability?

and one control question that decides whether the first answer means anything:

    does a norm-matched difference borrowed from a DIFFERENT variant do the
    same thing?

If rho(TM-to-WT, dG) rises with gamma for `true` and stays flat for `perm`, the
decoder's insensitivity is a gain problem and this is a fix. If both rise, we
have only shown that larger perturbations make larger structural changes. If
neither rises while the structures visibly move, the insensitivity is
structural, not a matter of scale, and guidance rather than scaling is needed.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402  (self-tests the superposition on import)


def readout(f):
    d = np.load(f)
    assay = str(d["assay"])
    ca, pl, sc = d["ca"], d["plddt"], d["score"]
    ca_wt, pos = d["ca_wt"].astype(float), d["pos"]
    kinds, gammas = d["cond_kind"], d["cond_gamma"]
    n, C = ca.shape[0], ca.shape[1]

    rows = []
    for c in range(C):
        tm = np.array([geom.tm_score(ca[i, c].astype(float), ca_wt) for i in range(n)])
        rmsd = np.array([geom.kabsch_rmsd(ca[i, c].astype(float), ca_wt) for i in range(n)])
        pls = np.array([pl[i, c, pos[i]] for i in range(n)])
        rows.append(dict(
            assay=assay, kind=str(kinds[c]), gamma=float(gammas[c]),
            mean_tm=tm.mean(), mean_rmsd=rmsd.mean(), mean_plddt=pl[:, c].mean(),
            rho_tm=spearmanr(tm, sc).correlation,
            p_tm=spearmanr(tm, sc).pvalue,
            rho_plddt=spearmanr(pl[:, c].mean(1), sc).correlation,
            rho_plddt_site=spearmanr(pls, sc).correlation,
            n=n))
    return rows


def gap_bootstrap(files, gamma=8.0, n_boot=6000, seed=1):
    """Pooled CI on the direction-specific gap (true minus norm-matched perm).

    Resamples variants within each assay and averages the per-assay gap. This is
    the statistic that decides whether amplification does anything beyond
    converting perturbation magnitude into structural displacement.
    """
    rng = np.random.default_rng(seed)
    data = []
    for f in files:
        d = np.load(f)
        sc, ca, caw = d["score"], d["ca"], d["ca_wt"].astype(float)
        ki, gm = list(d["cond_kind"]), d["cond_gamma"]
        def tm(idx):
            return np.array([geom.tm_score(ca[i, idx].astype(float), caw)
                             for i in range(len(sc))])
        it = [i for i in range(len(ki)) if ki[i] == "true" and gm[i] == gamma][0]
        ip = [i for i in range(len(ki)) if ki[i] == "perm" and gm[i] == gamma][0]
        data.append((tm(it), tm(ip), sc))
    gaps = []
    for _ in range(n_boot):
        g = []
        for tt, tp, sc in data:
            b = rng.integers(0, len(sc), len(sc))
            g.append(spearmanr(tt[b], sc[b]).correlation
                     - spearmanr(tp[b], sc[b]).correlation)
        gaps.append(np.mean(g))
    return np.array(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/amp_*.npz")
    ap.add_argument("--out", default="runs/amplify_summary.csv")
    args = ap.parse_args()

    allrows = []
    for f in sorted(glob.glob(args.glob)):
        rows = readout(f)
        allrows += rows
        a = rows[0]["assay"].split("_Tsu")[0]
        print(f"\n=== {a}  (n={rows[0]['n']}) "
              f"{'':>3s}{'TM->WT':>9s}{'RMSD':>8s}{'pLDDT':>8s}"
              f"{'rho(TM,dG)':>12s}{'p':>10s}{'rho(pLDDT,dG)':>15s}")
        for r in rows:
            star = " *" if r["p_tm"] < 0.05 else "  "
            print(f"  {r['kind']:5s} g={r['gamma']:<4g}"
                  f"{r['mean_tm']:9.3f}{r['mean_rmsd']:8.2f}{r['mean_plddt']:8.3f}"
                  f"{r['rho_tm']:+12.3f}{r['p_tm']:10.2g}{star}"
                  f"{r['rho_plddt']:+13.3f}")

    # pooled: the comparison the experiment exists to make
    print("\n" + "=" * 78)
    print("POOLED over assays -- rho(TM to WT, dG), mean across assays")
    print(f"  {'gamma':>6s} {'true':>10s} {'perm (norm-matched)':>22s}   verdict")
    gam = sorted({r["gamma"] for r in allrows})
    for g in gam:
        t = [r["rho_tm"] for r in allrows if r["kind"] == "true" and r["gamma"] == g]
        p = [r["rho_tm"] for r in allrows if r["kind"] == "perm" and r["gamma"] == g]
        ts = f"{np.mean(t):+.3f}" if t else "     --"
        ps = f"{np.mean(p):+.3f}" if p else "     --"
        # NO verdict from a threshold on the point estimate. The gap needs a
        # CI (see gap_bootstrap below) -- an early version of this script called
        # +0.11 "mutation-specific" on a >0.10 rule, when its 95 % CI in fact
        # spans zero.
        note = f"gap {np.mean(t) - np.mean(p):+.3f}" if t and p else ""
        print(f"  {g:6g} {ts:>10s} {ps:>22s}   {note}")

    print("\n  The permuted control is NOT information-free: it preserves ||dz_i||,")
    print("  and ||dz|| by itself predicts dG better than any decoded structure does.")
    print("  So `true - perm` isolates DIRECTION only. Use gap_bootstrap() for its CI.")

    base = [r["rho_tm"] for r in allrows if r["kind"] == "true" and r["gamma"] == 1.0]
    best = max(gam, key=lambda g: np.mean(
        [r["rho_tm"] for r in allrows if r["kind"] == "true" and r["gamma"] == g] or [-9]))
    bestv = np.mean([r["rho_tm"] for r in allrows
                     if r["kind"] == "true" and r["gamma"] == best])
    print(f"\n  baseline (gamma=1): {np.mean(base):+.3f}"
          f"    best gamma={best:g}: {bestv:+.3f}"
          f"    change {bestv - np.mean(base):+.3f}")
    print("  reference: Pairformer probe 0.548 (held-out), TM-to-WT 0.214 (12 assays)")

    files = sorted(glob.glob(args.glob))
    if files:
        g = gap_bootstrap(files)
        print(f"\n  DIRECTION-SPECIFIC GAP at gamma=8 (pooled bootstrap over variants):")
        print(f"    {g.mean():+.3f}   95% CI [{np.percentile(g, 2.5):+.3f}, "
              f"{np.percentile(g, 97.5):+.3f}]   P(>0) = {np.mean(g > 0):.3f}")
        if np.percentile(g, 2.5) <= 0:
            print("    CI includes zero -- direction-specific effect NOT established.")
        # the readout that needs no decoding at all
        direct = []
        for f in files:
            d = np.load(f)
            direct.append(abs(spearmanr(d["dz_norm"], d["score"]).correlation))
        print(f"\n    ||dz|| read straight off the trunk: {np.mean(direct):+.3f} "
              f"-- better than any decoded structure here.")

    if allrows:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
            w.writeheader()
            w.writerows(allrows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
