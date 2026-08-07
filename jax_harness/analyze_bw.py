"""Is the internal advantage about mutations, or only about positions?

The attribution analysis found that removing site identity destroyed more of
PC2's association with DMS than removing substitution identity did. Position-
grouped splits stop a probe memorising which sites are fragile, but they do not
stop it learning what makes a site fragile in general -- and "this model knows
which positions are sensitive" is a much weaker claim than "this model knows
what a given mutation does".

Held-out performance is therefore split into the two things it can be made of:

  between-position   Aggregate predictions and measurements to position means
                     and correlate those. Can the probe rank held-out SITES?
  within-position    Subtract each position's mean from both sides and pool the
                     residuals. Given a site, can the probe rank the handful of
                     substitutions AT it?

Both are computed on the same held-out rows from the same fit, so the split is
a decomposition of one number rather than two separate experiments.

The comparison that matters is internal against output WITHIN each component,
never between them. Within-position correlations are lower for every method and
necessarily so: only a few variants share a site, the spread of DMS within a
site is small, and measurement noise is a larger share of it. Reading "within is
lower than between" as a finding would be reading the noise floor.

If internal beats output only between positions, the claim narrows to site
sensitivity. If it also beats it within, the claim survives as stated.
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
from compare_internal_output import (grouped_split, output_matrix,  # noqa: E402
                                     ridge_fit, ridge_pred, select_k)

EPS = 1e-9


def predict(X, y, pos, tr, te, rng, ks=(8, 16, 32, 64),
            lams=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Held-out predictions under the project's canonical fitting recipe."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + EPS
    Xs = (X - mu) / sd
    ym, ys = y[tr].mean(), y[tr].std() + EPS
    yz = (y - ym) / ys
    itr, iva = grouped_split(pos[tr], 0.25, rng)
    best = (-np.inf, min(ks[0], X.shape[1]), lams[0])
    if iva.sum() >= 4 and itr.sum() >= 10:
        for k in ks:
            k = min(k, X.shape[1])
            idx = select_k(Xs[tr][itr], yz[tr][itr], k)
            for lam in lams:
                w = ridge_fit(Xs[tr][itr][:, idx], yz[tr][itr], lam)
                r = pi_stats.spearman(ridge_pred(w, Xs[tr][iva][:, idx]),
                                      yz[tr][iva])
                if np.isfinite(r) and abs(r) > best[0]:
                    best = (abs(r), k, lam)
    _, k, lam = best
    idx = select_k(Xs[tr], yz[tr], k)
    w = ridge_fit(Xs[tr][:, idx], yz[tr], lam)
    return ridge_pred(w, Xs[te][:, idx])


def decompose(pred, y, pos):
    """Split a set of held-out rows into between- and within-position parts."""
    up = np.unique(pos)
    pm_p = np.array([pred[pos == p].mean() for p in up])
    pm_y = np.array([y[pos == p].mean() for p in up])
    between = pi_stats.spearman(pm_p, pm_y) if len(up) >= 4 else np.nan
    # residuals, keeping only sites that actually hold more than one variant
    keep = np.array([(pos == p).sum() >= 2 for p in pos])
    rp, ry = pred.copy().astype(float), y.copy().astype(float)
    for p in up:
        m = pos == p
        if m.sum() >= 2:
            rp[m] -= rp[m].mean()
            ry[m] -= ry[m].mean()
    within = (pi_stats.spearman(rp[keep], ry[keep]) if keep.sum() >= 8
              else np.nan)
    return between, within, float(keep.mean()), int(len(up))


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/"
    ap.add_argument("--glob", default=R + "runs/gym3_*.npz")
    ap.add_argument("--tm-cache", default=R + "runs/tm_cache.npz")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    TM = np.load(a.tm_cache)
    A = {}
    for f in sorted(glob.glob(a.glob)):
        stem = Path(f).stem.split("_", 1)[1]
        d = np.load(f, allow_pickle=True)
        y, pos = np.asarray(d["score"], float), np.asarray(d["pos"])
        rich = output_matrix(d["ca"], np.asarray(d["ca_wt"], float),
                             np.asarray(TM[stem], float), d["plddt"],
                             d["plddt_site"], pos)
        dpl = (np.asarray(d["plddt_res"], float)
               - np.asarray(d["plddt_res_wt"], float))
        A[stem.split("_")[0]] = {
            "y": y, "pos": pos,
            "blocks": {
                "internal dz (128, one layer)":
                    np.asarray(d["dz_site"], float)[:, -1, :],
                "output pLDDT + rich": np.column_stack([dpl, rich]),
                "output rich (published, 10)": rich,
                "substitution chemistry (17)":
                    pi_chem.chem_matrix([str(m) for m in d["mutant"]]),
            }}
    names = sorted(A)
    BN = list(A[names[0]]["blocks"])
    print(f"{len(names)} assays\n")

    bet = {b: {} for b in BN}
    wit = {b: {} for b in BN}
    cov = []
    for n in names:
        r = A[n]
        for b in BN:
            vb, vw = [], []
            for s in range(a.seeds):
                rng = np.random.default_rng(s)
                tr, te = grouped_split(r["pos"], a.frac, rng)
                p = predict(r["blocks"][b], r["y"], r["pos"], tr, te, rng)
                bb, ww, frac, npos = decompose(p, r["y"][te], r["pos"][te])
                vb.append(bb); vw.append(ww)
                if b == BN[0]:
                    cov.append(frac)
            bet[b][n] = [float(np.nanmean(vb))]
            wit[b][n] = [float(np.nanmean(vw))]
    print(f"held-out rows usable for the within-position part: "
          f"{100*np.mean(cov):.1f}% (a site needs >=2 variants)\n")

    print(f"{'block':32s} {'between positions':>19s} {'within positions':>19s}")
    out = {"blocks": {}}
    for b in BN:
        pb = pi_stats.cluster_bootstrap(bet[b], n_boot=10000, seed=0,
                                        hierarchical=False)
        pw = pi_stats.cluster_bootstrap(wit[b], n_boot=10000, seed=0,
                                        hierarchical=False)
        out["blocks"][b] = {
            "between": {"mean": pb[0], "ci_lo": pb[1], "ci_hi": pb[2]},
            "within": {"mean": pw[0], "ci_lo": pw[1], "ci_hi": pw[2]}}
        print(f"{b:32s} {pb[0]:+8.3f} [{pb[1]:+.3f},{pb[2]:+.3f}] "
              f"{pw[0]:+8.3f} [{pw[1]:+.3f},{pw[2]:+.3f}]")

    print("\nPaired: internal minus each output block, per assay\n")
    ref = BN[0]
    gaps = {}
    for b in BN[1:]:
        row = {}
        for lab, src in (("between", bet), ("within", wit)):
            pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(
                {n: src[ref][n] for n in names}, {n: src[b][n] for n in names},
                n_boot=10000, seed=0, hierarchical=False)
            wins = sum(1 for n in names if src[ref][n][0] > src[b][n][0])
            row[lab] = {"gap": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins,
                        "n": len(names)}
            flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
            print(f"   vs {b:30s} {lab:8s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"{wins}/{len(names)}{flag}")
        gaps[b] = row
    out["internal_minus"] = gaps
    out["within_coverage"] = float(np.mean(cov))

    print("\n   Within-position values are lower for EVERY block and must be:")
    print("   few variants share a site, the DMS spread inside a site is small,")
    print("   and measurement noise is a larger share of it. Compare internal")
    print("   against output within a column, never across columns.")

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
