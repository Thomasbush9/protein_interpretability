"""Internal vs output, per model. Same protocol as the Boltz-2 probe.

Position-grouped 75/25 splits, 5 partitions per assay, so a variant's residue
never appears on both sides. Ridge on the internal features (selected on train
only); the output predictors are single unfitted numbers scored directly on the
same held-out rows. Every predictor sees IDENTICAL rows in every split, so the
comparison is paired.

Reports per model:  internal, TM-to-WT, pLDDT, pLDDT-at-site, position-only.
"""
from __future__ import annotations
import argparse, glob
from collections import defaultdict
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402  (tmtools lives here; login node only)

INTERNAL = ["kl_glob", "kl_site", "ent_glob", "ent_site", "ed_site"]


def grouped_split(pos, rng, frac=0.75):
    u = np.unique(pos); rng.shuffle(u)
    tr = set(u[:max(1, int(len(u) * frac))].tolist())
    m = np.array([p in tr for p in pos])
    return m, ~m


def ridge_fit(X, y, lam):
    Xb = np.column_stack([np.ones(len(X)), X])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1]); A[0, 0] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def ridge_pred(w, X):
    return np.column_stack([np.ones(len(X)), X]) @ w


def fit_internal(X, y, pos, tr, te, seed):
    """Internal-feature prediction for one split. THE single definition.

    lambda is tuned on an inner position-grouped split of the TRAINING rows
    only. Both the table and the figure call this, so they cannot report
    different numbers for the same data -- an earlier version of the figure
    hardcoded lambda = 1.0 while the table tuned it.
    """
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd
    best, best_r = 1.0, -9
    itr, ite = grouped_split(pos[tr], np.random.default_rng(seed))
    for lam in (0.1, 1.0, 10.0, 100.0):
        if ite.sum() < 4 or itr.sum() < 10:
            continue
        w = ridge_fit(Xs[tr][itr], y[tr][itr], lam)
        r = spearmanr(ridge_pred(w, Xs[tr][ite]), y[tr][ite]).correlation
        if np.isfinite(r) and r > best_r:
            best_r, best = r, lam
    w = ridge_fit(Xs[tr], y[tr], best)
    return spearmanr(ridge_pred(w, Xs[te]), y[te]).correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/gymm_*.npz")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    per = defaultdict(lambda: defaultdict(list))   # model -> predictor -> rhos
    wins = defaultdict(lambda: [0, 0])
    assays = defaultdict(set)

    for f in sorted(glob.glob(args.glob)):
        d = np.load(f)
        model, assay = str(d["model"]), str(d["assay"])
        assays[model].add(assay)
        y, pos = d["score"], d["pos"]
        # TM is computed here, not in the extractor: tmtools is absent from the
        # mosaic container, so the extractor saves coordinates instead.
        caw = d["ca_wt"].astype(float)
        tm_to_wt = np.array([geom.tm_score(c.astype(float), caw) for c in d["ca"]])
        X = np.column_stack([d[k] for k in INTERNAL])
        rng = np.random.default_rng(0)
        for s in range(args.splits):
            tr, te = grouped_split(pos, rng)
            if te.sum() < 8 or tr.sum() < 20:
                continue
            rho_int = fit_internal(X, y, pos, tr, te, s)
            per[model]["internal"].append(rho_int)
            rho_tm = spearmanr(tm_to_wt[te], y[te]).correlation
            for nm, v in (("TM to WT", tm_to_wt), ("pLDDT", d["plddt_mean"]),
                          ("pLDDT@site", d["plddt_site"])):
                per[model][nm].append(spearmanr(v[te], y[te]).correlation)
            # position-only baseline: train-set mean dG at the nearest train position
            tp, tv = pos[tr], y[tr]
            pred = np.array([tv[np.argmin(np.abs(tp - p))] for p in pos[te]])
            per[model]["position-only"].append(spearmanr(pred, y[te]).correlation)
            wins[model][0] += int(rho_int > rho_tm); wins[model][1] += 1

    order = ["internal", "TM to WT", "pLDDT", "pLDDT@site", "position-only"]
    print(f"{'model':10s}{'assays':>7s}{'splits':>7s}" +
          "".join(f"{k:>15s}" for k in order))
    for model in sorted(per):
        n_sp = len(per[model]["internal"])
        row = "".join(
            f"{np.nanmean(per[model][k]):>+15.3f}" for k in order)
        print(f"{model:10s}{len(assays[model]):>7d}{n_sp:>7d}{row}")
    print()
    for model in sorted(per):
        w, t = wins[model]
        gap = np.nanmean(per[model]["internal"]) - np.nanmean(per[model]["TM to WT"])
        # paired bootstrap on the gap
        a = np.array(per[model]["internal"]); b = np.array(per[model]["TM to WT"])
        rng = np.random.default_rng(0)
        bs = [np.nanmean(a[i] - b[i]) for i in
              (rng.integers(0, len(a), len(a)) for _ in range(4000))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {model:10s} internal beats TM in {w}/{t} splits; "
              f"gap {gap:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("\n  NOTE: internal here is FINAL-trunk distogram features only "
          "(5 features).\n  The Boltz-2 headline 0.548 used 256 per-layer "
          "features and is NOT comparable.")


if __name__ == "__main__":
    main()
