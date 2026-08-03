"""Deep cross-model probe readout: per-layer internal features vs the model's output.

Same protocol as the Boltz-2 headline -- position-grouped 75/25 splits, ridge on
train-selected features, identical rows for every predictor -- but now the
internal side is 4 quantities x every Pairformer layer, matching the headline's
construction rather than a 5-feature shortcut.

Feature counts differ by model (Boltz-2 64 layers, OF3 48, Protenix 16). That is
not a confound: each number is a PAIRED within-model comparison of internal
against output.
"""
from __future__ import annotations
import argparse, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402
from analyze_gym_multi import grouped_split, ridge_fit, ridge_pred  # noqa: E402

BLOCKS = ["kl_glob", "kl_site", "dz_site", "ds_site"]


def select_k(X, y, k):
    """Top-k features by |Spearman| on the TRAINING rows only."""
    r = np.array([abs(spearmanr(X[:, j], y).correlation) for j in range(X.shape[1])])
    r[~np.isfinite(r)] = 0
    return np.argsort(-r)[:k]


def fit_deep(X, y, pos, tr, te, seed, k=16):
    """Ridge on k train-selected features; lambda tuned on an inner grouped split."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd
    sel = select_k(Xs[tr], y[tr], k)                  # selection on TRAIN only
    Xs = Xs[:, sel]
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
    ap.add_argument("--glob", default="runs/deep_*.npz")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    per = defaultdict(lambda: defaultdict(list))
    gaps = defaultdict(list)
    meta = {}
    for f in sorted(glob.glob(args.glob)):
        if "smoke" in f:
            continue
        d = np.load(f)
        model = str(d["model"]); y, pos = d["score"], d["pos"]
        nL = int(d["n_layers"]); meta[model] = nL
        X = np.concatenate([d[b] for b in BLOCKS], axis=1)      # [n, 4*nL]
        caw = d["ca_wt"].astype(float)
        tm = np.array([geom.tm_score(c.astype(float), caw) for c in d["ca"]])
        rng = np.random.default_rng(0)
        for s in range(args.splits):
            tr, te = grouped_split(pos, rng)
            if te.sum() < 8 or tr.sum() < 20:
                continue
            ri = fit_deep(X, y, pos, tr, te, s)
            rt = spearmanr(tm[te], y[te]).correlation
            per[model]["internal (deep)"].append(ri)
            per[model]["TM to WT"].append(rt)
            per[model]["pLDDT"].append(spearmanr(d["plddt_mean"][te], y[te]).correlation)
            per[model]["pLDDT@site"].append(spearmanr(d["plddt_site"][te], y[te]).correlation)
            tp, tv = pos[tr], y[tr]
            pred = np.array([tv[np.argmin(np.abs(tp - p))] for p in pos[te]])
            per[model]["position-only"].append(spearmanr(pred, y[te]).correlation)
            gaps[model].append(ri - rt)

    ORDER = ["internal (deep)", "TM to WT", "pLDDT", "pLDDT@site", "position-only"]
    print(f"{'model':10s}{'layers':>7s}{'feats':>7s}{'splits':>7s}" +
          "".join(f"{k:>17s}" for k in ORDER))
    for m in sorted(per):
        nL = meta[m]
        print(f"{m:10s}{nL:>7d}{4*nL:>7d}{len(per[m]['TM to WT']):>7d}" +
              "".join(f"{np.nanmean(per[m][k]):>+17.3f}" for k in ORDER))
    print()
    rng = np.random.default_rng(0)
    for m in sorted(per):
        g = np.array(gaps[m])
        bs = np.array([np.nanmean(g[i]) for i in
                       (rng.integers(0, len(g), len(g)) for _ in range(4000))])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        w = sum(1 for a, b in zip(per[m]["internal (deep)"], per[m]["TM to WT"]) if a > b)
        print(f"  {m:10s} internal beats TM in {w}/{len(g)}; "
              f"gap {g.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"{'' if lo > 0 else '   (includes zero)'}")
    print("\n  Boltz-2 headline for reference: internal 0.548, TM 0.214, gap +0.335")
    print("  (12 assays x 250 variants x 256 features; these are 4 assays x 100 variants)")


if __name__ == "__main__":
    main()
