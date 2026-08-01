"""Probe ProteinGym stability from internal state: which layer, and does an MLP help?

Three questions, in order:

  1. Which single Pairformer layer's internal divergence best predicts measured
     stability? (the layer sweep -- the last layer is not assumed to be best)
  2. Does a linear model over all layers beat the best single layer?
  3. Does a small MLP beat the linear model?

All reported as Spearman on a **held-out split grouped by residue position**, not
a random split. A random split would put other substitutions at the same site in
both train and test, and position identity alone carries a lot of stability
signal (buried sites are intolerant), so a random split would flatter every
model and would not show whether the features generalise to unseen sites.

Baselines that matter:
  - the model's own confidence would be the natural comparator, but pLDDT is
    per-structure and near-constant across single mutants at this scale, so the
    honest baseline here is a position-only predictor (train-set mean score per
    position, which is what a random split would leak) and a global mean.

Hyperparameters (ridge lambda, MLP width, and how many features to keep) are
chosen on an inner position-grouped validation split of the *training* data
only, never on test. Without this the multi-feature models lose to the single
best feature purely from overfitting 256 features to a few hundred rows -- a
synthetic check with a planted signal showed exactly that failure, which is why
the tuning is here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def ridge_fit(X, y, lam=1.0):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def ridge_pred(w, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ w


def mlp_fit(X, y, hidden=32, epochs=400, lr=3e-3, seed=0, l2=1e-4):
    """Small 1-hidden-layer MLP, full-batch Adam, numpy only.

    Deliberately tiny: with a few hundred training variants a larger model would
    just memorise, and the question is whether the internal features carry
    predictive structure, not how well we can fit."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, 1)); b2 = np.zeros(1)
    ps = [W1, b1, W2, b2]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    yv = y.reshape(-1, 1)
    for t in range(1, epochs + 1):
        H = X @ W1 + b1
        A = np.maximum(H, 0)
        P = A @ W2 + b2
        r = P - yv
        gW2 = A.T @ r / len(X) + l2 * W2
        gb2 = r.mean(0)
        dA = (r @ W2.T) * (H > 0)
        gW1 = X.T @ dA / len(X) + l2 * W1
        gb1 = dA.mean(0)
        for i, g in enumerate([gW1, gb1, gW2, gb2]):
            ms[i] = b1_ * ms[i] + (1 - b1_) * g
            vs[i] = b2_ * vs[i] + (1 - b2_) * g * g
            mh = ms[i] / (1 - b1_ ** t); vh = vs[i] / (1 - b2_ ** t)
            ps[i] -= lr * mh / (np.sqrt(vh) + eps)
        W1, b1, W2, b2 = ps
    return lambda Z: (np.maximum(Z @ W1 + b1, 0) @ W2 + b2).ravel()


def grouped_split(pos, frac, rng):
    up = np.unique(pos); rng.shuffle(up)
    held = set(up[: max(1, int(round(len(up) * frac)))])
    m = np.array([p in held for p in pos])
    return ~m, m


def select_k(Xtr, ytr, k):
    """Top-k features by |Spearman| on the training rows only."""
    r = np.array([abs(spearman(Xtr[:, j], ytr)) for j in range(Xtr.shape[1])])
    return np.argsort(-r)[:k]


def tune_and_fit(X, y, pos, tr, rng):
    """Pick lambda / k / width on an inner grouped split, refit on all of train."""
    itr, iva = grouped_split(pos[tr], 0.25, rng)
    Xtr, ytr = X[tr][itr], y[tr][itr]
    Xva, yva = X[tr][iva], y[tr][iva]

    best_lin = (-1, None)
    for k in (8, 16, 32, 64, X.shape[1]):
        idx = select_k(Xtr, ytr, min(k, X.shape[1]))
        for lam in (0.1, 1.0, 10.0, 100.0, 1000.0):
            w = ridge_fit(Xtr[:, idx], ytr, lam)
            r = spearman(ridge_pred(w, Xva[:, idx]), yva)
            if abs(r) > best_lin[0]:
                best_lin = (abs(r), (k, lam, idx))
    k, lam, _ = best_lin[1]
    idx = select_k(X[tr], y[tr], min(k, X.shape[1]))
    w = ridge_fit(X[tr][:, idx], y[tr], lam)
    lin = lambda Z: ridge_pred(w, Z[:, idx])   # noqa: E731

    best_mlp = (-1, None)
    for k2 in (8, 16, 32):
        idx2 = select_k(Xtr, ytr, k2)
        for hid in (8, 16, 32):
            f = mlp_fit(Xtr[:, idx2], ytr, hidden=hid, seed=0)
            r = spearman(f(Xva[:, idx2]), yva)
            if abs(r) > best_mlp[0]:
                best_mlp = (abs(r), (k2, hid))
    k2, hid = best_mlp[1]
    idx2 = select_k(X[tr], y[tr], k2)
    f = mlp_fit(X[tr][:, idx2], y[tr], hidden=hid, seed=0)
    mlp = lambda Z: f(Z[:, idx2])              # noqa: E731

    return lin, mlp, {"ridge_k": int(k), "ridge_lambda": float(lam),
                      "mlp_k": int(k2), "mlp_hidden": int(hid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    results = {}
    for f in a.features:
        d = np.load(f, allow_pickle=True)
        X, y = d["X"], d["score"]
        L = int(d["n_layers"]); blocks = [str(b) for b in d["blocks"]]
        pos = d["pos"]
        name = Path(f).stem

        # position-grouped split: no residue appears in both train and test
        rng = np.random.default_rng(a.seed)
        tr, te = grouped_split(pos, 0.25, rng)

        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xs = (X - mu) / sd
        ym, ys = y[tr].mean(), y[tr].std() + 1e-8
        yz = (y - ym) / ys

        # 1. layer sweep. The *choice* of best layer is made on train; its
        # reported score is on test. Picking the argmax on test would be
        # selection-on-test across 256 candidates and would inflate it badly.
        sweep_tr, sweep_te = {}, {}
        for bi, blk in enumerate(blocks):
            sweep_tr[blk] = [spearman(X[tr, bi * L + l], y[tr]) for l in range(L)]
            sweep_te[blk] = [spearman(X[te, bi * L + l], y[te]) for l in range(L)]
        best_blk, best_lay = max(
            ((blk, int(np.argmax(np.abs(r)))) for blk, r in sweep_tr.items()),
            key=lambda t: abs(sweep_tr[t[0]][t[1]]))
        best = (best_blk, best_lay, float(abs(sweep_te[best_blk][best_lay])))
        sweep = sweep_te

        # 2/3. linear and MLP, both with hyperparameters tuned on an inner
        # grouped validation split of the training rows only
        lin, mlp, hp = tune_and_fit(Xs, yz, pos, tr, np.random.default_rng(a.seed + 1))
        rho_lin = spearman(lin(Xs[te]), y[te])
        rho_mlp = spearman(mlp(Xs[te]), y[te])

        # baseline: position-only (train mean per position); unseen positions
        # fall back to the global train mean -- which is the point of the split
        gmean = y[tr].mean()
        pm = {p: y[tr][pos[tr] == p].mean() for p in np.unique(pos[tr])}
        rho_pos = spearman(np.array([pm.get(p, gmean) for p in pos[te]]), y[te])

        results[name] = {
            "n": int(len(y)), "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_layers": L,
            "best_single": {"block": best[0], "layer": best[1], "rho_test": best[2],
                            "rho_train": float(abs(sweep_tr[best[0]][best[1]])),
                            "selected_on": "train"},
            "sweep_train": {k: [float(v) for v in vv] for k, vv in sweep_tr.items()},
            "rho_linear_all": rho_lin, "rho_mlp": rho_mlp,
            "rho_position_baseline": rho_pos, "hyperparams": hp,
            "sweep": {k: [float(v) for v in vv] for k, vv in sweep.items()},
        }
        print(f"\n=== {name} ===  n={len(y)} (train {tr.sum()} / test {te.sum()}, "
              f"position-grouped)")
        print(f"  best single feature : {best[0]} @ layer {best[1]:2d}  "
              f"(chosen on train)   rho_test = {best[2]:.3f}")
        for blk in blocks:
            rt = np.abs(sweep_tr[blk]); l = int(np.argmax(rt))
            print(f"     {blk:9s} train-best L{l:2d}  rho_test={abs(sweep_te[blk][l]):.3f}   "
                  f"final-layer rho_test={abs(sweep_te[blk][-1]):.3f}")
        print(f"  linear (k={hp['ridge_k']}, lam={hp['ridge_lambda']:g}) : rho = {rho_lin:.3f}")
        print(f"  small MLP (k={hp['mlp_k']}, h={hp['mlp_hidden']}) : rho = {rho_mlp:.3f}")
        print(f"  position-only base  : rho = {rho_pos:.3f}")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
