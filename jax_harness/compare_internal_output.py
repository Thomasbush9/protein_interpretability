"""Internal state vs the model's own outputs, same mutants, same protocol.

gym2 captures per-layer internal features, the predicted structure, and pLDDT
for every variant in one run, so this is the only fully like-for-like comparison
available. Everything is scored by Spearman on held-out *positions* (no residue
in both train and test); single-number predictors need no fitting and are scored
directly on the same test rows.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from geom import tm_score
from probe_gym import grouped_split, select_k, ridge_fit, ridge_pred, spearman

ap = argparse.ArgumentParser()
ap.add_argument("--features", nargs="+", required=True)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--out", required=True)
a = ap.parse_args()

res = {}
print(f"{'assay':8s} {'n':>4s} {'internal':>16s} {'TM_to_WT':>16s} {'pLDDT_site':>16s} "
      f"{'pLDDT_mean':>16s} {'pos_base':>16s}")
print("-" * 100)
for f in a.features:
    d = np.load(f, allow_pickle=True)
    name = Path(f).stem.replace("gym2_", "").split("_")[0]
    y, pos, seq = d["score"], d["pos"], str(d["wt_seq"])
    ca, ca_wt = d["ca"], d["ca_wt"]
    L = int(d["n_layers"])
    # internal feature block: per-layer scalars + per-layer ||dz|| at the site
    X = np.concatenate([d["kl_glob"], d["kl_site"],
                        np.linalg.norm(d["dz_site"], axis=-1),
                        np.linalg.norm(d["ds_site"], axis=-1)], axis=1)
    tm_wt = np.array([tm_score(ca[i], ca_wt, seq, seq) for i in range(len(y))])
    pl, pls = d["plddt"], d["plddt_site"]

    acc = {k: [] for k in ("internal", "tm", "pls", "pl", "base")}
    for s in range(a.seeds):
        tr, te = grouped_split(pos, 0.25, np.random.default_rng(s))
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xs = (X - mu) / sd
        idx = select_k(Xs[tr], y[tr], 16)
        w = ridge_fit(Xs[tr][:, idx], (y[tr] - y[tr].mean()) / (y[tr].std() + 1e-8), 1.0)
        acc["internal"].append(spearman(ridge_pred(w, Xs[te][:, idx]), y[te]))
        acc["tm"].append(spearman(tm_wt[te], y[te]))
        acc["pls"].append(spearman(pls[te], y[te]))
        acc["pl"].append(spearman(pl[te], y[te]))
        g = y[tr].mean(); pm = {p: y[tr][pos[tr] == p].mean() for p in np.unique(pos[tr])}
        acc["base"].append(spearman(np.array([pm.get(p, g) for p in pos[te]]), y[te]))
    fmt = lambda v: f"{np.mean(v):+.3f}+/-{np.std(v):.3f}"
    res[name] = {k: [float(x) for x in v] for k, v in acc.items()}
    print(f"{name:8s} {len(y):>4d} {fmt(acc['internal']):>16s} {fmt(acc['tm']):>16s} "
          f"{fmt(acc['pls']):>16s} {fmt(acc['pl']):>16s} {fmt(acc['base']):>16s}")
    res[name]["mean_tm_to_wt"] = float(tm_wt.mean())
    res[name]["plddt_wt"] = float(d["plddt_wt"])

print("\nSpearman on held-out positions, mean +/- sd over "
      f"{a.seeds} position-grouped splits. All predictors on identical rows.")
Path(a.out).write_text(json.dumps(res, indent=2)); print(f"wrote {a.out}")
