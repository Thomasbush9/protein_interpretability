"""Does the mutation live in the output ensemble rather than the output structure?

For each variant: `spread` = mean pairwise TM among its own K diffusion samples
(low = diverse ensemble), `tm_to_wt` = mean TM of its samples against the wild
type's. The wild type's own spread is the sampler's baseline and the only
reference against which a variant's spread means anything.

Headline comparison: does spread track measured stability better than tm_to_wt?
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from geom import tm_score

import pi_stats  # noqa: E402

# Tie-aware, and a standard partial correlation: the previous local versions
# ranked with argsort-of-argsort (which breaks ties by array order) and
# re-ranked their own residuals before correlating.
spearman = pi_stats.spearman


def partial_spearman(x, y, Zs):
    return pi_stats.partial_spearman(x, y, list(Zs))

ap = argparse.ArgumentParser()
ap.add_argument("--features", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
res = {}
for f in a.features:
    d = np.load(f, allow_pickle=True)
    name = Path(f).stem.replace("ens_", "")
    ca, ca_wt, y, pos = d["ca"], d["ca_wt"], d["score"], d["pos"]
    pl, seq, K = d["plddt"], str(d["wt_seq"]), int(d["samples"])
    n = len(y)
    wt_spread = float(np.mean([tm_score(ca_wt[i], ca_wt[j], seq, seq)
                               for i, j in itertools.combinations(range(K), 2)]))
    spread = np.array([np.mean([tm_score(ca[v][i], ca[v][j], seq, seq)
                                for i, j in itertools.combinations(range(K), 2)]) for v in range(n)])
    tm_wt = np.array([np.mean([tm_score(ca[v][i], ca_wt[j], seq, seq)
                               for i in range(K) for j in range(K)]) for v in range(n)])
    plm = pl.mean(axis=(1, 2))
    same = [pos]
    r = {"n": int(n), "K": K, "wt_spread": wt_spread,
         "rho_spread": spearman(spread, y), "rho_tm_to_wt": spearman(tm_wt, y),
         "rho_plddt": spearman(plm, y),
         "prho_spread": partial_spearman(spread, y, same),
         "prho_tm_to_wt": partial_spearman(tm_wt, y, same),
         "prho_plddt": partial_spearman(plm, y, same),
         "spread_mean": float(spread.mean()), "spread_min": float(spread.min()),
         "tm_to_wt_mean": float(tm_wt.mean()),
         "cond_rel_diff_mean": {str(k): float(np.nanmean(d["cond_rel_diff"][:, i]))
                                for i, k in enumerate(d["cond_names"])}}
    res[name] = r
    print(f"\n=== {name} ===  n={n}, K={K} samples/variant")
    print(f"  WT ensemble spread (baseline)   mean pairwise TM = {wt_spread:.4f}")
    print(f"  variant ensemble spread          mean {r['spread_mean']:.4f}  min {r['spread_min']:.4f}")
    print(f"  variant vs WT                    mean TM {r['tm_to_wt_mean']:.4f}")
    print("  correlation with measured stability (raw / partial on position):")
    print(f"    ensemble spread   rho {r['rho_spread']:+.3f} / {r['prho_spread']:+.3f}")
    print(f"    TM to wild type   rho {r['rho_tm_to_wt']:+.3f} / {r['prho_tm_to_wt']:+.3f}")
    print(f"    pLDDT             rho {r['rho_plddt']:+.3f} / {r['prho_plddt']:+.3f}")
    print(f"  conditioning |delta|/|wt|: {r['cond_rel_diff_mean']}")
Path(a.out).write_text(json.dumps(res, indent=2)); print(f"\nwrote {a.out}")
