"""Pool the steering runs: does PC2 outrank its own controls across proteins?

`analyze_steer.py` scores one assay and says, correctly, that with a handful of
random draws the comparison is descriptive rather than a p-value. That is the
right call for one protein and the wrong place to stop, because the power here
was never in more draws inside one assay -- it is in the number of independent
proteins that each put PC2 above their own controls.

The statistic is the ODD component of the response, per unit alpha, as
`analyze_steer` defines it:

    odd(a) = [f(+a) - f(-a)] / 2a

PC2 is the broadening axis, so if the model represents it as a signed quantity
then +alpha should broaden and -alpha should sharpen, and the response should be
odd in alpha. A direction that merely disturbs the computation has no privileged
sign and its response is even. Effect size cannot separate those; sign structure
can, which is why nothing here is reported as "PC2 moved the output more".

Per assay, PC2's odd component is ranked against the n random directions drawn
in that same assay. Under the null that PC2 is not special, its rank is uniform
on 1..n+1, so:

  sign test    the number of assays where PC2 ranks first, against
               Binomial(n_assays, 1/(n+1)). Exact tail, no asymptotics -- with
               twelve assays and eight controls the counts are small.
  rank sum     the mean normalised rank, which uses the near misses a sign test
               throws away, with a permutation null over per-assay rank draws.

PC1 is carried through the identical pipeline as a positive-control direction
that is NOT the stability axis: it is substitution volume, so if every component
outranks its controls the result is about components in general and not about
PC2.

  sbatch analysis.sbatch analyze_steer_pool.py --glob '../runs/steerall_*.npz' \
      --out ../runs/steer_pooled.json
"""

from __future__ import annotations

import argparse
import glob
import json
from math import comb
from pathlib import Path

import numpy as np

EPS = 1e-12
METRICS = [("d_sd_site", "distogram width at the injected site"),
           ("d_plddt_site", "pLDDT at the injected site")]


def odd_per_alpha(d, metric, direction, mode):
    """Mean over sites and |alpha| of [f(+a) - f(-a)] / 2a."""
    sel = (d["rec_dir"] == direction) & (d["rec_mode"] == mode)
    if not sel.any():
        return np.nan
    al, si, y = d["alpha"][sel], d["site"][sel], d[metric][sel]
    vals = []
    for s in np.unique(si):
        m = si == s
        a_s, y_s = al[m], y[m]
        for a in np.unique(a_s[a_s > 0]):
            p = y_s[np.isclose(a_s, a)]
            n = y_s[np.isclose(a_s, -a)]
            if p.size and n.size:
                vals.append((p.mean() - n.mean()) / (2 * a))
    return float(np.mean(vals)) if vals else np.nan


def binom_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), exact."""
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="sym",
                    help="row | sym | glob; sym is the mutation-shaped one")
    ap.add_argument("--n-perm", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"no files matched {a.glob}")
    rng = np.random.default_rng(a.seed)
    out = {"mode": a.mode, "assays": [], "metrics": {}}

    for metric, label in METRICS:
        rows, ranks, nrand = [], [], []
        for f in files:
            d = np.load(f, allow_pickle=True)
            name = str(d["assay"]).split("_")[0]
            dirs = [str(x) for x in d["dirs"]]
            rnd = [x for x in dirs if x.startswith("random")]
            if "PC2" not in dirs or not rnd:
                continue
            o_pc2 = odd_per_alpha(d, metric, "PC2", a.mode)
            o_pc1 = odd_per_alpha(d, metric, "PC1", a.mode)
            o_rnd = np.array([odd_per_alpha(d, metric, r, a.mode) for r in rnd])
            if np.isnan(o_pc2) or np.all(np.isnan(o_rnd)):
                continue
            # Rank by |odd|: the claim is that PC2 has MORE sign structure,
            # and its orientation is fixed by the report's own convention, so
            # a signed ranking would let the convention decide the answer.
            beat = int((np.abs(o_pc2) > np.abs(o_rnd)).sum())
            rk = len(o_rnd) - beat + 1          # 1 = PC2 largest
            rows.append({"assay": name, "pc2": o_pc2, "pc1": o_pc1,
                         "random_max": float(np.nanmax(np.abs(o_rnd))),
                         "n_random": len(o_rnd), "rank": rk})
            ranks.append(rk)
            nrand.append(len(o_rnd))

        if not rows:
            out["metrics"][metric] = {"error": "no usable runs"}
            continue
        ranks = np.array(ranks)
        n = len(ranks)
        k = int((ranks == 1).sum())
        p_first = 1.0 / (nrand[0] + 1)
        p_sign = binom_tail(k, n, p_first)

        # Permutation null on the mean normalised rank, drawing each assay's
        # rank uniformly from its own 1..n_i+1 so assays with different control
        # counts are not silently pooled as if they had the same.
        norm = np.array([(nr + 1 - r) / nr for r, nr in zip(ranks, nrand)])
        obs = float(norm.mean())
        draws = np.empty(a.n_perm)
        for i in range(a.n_perm):
            rr = np.array([rng.integers(1, nr + 2) for nr in nrand])
            draws[i] = np.mean([(nr + 1 - r) / nr for r, nr in zip(rr, nrand)])
        p_rank = float((draws >= obs).mean())

        print(f"\n=== {label}  ({metric}, mode={a.mode}) ===\n")
        print(f"  {'assay':8s} {'|odd| PC2':>11s} {'best random':>12s} "
              f"{'rank':>6s}")
        for r in rows:
            print(f"  {r['assay']:8s} {abs(r['pc2']):11.4f} "
                  f"{r['random_max']:12.4f} {r['rank']:4d}/{r['n_random']+1}")
        print(f"\n  PC2 ranks first in {k}/{n} assays "
              f"(chance {p_first:.3f} each)  ->  sign test p = {p_sign:.4g}")
        print(f"  mean normalised rank {obs:.3f} (0.5 = chance)  ->  "
              f"permutation p = {p_rank:.4g}")

        k1 = int(sum(1 for r in rows
                     if abs(r["pc1"]) > r["random_max"]))
        print(f"  positive control: PC1 beats its own controls in {k1}/{n} "
              f"assays (p = {binom_tail(k1, n, p_first):.4g})")
        if k1 >= k:
            print("  NOTE: PC1 does at least as well as PC2. Whatever this is\n"
                  "  measuring is a property of the components generally, not "
                  "of the\n  stability axis.")

        out["metrics"][metric] = {
            "label": label, "per_assay": rows, "n_assays": n,
            "pc2_first": k, "p_first_each": p_first, "p_sign": p_sign,
            "mean_norm_rank": obs, "p_rank": p_rank, "pc1_beats": k1,
            "p_sign_pc1": binom_tail(k1, n, p_first)}
        out["assays"] = [r["assay"] for r in rows]

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
