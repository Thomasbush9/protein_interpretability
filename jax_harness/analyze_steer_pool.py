"""Pool the steering runs — v2, descriptive. The v1 significance is withdrawn.

v1 of this script ranked PC2's |odd component| against eight random directions
per assay and reported an exact binomial on rank-first (p = 9.6e-04) plus a
mean-normalised-rank permutation p. The 2026-09-06 publication audit, and a
re-analysis of the archived steerall_*.npz, found four things wrong with
treating that as confirmatory:

  1. The controls are shared across assays. launch_steer.sh passes no
     assay-specific seed, so exp_steer.py draws the SAME eight standardized
     orientations (seed 0) in every assay, rescaled per assay. The binomial and
     uniform-rank nulls assume twelve independent comparisons; there are not
     twelve independent draws of anything.
  2. The response is not odd in alpha. Averaging odd/alpha over |a| in
     {3,10,30} mixes a large positive term at |a|=10 with a NEGATIVE term at
     |a|=30 (negative in 12/12 assays). Per-dose rank tests: |a|=3 -> 2/12
     (chance), |a|=10 -> 10/12, |a|=30 -> 0/12. At a = -30 the distogram
     BROADENS in all twelve assays.
  3. PC2 is the highest-GAIN direction, not the most sign-structured one. Its
     even component is rank 1 in 8/12 assays; normalising odd by even — the
     statistic report_svd already reports — puts PC2 first in 4/12, with
     odd <= ~9% of even for every direction, PC2 included.
  4. Forking paths: p = 9.6e-04 was the best of six mode x metric cells; `row`
     mode, the literal injection, is at chance. The binomial and rank tests are
     two functions of the same twelve ranks, not two lines of evidence.

An odd response is also generic first-order sensitivity —
f(z+av) - f(z-av) = 2a grad f . v + O(a^3) — so even a clean odd ranking would
show PC2 aligns with the local gradient, not that the model uses it
semantically.

So v2 reports the same quantities DESCRIPTIVELY, and reports everything at
once rather than one chosen cell:

  - odd and even components per dose, per direction, per assay
  - all three modes and both metrics — six cells, none privileged
  - ranks by |odd| (v1's statistic) AND by |odd|/|even| (the gain-corrected one)
  - signed per-dose observations (how many assays broaden at +a, at -a)
  - rank-based tail probabilities with an add-one Monte-Carlo floor, labelled
    descriptive: the independence their null assumes does not hold (point 1)

The archived per-assay numbers are unchanged; what changed is what they are
claimed to establish. The deletion null (analyze_ablate.py, ablate_v1.json) is
the companion piece and is reported beside this in the master report.

  sbatch analysis.sbatch analyze_steer_pool.py --glob '../runs/steerall_*.npz' \
      --out ../runs/steer_pooled_v2.json
"""

from __future__ import annotations

import argparse
import glob
from math import comb
from pathlib import Path

import numpy as np

import sys as _s
from pathlib import Path as _P
_s.path.insert(0, str(_P(__file__).parent))
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402

EPS = 1e-12
METRICS = [("d_sd_site", "distogram width at the injected site"),
           ("d_plddt_site", "pLDDT at the injected site")]
MODES = ["row", "sym", "glob"]


def components(d, metric, direction, mode):
    """Per-|alpha| odd and even components, meaned over sites.

    odd(a)  = [f(+a) - f(-a)] / 2a      first-order (signed) response
    even(a) = [f(+a) + f(-a)] / 2 - f(0)   magnitude response, sign-blind
    """
    sel = (d["rec_dir"] == direction) & (d["rec_mode"] == mode)
    if not sel.any():
        return {}
    al, si, y = d["alpha"][sel], d["site"][sel], d[metric][sel]
    per_dose = {}
    for a in sorted(np.unique(al[al > 0])):
        odds, evens = [], []
        for s in np.unique(si):
            m = si == s
            a_s, y_s = al[m], y[m]
            p = y_s[np.isclose(a_s, a)]
            n = y_s[np.isclose(a_s, -a)]
            z = y_s[np.isclose(a_s, 0.0)]
            if p.size and n.size:
                odds.append((p.mean() - n.mean()) / (2 * a))
                base = z.mean() if z.size else 0.0
                evens.append((p.mean() + n.mean()) / 2 - base)
        if odds:
            per_dose[float(a)] = {"odd": float(np.mean(odds)),
                                  "even": float(np.mean(evens))}
    return per_dose


def pooled(per_dose):
    """v1's pooled statistic (mean odd over doses) and its even counterpart."""
    if not per_dose:
        return np.nan, np.nan
    return (float(np.mean([v["odd"] for v in per_dose.values()])),
            float(np.mean([v["even"] for v in per_dose.values()])))


def binom_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), exact."""
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def rank_of(stat_pc2, stat_rnd):
    """1 = PC2 largest among itself and the randoms."""
    return int(len(stat_rnd) - (stat_pc2 > stat_rnd).sum() + 1)


def rank_cell(rows, key, n_perm, rng):
    """Rank-first count and mean normalised rank for one statistic.

    Tail probabilities use the v1 nulls (independent uniform ranks) with an
    add-one Monte-Carlo floor so nothing prints as 0. They are DESCRIPTIVE:
    the eight control orientations are identical across assays, so the
    independence both nulls assume does not hold.
    """
    ranks = np.array([r[key] for r in rows])
    nrand = np.array([r["n_random"] for r in rows])
    n = len(ranks)
    k = int((ranks == 1).sum())
    p_first = 1.0 / (nrand[0] + 1)
    norm = np.array([(nr + 1 - r) / nr for r, nr in zip(ranks, nrand)])
    obs = float(norm.mean())
    draws = np.empty(n_perm)
    for i in range(n_perm):
        rr = rng.integers(1, nrand + 2)
        draws[i] = np.mean([(nr + 1 - r) / nr for r, nr in zip(rr, nrand)])
    p_rank = float((1 + (draws >= obs).sum()) / (1 + n_perm))
    return {"first": k, "n": n, "chance_each": p_first,
            "binom_tail_descriptive": binom_tail(k, n, p_first),
            "mean_norm_rank": obs, "mc_tail_descriptive": p_rank}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"no files matched {a.glob}")
    rng = np.random.default_rng(a.seed)
    out = {"cells": {}, "assays": []}

    for metric, label in METRICS:
        for mode in MODES:
            rows = []
            for f in files:
                d = np.load(f, allow_pickle=True)
                name = str(d["assay"]).split("_")[0]
                dirs = [str(x) for x in d["dirs"]]
                rnd = [x for x in dirs if x.startswith("random")]
                if "PC2" not in dirs or not rnd:
                    continue
                pd_pc2 = components(d, metric, "PC2", mode)
                pd_pc1 = components(d, metric, "PC1", mode)
                pd_rnd = [components(d, metric, r, mode) for r in rnd]
                if not pd_pc2 or all(not p for p in pd_rnd):
                    continue
                o2, e2 = pooled(pd_pc2)
                o1, _ = pooled(pd_pc1)
                orn = np.array([pooled(p)[0] for p in pd_rnd])
                ern = np.array([pooled(p)[1] for p in pd_rnd])
                ratio2 = abs(o2) / (abs(e2) + EPS)
                ratior = np.abs(orn) / (np.abs(ern) + EPS)
                rows.append({
                    "assay": name, "n_random": len(rnd),
                    "pc2_odd": o2, "pc2_even": e2, "pc1_odd": o1,
                    "pc2_odd_per_dose": {str(k): v["odd"]
                                         for k, v in pd_pc2.items()},
                    "pc2_even_per_dose": {str(k): v["even"]
                                          for k, v in pd_pc2.items()},
                    "random_odd_max": float(np.nanmax(np.abs(orn))),
                    "rank_abs_odd": rank_of(abs(o2), np.abs(orn)),
                    "rank_odd_over_even": rank_of(ratio2, ratior),
                    "rank_abs_odd_per_dose": {
                        str(dose): rank_of(
                            abs(pd_pc2[dose]["odd"]),
                            np.abs(np.array([p[dose]["odd"] for p in pd_rnd
                                             if dose in p])))
                        for dose in pd_pc2},
                })
            if not rows:
                continue
            doses = sorted({float(k) for r in rows
                            for k in r["pc2_odd_per_dose"]})
            cell = {
                "label": label, "mode": mode, "per_assay": rows,
                "n_assays": len(rows),
                # v1's statistic, now one of several, none privileged
                "by_abs_odd": rank_cell(rows, "rank_abs_odd", a.n_perm, rng),
                "by_odd_over_even": rank_cell(rows, "rank_odd_over_even",
                                              a.n_perm, rng),
                "by_abs_odd_per_dose": {
                    str(dose): rank_cell(
                        [{"rank_d": r["rank_abs_odd_per_dose"][str(dose)],
                          "n_random": r["n_random"]} for r in rows
                         if str(dose) in r["rank_abs_odd_per_dose"]],
                        "rank_d", a.n_perm, rng)
                    for dose in doses},
                # the signed observations, per dose: how many assays respond
                # with positive odd component (broadening for d_sd_site)
                "signed_positive_per_dose": {
                    str(dose): int(sum(r["pc2_odd_per_dose"][str(dose)] > 0
                                       for r in rows
                                       if str(dose) in r["pc2_odd_per_dose"]))
                    for dose in doses},
            }
            out["cells"][f"{metric}:{mode}"] = cell
            out["assays"] = [r["assay"] for r in rows]

            print(f"\n=== {label}  ({metric}, mode={mode}) ===")
            print(f"  {'assay':8s} {'odd':>9s} {'even':>9s} "
                  f"{'rk|odd|':>8s} {'rk o/e':>7s}   per-dose odd")
            for r in rows:
                pd = "  ".join(f"{k}:{v:+.3f}"
                               for k, v in r["pc2_odd_per_dose"].items())
                print(f"  {r['assay']:8s} {r['pc2_odd']:+9.4f} "
                      f"{r['pc2_even']:+9.4f} "
                      f"{r['rank_abs_odd']:5d}/{r['n_random']+1} "
                      f"{r['rank_odd_over_even']:4d}/{r['n_random']+1}   {pd}")
            c = cell["by_abs_odd"]
            print(f"  |odd| (v1 statistic): first in {c['first']}/{c['n']}, "
                  f"mean norm rank {c['mean_norm_rank']:.3f}")
            c = cell["by_odd_over_even"]
            print(f"  |odd|/|even|:         first in {c['first']}/{c['n']}, "
                  f"mean norm rank {c['mean_norm_rank']:.3f}")
            for dose, c in cell["by_abs_odd_per_dose"].items():
                print(f"  |odd| at |a|={dose}: first in {c['first']}/{c['n']}")

    out["caveats"] = {
        "controls_shared_across_assays":
            "launch_steer.sh passes no per-assay seed; exp_steer.py draws the "
            "same 8 standardized orientations (seed 0) in every assay, "
            "rescaled per assay. Tail probabilities below any cell assume "
            "independent per-assay draws and are therefore descriptive.",
        "pc2_not_exchangeable":
            "PC2 was discovered on the pooled dz of these same twelve assays; "
            "exchangeability with isotropic Gaussian orientations under the "
            "null is not established. A covariance-matched null was not run.",
        "odd_is_first_order":
            "an odd response is generic first-order sensitivity "
            "(f(z+av)-f(z-av) = 2a grad.v + O(a^3)), not evidence of "
            "semantic use.",
        "dose_nonmonotone":
            "the pooled odd averages |a| in {3,10,30}; the |a|=30 term has "
            "the opposite sign in 12/12 assays (d_sd_site, sym) and |a|=3 "
            "ranks at chance. See by_abs_odd_per_dose.",
        "cells_reported": len(out["cells"]),
    }

    out["protocol"] = pi_protocol.protocol(
        script="analyze_steer_pool.py",
        design="intervention, DESCRIPTIVE (v2): PC2 injected into the FINAL "
               "z, odd/even components per dose, ranked against 8 random "
               "orientations shared across assays; all mode x metric cells "
               "reported; v1's confirmatory p-values withdrawn 2026-09-06",
        layer=pi_protocol.layers("final"),
        features=pi_protocol.features("PC2 direction, raw z units", 128, kept=1),
        source=a.glob, n_assays=len(out.get("assays", [])),
        statistic="odd [f(+a)-f(-a)]/2a and even [f(+a)+f(-a)]/2 - f(0), "
                  "per dose and pooled; ranks by |odd| and |odd|/|even|",
        test="none confirmatory; add-one MC tails labelled descriptive "
             "because the control orientations are shared across assays")
    pi_archive.write_result(a.out, out, protocol=out.pop("protocol"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
