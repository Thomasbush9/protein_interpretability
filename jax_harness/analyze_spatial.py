"""Section 4.7, reproducible: does divergence fall with 3-D distance from the mutation?

The claim under test is that a buried substitution's effect on the pair
representation is spatially organised -- large near the mutated residues in the
folded structure, small far from them -- while a surface substitution's is not.

This script exists because the reported numbers could not be reproduced. The
report gives rho = -0.564 for the core panel and +0.032 for the surface panel,
citing analyze_onset.py; but analyze_onset.py computes the TIMING of onset (a
divergence-weighted mean layer), not magnitude, and its archived output
(runs/onset_gfp.json) holds -0.503 / +0.188 raw and -0.398 / +0.004 partial.
Neither pair is what the report prints. Sweeping every plausible reconstruction
from the archived residue x layer matrices -- six aggregations, with and without
the mutated residues, raw and partial, under both the corrected rank statistic
and the buggy one the project used at the time -- produced nothing closer than
-0.513, so -0.564 came from inputs that were not archived.

Rather than pick whichever choice lands nearest the published number, this
reports the WHOLE specification curve. The reader sees how much the answer
depends on decisions nobody pre-registered, which for this result is the honest
summary: every specification agrees on the qualitative claim (core strongly
negative, surface indistinguishable from zero) and none of them reproduces the
specific value.

Definitions, stated because the report did not state them:

  response      the residue x layer matrix from exp_matrix.py. A residue's value
                at layer L is the MEAN SYMMETRIC KL over that residue's sampled
                partners -- a distogram divergence, not a ||dz||.
  aggregation   how the 64 layers collapse to one number per residue. Reported
                for all of: final layer, mean, sum, max, layer 45, last 8.
  predictor     3-D distance from the residue's CA to the NEAREST mutated
                residue's CA, in the wild-type structure.
  covariate     sequence distance to the nearest mutated residue, partialled
                out, because sequence-adjacent residues are also spatially close.
  exclusion     the mutated residues themselves are dropped in the primary
                specification; they sit at distance zero with the largest
                response by construction and would manufacture the correlation.

Inference is a circular-shift test along the chain, not a plain permutation.
Residues near each other in sequence have similar divergence, so an ordinary
shuffle builds a null with none of that smoothness and would declare almost any
profile significant. Sliding the response along the chain keeps the
autocorrelation and destroys only its registration with distance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_stats  # noqa: E402

AGGREGATIONS = {
    "final_layer": lambda r: r[:, -1],
    "mean_layers": lambda r: r.mean(1),
    "sum_layers": lambda r: r.sum(1),
    "max_layers": lambda r: r.max(1),
    "layer_45": lambda r: r[:, 45] if r.shape[1] > 45 else r[:, -1],
    "last_8_mean": lambda r: r[:, -8:].mean(1),
}


def legacy_partial_spearman(x, y, z):
    """The procedure this project used before the audit, kept to quantify it.

    Ranks are residualised and then RE-RANKED before correlating, which undoes
    part of the adjustment. Reported alongside the corrected value so the
    difference attributable to the bug is visible rather than asserted.
    """
    def rk(v):
        return np.argsort(np.argsort(v)).astype(float)

    def sp(a, b):
        ra, rb = rk(a), rk(b)
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / d) if d > 0 else float("nan")

    rx, ry, rz = rk(x), rk(y), rk(z)

    def res(a, b):
        b1 = np.stack([b, np.ones_like(b)], 1)
        c, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ c

    return sp(res(rx, rz), res(ry, rz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="runs/matrix_gfp.json",
                    help="exp_matrix.py output: residue x layer divergence")
    ap.add_argument("--onset", default="runs/onset_gfp.json",
                    help="analyze_onset.py output: supplies dist_3d / dist_seq")
    ap.add_argument("--primary-aggregation", default="final_layer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mat = json.load(open(a.matrix))
    ons = json.load(open(a.onset))
    panels = [m for m in mat["mutants"] if m in ons]
    if not panels:
        raise SystemExit(f"no panel is present in both {a.matrix} and {a.onset}")

    results = {"source": {"matrix": a.matrix, "onset": a.onset},
               "response": "mean symmetric KL over sampled partners, per residue "
                           "per layer (exp_matrix.py residue_by_layer)",
               "predictor": "3-D distance to nearest mutated residue (CA, WT)",
               "covariate": "sequence distance to nearest mutated residue",
               "primary_aggregation": a.primary_aggregation,
               "panels": {}}

    print(f"\nSpecification curve: partial Spearman(response, 3-D distance | "
          f"sequence distance)\n")
    hdr = f"{'aggregation':14s}{'excl. mut':10s}"
    for p in panels:
        hdr += f"{p[:14]:>16s}"
    print(hdr)
    print("-" * len(hdr))

    curve = {}
    for agg, fn in AGGREGATIONS.items():
        for excl in (True, False):
            row = {}
            for p in panels:
                rb = np.array(mat["mutants"][p]["residue_by_layer"], dtype=float)
                mut = list(mat["mutants"][p]["mutated_rows"])
                d3 = np.array(ons[p]["dist_3d"], dtype=float)
                ds = np.array(ons[p]["dist_seq"], dtype=float)
                v = fn(rb)
                keep = np.isfinite(d3) & np.isfinite(ds) & np.isfinite(v)
                if excl:
                    keep &= ~np.isin(np.arange(len(v)), mut)
                row[p] = {
                    "raw": pi_stats.spearman(v[keep], d3[keep]),
                    "partial": pi_stats.partial_spearman(v[keep], d3[keep], [ds[keep]]),
                    "partial_legacy_buggy": legacy_partial_spearman(
                        v[keep], d3[keep], ds[keep]),
                    "n_residues": int(keep.sum()),
                }
            curve[f"{agg}|excl={excl}"] = row
            print(f"{agg:14s}{str(excl):10s}" +
                  "".join(f"{row[p]['partial']:>+16.3f}" for p in panels))

    # ---- primary specification, with inference ---------------------------
    print(f"\nPrimary specification: {a.primary_aggregation}, mutated residues "
          f"excluded, circular-shift inference\n")
    fn = AGGREGATIONS[a.primary_aggregation]
    for p in panels:
        rb = np.array(mat["mutants"][p]["residue_by_layer"], dtype=float)
        mut = list(mat["mutants"][p]["mutated_rows"])
        d3 = np.array(ons[p]["dist_3d"], dtype=float)
        ds = np.array(ons[p]["dist_seq"], dtype=float)
        v = fn(rb)
        keep = (np.isfinite(d3) & np.isfinite(ds) & np.isfinite(v) &
                ~np.isin(np.arange(len(v)), mut))
        vk, d3k, dsk = v[keep], d3[keep], ds[keep]

        raw = pi_stats.spearman(vk, d3k)
        par = pi_stats.partial_spearman(vk, d3k, [dsk])
        obs_s, p_shift, sd_shift = pi_stats.circular_shift_test(vk, d3k, covars=[dsk])
        seq_only = pi_stats.spearman(vk, dsk)

        results["panels"][p] = {
            "n_residues": int(keep.sum()), "n_mutated": len(mut),
            "rho_raw_3d": raw, "rho_partial_3d_given_seq": par,
            "rho_raw_seq": seq_only,
            "circular_shift_p": p_shift, "circular_shift_null_sd": sd_shift,
        }
        print(f"  {p:18s} n={int(keep.sum()):3d}  raw 3-D {raw:+.3f}  "
              f"seq-only {seq_only:+.3f}  partial {par:+.3f}  "
              f"shift-test p={p_shift:.4f} (null sd {sd_shift:.3f})")

    results["specification_curve"] = curve

    # ---- what the report claims, against what is reproducible ------------
    par_vals = [results["panels"][p]["rho_partial_3d_given_seq"] for p in panels]
    all_partials = [r[p]["partial"] for r in curve.values() for p in panels]
    results["reported_in_report"] = {"core": -0.564, "surface": 0.032,
                                     "reproducible": False}
    results["partial_range_across_specifications"] = {
        "min": float(np.nanmin(all_partials)), "max": float(np.nanmax(all_partials))}
    print(f"\n  Report prints -0.564 (core) and +0.032 (surface) citing "
          f"analyze_onset.py.\n  Across every specification above, partial rho "
          f"spans [{np.nanmin(all_partials):+.3f}, {np.nanmax(all_partials):+.3f}]; "
          f"-0.564 is not\n  attained by any of them, so the published value came "
          f"from inputs that were\n  not archived and must be replaced by a "
          f"specification stated in the text.")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
