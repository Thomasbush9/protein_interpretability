"""Reproduce the headline: Spearman +0.758 for the internal 128-dim probe.

This is the number the report opens with -- "Boltz-2's internal state predicts
mutational stability far better than anything it emits" -- and it lives in
`transfer_full.json` as `predictors.internal_vec.mean`. The archive's own
producer, `analyze_transfer.py`, computes it alongside three other feature
blocks, a k-sweep, a rotated basis, bootstrap intervals and gap statistics. This
script computes only that one number, in the smallest form that still gets it
right, so the recipe is legible:

    features   dz_site at the FINAL layer -- the 128 pair channels at the
               mutated position. The DIRECTION the row moved, not how far.
    scaling    features and target z-scored WITHIN each assay
    protocol   leave-one-assay-out: fit on 11 assays, test on the 12th
    model      ridge, lambda = 10, intercept unpenalised
    statistic  within-assay Spearman on the held-out assay, meaned over the 12

Two details are load-bearing and easy to get wrong:

  * `dz_site[:, -1, :]` is the final layer. Feeding the per-layer NORM instead
    is a defensible shared feature space and it is what the `internal` block
    does -- it lands well below this figure, because a norm discards exactly
    the quantity this project is about.
  * z-scoring is per assay, not pooled. The channels mean the same thing in
    every protein, which is the shared-subspace result and what makes pooling
    them across assays well defined in the first place.

The expected value is asserted, not printed for eyeballing: this file is a
regression test for the headline that happens to be readable.

    uv run python experiments/analysis/reproduce_headline_transfer.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from protein_interpretability import artifacts
from protein_interpretability.analysis import statistics as st
from protein_interpretability.collection import Cohort

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")

EXPECTED = 0.7578272292356677          # transfer_full.json predictors.internal_vec.mean
LAM = 10.0                             # analyze_transfer's default --lam
TOL = 1e-6


def zscore(a: np.ndarray) -> np.ndarray:
    """Per-column standardisation. The 1e-9 matches the producer exactly; a
    constant column would otherwise divide by zero and poison the whole fit."""
    a = np.asarray(a, dtype=float)
    return (a - a.mean(0)) / (a.std(0) + 1e-9)


def ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge with an intercept that is NOT penalised.

    Shrinking the intercept toward zero would pull every prediction toward the
    origin of a target that was only centred within assay, which is a bias the
    leave-one-assay-out design would then read as poor transfer.
    """
    Xb = np.column_stack([X, np.ones(len(X))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([X, np.ones(len(X))]) @ w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default=str(W / "runs"))
    ap.add_argument("--lam", type=float, default=LAM)
    a = ap.parse_args()

    cohort = Cohort.load("basis_assays")
    cohort.verify()

    # ---- load: 128 final-layer channels and the assay score, per assay -----
    data = {}
    for assay in cohort:
        cap = artifacts.load_capture(Path(a.captures) / f"gym2s_{assay.id}.npz")
        dz = np.asarray(cap.field("dz_site"), float)      # [n, n_layers, 128]
        y = np.asarray(cap.field("score"), float)
        short = assay.id.split("_")[0]
        data[short] = {"X": zscore(dz[:, -1, :]), "y": y, "yz": zscore(y)}

    names = sorted(data)
    print(f"{len(names)} assays, {data[names[0]]['X'].shape[1]} channels\n")

    # ---- leave one assay out ----------------------------------------------
    per_assay = {}
    for held in names:
        train = [n for n in names if n != held]
        Xtr = np.concatenate([data[n]["X"] for n in train], 0)
        ytr = np.concatenate([data[n]["yz"] for n in train], 0)
        w = ridge(Xtr, ytr, a.lam)
        rho = st.spearman(predict(w, data[held]["X"]), data[held]["y"])
        per_assay[held] = float(rho)
        print(f"  hold out {held:6s}  train n={len(ytr):5d}  rho={rho:+.4f}")

    mean = float(np.mean(list(per_assay.values())))
    print(f"\n  internal_vec mean = {mean:+.6f}   (report: {EXPECTED:+.6f})")

    # ---- check against the archive, per assay and pooled -------------------
    archive = W / "report_master" / "data" / "transfer_full.json"
    if archive.exists():
        import json
        ref = json.loads(archive.read_text())["predictors"]["internal_vec"]
        worst, worst_at = 0.0, ""
        for k, v in ref["per_assay"].items():
            d = abs(per_assay[k] - v)
            if d > worst:
                worst, worst_at = d, k
        print(f"  worst per-assay difference {worst:.2e} at {worst_at}")
        if worst > TOL:
            print("  MISMATCH -- the recipe above no longer produces the "
                  "archived number")
            return 1

    if abs(mean - EXPECTED) > TOL:
        print(f"  MISMATCH: {abs(mean - EXPECTED):.2e} above tolerance {TOL:.0e}")
        return 1
    print("  reproduced to within 1e-6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
