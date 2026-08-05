"""Does the internal probe transfer to a protein it was never trained on?

Everything reported so far is WITHIN-assay decodability: train on some residue
positions of a protein, test on other positions of the same protein. That is the
right design for the comparison the paper makes -- internal against the same
model's own output, on identical rows -- but it leaves a question a reader will
ask immediately: is the Pairformer carrying a general stability signal, or does
each assay need its own decoder?

Leave-one-assay-out answers it. Train on 11 assays, test on the 12th, repeat.
All twelve Boltz-2 assays share the same 256-dimensional feature space (4
quantities x 64 layers), so the pooling is well defined.

Two normalisations are needed and neither is optional. Feature scales differ
between proteins -- ||dz|| depends on chain length and on the model's
representation scale for that fold -- so features are z-scored WITHIN each assay
before pooling. Targets differ in dynamic range for the same reason, so
DMS_score is z-scored within assay too. Without both, the ridge would spend its
capacity modelling which protein a row came from.

The comparison that matters here is NOT against a stability predictor; this
project does not compete on ProteinGym. It is whether internal still beats the
model's own emitted output when both have to generalise to an unseen protein.
So the transferred internal probe is scored against:

  TM to WT          unfitted, so transfer is free -- the same number either way
  output_rich       the 10 emitted quantities, transferred the same way
  chemistry         substitution chemistry, transferred the same way
  within-assay      the ordinary probe, as a ceiling reference

If internal transfers and the output side does not, the claim strengthens from
"decodable per assay" to "a general signal the coordinates do not carry". If
neither transfers, the honest description stays assay-specific decodability --
which does not damage the paired comparison, but does bound what can be said.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402
import pi_chem  # noqa: E402
import pi_stats  # noqa: E402
from compare_internal_output import (grouped_split, output_matrix,  # noqa: E402
                                     ridge_fit, ridge_pred, select_k)


def zscore(a):
    a = np.asarray(a, dtype=float)
    return (a - a.mean(0)) / (a.std(0) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = {}
    for f in a.features:
        d = np.load(f, allow_pickle=True)
        name = Path(f).stem.replace("gym2_", "").split("_")[0]
        y = d["score"]
        ca, ca_wt = d["ca"], d["ca_wt"]
        seq = str(d["wt_seq"])
        tm = np.array([geom.tm_score(c.astype(float), ca_wt.astype(float))
                       for c in ca])
        A[name] = {
            "y": y, "pos": d["pos"],
            "yz": zscore(y),
            "internal": zscore(np.concatenate(
                [d["kl_glob"], d["kl_site"],
                 np.linalg.norm(d["dz_site"], axis=-1),
                 np.linalg.norm(d["ds_site"], axis=-1)], axis=1)),
            "chemistry": zscore(pi_chem.chem_matrix([str(m) for m in d["mutant"]])),
            "output_rich": zscore(output_matrix(
                ca, ca_wt, tm, d["plddt"], d["plddt_site"], d["pos"])),
            "tm": tm,
        }
        print(f"  loaded {name:8s} n={len(y):4d}  internal dim "
              f"{A[name]['internal'].shape[1]}")

    names = sorted(A)
    dims = {A[n]["internal"].shape[1] for n in names}
    if len(dims) > 1:
        raise SystemExit(f"feature dimensions differ across assays: {dims}")

    BLOCKS = ["internal", "chemistry", "output_rich"]
    res = {b: {} for b in BLOCKS}
    res["TM_to_WT"] = {}
    res["internal_within"] = {}

    for held in names:
        tr_names = [n for n in names if n != held]
        for b in BLOCKS:
            Xtr = np.concatenate([A[n][b] for n in tr_names], 0)
            ytr = np.concatenate([A[n]["yz"] for n in tr_names], 0)
            k = min(a.k, Xtr.shape[1])
            idx = select_k(Xtr, ytr, k)
            w = ridge_fit(Xtr[:, idx], ytr, a.lam)
            pred = ridge_pred(w, A[held][b][:, idx])
            res[b][held] = pi_stats.spearman(pred, A[held]["y"])
        # unfitted comparator: transfer is free, it is the same number
        res["TM_to_WT"][held] = pi_stats.spearman(A[held]["tm"], A[held]["y"])
        # ceiling: the ordinary within-assay probe, position-grouped
        vals = []
        for s in range(5):
            tr, te = grouped_split(A[held]["pos"], 0.25, np.random.default_rng(s))
            X = A[held]["internal"]
            idx = select_k(X[tr], A[held]["y"][tr], a.k)
            w = ridge_fit(X[tr][:, idx], A[held]["yz"][tr], 1.0)
            vals.append(pi_stats.spearman(ridge_pred(w, X[te][:, idx]),
                                          A[held]["y"][te]))
        res["internal_within"][held] = float(np.nanmean(vals))

    ORDER = ["internal_within", "internal", "chemistry", "output_rich", "TM_to_WT"]
    LAB = {"internal_within": "internal (within-assay)",
           "internal": "internal TRANSFERRED", "chemistry": "chemistry TRANSFERRED",
           "output_rich": "output-rich TRANSFERRED", "TM_to_WT": "TM to WT"}
    print(f"\nLeave-one-assay-out: trained on 11 assays, tested on the 12th\n")
    print(f"{'held-out assay':16s}" + "".join(f"{LAB[k][:22]:>24s}" for k in ORDER))
    print("-" * (16 + 24 * len(ORDER)))
    for n in names:
        print(f"{n:16s}" + "".join(f"{res[k][n]:>+24.3f}" for k in ORDER))
    print("-" * (16 + 24 * len(ORDER)))
    print(f"{'mean':16s}" + "".join(
        f"{np.nanmean([res[k][n] for n in names]):>+24.3f}" for k in ORDER))

    print("\nAssay-level bootstrap over the 12 held-out assays\n")
    summary = {}
    for k in ORDER:
        pt, lo, hi, nk = pi_stats.cluster_bootstrap(
            {n: [res[k][n]] for n in names}, n_boot=10000, seed=0,
            hierarchical=False)
        summary[k] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                      "per_assay": {n: res[k][n] for n in names}}
        print(f"  {LAB[k]:26s} {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    print("\nPaired differences across the 12 held-out assays\n")
    gaps = {}
    for lab, ka, kb in (("transferred internal - TM", "internal", "TM_to_WT"),
                        ("transferred internal - output-rich", "internal", "output_rich"),
                        ("transferred internal - chemistry", "internal", "chemistry"),
                        ("within-assay - transferred", "internal_within", "internal")):
        pt, lo, hi, nk = pi_stats.paired_cluster_bootstrap(
            {n: [res[ka][n]] for n in names}, {n: [res[kb][n]] for n in names},
            n_boot=10000, seed=0, hierarchical=False)
        wins = sum(1 for n in names if res[ka][n] > res[kb][n])
        gaps[lab] = {"gap": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins,
                     "n_assays": len(names)}
        flag = "" if (np.isfinite(lo) and lo > 0) else "   <- includes zero"
        print(f"  {lab:36s} {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"{wins}/{len(names)} assays{flag}")

    Path(a.out).write_text(json.dumps(
        {"protocol": {"design": "leave-one-assay-out", "k": a.k, "lam": a.lam,
                      "normalisation": "features and target z-scored within assay",
                      "n_assays": len(names)},
         "predictors": summary, "gaps": gaps}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
