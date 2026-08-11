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


def zstats(a):
    a = np.asarray(a, dtype=float)
    return a.mean(0), a.std(0) + 1e-9


def zapply(a, mu, sd):
    return (np.asarray(a, dtype=float) - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--inductive", action="store_true",
                    help="scale held-out features with TRAINING-assay statistics "
                         "instead of the held-out assay's own")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--tm-cache",
                    default="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
                            "prot_interp_files/runs/tm_cache.npz",
                    help="tmtools is in the repo venv, not the container; "
                         "precompute_tm.py writes this")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    TM = np.load(a.tm_cache)
    A = {}
    for f in a.features:
        d = np.load(f, allow_pickle=True)
        # `.replace("gym2_", "")` silently collapsed every gym2s_* file to the
        # single key "gym2s", leaving one assay in the dict. Split on position.
        name = Path(f).stem.split("_")[1]
        y = d["score"]
        ca, ca_wt = d["ca"], d["ca_wt"]
        seq = str(d["wt_seq"])
        # TM comes from the cache: tmtools lives in the repo venv and JAX in
        # the container, and neither has the other. Recomputing it here would
        # simply fail; dropping it would weaken the output baseline, which is
        # the wrong direction to be sloppy in.
        stem = Path(f).stem.split("_", 1)[1]
        if stem not in TM:
            raise SystemExit(f"{stem}: not in {a.tm_cache}; run precompute_tm.py")
        tm = np.asarray(TM[stem], float)
        # RAW blocks are kept alongside the z-scored ones. The published
        # protocol scales every assay by its own statistics, which is
        # unsupervised but transductive -- the held-out assay's rows decide its
        # own scale. `--inductive` instead scales it with the training assays'
        # statistics, so nothing about the held-out protein enters the fit.
        raw = {
            "internal": np.concatenate(
                [d["kl_glob"], d["kl_site"],
                 np.linalg.norm(d["dz_site"], axis=-1),
                 np.linalg.norm(d["ds_site"], axis=-1)], axis=1),
            # The 128 pair channels at the final layer -- the DIRECTION, not
            # just how far the row moved. `internal` above feeds dz_site in as
            # a per-layer norm, which is a defensible shared feature space but
            # discards exactly the quantity this project is about, and it is
            # why the transferred internal number came out well below the
            # within-assay figures reported elsewhere. The channels mean the
            # same thing in every protein (that is the shared-subspace result),
            # so pooling them across assays is well defined.
            "internal_vec": np.asarray(d["dz_site"], float)[:, -1, :],
            "chemistry": pi_chem.chem_matrix([str(m) for m in d["mutant"]]),
            "output_rich": output_matrix(
                ca, ca_wt, tm, d["plddt"], d["plddt_site"], d["pos"]),
        }
        A[name] = {"y": y, "pos": d["pos"], "yz": zscore(y), "tm": tm,
                   "raw": raw, **{k: zscore(v) for k, v in raw.items()}}
        print(f"  loaded {name:8s} n={len(y):4d}  internal dim "
              f"{A[name]['internal'].shape[1]}")

    names = sorted(A)
    dims = {A[n]["internal"].shape[1] for n in names}
    if len(dims) > 1:
        raise SystemExit(f"feature dimensions differ across assays: {dims}")

    BLOCKS = ["internal", "internal_vec", "chemistry", "output_rich"]
    res = {b: {} for b in BLOCKS}
    res["TM_to_WT"] = {}
    res["internal_within"] = {}

    for held in names:
        tr_names = [n for n in names if n != held]
        for b in BLOCKS:
            if a.inductive:
                # one scale, learned on the training assays, applied to all
                mu, sd = zstats(np.concatenate([A[n]["raw"][b] for n in tr_names], 0))
                Xtr = np.concatenate([zapply(A[n]["raw"][b], mu, sd)
                                      for n in tr_names], 0)
                Xte = zapply(A[held]["raw"][b], mu, sd)
            else:
                Xtr = np.concatenate([A[n][b] for n in tr_names], 0)
                Xte = A[held][b]
            ytr = np.concatenate([A[n]["yz"] for n in tr_names], 0)
            k = min(a.k, Xtr.shape[1])
            idx = select_k(Xtr, ytr, k)
            w = ridge_fit(Xtr[:, idx], ytr, a.lam)
            pred = ridge_pred(w, Xte[:, idx])
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

    ORDER = ["internal_within", "internal", "internal_vec", "chemistry",
             "output_rich", "TM_to_WT"]
    LAB = {"internal_within": "internal (within-assay)",
           "internal_vec": "internal 128-dim TRANSFERRED",
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
                        ("internal 128-dim - output-rich", "internal_vec", "output_rich"),
                        ("internal 128-dim - chemistry", "internal_vec", "chemistry"),
                        ("internal 128-dim - internal norms", "internal_vec", "internal"),
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
         "normalisation_mode": "inductive (training-assay statistics)"
             if a.inductive else "transductive (each assay's own statistics)",
         "predictors": summary, "gaps": gaps}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
