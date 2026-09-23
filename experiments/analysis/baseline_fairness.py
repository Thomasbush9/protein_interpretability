"""Is the emitted baseline losing because it is weak, or because it is unfair?

Gate 1C of the 2026-09-06 audit. Three specific complaints, each answered by a
column here rather than by argument:

  1. FIXED LAMBDA. Every block has been fitted at lambda = 10. The same numeric
     ridge penalty is not the same effective capacity under different covariance
     spectra, so a 128-channel block and a 10-feature block are not equally
     regularized by it. Answer: an inner leave-one-TRAINING-protein-out sweep
     picks lambda per block, using training assays only, never the held-out one.

  2. NOT ACTUALLY NESTED. "Tripling the features buys almost nothing" was read
     as evidence that no useful geometry remains, but the 37-feature block is
     not the 10-feature block plus 27: `rmsd_local_8A` and `rmsd_local_12A`
     exist only in the 10. A union that is genuinely a superset is the only
     version of that argument that holds.

  3. A MISSING COMPARATOR. Effective strain -- local relative deformation --
     is the measure McBride et al. found informative for mutation effects, and
     it was never in the baseline. Prespecified here as three columns (at the
     site, mean, max), added to the union.

Everything is fitted on training proteins only; the held-out assay contributes
no label and no hyperparameter. Reported per model, leave-one-assay-out.

    uv run python experiments/analysis/baseline_fairness.py \
        --out $W/runs/baseline_fairness.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax_harness"))

import pi_archive                                              # noqa: E402
import pi_protocol                                             # noqa: E402
import pi_stats                                                # noqa: E402
from compare_internal_output import OUTPUT_FEATURES, output_matrix  # noqa: E402
from protein_interpretability import artifacts                 # noqa: E402
from protein_interpretability.analysis.emitted_geometry import (  # noqa: E402
    GEOMETRY_FEATURES, geometry_matrix,
)
from protein_interpretability.collection import Cohort         # noqa: E402

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
MODELS = ("boltz2", "of3", "protenix")
LAMBDAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
STRAIN_CUTOFF = 13.0          # McBride et al.'s neighbourhood radius
STRAIN_FEATURES = ["strain_site", "strain_mean", "strain_max"]
EPS = 1e-9


def strain_matrix(ca, ca_wt, pos):
    """Effective strain: mean relative change in local CA-CA distances.

    For residue i, over neighbours j within STRAIN_CUTOFF of i in the WILD
    TYPE, the mean of |d_ij(mut) - d_ij(wt)| / d_ij(wt). Frame-free by
    construction -- it is a function of distances, so no superposition is
    involved and no rigid-body difference can leak in.
    """
    ca_wt = np.asarray(ca_wt, float)
    dwt = np.linalg.norm(ca_wt[:, None] - ca_wt[None], axis=-1)
    n = len(ca_wt)
    nb = (dwt < STRAIN_CUTOFF) & ~np.eye(n, dtype=bool)
    rows = []
    for i in range(len(ca)):
        d = np.linalg.norm(np.asarray(ca[i], float)[:, None]
                           - np.asarray(ca[i], float)[None], axis=-1)
        rel = np.abs(d - dwt) / (dwt + EPS)
        es = np.array([rel[k][nb[k]].mean() if nb[k].any() else 0.0
                       for k in range(n)])
        p = int(pos[i]) if int(pos[i]) < n else 0
        rows.append([es[p], float(es.mean()), float(es.max())])
    return np.asarray(rows, float)


def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def _fit_predict(blocks, target, tr, held, lam):
    Xtr = np.concatenate(
        [(blocks[n] - blocks[n].mean(0)) / (blocks[n].std(0) + EPS)
         for n in tr], 0)
    ytr = np.concatenate([(target[n] - target[n].mean())
                          / (target[n].std() + EPS) for n in tr], 0)
    Xte = ((blocks[held] - blocks[held].mean(0))
           / (blocks[held].std(0) + EPS))
    return Xte @ ridge(Xtr, ytr, lam)


def loao(blocks, target, lam=None):
    """Leave-one-assay-out. If lam is None, an INNER sweep picks it per fold.

    The inner sweep leaves one TRAINING protein out, never the test protein,
    so the reported number still sees the held-out assay exactly once.
    """
    names = sorted(target)
    rho, chosen = {}, {}
    for held in names:
        tr = [n for n in names if n != held]
        if lam is None:
            scores = {}
            for cand in LAMBDAS:
                inner = []
                for v in tr:
                    tr2 = [n for n in tr if n != v]
                    inner.append(pi_stats.spearman(
                        _fit_predict(blocks, target, tr2, v, cand), target[v]))
                scores[cand] = float(np.nanmean(inner))
            best = max(scores, key=scores.get)
        else:
            best = lam
        chosen[held] = best
        rho[held] = pi_stats.spearman(
            _fit_predict(blocks, target, tr, held, best), target[held])
    return rho, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="heldout_assays")
    ap.add_argument("--captures", default=str(W / "runs/xmodel_layers"))
    ap.add_argument("--tm-cache", default=str(W / "runs/tm_heldout16_{model}.npz"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cohort = Cohort.load(a.cohort)
    cohort.verify()
    out = {"cohort": a.cohort, "lambdas": list(LAMBDAS), "models": {}}

    for model in MODELS:
        TM = np.load(a.tm_cache.format(model=model))
        internal, rich, geomb, strain, target = {}, {}, {}, {}, {}
        for assay in cohort:
            p = Path(a.captures) / f"xm_{model}_r1_{assay.id}.npz"
            if not p.exists() or assay.id not in TM:
                continue
            cap = artifacts.load_capture(p, require_vectors=True)
            k = assay.id.split("_")[0]
            ca, ca_wt = cap.field("ca"), cap.field("ca_wt")
            pl, pls = cap.field("plddt_mean"), cap.field("plddt_site")
            pos = cap.field("pos")
            tm = np.asarray(TM[assay.id], float)
            internal[k] = cap.pair_row(-1)
            rich[k] = output_matrix(ca, ca_wt, tm, pl, pls, pos)
            geomb[k] = geometry_matrix(ca, ca_wt, tm, pl, pls, pos)
            strain[k] = strain_matrix(ca, ca_wt, pos)
            target[k] = np.asarray(cap.field("score"), float)
        names = sorted(target)
        if not names:
            continue

        # The genuinely nested union: every column of the 10 and of the 37,
        # deduplicated by NAME so the two local RMSDs actually enter.
        seen, keep10 = set(GEOMETRY_FEATURES), []
        for i, nm in enumerate(OUTPUT_FEATURES):
            if nm not in seen:
                keep10.append(i)
        union = {k: np.concatenate([geomb[k], rich[k][:, keep10]], 1)
                 for k in names}
        union_str = {k: np.concatenate([union[k], strain[k]], 1) for k in names}
        added = [OUTPUT_FEATURES[i] for i in keep10]

        BLOCKS = {
            "rich10": rich, "geometry37": geomb, "strain3": strain,
            f"union{union[names[0]].shape[1]} (nested)": union,
            f"union+strain{union_str[names[0]].shape[1]}": union_str,
            "internal128": internal,
        }
        print(f"\n=== {model}  ({len(names)} assays) ===")
        print(f"  the 10-feature block contributes {len(added)} column(s) the "
              f"37 lacks: {added}")

        res = {"n_assays": len(names), "union_added_from_rich10": added,
               "fixed_lambda_10": {}, "inner_swept_lambda": {}}
        for lab, blk in BLOCKS.items():
            r_fix, _ = loao(blk, target, lam=10.0)
            r_swp, chosen = loao(blk, target, lam=None)
            for tag, r, extra in (("fixed_lambda_10", r_fix, None),
                                  ("inner_swept_lambda", r_swp, chosen)):
                pt, lo, hi, _ = pi_stats.cluster_bootstrap(
                    {k: [r[k]] for k in names}, n_boot=10000, seed=0,
                    hierarchical=False)
                res[tag][lab] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                                 "per_assay": r}
                if extra:
                    res[tag][lab]["lambda_per_fold"] = extra
            print(f"  {lab:26s} lam=10 {res['fixed_lambda_10'][lab]['mean']:+.3f}"
                  f"   swept {res['inner_swept_lambda'][lab]['mean']:+.3f}"
                  f"   (lambdas {sorted(set(chosen.values()))})")

        # the paired gap that matters, under both regimes
        res["gaps"] = {}
        best_out = f"union+strain{union_str[names[0]].shape[1]}"
        for tag in ("fixed_lambda_10", "inner_swept_lambda"):
            for other in ("rich10", "geometry37", best_out):
                ri = res[tag]["internal128"]["per_assay"]
                ro = res[tag][other]["per_assay"]
                pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(
                    {k: [ri[k]] for k in names}, {k: [ro[k]] for k in names},
                    n_boot=10000, seed=0, hierarchical=False)
                wins = sum(ri[k] > ro[k] for k in names)
                res["gaps"][f"{tag}: internal128 - {other}"] = {
                    "gap": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins,
                    "n": len(names)}
                flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
                print(f"    {tag:19s} internal - {other:22s} {pt:+.3f} "
                      f"[{lo:+.3f}, {hi:+.3f}]  {wins}/{len(names)}{flag}")
        out["models"][model] = res

    proto = pi_protocol.protocol(
        script="baseline_fairness.py",
        design="emitted-baseline fairness: fixed lambda=10 against an inner "
               "leave-one-TRAINING-protein-out lambda sweep per block; a "
               "genuinely nested union of the 10- and 37-feature blocks; and "
               "prespecified effective strain",
        layer=pi_protocol.layers("final"),
        features={"rich": pi_protocol.features("emitted summaries",
                                               len(OUTPUT_FEATURES)),
                  "geometry": pi_protocol.features("emitted geometry",
                                                   len(GEOMETRY_FEATURES)),
                  "strain": pi_protocol.features(
                      f"effective strain, {STRAIN_CUTOFF:g} A neighbourhood",
                      len(STRAIN_FEATURES)),
                  "internal": pi_protocol.features("dz_vec final pair row", 128)},
        source=f"{a.captures}/xm_<model>_r1_*.npz",
        n_assays=len(cohort), lambda_grid=list(LAMBDAS),
        note="the inner sweep never sees the held-out assay; strain is "
             "distance-based and so frame-free")
    pi_archive.write_result(a.out, out, protocol=proto, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
