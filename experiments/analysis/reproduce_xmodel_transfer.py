"""From captured vectors to the cross-model transfer numbers, in one file.

The result this reproduces: on 16 held-out proteins, a probe on the trunk beats
the richest description of the structure the model emits, in Boltz-2, OpenFold3
and Protenix alike. `analyze_transfer.py` computes it alongside four feature
blocks, a k-sweep, bootstrap intervals and gap statistics. This computes only
the two numbers the claim rests on, so the recipe is legible end to end:

    features   dz_vec at the FINAL layer -- the 128 pair channels at the
               mutated position. The DIRECTION the row moved, not how far.
    baseline   output_rich: ten features describing what the model EMITTED --
               TM to wild type, RMSDs global and in 8 A / 12 A shells around
               the site, displacement at the site and its max, radius-of-
               gyration change, pLDDT chain mean, pLDDT at the site, and their
               difference.
    scaling    features and target z-scored WITHIN each assay
    protocol   leave-one-assay-out: fit on 15 assays, test on the 16th
    model      ridge, lambda = 10, intercept unpenalised
    statistic  within-assay Spearman on the held-out assay, meaned over the 16

WHY THE FEATURES COME FROM ACCESSORS AND NOT FROM KEYS. `dz_site` is a 128-wide
vector in the gym2s family and a per-layer NORM in this one; `plddt` is spelled
`plddt_mean` here. Reading either by name gets the wrong quantity without
failing, which is how a truncated probe and a norm-where-a-vector-was-promised
both shipped. `pair_row(-1)` asks the array what it is.

WHAT THE NUMBERS MEAN, AND WHAT THEY DO NOT. The internal figure is not
comparable to the within-assay position-split figure from `analyze_xmodel_io`:
that one fits a separate probe per protein and holds out residue POSITIONS, so
nothing transfers between proteins. Only leave-one-assay-out answers "does this
work on a protein the probe has never seen". The two must not be drawn on one
axis.

Nothing here is a ranking across models. Depths, distogram grids and alignment
handling all differ; only the within-model gap is meaningful.

    uv run python experiments/analysis/reproduce_xmodel_transfer.py --model boltz2
    uv run python experiments/analysis/reproduce_xmodel_transfer.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# `output_matrix` is imported rather than reimplemented. It superposes each
# variant onto the wild type with a Kabsch fit and reads displacements out of
# that frame; a second implementation would be a second thing to keep correct,
# and the whole point of this file is that the recipe is the one that ran.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax_harness"))

from compare_internal_output import output_matrix              # noqa: E402
from protein_interpretability import artifacts                 # noqa: E402
from protein_interpretability.analysis.probes import (         # noqa: E402
    leave_one_group_out,
)
from protein_interpretability.collection import Cohort         # noqa: E402

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
MODELS = ("boltz2", "of3", "protenix")
LAM = 10.0
TOL = 1e-6


def load_blocks(model, cohort, captures, tm_cache, run="r1"):
    """One entry per assay: the internal vectors, the emitted features, the score."""
    TM = np.load(tm_cache)
    internal, output, target = {}, {}, {}
    for assay in cohort:
        path = Path(captures) / f"xm_{model}_{run}_{assay.id}.npz"
        if not path.exists():
            raise SystemExit(f"missing {path.name}; collect it before analysing")
        cap = artifacts.load_capture(path, require_meta=True, require_vectors=True)
        if assay.id not in TM:
            raise SystemExit(
                f"{assay.id}: no TM in {Path(tm_cache).name}. Run "
                f"jax_harness/precompute_tm.py over these captures first -- TM "
                f"is computed from the model's OWN coordinates, so the cache is "
                f"per model and cannot be borrowed from another.")
        key = assay.id.split("_")[0]
        # The direction, found wherever this family keeps it.
        internal[key] = cap.pair_row(-1)                       # (n, 128)
        output[key] = output_matrix(
            cap.field("ca"), cap.field("ca_wt"), np.asarray(TM[assay.id], float),
            cap.field("plddt_mean"), cap.field("plddt_site"), cap.field("pos"))
        target[key] = np.asarray(cap.field("score"), float)
    return internal, output, target


def run_model(model, a, cohort):
    internal, output, target = load_blocks(
        model, cohort, a.captures, W / "runs" / f"tm_heldout16_{model}.npz")
    names = sorted(internal)

    # Leave one assay out, twice: once on the trunk, once on what was emitted.
    # The standardisation, the unpenalised intercept and the pooling live in the
    # library, so this file states the science and not the arithmetic.
    rho_int = leave_one_group_out(
        {k: {"X": internal[k], "y": target[k]} for k in names}, lam=a.lam)
    rho_out = leave_one_group_out(
        {k: {"X": output[k], "y": target[k]} for k in names}, lam=a.lam)

    gaps = {k: rho_int[k] - rho_out[k] for k in names}
    mean_int = float(np.mean([rho_int[k] for k in names]))
    mean_out = float(np.mean([rho_out[k] for k in names]))

    print(f"\n=== {model}  ({len(names)} assays, "
          f"{internal[names[0]].shape[1]} channels, lam={a.lam:g}) ===")
    print(f"{'assay':9s} {'internal':>9s} {'emitted':>9s} {'gap':>9s}  phenotype")
    stability = {x.id.split("_")[0]: "Tsuboyama_2023" in x.id for x in cohort}
    for k in sorted(names, key=lambda n: -gaps[n]):
        print(f"{k:9s} {rho_int[k]:>+9.3f} {rho_out[k]:>+9.3f} {gaps[k]:>+9.3f}"
              f"  {'stability' if stability[k] else 'OTHER'}")
    print(f"{'mean':9s} {mean_int:>+9.3f} {mean_out:>+9.3f} "
          f"{np.mean(list(gaps.values())):>+9.3f}")

    # The phenotype split: the whole of Boltz-2's fall from +0.758 is here.
    s = [rho_int[k] for k in names if stability[k]]
    o = [rho_int[k] for k in names if not stability[k]]
    if s and o:
        print(f"  internal on stability  {np.mean(s):+.3f}  (n={len(s)})")
        print(f"  internal on other      {np.mean(o):+.3f}  (n={len(o)})"
              f"   <- the four from other labs")

    # Check against the archived producer, per assay and pooled.
    ref_path = W / "runs" / f"transfer_heldout16_{model}.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())["predictors"]
        worst = max(abs(rho_int[k] - v)
                    for k, v in ref["internal_vec"]["per_assay"].items())
        worst_o = max(abs(rho_out[k] - v)
                      for k, v in ref["output_rich"]["per_assay"].items())
        print(f"  vs {ref_path.name}: worst per-assay difference "
              f"{max(worst, worst_o):.2e}")
        return max(worst, worst_o) <= TOL
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="boltz2")
    ap.add_argument("--all", action="store_true", help="every model in turn")
    ap.add_argument("--cohort", default="heldout_assays")
    ap.add_argument("--captures", default=str(W / "runs" / "xmodel_layers"))
    ap.add_argument("--lam", type=float, default=LAM)
    a = ap.parse_args()

    cohort = Cohort.load(a.cohort)
    cohort.verify()
    print(f"{cohort.name}: {len(cohort)} assays, inputs verified")

    ok = all(run_model(m, a, cohort) for m in (MODELS if a.all else [a.model]))
    print("\nreproduced to within 1e-6" if ok else
          "\nMISMATCH -- this recipe no longer produces the archived numbers")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
