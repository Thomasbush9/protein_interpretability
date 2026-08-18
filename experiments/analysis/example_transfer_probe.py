"""A worked example: does the mutation signal predict stability, held out by assay?

This is a template rather than a result. It is deliberately the smallest script
that still does every part of the contract, so it can be copied and changed:

    cohort  ->  verify  ->  read captures  ->  statistic  ->  archived result

Runs on the login node: it loads no model and reads only artifacts.

    uv run python experiments/analysis/example_transfer_probe.py \
        --out /tmp/example_transfer.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from protein_interpretability import artifacts
from protein_interpretability.analysis import statistics as st
from protein_interpretability.collection import Cohort
from protein_interpretability.experiments import protocol as P

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")

PROTOCOL_NOTE = (
    "EXAMPLE, not a reported result: a single magnitude predictor with no "
    "cross-validation and no permutation test. It is here to show the shape of "
    "a script, and it carries a protocol block for the same reason every result "
    "does -- so nobody can quote it without seeing what it is."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="basis_assays")
    ap.add_argument("--captures", default=str(W / "runs"))
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # 1. State the cohort, and check it is still the one the manifest describes.
    #    Cheap, and the failure it catches -- an alignment or assay table
    #    rewritten in place -- is otherwise silent.
    cohort = Cohort.load(a.cohort)
    cohort.verify()
    print(f"{cohort.name}: {len(cohort)} assays, inputs verified")

    # 2. Read the captures through the seam, so a missing field fails loudly.
    rows = []
    for assay in cohort:
        path = Path(a.captures) / f"gym2s_{assay.id}.npz"
        if not path.exists():
            print(f"  skip {assay.id}: no capture at {path.name}")
            continue
        cap = artifacts.load_capture(path)
        dz = np.asarray(cap.field("dz_site"))        # [n_variants, n_layers, 128]
        # The assay score travels WITH the capture, one row per variant, rather
        # than being re-read from the CSV and re-aligned by mutant name. The
        # capture also carries `mutant` and `pos` if you need to join on them.
        score = np.asarray(cap.field("score"), float)

        # A deliberately trivial predictor: how far the pair row moved.
        pred = np.linalg.norm(dz[:, a.layer, :], axis=-1)
        rho = st.spearman(pred, score)
        rows.append({"assay": assay.id, "n": int(len(score)), "spearman": rho})
        print(f"  {assay.id:34s} n={len(score):5d}  rho={rho:+.3f}")

    if not rows:
        raise SystemExit("no captures found; nothing to report")

    # 3. Pool with an interval that respects the clustering by assay.
    # Clusters are ASSAYS, so an assay contributes once no matter how many
    # variants it has. Returns (point, lo, hi, n_clusters).
    per_assay = {r["assay"]: [r["spearman"]] for r in rows}
    mean, lo, hi, n_clusters = st.cluster_bootstrap(per_assay, n_boot=10000,
                                                    seed=0)

    result = {
        "per_assay": rows,
        "pooled": {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_assays": len(rows)},
    }

    # 4. Write it through the seam. `protocol` is required, and it is what makes
    #    the number comparable to anything else later.
    artifacts.write_result(
        Path(a.out), result,
        protocol=P.protocol(
            # script, design, layer, features, source and n_assays are all
            # REQUIRED -- protocol() raises rather than writing a block that
            # cannot say what the number is comparable to.
            script=Path(__file__).name,
            design="per-assay Spearman of a dz magnitude against the assay "
                   "score, pooled by cluster bootstrap over assays",
            layer=P.layers("final" if a.layer == -1 else str(a.layer)),
            features=P.features("dz_site row magnitude", 128, kept=1),
            source=str(Path(a.captures) / "gym2s_<assay>.npz"),
            n_assays=len(rows),
            cohort=cohort.name,
            n_boot=10000,
            seed=0,
            note=PROTOCOL_NOTE,
        ))
    print(f"\npooled rho = {mean:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(rows)} assays")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
