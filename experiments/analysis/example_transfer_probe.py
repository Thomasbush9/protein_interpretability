"""A worked example: does the mutation signal predict stability, held out by assay?

This is a template rather than a result. It is deliberately the smallest script
that still does every part of the contract, so it can be copied and changed:

    cohort  ->  verify  ->  read captures  ->  statistic  ->  archived result

Runs on the login node: it loads no model and reads only artifacts.

    uv run python experiments/analysis/example_transfer_probe.py \
        --out /tmp/example_transfer.json

A MISSING CAPTURE IS A REFUSAL. This script used to print `skip` and carry on,
so a named twelve-assay cohort silently became an eleven-assay result whose
protocol block recorded n_assays: 11 -- true, and unreadable as an omission
unless you knew twelve were asked for. The pooled interval is over assays, so a
dropped assay moves the number the file reports. `--allow-partial` still allows
the exploratory run, and records exactly which assays were missing in both the
result and its protocol, so a partial result is partial on its face.
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


def capture_path(captures_dir, assay_id) -> Path:
    return Path(captures_dir) / f"gym2s_{assay_id}.npz"


def missing_captures(cohort, captures_dir) -> list[str]:
    """Which assays of the cohort have no capture on disk. Checked up front.

    Answerable before any array is read, so the refusal happens before the first
    per-assay number exists to be attached to.
    """
    return [assay.id for assay in cohort
            if not capture_path(captures_dir, assay.id).exists()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="basis_assays")
    ap.add_argument("--captures", default=str(W / "runs"))
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--allow-partial", action="store_true",
                    help="analyse the assays that ARE present. The omissions "
                         "are recorded in the result and the protocol; the "
                         "number is not comparable to a complete run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    # 1. State the cohort, and check it is still the one the manifest describes.
    #    Cheap, and the failure it catches -- an alignment or assay table
    #    rewritten in place -- is otherwise silent.
    cohort = Cohort.load(a.cohort)
    cohort.verify()
    print(f"{cohort.name}: {len(cohort)} assays, inputs verified")

    # 2. The cohort is the claim. An assay with no capture is a hole in it, and
    #    the pooled interval is over assays -- so dropping one changes the
    #    reported number, not just the sample size.
    missing = missing_captures(cohort, a.captures)
    if missing and not a.allow_partial:
        raise SystemExit(
            f"{len(missing)} of {len(cohort)} assays in cohort "
            f"{cohort.name!r} have no capture under {a.captures}: "
            f"{missing}\nCollect them, or pass --allow-partial to analyse the "
            f"rest and have the omission recorded in the result.")
    if missing:
        print(f"  PARTIAL: {len(missing)} assay(s) missing: {missing}")

    # 3. Read the captures through the seam, so a missing field fails loudly.
    rows = []
    for assay in cohort:
        path = capture_path(a.captures, assay.id)
        if not path.exists():
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

    # The count is checked against the cohort rather than against itself: a
    # capture that failed to load, or a duplicate id, would otherwise leave a
    # short table that reads exactly like a complete one.
    if len(rows) != len(cohort) - len(missing):
        raise SystemExit(
            f"{len(rows)} assay results from {len(cohort) - len(missing)} "
            f"captures found. Something was dropped between the file list and "
            f"the table; that is not a result.")

    # 4. Pool with an interval that respects the clustering by assay.
    # Clusters are ASSAYS, so an assay contributes once no matter how many
    # variants it has. Returns (point, lo, hi, n_clusters).
    per_assay = {r["assay"]: [r["spearman"]] for r in rows}
    mean, lo, hi, n_clusters = st.cluster_bootstrap(per_assay, n_boot=10000,
                                                    seed=0)

    result = {
        "per_assay": rows,
        "pooled": {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_assays": len(rows)},
        # Machine-readable, next to the number it changed.
        "partial": bool(missing),
        "cohort_size": len(cohort),
        "missing_assays": missing,
    }

    # 5. Write it through the seam. `protocol` is required, and it is what makes
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
            # n_assays alone cannot say whether 11 was the cohort or what was
            # left of it, so the cohort size and the omissions travel with it.
            cohort_size=len(cohort),
            partial=bool(missing),
            missing_assays=missing,
            n_boot=10000,
            seed=0,
            note=PROTOCOL_NOTE,
        ))
    print(f"\npooled rho = {mean:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(rows)} "
          f"of {len(cohort)} assays")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
