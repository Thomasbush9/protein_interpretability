#!/usr/bin/env python
"""Sliding-window spatial divergence profile along the sequence.

For each mutant × layer × step, slides a window across residues and reports
the mean per-residue cosine divergence inside each window. Position-agnostic
(does not use mutation sites), so the result is a spatial profile that lets
you see where divergence concentrates along the protein.

Usage
-----
    python scripts/compute_spatial_divergence.py \
        --ref /path/to/wildtype/hidden_reps.pt \
        --mutants /path/to/high_effect/hidden_reps \
        --label high_effect \
        --output /path/to/output_dir \
        --window 20 --stride 10 --trim-right 20

Output: ``<output_dir>/<label>_spatial_divergence.parquet`` (and .csv).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from protein_interpretability.rsa.batch import (
    compute_spatial_divergence,
    discover_mutant_dirs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Sliding-window spatial divergence profile.",
    )
    parser.add_argument(
        "--ref", type=Path, required=True,
        help="Path to reference hidden_reps.pt (e.g. wildtype).",
    )
    parser.add_argument(
        "--mutants", type=Path, required=True,
        help="Directory containing seq_XXXXX/hidden_reps.pt subdirs.",
    )
    parser.add_argument(
        "--label", type=str, required=True,
        help="Class label for this set (e.g. 'high_effect', 'neutral').",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory (will be created if needed).",
    )
    parser.add_argument("--window", type=int, default=20, help="Window size (default: 20).")
    parser.add_argument("--stride", type=int, default=10, help="Stride between windows (default: 10).")
    parser.add_argument("--trim-left", type=int, default=0, help="Residues to exclude at N-terminus.")
    parser.add_argument("--trim-right", type=int, default=20, help="Residues to exclude at C-terminus.")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--site", type=str, default="layer_s")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Skip writing CSV (parquet only). CSV can be large.",
    )
    args = parser.parse_args()

    if not args.ref.exists():
        logger.error("Reference file not found: %s", args.ref)
        sys.exit(1)
    if not args.mutants.is_dir():
        logger.error("Mutants directory not found: %s", args.mutants)
        sys.exit(1)

    dirs = discover_mutant_dirs(args.mutants)
    logger.info("Found %d mutant directories", len(dirs))
    if not dirs:
        logger.error("No hidden_reps.pt found")
        sys.exit(1)

    df = compute_spatial_divergence(
        wt_reps_path=args.ref,
        mutant_dirs={args.label: dirs},
        window=args.window,
        stride=args.stride,
        trim_left=args.trim_left,
        trim_right=args.trim_right,
        metric=args.metric,
        site=args.site,
        device=args.device,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.label}_spatial_divergence"
    parquet_path = args.output / f"{stem}.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved %d rows to %s", len(df), parquet_path)

    if not args.no_csv:
        csv_path = args.output / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved CSV to %s (%.1f MB)", csv_path, csv_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
