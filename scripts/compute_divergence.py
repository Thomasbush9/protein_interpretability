#!/usr/bin/env python
"""Compute per-layer, per-step divergence from a reference (WT) sequence.

Usage
-----
    python scripts/compute_divergence.py \
        --ref /path/to/wildtype/hidden_reps.pt \
        --mutants /path/to/high_effect/hidden_reps \
        --label high_effect \
        --tsv /path/to/high_effect/selected.tsv \
        --output /path/to/output_dir

The mutants directory should contain seq_XXXXX subdirectories, each with
a ``hidden_reps.pt`` file.  The script discovers them automatically and
preserves the seq_XXXXX id in the output.

When ``--tsv`` is provided, mutation positions are parsed from the TSV
and used to compute site-specific and windowed divergence metrics.

Output: ``<output_dir>/<label>_divergence.csv``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from protein_interpretability.rsa.batch import (
    compute_divergence,
    discover_mutant_dirs,
    load_mutation_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Compute distance-to-reference across layers and recycling steps.",
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
        "--tsv", type=Path, default=None,
        help="Path to selected.tsv with mutation positions (optional).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory (will be created if needed).",
    )
    parser.add_argument(
        "--metric", type=str, default="cosine",
        choices=["cosine", "l2"],
        help="Per-residue metric (default: cosine).",
    )
    parser.add_argument(
        "--windows", nargs="+", type=int, default=[0, 5, 10],
        help="Window half-widths around mutation site (default: 0 5 10).",
    )
    parser.add_argument(
        "--site", type=str, default="layer_s",
        help="Activation site (default: layer_s).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device (default: cpu).",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.ref.exists():
        logger.error("Reference file not found: %s", args.ref)
        sys.exit(1)
    if not args.mutants.is_dir():
        logger.error("Mutants directory not found: %s", args.mutants)
        sys.exit(1)

    # Load mutation map if provided
    mutation_map = None
    if args.tsv is not None:
        if not args.tsv.exists():
            logger.error("TSV file not found: %s", args.tsv)
            sys.exit(1)
        mutation_map = load_mutation_map(args.tsv)
        logger.info("Loaded mutation positions for %d sequences", len(mutation_map))

    # Discover mutant dirs
    dirs = discover_mutant_dirs(args.mutants)
    logger.info("Found %d mutant directories in %s", len(dirs), args.mutants)
    if not dirs:
        logger.error("No hidden_reps.pt files found under %s", args.mutants)
        sys.exit(1)

    # Compute
    df = compute_divergence(
        wt_reps_path=args.ref,
        mutant_dirs={args.label: dirs},
        mutation_map=mutation_map,
        metric=args.metric,
        windows=args.windows,
        site=args.site,
        device=args.device,
    )

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"{args.label}_divergence.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
