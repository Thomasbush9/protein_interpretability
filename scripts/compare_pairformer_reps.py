#!/usr/bin/env python3
"""Compare each ``seq_XXXXX/hidden_reps.pt`` in a directory to a reference.

For a chosen recycling step, one or more layer-types (``layer_z``,
``tri_att_start``, ``tri_att_end``, ``layer_s``), and metric (``cosine``,
``l2``, ``norm_diff``), writes one CSV per (seq, layer-type) with
per-layer summary stats.

Usage
-----
::

    python scripts/compare_pairformer_reps.py \\
        --ref   /path/to/wt/hidden_reps.pt \\
        --mutants /path/to/p40/pairformer \\
        --step 0 \\
        --layer-type layer_z tri_att_start \\
        --metric cosine \\
        --output /path/to/out_dir

Output
------
``<output>/seq_XXXXX__<layer_type>.csv`` — columns: ``layer, mean,
median, min, max, std, n_positions``. Aggregation across seqs is left
to the caller.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from protein_interpretability.rsa.pair_metrics import (
    METRICS,
    PAIR_SITES,
    TOKEN_SITES,
    compare_reps,
    load_reps,
)


_SEQ_DIR_RE = re.compile(r"seq_\d+")


def discover_seq_dirs(base: Path) -> list[Path]:
    """Return ``seq_XXXXX`` dirs containing ``hidden_reps.pt`` (sorted)."""
    return sorted(
        d for d in base.iterdir()
        if d.is_dir()
        and _SEQ_DIR_RE.fullmatch(d.name)
        and (d / "hidden_reps.pt").exists()
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ref", required=True, type=Path,
                    help="Reference hidden_reps.pt (e.g. wildtype).")
    ap.add_argument("--mutants", required=True, type=Path,
                    help="Directory containing seq_XXXXX/hidden_reps.pt subdirs.")
    ap.add_argument("--step", required=True, type=int,
                    help="Recycling step index (maps to step_<N> key).")
    ap.add_argument("--layer-type", nargs="+", default=["layer_z"],
                    help=f"One or more site/layer types (default: layer_z). "
                         f"Pairwise: {PAIR_SITES}. Per-token: {TOKEN_SITES}.")
    ap.add_argument("--metric", default="cosine", choices=sorted(METRICS.keys()),
                    help="Per-position metric (default: cosine).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory for per-seq CSVs.")
    ap.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    args = ap.parse_args()

    if not args.ref.exists():
        print(f"[error] ref not found: {args.ref}", file=sys.stderr)
        sys.exit(1)
    if not args.mutants.is_dir():
        print(f"[error] mutants dir not found: {args.mutants}", file=sys.stderr)
        sys.exit(1)

    seq_dirs = discover_seq_dirs(args.mutants)
    if not seq_dirs:
        print(f"[error] no seq_XXXXX/hidden_reps.pt under {args.mutants}",
              file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    ref = load_reps(args.ref, device=args.device)

    print(f"[info] ref={args.ref}")
    print(f"[info] step={args.step} layer_types={args.layer_type} metric={args.metric}")
    print(f"[info] {len(seq_dirs)} seq dirs x {len(args.layer_type)} layer_types "
          f"-> {args.output}")

    n_written = 0
    for i, d in enumerate(seq_dirs, 1):
        mut = load_reps(d / "hidden_reps.pt", device=args.device)
        for lt in args.layer_type:
            df = compare_reps(
                ref, mut,
                step=args.step,
                layer_type=lt,
                metric=args.metric,
            )
            df.to_csv(args.output / f"{d.name}__{lt}.csv", index=False)
            n_written += 1
        if i % 10 == 0 or i == len(seq_dirs):
            print(f"[done] {i}/{len(seq_dirs)} seqs")

    print(f"[ok] wrote {n_written} CSVs to {args.output}")


if __name__ == "__main__":
    main()
