"""Augment a pairs manifest with a `fold_switch_region` column.

Porter's TableS1 has one row per fold-switcher pair with column C =
"Sequence of fold-switching region". This script joins that column into
the project's pairs manifest (typically `identities.tsv`), keyed by
`idx_tableS1`.

Usage:
    uv run python scripts/foldswitch_make_regions_manifest.py \\
        --tables1 /path/to/TableS1.xlsx \\
        --manifest /path/to/identities.tsv \\
        --out /path/to/identities_with_regions.tsv

Out-of-set rows (with no idx_tableS1 entry in TableS1) are written
through unchanged with an empty region; fill them in manually.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables1", type=Path, required=True,
                        help="Path to Porter TableS1.xlsx")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Existing pairs manifest TSV "
                             "(seq_id, idx_tableS1, fold1, fold2, ...)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output TSV (input columns + fold_switch_region)")
    args = parser.parse_args()

    # TableS1 has header rows: row0 = column labels ("Fold1", "Fold2",
    # "Sequence of fold-switching region"). idx_tableS1 in the project's
    # manifest is 1-based by row order in TableS1, so we just enumerate.
    t = pd.read_excel(args.tables1, sheet_name=0, header=0)
    region_col = next(
        (c for c in t.columns if "fold-switching" in str(c).lower() or "fold switching" in str(c).lower()),
        t.columns[-1],
    )
    print(f"TableS1: using column {region_col!r} for fold-switch region")
    regions = {i + 1: str(s).strip() for i, s in enumerate(t[region_col].tolist())}
    print(f"  loaded {sum(1 for v in regions.values() if v)} non-empty regions")

    with args.manifest.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "fold_switch_region" not in fieldnames:
        fieldnames.append("fold_switch_region")

    n_filled = 0
    for row in rows:
        idx_raw = row.get("idx_tableS1", "").strip()
        if not idx_raw.isdigit():
            row.setdefault("fold_switch_region", "")
            continue
        idx = int(idx_raw)
        region = regions.get(idx, "")
        row["fold_switch_region"] = region
        if region:
            n_filled += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out} ({n_filled} with region).")


if __name__ == "__main__":
    main()
