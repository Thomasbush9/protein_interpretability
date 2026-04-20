from __future__ import annotations

import csv
import re
from argparse import ArgumentParser
from pathlib import Path

from protein_interpretability.scoring import path_rmsd, path_tm_score

SEQUENCE_INDEX_PATTERN = re.compile(r"seq_(\d+)")
SUPPORTED_STRUCTURE_SUFFIXES = (".cif", ".pdb")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Score all predicted .cif/.pdb structures in a directory against a "
            "reference structure and save TM-score/RMSD to CSV."
        )
    )
    parser.add_argument(
        "--ref",
        type=Path,
        required=True,
        help="Path to the reference .cif or .pdb structure.",
    )
    parser.add_argument(
        "--predicted-dir",
        type=Path,
        required=True,
        help="Directory to search recursively for predicted .cif/.pdb files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the CSV file will be written.",
    )
    parser.add_argument(
        "--output-name",
        default="structure_scores.csv",
        help="Name of the output CSV file.",
    )
    parser.add_argument(
        "--chain-id",
        default=None,
        help="Optional chain identifier to score if the structures are multichain.",
    )
    parser.add_argument(
        "--normalize-by",
        choices=("reference", "predicted"),
        default="reference",
        help="Which structure length to use for TM-score normalization.",
    )
    return parser


def find_structure_files(predicted_dir: Path) -> list[Path]:
    paths = (
        path
        for suffix in SUPPORTED_STRUCTURE_SUFFIXES
        for path in predicted_dir.rglob(f"*{suffix}")
    )
    return sorted(path for path in paths if path.is_file())


def extract_sequence_idx(path: Path) -> int:
    match = SEQUENCE_INDEX_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(
            f"Could not parse sequence index from '{path.name}'. "
            "Expected a name like 'seq_19851_model_24.cif' or "
            "'seq_19851_model_24.pdb'."
        )
    return int(match.group(1))


def score_predictions(
    reference_path: Path,
    predicted_paths: list[Path],
    chain_id: str | None,
    normalize_by: str,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []

    for predicted_path in predicted_paths:
        sequence_idx = extract_sequence_idx(predicted_path)
        rows.append(
            {
                "sequence_idx": sequence_idx,
                "predicted_path": str(predicted_path),
                "tm_score": path_tm_score(
                    reference_path=reference_path,
                    predicted_path=predicted_path,
                    chain_id=chain_id,
                    normalize_by=normalize_by,
                ),
                "rmsd": path_rmsd(
                    reference_path=reference_path,
                    predicted_path=predicted_path,
                    chain_id=chain_id,
                ),
            }
        )

    rows.sort(key=lambda row: int(row["sequence_idx"]))
    return rows


def write_scores(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence_idx", "predicted_path", "tm_score", "rmsd"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    if not args.ref.is_file():
        raise FileNotFoundError(f"Reference structure not found: {args.ref}")
    if not args.predicted_dir.is_dir():
        raise NotADirectoryError(
            f"Predicted structure directory not found: {args.predicted_dir}"
        )

    predicted_paths = find_structure_files(args.predicted_dir)
    if not predicted_paths:
        raise FileNotFoundError(
            f"No .cif or .pdb files were found under {args.predicted_dir}."
        )

    output_path = args.output_dir / args.output_name
    rows = score_predictions(
        reference_path=args.ref,
        predicted_paths=predicted_paths,
        chain_id=args.chain_id,
        normalize_by=args.normalize_by,
    )
    write_scores(rows, output_path)
    print(f"Wrote {len(rows)} scores to {output_path}")


if __name__ == "__main__":
    main()
