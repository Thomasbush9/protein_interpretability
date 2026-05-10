from __future__ import annotations

import csv
import re
from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlretrieve

from protein_interpretability.scoring import (
    extract_residue_coordinates,
    extract_residue_sequence,
    load_structure,
    path_rmsd,
    path_tm_score,
    rmsd,
    tm_score,
)

SEQUENCE_INDEX_PATTERN = re.compile(r"seq_(\d+)")
SUPPORTED_STRUCTURE_SUFFIXES = (".cif", ".pdb")
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb}.cif"
PAIRS_FIELDNAMES = [
    "seq_id",
    "idx_tableS1",
    "fold1",
    "fold2",
    "chain_used",
    "primary_fold",
    "seq_len",
    "predicted_path",
    "tm_g1",
    "tm_g2",
    "rmsd_g1",
    "rmsd_g2",
]


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Score predicted .cif/.pdb structures and save TM-score/RMSD to CSV. "
            "Two modes: (1) single-reference (--ref) — score a cohort against one "
            "structure; (2) pairs (--pairs-manifest) — score each prediction against "
            "two refs (G1, G2) drawn from a manifest, e.g. fold-switch G1/G2 pairs."
        )
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=None,
        help="Single-reference mode: path to one .cif/.pdb to score everything against.",
    )
    parser.add_argument(
        "--pairs-manifest",
        type=Path,
        default=None,
        help=(
            "Pairs mode: TSV with columns "
            "seq_id, idx_tableS1, fold1, fold2, chain_used, ... "
            "Each prediction is scored against fold1 (G1) and fold2 (G2)."
        ),
    )
    parser.add_argument(
        "--refs-dir",
        type=Path,
        default=None,
        help="Pairs mode: directory holding <pdb>.cif reference files.",
    )
    parser.add_argument(
        "--download-missing-refs",
        action="store_true",
        help="Pairs mode: fetch missing <pdb>.cif from RCSB into --refs-dir.",
    )
    parser.add_argument(
        "--predicted-dir",
        type=Path,
        required=True,
        help="Directory to search recursively for predicted .cif/.pdb files.",
    )
    parser.add_argument(
        "--model-subdir",
        default=None,
        help=(
            "Optional model subdirectory to restrict matches to, such as "
            "'boltz' or 'esmfold'."
        ),
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
        help="Single-reference mode: chain id if structures are multichain.",
    )
    parser.add_argument(
        "--normalize-by",
        choices=("reference", "predicted"),
        default="reference",
        help="Which structure length to use for TM-score normalization.",
    )
    return parser


def find_structure_files(
    predicted_dir: Path, model_subdir: str | None = None
) -> list[Path]:
    paths = (
        path
        for suffix in SUPPORTED_STRUCTURE_SUFFIXES
        for path in predicted_dir.rglob(f"*{suffix}")
    )
    if model_subdir is not None:
        paths = (
            path
            for path in paths
            if any(part == model_subdir for part in path.relative_to(predicted_dir).parts)
        )
    return sorted(path for path in paths if path.is_file())


def extract_sequence_idx(path: Path) -> int | str:
    candidates = (path.stem, *path.parts)
    for candidate in candidates:
        match = SEQUENCE_INDEX_PATTERN.search(candidate)
        if match is not None:
            label = match.group(1)
            return int(label) if label.isdigit() else label

    raise ValueError(
        f"Could not parse sequence index from '{path}'. "
        "Expected a name like 'seq_19851_model_24.cif' or "
        "a parent directory like 'seq_19851/esmfold/structure.pdb'."
    )


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

    def sort_key(row: dict[str, str | int | float]) -> tuple[int, int | str]:
        idx = row["sequence_idx"]
        return (1, idx) if isinstance(idx, int) else (0, str(idx))

    rows.sort(key=sort_key)
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


def parse_fold_field(fold: str) -> tuple[str, str]:
    """Parse a TableS1 fold token like '2frhA' into ('2frh', 'A')."""
    fold = fold.strip()
    if len(fold) < 5:
        raise ValueError(
            f"Cannot parse fold field {fold!r}; expected '<pdb><chain>' (e.g. '2frhA')."
        )
    return fold[:4].lower(), fold[4:]


def ensure_reference_cif(
    pdb_id: str, refs_dir: Path, *, download_missing: bool
) -> Path:
    """Return path to refs_dir/<pdb>.cif, optionally fetching from RCSB."""
    refs_dir.mkdir(parents=True, exist_ok=True)
    target = refs_dir / f"{pdb_id}.cif"
    if target.exists():
        return target
    if not download_missing:
        raise FileNotFoundError(
            f"Reference {target} missing. Pass --download-missing-refs or stage it manually."
        )
    url = RCSB_CIF_URL.format(pdb=pdb_id)
    print(f"[refs] downloading {url}")
    urlretrieve(url, target)
    return target


def find_prediction_for_seq(
    predicted_dir: Path, seq_id: str, model_subdir: str | None
) -> Path | None:
    """Locate the prediction .cif for a given seq_id, e.g.
    outputs/sequences/seq_NNNNN/boltz/seq_NNNNN_model_XX.cif."""
    candidates: list[Path] = []
    for suffix in SUPPORTED_STRUCTURE_SUFFIXES:
        candidates.extend(predicted_dir.rglob(f"{seq_id}/**/*{suffix}"))
    if model_subdir is not None:
        candidates = [p for p in candidates if model_subdir in p.parts]
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    # Stable ordering: prefer 'model_0' if present, else lexicographic.
    candidates.sort(key=lambda p: ("model_0" not in p.stem, str(p)))
    return candidates[0]


def score_pair(
    predicted_path: Path,
    ref1_path: Path,
    chain1: str,
    ref2_path: Path,
    chain2: str,
    normalize_by: str,
) -> tuple[float, float, float, float]:
    """Score one prediction against two references with their auth chains."""
    pred_struct = load_structure(predicted_path)
    pred_coords, _ = extract_residue_coordinates(pred_struct, chain_id=None)
    pred_seq = extract_residue_sequence(pred_struct, chain_id=None)

    def _score_against(ref_path: Path, chain: str) -> tuple[float, float]:
        ref_struct = load_structure(ref_path)
        ref_coords, _ = extract_residue_coordinates(ref_struct, chain_id=chain)
        ref_seq = extract_residue_sequence(ref_struct, chain_id=chain)
        tm = tm_score(
            ref_coords, pred_coords, ref_seq, pred_seq, normalize_by=normalize_by
        )
        rmsd_val = rmsd(ref_coords, pred_coords, ref_seq, pred_seq)
        return tm, rmsd_val

    tm_g1, rmsd_g1 = _score_against(ref1_path, chain1)
    tm_g2, rmsd_g2 = _score_against(ref2_path, chain2)
    return tm_g1, tm_g2, rmsd_g1, rmsd_g2


def score_pairs(
    predicted_dir: Path,
    pairs_manifest: Path,
    refs_dir: Path,
    *,
    model_subdir: str | None,
    normalize_by: str,
    download_missing: bool,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    with pairs_manifest.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"seq_id", "fold1", "fold2"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(
                f"--pairs-manifest is missing required columns: {sorted(missing_cols)}"
            )

        for entry in reader:
            seq_id = entry["seq_id"].strip()
            fold1 = entry["fold1"].strip()
            fold2 = entry["fold2"].strip()
            if not (seq_id and fold1 and fold2):
                continue
            chain_used = entry.get("chain_used", "").strip() or fold1
            primary_fold = "fold2" if chain_used == fold2 else "fold1"
            seq_len_raw = entry.get("seq_len", "").strip()
            seq_len: int | str = int(seq_len_raw) if seq_len_raw.isdigit() else ""

            predicted_path = find_prediction_for_seq(
                predicted_dir, seq_id, model_subdir
            )
            if predicted_path is None:
                print(f"[skip] {seq_id}: no prediction found under {predicted_dir}")
                continue

            pdb1, chain1 = parse_fold_field(fold1)
            pdb2, chain2 = parse_fold_field(fold2)
            ref1 = ensure_reference_cif(
                pdb1, refs_dir, download_missing=download_missing
            )
            ref2 = ensure_reference_cif(
                pdb2, refs_dir, download_missing=download_missing
            )

            try:
                tm_g1, tm_g2, rmsd_g1, rmsd_g2 = score_pair(
                    predicted_path,
                    ref1,
                    chain1,
                    ref2,
                    chain2,
                    normalize_by=normalize_by,
                )
            except Exception as exc:  # noqa: BLE001 — report and continue cohort
                print(f"[error] {seq_id}: {type(exc).__name__}: {exc}")
                continue

            rows.append(
                {
                    "seq_id": seq_id,
                    "idx_tableS1": entry.get("idx_tableS1", ""),
                    "fold1": fold1,
                    "fold2": fold2,
                    "chain_used": chain_used,
                    "primary_fold": primary_fold,
                    "seq_len": seq_len,
                    "predicted_path": str(predicted_path),
                    "tm_g1": tm_g1,
                    "tm_g2": tm_g2,
                    "rmsd_g1": rmsd_g1,
                    "rmsd_g2": rmsd_g2,
                }
            )

    rows.sort(key=lambda row: str(row["seq_id"]))
    return rows


def write_pair_scores(
    rows: list[dict[str, str | int | float]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    if not args.predicted_dir.is_dir():
        raise NotADirectoryError(
            f"Predicted structure directory not found: {args.predicted_dir}"
        )
    if (args.ref is None) == (args.pairs_manifest is None):
        parser.error("Provide exactly one of --ref or --pairs-manifest.")

    output_path = args.output_dir / args.output_name

    if args.pairs_manifest is not None:
        if args.refs_dir is None:
            parser.error("--pairs-manifest requires --refs-dir.")
        if not args.pairs_manifest.is_file():
            raise FileNotFoundError(
                f"Pairs manifest not found: {args.pairs_manifest}"
            )
        rows = score_pairs(
            predicted_dir=args.predicted_dir,
            pairs_manifest=args.pairs_manifest,
            refs_dir=args.refs_dir,
            model_subdir=args.model_subdir,
            normalize_by=args.normalize_by,
            download_missing=args.download_missing_refs,
        )
        if not rows:
            raise RuntimeError(
                "No pairs were scored. Check predicted-dir, manifest, and model-subdir."
            )
        write_pair_scores(rows, output_path)
        print(f"Wrote {len(rows)} pair scores to {output_path}")
        return

    if not args.ref.is_file():
        raise FileNotFoundError(f"Reference structure not found: {args.ref}")

    predicted_paths = find_structure_files(
        args.predicted_dir, model_subdir=args.model_subdir
    )
    if not predicted_paths:
        model_detail = (
            f" under model subdir '{args.model_subdir}'"
            if args.model_subdir is not None
            else ""
        )
        raise FileNotFoundError(
            f"No .cif or .pdb files were found under {args.predicted_dir}"
            f"{model_detail}."
        )

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
