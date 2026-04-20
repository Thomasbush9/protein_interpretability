import copy
from pathlib import Path

import pytest
from Bio.PDB import PDBIO

from protein_interpretability.score_sequences import (
    extract_sequence_idx,
    find_structure_files,
    score_predictions,
)
from protein_interpretability.scoring import load_structure


ASSET_STRUCTURE = (
    Path(__file__).resolve().parents[1] / "assets" / "seq_02765_model_24.cif"
)


def _write_pdb_from_asset(destination: Path) -> None:
    structure = copy.deepcopy(load_structure(ASSET_STRUCTURE))
    for index, chain in enumerate(structure.get_chains()):
        chain.id = chr(ord("A") + index)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(destination))


def test_find_structure_files_includes_cif_and_pdb(tmp_path: Path) -> None:
    cif_dir = tmp_path / "seq_00002" / "predictions"
    cif_dir.mkdir(parents=True)
    cif_path = cif_dir / "seq_00002_model_0.cif"
    cif_path.write_bytes(ASSET_STRUCTURE.read_bytes())

    pdb_dir = tmp_path / "seq_00001" / "predictions"
    pdb_dir.mkdir(parents=True)
    pdb_path = pdb_dir / "seq_00001_model_0.pdb"
    _write_pdb_from_asset(pdb_path)

    ignored_path = tmp_path / "seq_00003_model_0.txt"
    ignored_path.write_text("ignore me")

    assert find_structure_files(tmp_path) == [pdb_path, cif_path]


def test_find_structure_files_can_filter_by_model_subdir(tmp_path: Path) -> None:
    boltz_dir = tmp_path / "seq_00002" / "boltz"
    boltz_dir.mkdir(parents=True)
    boltz_path = boltz_dir / "seq_00002_model_0.cif"
    boltz_path.write_bytes(ASSET_STRUCTURE.read_bytes())

    esmfold_dir = tmp_path / "seq_00001" / "esmfold"
    esmfold_dir.mkdir(parents=True)
    esmfold_path = esmfold_dir / "structure.pdb"
    _write_pdb_from_asset(esmfold_path)

    assert find_structure_files(tmp_path, model_subdir="esmfold") == [esmfold_path]
    assert find_structure_files(tmp_path, model_subdir="boltz") == [boltz_path]


def test_score_predictions_supports_mixed_prediction_formats(tmp_path: Path) -> None:
    cif_path = tmp_path / "seq_00010_model_0.cif"
    cif_path.write_bytes(ASSET_STRUCTURE.read_bytes())

    pdb_path = tmp_path / "seq_00002_model_0.pdb"
    _write_pdb_from_asset(pdb_path)

    rows = score_predictions(
        reference_path=ASSET_STRUCTURE,
        predicted_paths=[cif_path, pdb_path],
        chain_id=None,
        normalize_by="reference",
    )

    assert [row["sequence_idx"] for row in rows] == [2, 10]
    assert [Path(str(row["predicted_path"])).suffix for row in rows] == [
        ".pdb",
        ".cif",
    ]
    assert rows[0]["tm_score"] == pytest.approx(1.0, abs=1e-6)
    assert rows[1]["tm_score"] == pytest.approx(1.0, abs=1e-6)
    assert rows[0]["rmsd"] == pytest.approx(0.0, abs=1e-3)
    assert rows[1]["rmsd"] == pytest.approx(0.0, abs=1e-6)


def test_extract_sequence_idx_uses_parent_directory_when_needed() -> None:
    path = Path("seq_19851/esmfold/structure.pdb")

    assert extract_sequence_idx(path) == 19851


def test_extract_sequence_idx_mentions_supported_extensions() -> None:
    path = Path("model_0.pdb")

    try:
        extract_sequence_idx(path)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected extract_sequence_idx to raise ValueError")

    assert ".cif" in message
    assert "seq_19851/esmfold/structure.pdb" in message
