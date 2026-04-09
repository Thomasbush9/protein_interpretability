import copy
from pathlib import Path

import pytest
from Bio.PDB import PDBIO
from Bio.PDB.Chain import Chain

from protein_interpretability.scoring import (
    load_structure,
    path_rmsd,
    path_tm_score,
    structure_rmsd,
    structure_tm_score,
)


ASSET_STRUCTURE = (
    Path(__file__).resolve().parents[1] / "assets" / "seq_02765_model_24.cif"
)


def test_load_structure_supports_mmcif_and_pdb(tmp_path: Path) -> None:
    cif_structure = load_structure(ASSET_STRUCTURE)
    cif_chains = list(cif_structure.get_chains())
    assert cif_chains
    assert cif_structure.id == "seq_02765_model_24"

    pdb_ready_structure = copy.deepcopy(cif_structure)
    for index, chain in enumerate(pdb_ready_structure.get_chains()):
        chain.id = chr(ord("A") + index)

    pdb_path = tmp_path / "seq_02765_model_24.pdb"
    io = PDBIO()
    io.set_structure(pdb_ready_structure)
    io.save(str(pdb_path))

    pdb_structure = load_structure(pdb_path)
    pdb_chains = list(pdb_structure.get_chains())

    assert pdb_chains
    assert len(cif_chains) == len(pdb_chains)
    assert pdb_structure.id == "seq_02765_model_24"


def test_path_tm_score_self_comparison_is_one() -> None:
    score = path_tm_score(ASSET_STRUCTURE, ASSET_STRUCTURE)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_path_rmsd_self_comparison_is_zero() -> None:
    value = path_rmsd(ASSET_STRUCTURE, ASSET_STRUCTURE)
    assert value == pytest.approx(0.0, abs=1e-6)


def test_structure_tm_score_and_rmsd_self_comparison() -> None:
    structure = load_structure(ASSET_STRUCTURE)
    assert structure_tm_score(structure, structure) == pytest.approx(1.0, abs=1e-6)
    assert structure_rmsd(structure, structure) == pytest.approx(0.0, abs=1e-6)


def test_multi_chain_structure_requires_explicit_chain_id() -> None:
    structure = copy.deepcopy(load_structure(ASSET_STRUCTURE))
    model = next(structure.get_models())

    # Duplicate the single chain under a new id so the structure has two chains.
    original_chain = next(model.get_chains())
    original_chain.id = "A"
    extra = Chain("B")
    for residue in original_chain.get_residues():
        extra.add(residue.copy())
    model.add(extra)

    with pytest.raises(ValueError, match="2 chains"):
        structure_tm_score(structure, structure)

    # Explicit chain id should resolve the ambiguity.
    assert structure_tm_score(structure, structure, chain_id="A") == pytest.approx(
        1.0, abs=1e-6
    )
