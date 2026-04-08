import copy
from pathlib import Path

from Bio.PDB import PDBIO

from protein_interpretability.scoring import load_structure


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
