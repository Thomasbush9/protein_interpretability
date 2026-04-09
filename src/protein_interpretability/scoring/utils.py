from pathlib import Path

import numpy as np
from Bio.Data import IUPACData
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure
from tmtools import tm_align


def load_structure(path: str | Path, structure_id: str | None = None) -> Structure:
    """Load a protein structure from a PDB or mmCIF file."""
    structure_path = Path(path)
    suffix = structure_path.suffix.lower()

    if suffix in {".pdb", ".ent"}:
        parser = PDBParser(QUIET=True)
    elif suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(
            f"Unsupported structure format '{suffix}'. Expected .pdb, .ent, .cif, or .mmcif."
        )

    return parser.get_structure(structure_id or structure_path.stem, str(structure_path))


def extract_residue_coordinates(
    structure: Structure,
    chain_id: str | None = None,
    model_index: int = 0,
    atom_name: str = "CA",
) -> tuple[np.ndarray, list[tuple[str, int, str]]]:
    """Extract per-residue coordinates for one atom from a structure model.

    Returns coordinates with shape ``(N, 3)`` and residue identifiers as
    ``(chain_id, residue_number, insertion_code)`` tuples. Hetero residues are
    skipped, as are residues that do not contain the requested atom.
    """
    models = list(structure.get_models())
    if not models:
        raise ValueError("Structure does not contain any models.")
    if model_index >= len(models):
        raise IndexError(
            f"Model index {model_index} is out of range for {len(models)} model(s)."
        )

    model = models[model_index]
    if chain_id is None:
        chains = list(model.get_chains())
    else:
        if chain_id not in model:
            raise ValueError(f"Chain '{chain_id}' not found in model {model.id}.")
        chains = [model[chain_id]]

    coordinates, _, residue_ids = _extract_chain_data(chains, atom_name=atom_name)
    if not len(coordinates):
        raise ValueError(
            f"No polymer residues with atom '{atom_name}' were found in the selected structure."
        )

    return coordinates, residue_ids


def extract_residue_sequence(
    structure: Structure,
    chain_id: str | None = None,
    model_index: int = 0,
    atom_name: str = "CA",
) -> str:
    """Extract the residue sequence corresponding to the selected atom trace."""
    models = list(structure.get_models())
    if not models:
        raise ValueError("Structure does not contain any models.")
    if model_index >= len(models):
        raise IndexError(
            f"Model index {model_index} is out of range for {len(models)} model(s)."
        )

    model = models[model_index]
    if chain_id is None:
        chains = list(model.get_chains())
    else:
        if chain_id not in model:
            raise ValueError(f"Chain '{chain_id}' not found in model {model.id}.")
        chains = [model[chain_id]]

    _, sequence, residue_ids = _extract_chain_data(chains, atom_name=atom_name)
    if not residue_ids:
        raise ValueError(
            f"No polymer residues with atom '{atom_name}' were found in the selected structure."
        )
    return sequence


def _extract_chain_data(
    chains,
    atom_name: str,
) -> tuple[np.ndarray, str, list[tuple[str, int, str]]]:
    coordinates: list[np.ndarray] = []
    sequence: list[str] = []
    residue_ids: list[tuple[str, int, str]] = []

    for chain in chains:
        for residue in chain.get_residues():
            hetflag, residue_number, insertion_code = residue.id
            if hetflag.strip():
                continue
            if atom_name not in residue:
                continue

            coordinates.append(residue[atom_name].coord.astype(np.float64, copy=True))
            sequence.append(_resname_to_one_letter(residue.resname))
            residue_ids.append((chain.id, int(residue_number), insertion_code.strip()))

    if not coordinates:
        return np.empty((0, 3), dtype=np.float64), "", []
    return np.stack(coordinates), "".join(sequence), residue_ids


def _resname_to_one_letter(resname: str) -> str:
    return IUPACData.protein_letters_3to1_extended.get(
        resname[0].upper() + resname[1:].lower(), "X"
    )


def tm_score(
    reference_coords: np.ndarray,
    predicted_coords: np.ndarray,
    reference_sequence: str,
    predicted_sequence: str,
    normalize_by: str = "reference",
) -> float:
    """Compute the canonical TM-score via TM-align."""
    result = _tm_align_result(
        reference_coords,
        predicted_coords,
        reference_sequence,
        predicted_sequence,
    )
    if normalize_by == "reference":
        return float(result.tm_norm_chain1)
    if normalize_by == "predicted":
        return float(result.tm_norm_chain2)
    raise ValueError("normalize_by must be either 'reference' or 'predicted'.")


def rmsd(
    reference_coords: np.ndarray,
    predicted_coords: np.ndarray,
    reference_sequence: str,
    predicted_sequence: str,
) -> float:
    """Compute RMSD from the canonical TM-align structural alignment."""
    result = _tm_align_result(
        reference_coords,
        predicted_coords,
        reference_sequence,
        predicted_sequence,
    )
    return float(result.rmsd)


def _tm_align_result(
    reference_coords: np.ndarray,
    predicted_coords: np.ndarray,
    reference_sequence: str,
    predicted_sequence: str,
):
    """Run TM-align and return the raw result object."""
    reference = np.asarray(reference_coords, dtype=np.float64, order="C")
    predicted = np.asarray(predicted_coords, dtype=np.float64, order="C")

    if reference.ndim != 2 or predicted.ndim != 2:
        raise ValueError("TM-score expects coordinate arrays with shape (N, 3).")
    if reference.shape[1] != 3 or predicted.shape[1] != 3:
        raise ValueError("TM-score expects 3D coordinates.")
    if len(reference) != len(reference_sequence):
        raise ValueError("reference_coords and reference_sequence must have matching lengths.")
    if len(predicted) != len(predicted_sequence):
        raise ValueError("predicted_coords and predicted_sequence must have matching lengths.")

    return tm_align(reference, predicted, reference_sequence, predicted_sequence)


def structure_tm_score(
    reference_structure: Structure,
    predicted_structure: Structure,
    chain_id: str | None = None,
    atom_name: str = "CA",
    normalize_by: str = "reference",
) -> float:
    """Compute the canonical TM-score between two structures with TM-align."""
    if chain_id is None:
        ref_chains = list(reference_structure.get_chains())
        pred_chains = list(predicted_structure.get_chains())
        if len(ref_chains) == 1 and len(pred_chains) == 1:
            ref_coords, ref_sequence, _ = _extract_chain_data(
                [ref_chains[0]], atom_name=atom_name
            )
            pred_coords, pred_sequence, _ = _extract_chain_data(
                [pred_chains[0]], atom_name=atom_name
            )
        else:
            ref_coords, ref_sequence = _extract_coords_and_sequence(
                reference_structure, chain_id=chain_id, atom_name=atom_name
            )
            pred_coords, pred_sequence = _extract_coords_and_sequence(
                predicted_structure, chain_id=chain_id, atom_name=atom_name
            )
    else:
        ref_coords, ref_sequence = _extract_coords_and_sequence(
            reference_structure, chain_id=chain_id, atom_name=atom_name
        )
        pred_coords, pred_sequence = _extract_coords_and_sequence(
            predicted_structure, chain_id=chain_id, atom_name=atom_name
        )
    return tm_score(
        ref_coords,
        pred_coords,
        ref_sequence,
        pred_sequence,
        normalize_by=normalize_by,
    )


def structure_rmsd(
    reference_structure: Structure,
    predicted_structure: Structure,
    chain_id: str | None = None,
    atom_name: str = "CA",
) -> float:
    """Compute RMSD from the canonical TM-align alignment between two structures."""
    ref_coords, pred_coords, ref_sequence, pred_sequence = _extract_pair_inputs(
        reference_structure,
        predicted_structure,
        chain_id=chain_id,
        atom_name=atom_name,
    )
    return rmsd(ref_coords, pred_coords, ref_sequence, pred_sequence)


def path_tm_score(
    reference_path: str | Path,
    predicted_path: str | Path,
    chain_id: str | None = None,
    atom_name: str = "CA",
    normalize_by: str = "reference",
) -> float:
    """Load two structures and compute a TM-score between them."""
    reference_structure = load_structure(reference_path)
    predicted_structure = load_structure(predicted_path)
    return structure_tm_score(
        reference_structure,
        predicted_structure,
        chain_id=chain_id,
        atom_name=atom_name,
        normalize_by=normalize_by,
    )


def path_rmsd(
    reference_path: str | Path,
    predicted_path: str | Path,
    chain_id: str | None = None,
    atom_name: str = "CA",
) -> float:
    """Load two structures and compute RMSD from the TM-align alignment."""
    reference_structure = load_structure(reference_path)
    predicted_structure = load_structure(predicted_path)
    return structure_rmsd(
        reference_structure,
        predicted_structure,
        chain_id=chain_id,
        atom_name=atom_name,
    )


def _extract_pair_inputs(
    reference_structure: Structure,
    predicted_structure: Structure,
    chain_id: str | None,
    atom_name: str,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if chain_id is None:
        ref_chains = list(reference_structure.get_chains())
        pred_chains = list(predicted_structure.get_chains())
        if len(ref_chains) == 1 and len(pred_chains) == 1:
            ref_coords, ref_sequence, _ = _extract_chain_data(
                [ref_chains[0]], atom_name=atom_name
            )
            pred_coords, pred_sequence, _ = _extract_chain_data(
                [pred_chains[0]], atom_name=atom_name
            )
            return ref_coords, pred_coords, ref_sequence, pred_sequence

    ref_coords, ref_sequence = _extract_coords_and_sequence(
        reference_structure, chain_id=chain_id, atom_name=atom_name
    )
    pred_coords, pred_sequence = _extract_coords_and_sequence(
        predicted_structure, chain_id=chain_id, atom_name=atom_name
    )
    return ref_coords, pred_coords, ref_sequence, pred_sequence


def _extract_coords_and_sequence(
    structure: Structure,
    chain_id: str | None,
    atom_name: str,
) -> tuple[np.ndarray, str]:
    coords, _ = extract_residue_coordinates(
        structure, chain_id=chain_id, atom_name=atom_name
    )
    sequence = extract_residue_sequence(
        structure, chain_id=chain_id, atom_name=atom_name
    )
    if len(coords) != len(sequence):
        raise ValueError("Coordinate and sequence lengths must match for TM-align.")
    return coords, sequence
