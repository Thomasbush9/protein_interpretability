#!/usr/bin/env python3
"""Sanity-check chromophore-block preservation between WT and mutant Boltz predictions.

Verifies the experimental premise of the chromophore-block gradient-attribution
experiment (`log/2026-05-06-chromophore-attribution-pivot.md`):

1. Parses WT and mutant Boltz-predicted .cif files.
2. Cross-checks residue identities against the YAML sequences.
3. Reports which tentative-B positions are mutated in the mutant.
4. Computes Kabsch-aligned CA-RMSD on tentative B and on an expanded B
   (= tentative + heavy-atom shell within `--shell_radius` Å of the
   chromophore triad in the WT predicted structure).
5. Persists the final residue list to JSON for downstream attribution scripts.

Usage::

    python scripts/analyze_chromophore_block.py \\
        --wt_cif  /path/to/wt.cif  \\
        --mut_cif /path/to/mut.cif \\
        --wt_yaml /path/to/wt.yaml \\
        --mut_yaml /path/to/mut.yaml \\
        --out     /path/to/chromo_block.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from Bio.PDB import MMCIFParser
from Bio.SVDSuperimposer import SVDSuperimposer


# Tentative chromophore-block residues (1-indexed canonical GFP numbering).
# Chromophore triad + immediate environment, drawn from GFP fluorescence-
# mutagenesis literature.
TENTATIVE_B = [46, 65, 66, 67, 96, 145, 148, 165, 203, 205, 222]
CHROMOPHORE_TRIAD = [65, 66, 67]
SHELL_RADIUS_A = 6.0

_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wt_cif", type=Path, required=True)
    p.add_argument("--mut_cif", type=Path, required=True)
    p.add_argument("--wt_yaml", type=Path, required=True)
    p.add_argument("--mut_yaml", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shell_radius", type=float, default=SHELL_RADIUS_A)
    return p.parse_args()


def load_seq(yaml_path: Path) -> str:
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg["sequences"][0]["protein"]["sequence"]


def parse_chain(cif_path: Path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    model = next(structure.get_models())
    chain = next(model.get_chains())
    return chain


def residues_by_pos(chain) -> dict[int, object]:
    """Map 1-indexed sequence position -> Bio.PDB Residue, skipping heteroatoms."""
    out: dict[int, object] = {}
    pos = 0
    for res in chain.get_residues():
        het, _resseq, _icode = res.id
        if het.strip():
            continue
        pos += 1
        out[pos] = res
    return out


def verify_against_seq(by_pos: dict[int, object], seq: str, label: str) -> None:
    if len(by_pos) != len(seq):
        raise SystemExit(
            f"[{label}] residue-count mismatch: cif={len(by_pos)} yaml={len(seq)}"
        )
    bad = []
    for pos, res in by_pos.items():
        cif_aa = _3to1.get(res.get_resname(), "?")
        if cif_aa != seq[pos - 1]:
            bad.append((pos, cif_aa, seq[pos - 1]))
    if bad:
        raise SystemExit(
            f"[{label}] residue-identity mismatches at {len(bad)} positions; "
            f"first: {bad[:5]}"
        )


def heavy_atoms(res):
    return [a for a in res.get_atoms() if a.element != "H"]


def ca_rmsd(a_by_pos, b_by_pos, positions: list[int]) -> float:
    coords_a = np.array([a_by_pos[p]["CA"].coord for p in positions])
    coords_b = np.array([b_by_pos[p]["CA"].coord for p in positions])
    sup = SVDSuperimposer()
    sup.set(coords_a, coords_b)
    sup.run()
    return float(sup.get_rms())


def shell_around_triad(by_pos, triad: list[int], radius: float) -> set[int]:
    triad_atoms = [a for p in triad for a in heavy_atoms(by_pos[p])]
    triad_coords = np.stack([a.coord for a in triad_atoms])  # (T, 3)
    found: set[int] = set()
    for pos, res in by_pos.items():
        if pos in triad:
            continue
        for a in heavy_atoms(res):
            if (np.linalg.norm(triad_coords - a.coord, axis=1) < radius).any():
                found.add(pos)
                break
    return found


def main() -> None:
    args = parse_args()

    wt_seq = load_seq(args.wt_yaml)
    mut_seq = load_seq(args.mut_yaml)
    if len(wt_seq) != len(mut_seq):
        raise SystemExit(
            f"YAML sequence length mismatch: wt={len(wt_seq)} mut={len(mut_seq)}"
        )

    wt_by_pos = residues_by_pos(parse_chain(args.wt_cif))
    mut_by_pos = residues_by_pos(parse_chain(args.mut_cif))
    verify_against_seq(wt_by_pos, wt_seq, "wt")
    verify_against_seq(mut_by_pos, mut_seq, "mut")
    print(f"[ok] cif/yaml residue numbering consistent ({len(wt_seq)} residues)")

    # --- Tentative B audit
    print("\n[tentative B] residue identities (WT vs mut):")
    print(f"  {'pos':>4}  {'wt':>3}  {'mut':>3}  status")
    n_mut = 0
    for pos in TENTATIVE_B:
        wt_aa = wt_seq[pos - 1]
        mut_aa = mut_seq[pos - 1]
        flag = "MUTATED" if wt_aa != mut_aa else ""
        if flag:
            n_mut += 1
        print(f"  {pos:>4}  {wt_aa:>3}  {mut_aa:>3}  {flag}")
    print(f"  ({n_mut}/{len(TENTATIVE_B)} tentative-B residues mutated in mutant)")

    rmsd_tent = ca_rmsd(wt_by_pos, mut_by_pos, TENTATIVE_B)
    print(
        f"\n[CA-RMSD] WT-pred vs mut-pred on tentative B "
        f"(n={len(TENTATIVE_B)}): {rmsd_tent:.3f} Å"
    )

    # --- Shell expansion
    shell = shell_around_triad(wt_by_pos, CHROMOPHORE_TRIAD, args.shell_radius)
    expanded = sorted(set(TENTATIVE_B) | shell)
    print(
        f"\n[shell] residues with any heavy atom within "
        f"{args.shell_radius:.1f} Å of chromophore triad (excl. triad): {len(shell)}"
    )
    print(
        f"[expanded B] tentative ({len(TENTATIVE_B)}) ∪ shell ({len(shell)}) "
        f"= {len(expanded)} residues"
    )

    print(f"\n  {'pos':>4}  {'wt':>3}  {'mut':>3}  {'source':>10}  status")
    for pos in expanded:
        wt_aa = wt_seq[pos - 1]
        mut_aa = mut_seq[pos - 1]
        is_tent = pos in TENTATIVE_B
        is_shell = pos in shell
        source = (
            "tent+shell" if is_tent and is_shell
            else "tentative" if is_tent
            else "shell"
        )
        flag = "MUTATED" if wt_aa != mut_aa else ""
        print(f"  {pos:>4}  {wt_aa:>3}  {mut_aa:>3}  {source:>10}  {flag}")

    rmsd_exp = ca_rmsd(wt_by_pos, mut_by_pos, expanded)
    print(
        f"\n[CA-RMSD] WT-pred vs mut-pred on expanded B "
        f"(n={len(expanded)}): {rmsd_exp:.3f} Å"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(
            {
                "tentative_B": TENTATIVE_B,
                "chromophore_triad": CHROMOPHORE_TRIAD,
                "shell_radius_A": args.shell_radius,
                "shell": sorted(shell),
                "expanded_B": expanded,
                "wt_cif": str(args.wt_cif),
                "mut_cif": str(args.mut_cif),
                "wt_yaml": str(args.wt_yaml),
                "mut_yaml": str(args.mut_yaml),
                "rmsd_tentative_B_A": rmsd_tent,
                "rmsd_expanded_B_A": rmsd_exp,
            },
            f,
            indent=2,
        )
    print(f"\n[ok] saved residue list -> {args.out}")


if __name__ == "__main__":
    main()
