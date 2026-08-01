"""Predict a wild-type structure and dump CA coordinates as a minimal cif.

`build_dataset.py` derives burial from a cif, and `parse_cif_ca` only ever reads
CA rows, so a CA-only cif is sufficient and avoids depending on any external
structure. Geometry from this path was validated in `check_predict.py`
(CA-CA 3.80 A, Rg matching the independent PyTorch Boltz-2 prediction), so it is
trustworthy for burial even though the TM-score utility is not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}

HEADER = """data_pred
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--seq", required=True, help="sequence, for residue names")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mosaic.losses.boltz2 import boltz2_forward_from_trunk

    model = pi.load_model(subsample_msa=False)
    feats, h = pi.load_features(Path(args.yaml).read_text())
    key = jax.random.key(0)
    emb = model.embed_inputs(feats)
    trunk = pi.run_trunk(model, emb, feats, recycling_steps=args.recycles,
                         key=key, deterministic=True, capture_last=False)
    out = boltz2_forward_from_trunk(
        model, feats, emb, trunk["trunk_state"],
        num_sampling_steps=args.sampling_steps, deterministic=True, key=key,
    )
    mask = np.asarray(feats["token_pad_mask"][0]).astype(bool)
    ca = np.asarray(out.backbone_coordinates)[mask][:, 1]
    plddt = float(np.asarray(out.plddt)[mask].mean())
    seq = args.seq
    if len(seq) != len(ca):
        raise SystemExit(f"sequence length {len(seq)} != {len(ca)} CA atoms")

    step = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))
    print(f"N={len(ca)}  pLDDT={plddt:.3f}  CA-CA median={np.median(step):.2f} A  Rg={rg:.2f} A")
    if not (3.5 < np.median(step) < 4.1):
        raise SystemExit("CA-CA spacing is not ~3.8 A -- refusing to write a bad structure")

    lines = [HEADER]
    for i, (aa, xyz) in enumerate(zip(seq, ca), start=1):
        lines.append(
            f"ATOM {i} CA {ONE_TO_THREE.get(aa,'UNK')} A {i} "
            f"{xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}\n"
        )
    lines.append("#\n")
    Path(args.out).write_text("".join(lines))
    print(f"wrote {args.out}")
    h.cleanup()


if __name__ == "__main__":
    main()
