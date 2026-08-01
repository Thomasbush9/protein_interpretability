"""Diagnose exp_predict: is the coordinate extraction / TM-score sound?

A single buried mutation reporting TM=0.22 against the wild type while the
distogram moves by <0.3 A is internally inconsistent, so before treating it as
a result we check the pipeline against things with known answers:

  1. consecutive CA-CA distance should be ~3.8 A. If it isn't, the axis-1 index
     of backbone_coordinates is not CA (or the array is not per-token).
  2. radius of gyration for a 238-residue beta barrel should be ~15-18 A. A
     much larger value means the chain is not folded / not what we think.
  3. TM-score of our JAX wild-type prediction against the *independent* PyTorch
     Boltz-2 prediction of the same sequence (ProtForge run, 34073_model_0.cif).
     Both are Boltz-2 on the same input, so this should be high (>0.8). This is
     the load-bearing check: it validates extraction and TM together, against
     an artefact produced by a completely separate code path.
  4. TM of WT against a randomly permuted copy of itself, as a floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import parse_cif_ca  # noqa: E402
from geom import kabsch, tm_score  # noqa: E402

NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/runs/predict_gfp_ca.npz"
)
REF_CIF = Path(
    "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/test_protforge/outputs/sequences/34073/boltz/34073_model_0.cif"
)


def geom(name, ca):
    step = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    rg = np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean())
    print(
        f"  {name:18s} n={len(ca):4d}  CA-CA median {np.median(step):6.2f} A "
        f"(iqr {np.percentile(step,25):.2f}-{np.percentile(step,75):.2f})  Rg {rg:6.2f} A"
    )
    return np.median(step), rg


def main():
    d = np.load(NPZ)
    print(f"structures in {NPZ.name}: {list(d.keys())}\n")

    print("1+2. geometry sanity (expect CA-CA ~3.8 A, Rg ~15-18 A for GFP)")
    for k in d.files:
        geom(k, d[k])

    wt = d["gfp_wt"]
    ref_seq, ref_ca = parse_cif_ca(REF_CIF)
    print("\n  reference (PyTorch Boltz-2, ProtForge run)")
    geom("34073_model_0", ref_ca)

    print("\n3. JAX WT vs independent PyTorch Boltz-2 WT prediction")
    if len(ref_ca) != len(wt):
        print(f"   length mismatch {len(ref_ca)} vs {len(wt)} -- truncating to common prefix")
        n = min(len(ref_ca), len(wt))
        ref_ca, wt2 = ref_ca[:n], wt[:n]
    else:
        wt2 = wt
    tm = tm_score(wt2, ref_ca)
    P, Q = wt2, ref_ca
    pc, qc = P.mean(0), Q.mean(0)
    R = kabsch(P - pc, Q - qc)
    rmsd = float(np.sqrt((((P - pc) @ R.T - (Q - qc)) ** 2).sum(1).mean()))
    print(f"   TM = {tm:.4f}   RMSD = {rmsd:.2f} A")
    print("   -> TM > 0.8 means extraction + TM-score are sound.")
    print("   -> TM low means the bug is in this pipeline, not in the model.")

    print("\n4. floor: WT vs shuffled WT")
    rng = np.random.default_rng(0)
    print(f"   TM = {tm_score(wt[rng.permutation(len(wt))], wt):.4f}")


if __name__ == "__main__":
    main()
