"""Experiment 0 -- the behavioural anchor: what does Boltz-2 actually predict?

Everything else in this harness measures the trunk (distogram / pair
representation). That is where the mechanism lives, but it is not the model's
output. This script runs the full model -- trunk, diffusion, confidence -- for
the whole cohort and asks the plain question:

    given a GFP whose buried core has been replaced by charged residues,
    does Boltz-2 still predict the barrel, and how confident is it?

Because every sequence in the cohort has identical length and residue
correspondence to the wild type, TM-score needs no alignment search: the
residue mapping is the identity, so an optimal superposition (Kabsch) over the
common CA set gives the exact TM-score. That is a genuine simplification here,
not an approximation -- it would be wrong for the fold-switch cohort, where
tmtools is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
from geom import kabsch, tm_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coords-out", default=None)
    args = ap.parse_args()

    from mosaic.losses.boltz2 import boltz2_forward_from_trunk

    data = Path(args.data)
    manifest = {r["id"]: r for r in csv.DictReader((data / "manifest.csv").open())}
    ids = [args.wt] + [i for i in manifest if i != args.wt]

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    coords, rows = {}, []
    for cid in ids:
        feats, h = pi.load_features((data / "yamls" / f"{cid}.yaml").read_text())
        emb = model.embed_inputs(feats)
        trunk = pi.run_trunk(
            model, emb, feats, recycling_steps=args.recycles, key=key,
            deterministic=True, capture_last=False,
        )
        out = boltz2_forward_from_trunk(
            model, feats, emb, trunk["trunk_state"],
            num_sampling_steps=args.sampling_steps, deterministic=True, key=key,
        )
        mask = np.asarray(feats["token_pad_mask"][0]).astype(bool)
        # StructureModelOutput carries no batch dim: backbone_coordinates is
        # [N, 4, 3] (N, CA, C, O) and plddt is [N].
        bb = np.asarray(out.backbone_coordinates)
        ca = bb[mask][:, 1]
        plddt = float(np.asarray(out.plddt)[mask].mean())
        coords[cid] = ca

        tm = tm_score(ca, coords[args.wt]) if cid != args.wt else 1.0
        rmsd = float("nan")
        if cid != args.wt:
            P, Q = ca, coords[args.wt]
            pc, qc = P.mean(0), Q.mean(0)
            R = kabsch(P - pc, Q - qc)
            rmsd = float(np.sqrt((((P - pc) @ R.T - (Q - qc)) ** 2).sum(1).mean()))

        m = manifest[cid]
        rows.append({
            "id": cid, "mode": m["mode"], "n_mut": int(m["n_mut"]),
            "tm_to_wt": tm, "rmsd_to_wt": rmsd, "plddt": plddt,
        })
        print(
            f"  {cid:18s} n={m['n_mut']:>2s} {m['mode']:7s} "
            f"TM={tm:.4f}  RMSD={rmsd:6.2f} A  pLDDT={plddt:.3f}",
            flush=True,
        )
        h.cleanup()

    Path(args.out).write_text(json.dumps(rows, indent=2))
    if args.coords_out:
        np.savez_compressed(args.coords_out, **coords)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
