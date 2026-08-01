"""Experiment 1 -- which input route decides the structure when sequence and MSA disagree.

For a mutant M carrying the wild-type alignment, we ask two complementary
causal questions about each of the five routes (see pi_paths.ROUTES):

  RESTORE  run M, but source route r from WT.
           How much of the mutation's effect on the prediction does r carry?
           necessity(r) = 1 - ||D(patched) - D(WT)|| / ||D(M) - D(WT)||
           r=1 means route r alone carried the entire mutation signal.

  INJECT   run WT, but source route r from M.
           Is r on its own enough to move the prediction?
           sufficiency(r) = ||D(patched) - D(WT)|| / ||D(M) - D(WT)||

Distances are mean |dE[d]| in Angstrom over valid off-diagonal residue pairs,
which is interpretable in physical units rather than in logit space.

Sanity checks are run first and are not optional: patching *no* routes must
reproduce M exactly, and patching *all* routes must reproduce WT exactly. If
either fails, the decomposition is not a decomposition and the numbers below
mean nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
import pi_paths as pp  # noqa: E402


def emap(model, z, mask):
    """Expected CA-CA distance map (A) over valid tokens."""
    d = np.asarray(pi.expected_distance(pi.logit_lens(model, z)))
    return d[np.ix_(mask, mask)]


def dist(a, b):
    """Mean |dE[d]| over off-diagonal pairs, in Angstrom."""
    n = a.shape[0]
    off = ~np.eye(n, dtype=bool)
    return float(np.abs(a - b)[off].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir with yamls/ + manifest.csv")
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--mutants", default="gfp_core_32,gfp_core_08,gfp_surface_32")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    key = jax.random.key(0)
    R = dict(recycling_steps=args.recycles, key=key, deterministic=True)

    feats_wt, h_wt = pi.load_features((data / "yamls" / f"{args.wt}.yaml").read_text())
    mask = np.asarray(feats_wt["token_pad_mask"][0]).astype(bool)
    emb_wt = model.embed_inputs(feats_wt)
    D_wt = emap(model, pp.run_trunk(model, emb_wt, feats_wt, capture_last=False, **R)["trunk_state"].z, mask)
    print(f"[{time.time()-t0:6.1f}s] WT done (N={mask.sum()}, S={feats_wt['msa'].shape[1]})", flush=True)

    results = []
    for mid in args.mutants.split(","):
        mid = mid.strip()
        feats_m, h_m = pi.load_features((data / "yamls" / f"{mid}.yaml").read_text())
        emb_m = model.embed_inputs(feats_m)
        D_m = emap(model, pp.run_trunk(model, emb_m, feats_m, capture_last=False, **R)["trunk_state"].z, mask)
        total = dist(D_m, D_wt)
        print(f"\n=== {mid} ===   ||D(M)-D(WT)|| = {total:.4f} A", flush=True)

        # --- sanity: the decomposition must bracket the two endpoints ------
        d_none = dist(emap(model, pp.patch(model, feats_m, feats_wt, (), **R)["trunk_state"].z, mask), D_m)
        d_all = dist(
            emap(model, pp.patch(model, feats_m, feats_wt, pp.ROUTES, **R)["trunk_state"].z, mask), D_wt
        )
        print(f"  sanity: patch(none) vs M = {d_none:.2e} A ; patch(all) vs WT = {d_all:.2e} A")
        if total > 1e-6 and max(d_none, d_all) > 0.05 * total:
            print("  WARNING: endpoints do not close -- routes are not exhaustive for this pair")

        row = {
            "mutant": mid, "total_A": total,
            "sanity_none_vs_M": d_none, "sanity_all_vs_WT": d_all,
        }

        for r in pp.ROUTES:
            z_res = pp.patch(model, feats_m, feats_wt, (r,), **R)["trunk_state"].z
            z_inj = pp.patch(model, feats_wt, feats_m, (r,), **R)["trunk_state"].z
            nec = dist(emap(model, z_res, mask), D_wt)
            suf = dist(emap(model, z_inj, mask), D_wt)
            row[f"restore_{r}_resid_A"] = nec
            row[f"restore_{r}_necessity"] = 1 - nec / total if total > 1e-9 else float("nan")
            row[f"inject_{r}_move_A"] = suf
            row[f"inject_{r}_sufficiency"] = suf / total if total > 1e-9 else float("nan")
            print(
                f"  {r:10s}  necessity {row[f'restore_{r}_necessity']:+.3f}"
                f"   sufficiency {row[f'inject_{r}_sufficiency']:+.3f}"
                f"   (resid {nec:.4f} A, move {suf:.4f} A)",
                flush=True,
            )

        results.append(row)
        h_m.cleanup()

    outp.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {outp}", flush=True)
    h_wt.cleanup()


if __name__ == "__main__":
    main()
