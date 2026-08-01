"""Experiment 9 -- does the trunk difference survive into coordinates?

Everything measured so far is the distogram: a readout head. The structure is
produced by AtomDiffusion conditioned on (s_trunk, z, s_inputs). The link from
"the pair representation barely changed" to "the predicted structure barely
changed" has never actually been measured, and it is the last gap in the chain.

The earlier attempt failed for a reason worth not repeating: with a *single*
diffusion sample, two runs of the same model on the same input agreed at only
TM 0.70, so every mutant-vs-WT number sat at or below the sampler's own noise
floor and carried no information.

The fix is to measure the noise floor explicitly rather than hope it is small.
For each condition we draw K structures with different diffusion keys from the
*same* (deterministic) trunk state, which gives:

  within(c)     mean pairwise TM among the K samples of condition c
                -- the sampler's reproducibility, i.e. the noise floor

  between(c,WT) mean TM over all K x K pairs of (condition c, wild type)

A mutation has a detectable effect on the *structure* only if
between(c,WT) is meaningfully below within(WT). Reporting both, with the
scramble control as the upper bound on what any query-side change can do, makes
the comparison honest in a way a single-sample TM never can be.

Sequences are residue-aligned and equal length, so TM needs no alignment search.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
from geom import tm_score  # noqa: E402


def sample_structures(model, feats, *, recycles, n_samples, sampling_steps, key):
    """One deterministic trunk, then n_samples diffusion draws with distinct keys."""
    from mosaic.losses.boltz2 import boltz2_forward_from_trunk

    emb = model.embed_inputs(feats)
    trunk = pi.run_trunk(
        model, emb, feats, recycling_steps=recycles, key=key,
        deterministic=True, capture_last=False,
    )
    mask = np.asarray(feats["token_pad_mask"][0]).astype(bool)
    cas, plddts = [], []
    for i in range(n_samples):
        out = boltz2_forward_from_trunk(
            model, feats, emb, trunk["trunk_state"],
            num_sampling_steps=sampling_steps, deterministic=True,
            key=jax.random.fold_in(jax.random.key(1000 + i), i),
        )
        bb = np.asarray(out.backbone_coordinates)
        cas.append(bb[mask][:, 1])
        plddts.append(float(np.asarray(out.plddt)[mask].mean()))
    return np.stack(cas), np.array(plddts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--conditions", default="gfp_core_32,gfp_surface_32,gfp_scramble")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    ids = [args.wt] + [c.strip() for c in args.conditions.split(",")]
    coords, plddt, handles = {}, {}, []
    for cid in ids:
        feats, h = pi.load_features((data / "yamls" / f"{cid}.yaml").read_text())
        handles.append(h)
        coords[cid], plddt[cid] = sample_structures(
            model, feats, recycles=args.recycles, n_samples=args.samples,
            sampling_steps=args.sampling_steps, key=key,
        )
        print(f"[{time.time()-t0:6.1f}s] {cid}: {args.samples} samples, "
              f"pLDDT {plddt[cid].mean():.3f}", flush=True)

    def within(c):
        v = [tm_score(coords[c][i], coords[c][j])
             for i, j in itertools.combinations(range(args.samples), 2)]
        return float(np.mean(v)), float(np.std(v))

    def between(c, d):
        v = [tm_score(coords[c][i], coords[d][j])
             for i in range(args.samples) for j in range(args.samples)]
        return float(np.mean(v)), float(np.std(v))

    w_wt, w_wt_sd = within(args.wt)
    out = {"samples": args.samples, "recycles": args.recycles,
           "within": {args.wt: [w_wt, w_wt_sd]}, "between": {}, "plddt": {}}

    print(f"\n  NOISE FLOOR: within-{args.wt} mean pairwise TM = {w_wt:.4f} +/- {w_wt_sd:.4f}")
    print(f"  (this is the ceiling any between-condition TM can be compared against)\n")

    for cid in ids[1:]:
        wi, wi_sd = within(cid)
        be, be_sd = between(cid, args.wt)
        out["within"][cid] = [wi, wi_sd]
        out["between"][cid] = [be, be_sd]
        out["plddt"][cid] = [float(plddt[cid].mean()), float(plddt[cid].std())]
        gap = w_wt - be
        print(
            f"  {cid:16s} within {wi:.4f}   between-vs-WT {be:.4f} +/- {be_sd:.4f}"
            f"   gap below noise floor {gap:+.4f}"
            f"   {'DETECTABLE' if gap > 2 * w_wt_sd else 'not distinguishable from noise'}",
            flush=True,
        )
    out["plddt"][args.wt] = [float(plddt[args.wt].mean()), float(plddt[args.wt].std())]

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    for h in handles:
        h.cleanup()


if __name__ == "__main__":
    main()
