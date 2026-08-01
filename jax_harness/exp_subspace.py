"""Experiment 5 -- is the mutation signal destroyed, or moved where the readout cannot see it?

exp_layers showed that at the final Pairformer layers the mutant's pair
representation is *further* from wild type than at any earlier layer, while the
distogram read off it is *closest* to wild type. Two explanations:

  DESTROYED   the mutation-relevant component of z shrinks; the residual
              difference is unrelated noise.
  UNREAD      the difference is as large as ever but has moved into directions
              the distogram head does not look at.

This is decidable exactly, with no probe to train, because the head is

    distogram(z) = W @ (z + z^T)

so it is blind to two things by construction:

  1. the **antisymmetric** part of z -- (z - z^T)/2 is annihilated by z + z^T;
  2. anything in the **null space of W** (W is [64, 128], so at least half of
     the 128 channels of the symmetric part are unread too).

So decompose dz = z_mut - z_wt at each layer into
    readable    = P_rowspace(W) [ symmetric part of dz ]
    unreadable  = antisymmetric part + null-space part of the symmetric part
and report the readable fraction of the squared norm. If the UNREAD story is
right, that fraction falls across the suppression band while |dz| does not.

Sampling: capturing full z for 64 layers is ~1.9 GB per run, so we capture a
fixed random set of residue pairs instead -- both (i,j) and (j,i), since the
symmetric/antisymmetric split needs the transpose partner.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
from joltz import TrunkState  # noqa: E402


def capture_pairs(model, feats, ii, jj, *, recycles, key):
    """Run the trunk, capturing z at the given (i,j) index pairs per layer.

    Returns [n_layers, n_pairs, D].
    """
    emb = model.embed_inputs(feats)
    state = TrunkState(s=jnp.zeros_like(emb.s_init), z=jnp.zeros_like(emb.z_init))
    for i in range(recycles - 1):
        state = pi.iteration(model, state, emb, feats, key=jax.random.fold_in(key, i))

    mask = feats["token_pad_mask"]
    pair_mask = mask[:, :, None] * mask[:, None, :]
    k = jax.random.fold_in(key, recycles - 1)

    s = emb.s_init + model.s_recycle(model.s_norm(state.s))
    z = emb.z_init + model.z_recycle(model.z_norm(state.z))
    z = z + model.template_module(z, feats, pair_mask, deterministic=True, key=k)
    z = z + model.msa_module(
        z, emb.s_inputs, feats, deterministic=True, key=jax.random.fold_in(k, 0)
    )

    s, z, per_layer = pi.pairformer_capture(
        model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=True,
        reduce_fn=lambda s_, z_: z_[0][ii, jj],
    )
    return per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--mutants", default="gfp_core_32,gfp_surface_32")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-pairs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    f_wt, h = pi.load_features((data / "yamls" / f"{args.wt}.yaml").read_text())
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask)[0]

    rng = np.random.default_rng(args.seed)
    a = rng.choice(valid, args.n_pairs)
    b = rng.choice(valid, args.n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    # capture (i,j) and (j,i) together so the transpose partner is available
    ii = jnp.asarray(np.concatenate([a, b]))
    jj = jnp.asarray(np.concatenate([b, a]))
    m = len(a)
    print(f"[{time.time()-t0:6.1f}s] sampling {m} residue pairs (+transposes)", flush=True)

    # readout geometry: W acts on the symmetrised z
    W = np.asarray(model.distogram_module.distogram.weight)  # [out, D]
    D = W.shape[1]
    # orthonormal basis for rowspace(W)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    rank = int((S > S.max() * 1e-6).sum())
    B = Vt[:rank]  # [rank, D]
    print(f"  distogram weight {W.shape}, rowspace rank {rank} of {D}", flush=True)

    zr = np.asarray(capture_pairs(model, f_wt, ii, jj, recycles=args.recycles, key=key))
    print(f"[{time.time()-t0:6.1f}s] WT captured {zr.shape}", flush=True)

    out = {"rank": rank, "dim": D, "n_pairs": m, "mutants": {}}

    for mid in args.mutants.split(","):
        mid = mid.strip()
        f_m, hm = pi.load_features((data / "yamls" / f"{mid}.yaml").read_text())
        zm = np.asarray(capture_pairs(model, f_m, ii, jj, recycles=args.recycles, key=key))

        dz = zm - zr                      # [L, 2m, D]
        fwd, rev = dz[:, :m], dz[:, m:]   # (i,j) and (j,i)
        sym = 0.5 * (fwd + rev)
        anti = 0.5 * (fwd - rev)

        sym_read = sym @ B.T              # component inside rowspace(W)
        n_tot = (fwd ** 2).sum(axis=(1, 2))
        n_sym = (sym ** 2).sum(axis=(1, 2))
        n_anti = (anti ** 2).sum(axis=(1, 2))
        n_read = (sym_read ** 2).sum(axis=(1, 2))

        rec = {
            "dz_norm": [float(v) for v in np.sqrt(n_tot)],
            "frac_readable": [float(v) for v in n_read / np.maximum(n_sym + n_anti, 1e-12)],
            "frac_antisym": [float(v) for v in n_anti / np.maximum(n_sym + n_anti, 1e-12)],
            "readable_norm": [float(v) for v in np.sqrt(n_read)],
        }
        out["mutants"][mid] = rec

        fr = np.array(rec["frac_readable"])
        rn = np.array(rec["readable_norm"])
        dn = np.array(rec["dz_norm"])
        print(f"\n=== {mid} ===")
        print(f"  |dz|              L0 {dn[0]:8.2f}  peak L{dn.argmax():2d} {dn.max():8.2f}  final {dn[-1]:8.2f}")
        print(f"  readable fraction L0 {fr[0]:.4f}  at L34 {fr[34]:.4f}  final {fr[-1]:.4f}")
        print(f"  |readable part|   L0 {rn[0]:8.2f}  peak L{rn.argmax():2d} {rn.max():8.2f}  final {rn[-1]:8.2f}")
        print("  readable fraction every 8th layer: "
              + " ".join(f"{v:.3f}" for v in fr[::8]), flush=True)
        hm.cleanup()

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    h.cleanup()


if __name__ == "__main__":
    main()
