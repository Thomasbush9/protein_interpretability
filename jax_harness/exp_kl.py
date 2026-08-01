"""Experiment 6 -- is the mutation signal suppressed, or is the readout just saturating?

exp_subspace killed the "rotated out of the readout" story: the component of
dz inside the distogram head's rowspace does not shrink across the stack -- it
*grows*, and its share of |dz| is flat (~0.28) from layer 0 to layer 63. Yet
exp_layers showed mean |dE[d]| falling from 0.83 A at L34 to 0.31 A at L63.

Both cannot be describing the same thing, because E[d] is a *nonlinear* readout:

    E[d] = sum_b softmax(logits)_b * centre_b

As the trunk grows more confident, the distogram distribution sharpens, and the
same perturbation in logit space moves E[d] less. So a falling |dE[d]| is
consistent with a mutation signal that is not being suppressed at all -- it can
be an artefact of measuring in Angstrom against a sharpening distribution.

The scale-free measure is a divergence between the two distributions:

    symmetric KL( p_mut || p_wt )   per residue pair, per layer

KL is invariant to how peaked the distribution is in the way E[d] is not. If KL
*also* falls across L37-45, suppression is real and the exp_layers/exp_sublayers
conclusions stand (with E[d] merely exaggerating it). If KL is flat or rising
while E[d] falls, then nothing is being suppressed and the whole "band of
suppressing layers" reading is a readout artefact.

Entropy of the wild-type distogram per layer is reported alongside, because
that is the quantity the saturation explanation predicts should be falling.
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
from joltz import TrunkState  # noqa: E402


def capture_logits(model, feats, ii, jj, *, recycles, key):
    """Per-layer distogram logits at the given (i,j) pairs -> [L, n_pairs, 64]."""
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

    def reduce_fn(s_, z_):
        return model.distogram_module(z_)[0, :, :, 0, :][ii, jj]

    s, z, per_layer = pi.pairformer_capture(
        model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=True, reduce_fn=reduce_fn,
    )
    return per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--mutants", default="gfp_core_32,gfp_surface_32")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-pairs", type=int, default=6000)
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
    ii, jj = jnp.asarray(a[keep]), jnp.asarray(b[keep])
    print(f"[{time.time()-t0:6.1f}s] {len(a[keep])} pairs", flush=True)

    lw = np.asarray(capture_logits(model, f_wt, ii, jj, recycles=args.recycles, key=key))
    print(f"[{time.time()-t0:6.1f}s] WT logits {lw.shape}", flush=True)

    centres = np.asarray(pi.BIN_CENTRES)

    def softmax(x):
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)

    pw = softmax(lw)
    ent_w = -(pw * np.log(pw + 1e-12)).sum(-1).mean(axis=1)   # [L]
    out = {"wt_entropy_nats": [float(v) for v in ent_w], "mutants": {}}

    for mid in args.mutants.split(","):
        mid = mid.strip()
        f_m, hm = pi.load_features((data / "yamls" / f"{mid}.yaml").read_text())
        lm = np.asarray(capture_logits(model, f_m, ii, jj, recycles=args.recycles, key=key))
        pm = softmax(lm)

        skl = ((pm - pw) * (np.log(pm + 1e-12) - np.log(pw + 1e-12))).sum(-1).mean(axis=1)
        ed_w = (pw * centres).sum(-1)
        ed_m = (pm * centres).sum(-1)
        d_ed = np.abs(ed_m - ed_w).mean(axis=1)
        d_logit = np.abs(lm - lw).mean(axis=(1, 2))

        rec = {
            "sym_kl": [float(v) for v in skl],
            "d_ed_A": [float(v) for v in d_ed],
            "d_logit": [float(v) for v in d_logit],
        }
        out["mutants"][mid] = rec

        k34, k63 = skl[34], skl[-1]
        print(f"\n=== {mid} ===")
        print(f"  symmetric KL   L0 {skl[0]:.4f}  peak L{skl.argmax():2d} {skl.max():.4f}"
              f"  L34 {k34:.4f}  final {k63:.4f}   (final/peak {k63/max(skl.max(),1e-12):.3f})")
        print(f"  mean |dE[d]|   L0 {d_ed[0]:.4f}  peak L{d_ed.argmax():2d} {d_ed.max():.4f}"
              f"  final {d_ed[-1]:.4f}   (final/peak {d_ed[-1]/max(d_ed.max(),1e-12):.3f})")
        print(f"  mean |dlogit|  L0 {d_logit[0]:.4f}  final {d_logit[-1]:.4f}")
        print(f"  WT entropy     L0 {ent_w[0]:.4f}  L34 {ent_w[34]:.4f}  final {ent_w[-1]:.4f} nats")
        print("  KL every 8th layer: " + " ".join(f"{v:.4f}" for v in skl[::8]), flush=True)
        hm.cleanup()

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    h.cleanup()


if __name__ == "__main__":
    main()
