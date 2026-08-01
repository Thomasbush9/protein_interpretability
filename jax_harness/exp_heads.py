"""Experiment 11 -- which triangle-attention heads carry the spreading?

exp_sublayers showed the four triangle operations amplify the mutation's
distributional footprint while transition_z trims it. Triangle *attention*
(start and end node) is multi-head -- 4 heads in Boltz-2 -- so the amplification
may be spread evenly or concentrated in a few heads. If concentrated, those
heads are the natural next ablation target and a much sharper claim than
"triangle attention".

Method: joltz's TriangleAttention contracts the per-head attention with values
inside its __call__, so head-resolved output is not exposed. Rather than patch
it, we ablate: zero one head's output-projection slice at a time, across the
whole stack, and measure the change in final KL(mutant || WT). A head that
carries the mutation's amplification should, when removed, reduce final KL more
than its peers.

Zeroing acts on `stacked_parameters`, so all 64 layers' copies of that head go
at once -- this asks "is head h special", not "is head h at layer L special".
Head-and-layer resolution is a 4 x 64 sweep and only worth it if the head
marginal is uneven.

Caveat carried from exp_ablate: deleting a component perturbs the trunk beyond
the quantity of interest, so the *ranking* across heads is the interpretable
output, not the absolute drop. A no-op condition and a whole-module ablation
bracket the scale.
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


def capture_logits(model, feats, ii, jj, *, recycles, key):
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
    s, z, per = pi.pairformer_capture(
        model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=True,
        reduce_fn=lambda s_, z_: model.distogram_module(z_)[0, :, :, 0, :][ii, jj],
    )
    return np.asarray(per)[-1]        # final layer only


def skl(la, lb):
    def sm(x):
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)
    pa, pb = sm(la), sm(lb)
    return float(((pa - pb) * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum(-1).mean())


def ablate_head(model, which, head, n_heads):
    """Zero one head's slice of a triangle-attention output projection.

    joltz's TriangleAttention wraps an `Attention` as `.mha`, whose `linear_o`
    maps [no_heads * c_hidden] -> c_out. The columns belonging to head h are a
    contiguous block, so zeroing that block removes head h's contribution
    exactly while leaving the others untouched.
    """
    def sel(m):
        return getattr(m.pairformer_module.stacked_parameters, which).mha.linear_o.weight

    w = sel(model)                       # [n_layers, out, heads*head_dim]
    hd = w.shape[-1] // n_heads
    new = w.at[:, :, head * hd:(head + 1) * hd].set(0.0)
    return eqx.tree_at(sel, model, new)


def ablate_module(model, which):
    def sel(m):
        return getattr(m.pairformer_module.stacked_parameters, which).mha.linear_o.weight
    return eqx.tree_at(sel, model, jnp.zeros_like(sel(model)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--mutant", default="gfp_core_32")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-pairs", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)

    n_heads = int(model.pairformer_module.static.tri_att_start.no_heads)
    print(f"[{time.time()-t0:6.1f}s] model loaded; triangle attention heads = {n_heads}",
          flush=True)

    f_wt, h = pi.load_features((data / "yamls" / f"{args.wt}.yaml").read_text())
    f_m, hm = pi.load_features((data / "yamls" / f"{args.mutant}.yaml").read_text())
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask)[0]
    rng = np.random.default_rng(0)
    a, b = rng.choice(valid, args.n_pairs), rng.choice(valid, args.n_pairs)
    keep = a != b
    ii, jj = jnp.asarray(a[keep]), jnp.asarray(b[keep])

    def final_kl(m):
        lw = capture_logits(m, f_wt, ii, jj, recycles=args.recycles, key=key)
        lm = capture_logits(m, f_m, ii, jj, recycles=args.recycles, key=key)
        return skl(lm, lw)

    base = final_kl(model)
    print(f"  intact                         final KL {base:.4f}", flush=True)
    out = {"intact": base, "n_heads": n_heads, "heads": {}, "modules": {}}

    for which in ("tri_att_start", "tri_att_end"):
        v = final_kl(ablate_module(model, which))
        out["modules"][which] = v
        print(f"  ablate all of {which:14s} final KL {v:.4f}   (x{v/base:.3f})", flush=True)

    for which in ("tri_att_start", "tri_att_end"):
        for hh in range(n_heads):
            v = final_kl(ablate_head(model, which, hh, n_heads))
            out["heads"][f"{which}_h{hh}"] = v
            print(f"  ablate {which}  head {hh}      final KL {v:.4f}   (x{v/base:.3f})",
                  flush=True)

    print("\n  ranked by effect (most reduction first):")
    for k, v in sorted(out["heads"].items(), key=lambda kv: kv[1]):
        print(f"    {k:22s} {v:.4f}   (x{v/base:.3f})")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    h.cleanup(); hm.cleanup()


if __name__ == "__main__":
    main()
