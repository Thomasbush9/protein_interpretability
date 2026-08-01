"""Experiment 4 -- which operation inside the Pairformer layer does the suppressing?

exp_layers localised the attenuation of the mutation signal to a band around
Pairformer layers 37-45, in the same place regardless of the mutation. Each
layer is five sequential writes into z:

    z += tri_mul_out(z)
    z += tri_mul_in(z)
    z += tri_att_start(z)
    z += tri_att_end(z)
    z += transition_z(z)

(then s reads z via AttentionPairBias, but s never writes back into z, so the
distogram depends only on the five above.)

This reruns the stack capturing distogram logits after *each* of the five, for wild type and
mutant, so the per-sublayer change in divergence

    delta_op(L) = div(after op at layer L) - div(before op at layer L)

is available for all 64 layers. Summed over the five ops it telescopes to the
per-layer change exp_layers already reported, which is the built-in consistency
check: `--check` asserts the final divergence matches the known value.

Divergence is symmetric KL, NOT mean |dE[d]|. exp_kl showed that E[d] in
Angstrom falls across the last eight layers purely because the distogram
sharpens (entropy 2.16 -> 0.81 nats), while the actual distributional
difference nearly doubles. The first version of this experiment used E[d] and
so could not distinguish "transition_z suppresses the mutation" from
"transition_z sharpens the distogram". This version can.
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
import joltz  # noqa: E402
from joltz import TrunkState  # noqa: E402

SAMP_I = SAMP_J = None  # set in main()
OPS = ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end", "transition_z")


def pairformer_sublayer_capture(model, pf, s, z, mask, pair_mask, *, key, deterministic=True):
    """Mirror of joltz PairformerLayer2.__call__, capturing E[d] after each write.

    Returns (s, z, per_layer) where per_layer[op] has shape [n_layers, N, N].
    """

    def ed(z_):
        # raw logits, not E[d]: exp_kl showed that E[d] in Angstrom is not a
        # valid yardstick across depth once the distogram starts sharpening.
        # Sub-sampled pairs keep 5 ops x 64 layers affordable.
        return model.distogram_module(z_)[0, :, :, 0, :][SAMP_I, SAMP_J]

    @jax.checkpoint
    def body(carry, params):
        s_, z_, k_ = carry
        layer = eqx.combine(pf.static, params)
        out = {}

        drop, k_ = joltz.get_dropout_mask(layer.dropout, z_, not deterministic, key=k_)
        z_ = z_ + drop * layer.tri_mul_out(z_, pair_mask)
        out["tri_mul_out"] = ed(z_)

        drop, k_ = joltz.get_dropout_mask(layer.dropout, z_, not deterministic, key=k_)
        z_ = z_ + drop * layer.tri_mul_in(z_, pair_mask)
        out["tri_mul_in"] = ed(z_)

        drop, k_ = joltz.get_dropout_mask(layer.dropout, z_, not deterministic, key=k_)
        z_ = z_ + drop * layer.tri_att_start(z_, pair_mask)
        out["tri_att_start"] = ed(z_)

        drop, k_ = joltz.get_dropout_mask(
            layer.dropout, z_, not deterministic, key=k_, columnwise=True
        )
        z_ = z_ + drop * layer.tri_att_end(z_, pair_mask)
        out["tri_att_end"] = ed(z_)

        z_ = z_ + layer.transition_z(z_)
        out["transition_z"] = ed(z_)

        s_normed = layer.pre_norm_s(s_)
        s_ = s_ + layer.attention(s=s_normed, z=z_, mask=mask, k_in=s_normed)
        s_ = s_ + layer.transition_s(s_)
        s_ = layer.s_post_norm(s_)
        return (s_, z_, k_), out

    (s, z, _), per_layer = jax.lax.scan(body, (s, z, key), pf.stacked_parameters)
    return s, z, per_layer


def run(model, feats, *, recycles, key):
    """Trunk with sublayer capture on the final recycle."""
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
    # z entering the pairformer, i.e. the "before layer 0" reference point
    z_in = z
    s, z, per_layer = pairformer_sublayer_capture(
        model, model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=True,
    )
    return {"per_layer": per_layer, "z_final": z, "ed_in": model.distogram_module(z_in)[0, :, :, 0, :][SAMP_I, SAMP_J]}


def _skl(la, lb):
    """Symmetric KL between two sets of distogram logits [..., bins]."""
    def sm(x):
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)
    pa, pb = sm(la), sm(lb)
    return ((pa - pb) * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--mutants", default="gfp_core_32,gfp_surface_32")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--band", default="37,45", help="layer band to summarise, inclusive")
    ap.add_argument("--n-pairs", type=int, default=6000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    lo, hi = (int(x) for x in args.band.split(","))
    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    f_wt, h = pi.load_features((data / "yamls" / f"{args.wt}.yaml").read_text())
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask)[0]
    rng = np.random.default_rng(0)
    a, b = rng.choice(valid, args.n_pairs), rng.choice(valid, args.n_pairs)
    keep = a != b
    global SAMP_I, SAMP_J
    SAMP_I, SAMP_J = jnp.asarray(a[keep]), jnp.asarray(b[keep])
    ref = run(model, f_wt, recycles=args.recycles, key=key)
    print(f"[{time.time()-t0:6.1f}s] WT captured ({len(a[keep])} pairs)", flush=True)

    out = {"band": [lo, hi], "ops": list(OPS), "mutants": {}}

    for mid in args.mutants.split(","):
        mid = mid.strip()
        f_m, hm = pi.load_features((data / "yamls" / f"{mid}.yaml").read_text())
        cur = run(model, f_m, recycles=args.recycles, key=key)

        # divergence after each op, per layer: [n_layers]
        div = {
            op: _skl(np.asarray(cur["per_layer"][op]), np.asarray(ref["per_layer"][op])).mean(axis=1)
            for op in OPS
        }
        div_in = float(_skl(np.asarray(cur["ed_in"]), np.asarray(ref["ed_in"])).mean())

        # delta contributed by each op: divergence after it minus divergence
        # after the previous write (the previous op, or the previous layer's last op)
        prev_layer_end = np.concatenate([[div_in], div[OPS[-1]][:-1]])
        deltas = {}
        prev = prev_layer_end
        for op in OPS:
            deltas[op] = div[op] - prev
            prev = div[op]

        band = slice(lo, hi + 1)
        band_sum = {op: float(deltas[op][band].sum()) for op in OPS}
        total_sum = {op: float(deltas[op].sum()) for op in OPS}

        out["mutants"][mid] = {
            "div_entering_pairformer": div_in,
            "div_after_op": {op: [float(v) for v in div[op]] for op in OPS},
            "delta_per_op": {op: [float(v) for v in deltas[op]] for op in OPS},
            "band_sum": band_sum,
            "total_sum": total_sum,
        }

        print(f"\n=== {mid} ===  entering pairformer KL {div_in:.4f}, "
              f"leaving KL {div[OPS[-1]][-1]:.4f}", flush=True)
        print(f"  net change contributed by each op, summed over layers {lo}-{hi} "
              "(negative = suppresses, units: symmetric KL):")
        for op, v in sorted(band_sum.items(), key=lambda kv: kv[1]):
            print(f"    {op:16s} {v:+.4f}", flush=True)
        print("  and over all 64 layers:")
        for op, v in sorted(total_sum.items(), key=lambda kv: kv[1]):
            print(f"    {op:16s} {v:+.4f}", flush=True)
        hm.cleanup()

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    h.cleanup()


if __name__ == "__main__":
    main()
