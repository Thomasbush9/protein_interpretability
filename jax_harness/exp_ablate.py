"""Experiment 7 -- causal test: is `transition_z` responsible for the L37-45 dip?

exp_sublayers (in KL) attributes the transient reduction of the mutation's
distributional footprint at layers 37-45 to `transition_z`, the per-pair
feed-forward. That is an attribution, not a cause: it says where the divergence
went down, not that removing the op would change anything.

Here we remove it. `Pairformer2` keeps its 64 layers as a *stacked* parameter
pytree, so zeroing `transition_z`'s output weight on a chosen slice of layers
is one `eqx.tree_at` on `stacked_parameters` -- no hooks, no re-tracing of the
model, and the untouched layers are bit-identical.

Zeroing the final projection of the Transition block makes its residual
contribution exactly zero, i.e. `z += 0`, which is a clean deletion of that
write rather than a perturbation of it.

Predictions if the attribution is causal:
  - ablating layers 37-45 removes (most of) the dip: KL through the band should
    no longer fall, and final KL should rise;
  - ablating a *matched control band* of the same width elsewhere (e.g. 10-18,
    where transition_z's contribution is small) should do much less;
  - ablating all 64 layers should raise final KL a lot -- transition_z is the
    only op that ever reduces divergence.
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


def ablate_transition_z(model, layers):
    """Return a model with transition_z's residual contribution zeroed on `layers`.

    joltz's Transition is `fc3(silu(fc1(v)) * fc2(v))`, so **fc3** is the output
    projection -- fc1/fc2 are the gated inner branches. Zeroing fc3's weight
    (and bias, if it has one) makes the block emit exactly zero, i.e. deletes
    the `z += transition_z(z)` write rather than perturbing it. Layers not in
    `layers` are bit-identical because only the selected slice of the stacked
    parameter array is touched.
    """
    idx = jnp.asarray(sorted(layers))

    def out_w(m):
        return m.pairformer_module.stacked_parameters.transition_z.fc3.weight

    model = eqx.tree_at(out_w, model, out_w(model).at[idx].set(0.0))

    def out_b(m):
        return m.pairformer_module.stacked_parameters.transition_z.fc3.bias

    if out_b(model) is not None:
        model = eqx.tree_at(out_b, model, out_b(model).at[idx].set(0.0))
    return model


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
    s, z, per_layer = pi.pairformer_capture(
        model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=True,
        reduce_fn=lambda s_, z_: model.distogram_module(z_)[0, :, :, 0, :][ii, jj],
    )
    return per_layer


def skl(la, lb):
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
    ap.add_argument("--mutant", default="gfp_core_32")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-pairs", type=int, default=5000)
    ap.add_argument("--band", default="37,45")
    ap.add_argument("--control-band", default="10,18")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.band.split(","))
    clo, chi = (int(x) for x in args.control_band.split(","))
    data = Path(args.data)
    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    f_wt, h = pi.load_features((data / "yamls" / f"{args.wt}.yaml").read_text())
    f_m, hm = pi.load_features((data / "yamls" / f"{args.mutant}.yaml").read_text())
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask)[0]
    rng = np.random.default_rng(0)
    a, b = rng.choice(valid, args.n_pairs), rng.choice(valid, args.n_pairs)
    keep = a != b
    ii, jj = jnp.asarray(a[keep]), jnp.asarray(b[keep])

    conditions = {
        "intact": [],
        f"ablate_band_{lo}_{hi}": list(range(lo, hi + 1)),
        f"ablate_control_{clo}_{chi}": list(range(clo, chi + 1)),
        "ablate_all": list(range(64)),
    }

    out = {"band": [lo, hi], "control_band": [clo, chi], "conditions": {}}
    for name, layers in conditions.items():
        m = model if not layers else ablate_transition_z(model, layers)
        lw = np.asarray(capture_logits(m, f_wt, ii, jj, recycles=args.recycles, key=key))
        lm = np.asarray(capture_logits(m, f_m, ii, jj, recycles=args.recycles, key=key))
        kl = skl(lm, lw).mean(axis=1)
        band_change = float(kl[hi] - kl[lo - 1])
        out["conditions"][name] = {
            "kl": [float(v) for v in kl],
            "kl_final": float(kl[-1]),
            "kl_change_across_band": band_change,
            "n_layers_ablated": len(layers),
        }
        print(
            f"  {name:24s} KL(L{lo-1})={kl[lo-1]:.4f} KL(L{hi})={kl[hi]:.4f} "
            f"change across band {band_change:+.4f}   final KL {kl[-1]:.4f}",
            flush=True,
        )

    base = out["conditions"]["intact"]
    print("\n  vs intact:")
    for name, rec in out["conditions"].items():
        if name == "intact":
            continue
        print(
            f"    {name:24s} band change {rec['kl_change_across_band']:+.4f} "
            f"(intact {base['kl_change_across_band']:+.4f})   "
            f"final KL x{rec['kl_final']/base['kl_final']:.3f}",
            flush=True,
        )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    h.cleanup(); hm.cleanup()


if __name__ == "__main__":
    main()
