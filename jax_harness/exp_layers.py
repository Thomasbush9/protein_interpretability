"""Experiment 3 -- where in the 64-layer Pairformer does the mutation signal live or die?

Two hypotheses make opposite predictions about the per-layer trace of
"how different is the mutant's pair representation from the wild-type's":

  NEVER WRITTEN   divergence is small at every layer. The query's contribution
                  to z is simply outcompeted by the MSA write from the start;
                  there is nothing to erase.

  WRITTEN THEN ERASED  divergence rises in early/mid layers and then falls.
                  The model does register the mutation and subsequently
                  suppresses it -- an active computation, and a much more
                  interesting claim.

We read this out two ways per layer L:

  z_div    mean |‖z_mut‖ - ‖z_wt‖| over valid pairs (representation-space)
  ed_div   mean |E[d]_mut - E[d]_wt| via the distogram head applied to the
           layer-L pair representation (structure-space "logit lens")

Caveat on the logit lens: the distogram head was trained on the *final* z, so
applying it mid-stack is an out-of-distribution probe. Absolute values at early
layers are not meaningful; the WT-vs-mutant *difference* at a fixed layer is,
because both runs are probed identically. z_div is reported alongside precisely
because it needs no such assumption.

Also emits the same trace across the 4 MSA blocks, and the per-block
OuterProductMean magnitude -- the size of the MSA's write into z.
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


def make_reduce(model):
    """Per-layer capture: keep the structure-space read-out, not raw z.

    Full z for 64 layers is 64 x N^2 x 128 floats (~1.9 GB at N=238); the
    distogram projection is 64 x N^2 (~15 MB) and is what we actually analyse.
    """

    def reduce_fn(s, z):
        logits = model.distogram_module(z)[0, :, :, 0, :]
        return {
            "z_norm": jnp.linalg.norm(z[0], axis=-1),
            "ed": pi.expected_distance(logits),
            "cp": pi.contact_prob(logits),
            "s_norm": jnp.linalg.norm(s[0], axis=-1),
        }

    return reduce_fn


def run_capture(model, feats, *, recycles, key, reduce_fn):
    """run_trunk, but with a custom pairformer reduce_fn."""
    emb = model.embed_inputs(feats)
    from joltz import TrunkState

    state = TrunkState(s=jnp.zeros_like(emb.s_init), z=jnp.zeros_like(emb.z_init))
    for i in range(recycles - 1):
        state = pi.iteration(model, state, emb, feats, key=jax.random.fold_in(key, i))

    mask = feats["token_pad_mask"]
    pair_mask = mask[:, :, None] * mask[:, None, :]
    k = jax.random.fold_in(key, recycles - 1)

    s = emb.s_init + model.s_recycle(model.s_norm(state.s))
    z = emb.z_init + model.z_recycle(model.z_norm(state.z))
    z = z + model.template_module(z, feats, pair_mask, deterministic=True, key=k)
    z_msa, msa_layers = pi.msa_module_capture(
        model.msa_module, z, emb.s_inputs, feats,
        key=jax.random.fold_in(k, 0), deterministic=True,
    )
    z = z + z_msa
    s, z, pf = pi.pairformer_capture(
        model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=True, reduce_fn=reduce_fn,
    )
    return {"pf": pf, "msa": msa_layers, "final_z": z}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--mutants", default="gfp_core_32,gfp_core_08,gfp_surface_32")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    reduce_fn = make_reduce(model)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    f_wt, h = pi.load_features((data / "yamls" / f"{args.wt}.yaml").read_text())
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    n = int(mask.sum())
    off = ~np.eye(n, dtype=bool)
    ref = run_capture(model, f_wt, recycles=args.recycles, key=key, reduce_fn=reduce_fn)
    print(f"[{time.time()-t0:6.1f}s] WT captured (N={n})", flush=True)

    def sub(a):
        a = np.asarray(a)
        return a[..., mask, :][..., :, mask]

    out = {"n_tokens": n, "wt": args.wt, "mutants": {}}
    out["opm_per_block_wt"] = [
        float(x) for x in sub(ref["msa"]["opm_norm"]).mean(axis=(1, 2))
    ]

    for mid in args.mutants.split(","):
        mid = mid.strip()
        f_m, hm = pi.load_features((data / "yamls" / f"{mid}.yaml").read_text())
        cur = run_capture(model, f_m, recycles=args.recycles, key=key, reduce_fn=reduce_fn)

        rec = {}
        for name, a, b in (
            ("pf_z_div", sub(cur["pf"]["z_norm"]), sub(ref["pf"]["z_norm"])),
            ("pf_ed_div", sub(cur["pf"]["ed"]), sub(ref["pf"]["ed"])),
            ("msa_z_div", sub(cur["msa"]["z_norm"]), sub(ref["msa"]["z_norm"])),
        ):
            rec[name] = [float(v) for v in np.abs(a - b)[:, off].mean(axis=1)]
        rec["pf_cp_flip"] = [
            float(v)
            for v in ((sub(cur["pf"]["cp"]) > 0.5) != (sub(ref["pf"]["cp"]) > 0.5))[:, off].mean(axis=1)
        ]
        rec["pf_s_div"] = [
            float(v)
            for v in np.abs(
                np.asarray(cur["pf"]["s_norm"])[:, mask] - np.asarray(ref["pf"]["s_norm"])[:, mask]
            ).mean(axis=1)
        ]
        out["mutants"][mid] = rec

        ed = rec["pf_ed_div"]
        peak = int(np.argmax(ed))
        print(
            f"\n{mid}: E[d] divergence across 64 pairformer layers\n"
            f"   layer 0 {ed[0]:.4f} | peak L{peak} {ed[peak]:.4f} | final {ed[-1]:.4f} A"
            f"   (final/peak = {ed[-1]/max(ed[peak],1e-9):.3f})",
            flush=True,
        )
        print("   " + " ".join(f"{v:.3f}" for v in ed[::4]) + "   (every 4th layer)", flush=True)
        print(f"   MSA blocks z-div: {[round(v,4) for v in rec['msa_z_div']]}", flush=True)
        hm.cleanup()

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    h.cleanup()


if __name__ == "__main__":
    main()
