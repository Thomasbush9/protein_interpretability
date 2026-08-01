"""Experiment 8 -- is the mutation signal actually amplified, or does everything just grow?

I reported that the mutation's footprint grows through the trunk (|dz| x2.1,
KL x5.6). Both are growth relative to their own value at layer 0, with no
denominator. If the pair representation as a whole is also growing, and if the
distogram as a whole is also moving, then "amplified" may be an artefact of an
absent normalisation -- exactly the same class of error as measuring in
Angstrom against a sharpening distribution.

Three denominators, per layer:

  rel_dz    ||z_mut - z_wt|| / ||z_wt||
            the mutation's share of the representation's own magnitude.

  rel_kl    KL(mutant || WT)  /  KL(WT at this layer || WT at the final layer)
            the mutation's effect measured against how much the wild-type
            distogram is itself still changing at that depth. If the model is
            still substantially revising its own prediction, a given KL means
            less than the same KL once the prediction has settled.

  frac_of_scramble
            KL(mutant || WT) / KL(scrambled-sequence || WT)
            the mutation's effect as a fraction of what a *completely different*
            sequence does to the same MSA. This is the "how big is big" scale:
            a scrambled query with the wild-type alignment is the maximal
            query-side perturbation available without touching the MSA.

If rel_dz and frac_of_scramble are flat or falling while the raw numbers rise,
then nothing is being amplified and the §8 reading needs the same correction
that §3 did.
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


def capture(model, feats, ii, jj, *, recycles, key):
    """Per-layer z and distogram logits at sampled pairs."""
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
        return {
            "z": z_[0][ii, jj],
            "logits": model.distogram_module(z_)[0, :, :, 0, :][ii, jj],
        }

    s, z, per = pi.pairformer_capture(
        model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=True, reduce_fn=reduce_fn,
    )
    return per


def sm(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def skl(la, lb):
    pa, pb = sm(la), sm(lb)
    return ((pa - pb) * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum(-1).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--mutants", default="gfp_core_32,gfp_surface_32")
    ap.add_argument("--scramble", default=None,
                    help="id of a scrambled-sequence control sharing the WT MSA")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-pairs", type=int, default=5000)
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
    rng = np.random.default_rng(0)
    a, b = rng.choice(valid, args.n_pairs), rng.choice(valid, args.n_pairs)
    keep = a != b
    ii, jj = jnp.asarray(a[keep]), jnp.asarray(b[keep])

    ref = capture(model, f_wt, ii, jj, recycles=args.recycles, key=key)
    zw = np.asarray(ref["z"])          # [L, P, D]
    lw = np.asarray(ref["logits"])     # [L, P, B]
    zw_norm = np.linalg.norm(zw, axis=-1).mean(-1)          # [L]
    # how much is the WT distogram still changing at this depth, vs its own end?
    kl_wt_to_final = np.array([skl(lw[i], lw[-1]) for i in range(lw.shape[0])])
    print(f"[{time.time()-t0:6.1f}s] WT captured", flush=True)

    scr = None
    if args.scramble:
        f_s, hs = pi.load_features((data / "yamls" / f"{args.scramble}.yaml").read_text())
        cs = capture(model, f_s, ii, jj, recycles=args.recycles, key=key)
        scr = {"z": np.asarray(cs["z"]), "logits": np.asarray(cs["logits"])}
        scr["kl"] = np.array([skl(scr["logits"][i], lw[i]) for i in range(lw.shape[0])])
        scr["dz"] = np.linalg.norm(scr["z"] - zw, axis=-1).mean(-1)
        print(f"  scramble control: KL L0 {scr['kl'][0]:.4f} -> L63 {scr['kl'][-1]:.4f}", flush=True)
        hs.cleanup()

    out = {"wt_z_norm": [float(v) for v in zw_norm],
           "kl_wt_to_final": [float(v) for v in kl_wt_to_final], "mutants": {}}
    if scr is not None:
        out["scramble_kl"] = [float(v) for v in scr["kl"]]
        out["scramble_dz"] = [float(v) for v in scr["dz"]]

    for mid in args.mutants.split(","):
        mid = mid.strip()
        f_m, hm = pi.load_features((data / "yamls" / f"{mid}.yaml").read_text())
        cm = capture(model, f_m, ii, jj, recycles=args.recycles, key=key)
        zm, lm = np.asarray(cm["z"]), np.asarray(cm["logits"])

        dz = np.linalg.norm(zm - zw, axis=-1).mean(-1)
        rel_dz = dz / zw_norm
        kl = np.array([skl(lm[i], lw[i]) for i in range(lw.shape[0])])

        rec = {
            "dz": [float(v) for v in dz],
            "rel_dz": [float(v) for v in rel_dz],
            "kl": [float(v) for v in kl],
        }
        if scr is not None:
            rec["frac_of_scramble_kl"] = [float(v) for v in kl / np.maximum(scr["kl"], 1e-12)]
            rec["frac_of_scramble_dz"] = [float(v) for v in dz / np.maximum(scr["dz"], 1e-12)]
        out["mutants"][mid] = rec

        print(f"\n=== {mid} ===")
        print(f"  raw  |dz|      L0 {dz[0]:8.2f} -> L63 {dz[-1]:8.2f}   (x{dz[-1]/dz[0]:.2f})")
        print(f"  ||z_wt||       L0 {zw_norm[0]:8.2f} -> L63 {zw_norm[-1]:8.2f}   (x{zw_norm[-1]/zw_norm[0]:.2f})")
        print(f"  RELATIVE |dz|/||z||  L0 {rel_dz[0]:.4f} -> L34 {rel_dz[34]:.4f} -> L63 {rel_dz[-1]:.4f}"
              f"   (x{rel_dz[-1]/rel_dz[0]:.2f})")
        print(f"  raw KL         L0 {kl[0]:.4f} -> L63 {kl[-1]:.4f}   (x{kl[-1]/kl[0]:.2f})")
        if scr is not None:
            f = np.array(rec["frac_of_scramble_kl"])
            print(f"  KL as fraction of scramble  L0 {f[0]:.4f} -> L34 {f[34]:.4f} -> L63 {f[-1]:.4f}"
                  f"   (x{f[-1]/f[0]:.2f})")
        print("  rel_dz every 8th layer: " + " ".join(f"{v:.4f}" for v in rel_dz[::8]), flush=True)
        hm.cleanup()

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    h.cleanup()


if __name__ == "__main__":
    main()
