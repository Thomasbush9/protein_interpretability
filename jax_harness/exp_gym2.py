"""Experiment 14 -- pLDDT and per-layer representations for ProteinGym variants.

Framing: this is not a ProteinGym leaderboard attempt. The question is *where in
the model the picture of a mutation stops matching the measured reality*, and
what kind of mutation it stops matching for. That needs three things the first
gym run did not have:

  1. **pLDDT per variant** -- the model's own confidence, so the internal state
     can be compared against what the model actually reports. Requires the
     diffusion + confidence path, not just the trunk.
  2. **Per-layer representation vectors**, not just scalar divergences, so a
     representational-similarity analysis can compare the *geometry* of the
     model's variant space against the geometry of measured stability.
  3. **pLDDT at the mutated residue**, not only the chain mean -- a single
     substitution barely moves a 63-residue average.

Captured per variant, per layer:
    z_site  [128]   pair-representation row at the mutated position
    s_site  [384]   single representation at the mutated position
    kl_glob, kl_site  scalars, as before (recomputed here so this run is
                      internally consistent with the capped alignment)

The alignment is capped (`--msa-cap`) because Boltz re-parses it per variant and
that, not the GPU, sets the wall-clock. Capping also makes MSA depth an exactly
controlled quantity across variants, which the comparison wants anyway. Features
from this run are therefore NOT directly comparable to the uncapped first run;
everything downstream should use one or the other, not a mix.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m, skl  # noqa: E402
from joltz import TrunkState  # noqa: E402


def trunk_capture(model, feats, ii, jj, row, *, recycles, key):
    """Per-layer logits at sampled pairs + z/s at the mutated position.

    `row` is the index of the mutated residue among valid tokens, resolved by
    the caller so the reduce_fn stays a pure function of the traced arrays.
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

    def reduce_fn(s_, z_):
        return {
            "logits": model.distogram_module(z_)[0, :, :, 0, :][ii, jj],
            "z_site": z_[0].mean(axis=1)[row],
            "s_site": s_[0][row],
        }

    s, z, per = pi.pairformer_capture(
        model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=True, reduce_fn=reduce_fn,
    )
    return emb, TrunkState(s=s, z=z), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-variants", type=int, default=250)
    ap.add_argument("--n-pairs", type=int, default=1500)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mosaic.losses.boltz2 import boltz2_forward_from_trunk

    rows = [r for r in csv.DictReader(open(Path(args.assay_dir) / f"{args.assay}.csv"))
            if ":" not in r["mutant"]]
    wt = list(rows[0]["mutated_sequence"])
    m0 = re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"])
    wt[int(m0.group(2)) - 1] = m0.group(1)
    wt = "".join(wt)

    rng = np.random.default_rng(args.seed)
    if len(rows) > args.n_variants:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.n_variants, replace=False))]

    work = Path(args.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)
    src = Path(args.a3m)

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] {args.assay} len={len(wt)} n={len(rows)} "
          f"msa_cap={args.msa_cap}", flush=True)

    def featurise(seq, tag):
        a3m = work / "msa" / f"{tag}.a3m"
        graft_a3m(a3m, src, seq, wt, cap=args.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=a3m.resolve()))
        return pi.load_features(y.read_text())

    f_wt, h = featurise(wt, "wt")
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask)[0]
    pos_of = {int(r): k for k, r in enumerate(valid)}
    a, b = rng.choice(valid, args.n_pairs), rng.choice(valid, args.n_pairs)
    keep = a != b
    ii_np, jj_np = a[keep], b[keep]
    ii, jj = jnp.asarray(ii_np), jnp.asarray(jj_np)

    def structure_of(emb, trunk, feats, row):
        """pLDDT (chain mean and at the mutated residue) plus CA coordinates.

        Coordinates are needed to build a *structure-space* RDM over variants:
        the question is how much of the variant-to-variant structure present in
        the pairformer survives into the predicted geometry, and that requires
        comparing variants to each other, not just to wild type."""
        out = boltz2_forward_from_trunk(
            model, feats, emb, trunk, num_sampling_steps=args.sampling_steps,
            deterministic=True, key=jax.random.fold_in(key, 7),
        )
        p = np.asarray(out.plddt)[mask]
        ca = np.asarray(out.backbone_coordinates)[mask][:, 1].astype(np.float32)
        return float(p.mean()), float(p[row]), ca

    emb_w, tr_w, ref = trunk_capture(model, f_wt, ii, jj, 0, recycles=args.recycles, key=key)
    lw = np.asarray(ref["logits"])
    L = lw.shape[0]
    pl_wt_mean, _, ca_wt = structure_of(emb_w, tr_w, f_wt, 0)
    print(f"[{time.time()-t0:6.1f}s] WT: {L} layers, pLDDT {pl_wt_mean:.3f}", flush=True)
    h.cleanup()

    Z, S, KLg, KLs, PL, PLs, CA, DIS, meta = [], [], [], [], [], [], [], [], []
    for n, r in enumerate(rows):
        mo = re.match(r"([A-Z])(\d+)([A-Z])", r["mutant"])
        p0 = int(mo.group(2)) - 1
        if p0 not in pos_of:
            continue
        row = pos_of[p0]
        f_m, hm = featurise(r["mutated_sequence"], "mut")
        # the WT reference for z_site/s_site must be taken at the SAME residue
        _, _, refr = trunk_capture(model, f_wt, ii, jj, row, recycles=args.recycles, key=key)
        emb_m, tr_m, cur = trunk_capture(model, f_m, ii, jj, row, recycles=args.recycles, key=key)
        kl = skl(np.asarray(cur["logits"]), lw)
        at = (ii_np == p0) | (jj_np == p0)
        pm, ps, ca = structure_of(emb_m, tr_m, f_m, row)
        # distogram at the trunk output, as a probability vector over the
        # sampled pairs -- the last representation before the structure module
        dis = np.asarray(cur["logits"])[-1]

        Z.append((np.asarray(cur["z_site"]) - np.asarray(refr["z_site"])).astype(np.float32))
        S.append((np.asarray(cur["s_site"]) - np.asarray(refr["s_site"])).astype(np.float32))
        KLg.append(kl.mean(1).astype(np.float32))
        KLs.append((kl[:, at].mean(1) if at.any() else np.zeros(L)).astype(np.float32))
        PL.append(pm); PLs.append(ps); CA.append(ca); DIS.append(dis.astype(np.float32))
        meta.append((r["mutant"], p0, float(r["DMS_score"]), int(r["DMS_score_bin"])))
        hm.cleanup()
        if (n + 1) % 25 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(rows)}", flush=True)

    np.savez_compressed(
        args.out, dz_site=np.stack(Z), ds_site=np.stack(S),
        kl_glob=np.stack(KLg), kl_site=np.stack(KLs),
        plddt=np.array(PL), plddt_site=np.array(PLs),
        plddt_wt=pl_wt_mean, n_layers=L,
        ca=np.stack(CA), ca_wt=ca_wt, disto=np.stack(DIS),
        disto_wt=lw[-1].astype(np.float32),
        score=np.array([m[2] for m in meta]), bin=np.array([m[3] for m in meta]),
        pos=np.array([m[1] for m in meta]), mutant=np.array([m[0] for m in meta]),
        wt_seq=wt,
    )
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}  n={len(meta)}", flush=True)


if __name__ == "__main__":
    main()
