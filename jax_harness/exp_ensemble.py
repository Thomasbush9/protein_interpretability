"""Experiment 15 -- is the mutation in the output *ensemble* rather than the output *structure*?

Diffusion is a sampler, not a function. Everything reported about structural
invariance -- ours and the literature's -- compares one predicted structure per
sequence. But the pattern we found one stage upstream was shape-not-location: at
the distogram, doubling the mutation load moved symmetric KL by 143 % and the
mean expected distance by 5 %. If the same holds at the structure module, a
destabilising mutation would not shift the predicted structure so much as widen
the distribution the sampler draws from -- and a single sample cannot see that.

For each variant we run the trunk once (deterministic) and then draw K
structures from the *same* trunk state with K different diffusion noise keys.
That isolates sampler variability: everything upstream is held fixed, so any
spread is the structure module's own.

Measured per variant:
  spread        mean pairwise TM among its own K samples (LOW = diverse ensemble)
  tm_to_wt      mean TM of its samples against the wild type's samples
  plddt, plddt_site
  cond_*        L2 norm of the difference from wild type in each diffusion
                conditioning tensor -- a cheap first look at whether the
                conditioning even carries the mutation before the sampler sees it

The test: does `spread` correlate with measured stability better than
`tm_to_wt` does? If yes, the information is in the output ensemble and the
single-structure comparisons everyone reports are looking in the wrong place.

The wild type's own ensemble spread is the reference: it is the sampler's
baseline variability, and no variant's spread means anything except relative
to it.

**beta sweep.** Boltz-sample (Steering Conformational Sampling in Boltz-2 via
Pair Representation Scaling, bioRxiv 2026.01.23.701250) shows that rescaling the
latent pair representation by a global scalar beta markedly increases multi-state
recovery -- i.e. the *default* sampler under-explores what the pair
representation supports. That is a confound for this experiment: at beta=1 a
null result would be ambiguous between "the mutation is not in the ensemble" and
"the ensemble is too narrow for anything to show". So z is scaled by beta before
the diffusion conditioning and the test is repeated across beta. The informative
outcome is whether mutation information becomes visible in the ensemble only
once the sampler is widened.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m  # noqa: E402
from joltz import TrunkState  # noqa: E402


class ScaledConditioning(eqx.Module):
    """Wraps DiffusionConditioning and scales the pair-derived attention biases.

    These three tensors are added to attention logits inside the diffusion
    transformer, so scaling them changes how strongly pairwise information
    steers the sampler relative to the noise. That is downstream of
    PairwiseConditioning's LayerNorm, which is what made the naive `z_trunk`
    scaling a no-op. Wrapping the module (rather than reimplementing the
    sampling path) means the library's own forward is used unchanged.
    """

    inner: eqx.Module
    beta: float = eqx.field(static=True)

    def __call__(self, s_trunk, z_trunk, relative_position_encoding, feats):
        q, c, to_keys, aeb, adb, ttb = self.inner(
            s_trunk, z_trunk, relative_position_encoding, feats)
        b = self.beta
        return q, c, to_keys, aeb * b, adb * b, ttb * b


def scale_conditioning(model, beta):
    if beta == 1.0:
        return model
    return eqx.tree_at(lambda m: m.diffusion_conditioning, model,
                       ScaledConditioning(model.diffusion_conditioning, beta),
                       is_leaf=lambda x: x is model.diffusion_conditioning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-variants", type=int, default=120)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="scale on the pairwise signal reaching the structure module")
    ap.add_argument("--beta-mode", choices=("z_trunk", "bias"), default="bias",
                    help="z_trunk: scale z before DiffusionConditioning -- NOTE this is "
                         "very nearly a no-op, because PairwiseConditioning begins with "
                         "nn.LayerNorm(concat([z_trunk, rel_pos])) which absorbs a global "
                         "scale (measured: beta 1->2 moved WT ensemble spread 0.9900->0.9911 "
                         "and left the conditioning delta identical to 7 dp). "
                         "bias: scale the pair-derived attention biases that the diffusion "
                         "transformer actually consumes -- downstream of that LayerNorm, and "
                         "the quantity that sets the signal-to-noise of pairwise couplings")
    ap.add_argument("--msa-cap", type=int, default=2048)
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
    sample_model = (scale_conditioning(model, args.beta)
                    if args.beta_mode == "bias" else model)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] {args.assay} len={len(wt)} n={len(rows)} "
          f"K={args.samples} samples/variant", flush=True)

    def featurise(seq, tag):
        a3m = work / "msa" / f"{tag}.a3m"
        graft_a3m(a3m, src, seq, wt, cap=args.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=a3m.resolve()))
        return pi.load_features(y.read_text())

    def run_variant(feats):
        """One deterministic trunk, K diffusion draws, plus conditioning tensors."""
        emb = model.embed_inputs(feats)
        tr = pi.run_trunk(model, emb, feats, recycling_steps=args.recycles,
                          key=key, deterministic=True, capture_last=False)
        st = tr["trunk_state"]
        if args.beta != 1.0 and args.beta_mode == "z_trunk":
            st = TrunkState(s=st.s, z=st.z * args.beta)
        cond = model.diffusion_conditioning(
            st.s, st.z, emb.relative_position_encoding, feats)
        mask = np.asarray(feats["token_pad_mask"][0]).astype(bool)
        cas, pls = [], []
        for i in range(args.samples):
            k_i = jax.random.fold_in(jax.random.key(5000 + i), i)
            out = boltz2_forward_from_trunk(
                sample_model, feats, emb, st, num_sampling_steps=args.sampling_steps,
                deterministic=True, key=k_i,
            )
            cas.append(np.asarray(out.backbone_coordinates)[mask][:, 1].astype(np.float32))
            pls.append(np.asarray(out.plddt)[mask])
        return np.stack(cas), np.stack(pls), cond, mask

    f_wt, h = featurise(wt, "wt")
    ca_wt, pl_wt, cond_wt, mask = run_variant(f_wt)
    valid = np.where(mask)[0]
    pos_of = {int(r): k for k, r in enumerate(valid)}
    print(f"[{time.time()-t0:6.1f}s] WT ensemble done, pLDDT {pl_wt.mean():.3f}", flush=True)
    h.cleanup()

    # only the first three conditioning outputs are dense arrays comparable
    # across runs; `to_keys` is a closure and the atom biases are ragged
    cond_names = ["q", "c", "token_trans_bias"]
    cond_idx = [0, 1, 5]

    CA, PL, COND, meta = [], [], [], []
    for n, r in enumerate(rows):
        mo = re.match(r"([A-Z])(\d+)([A-Z])", r["mutant"])
        p0 = int(mo.group(2)) - 1
        if p0 not in pos_of:
            continue
        f_m, hm = featurise(r["mutated_sequence"], "mut")
        ca, pl, cond, _ = run_variant(f_m)
        dif = []
        for ci in cond_idx:
            try:
                a = np.asarray(cond[ci], dtype=np.float32)
                b = np.asarray(cond_wt[ci], dtype=np.float32)
                dif.append(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9)))
            except Exception:
                dif.append(float("nan"))
        CA.append(ca); PL.append(pl); COND.append(dif)
        meta.append((r["mutant"], p0, float(r["DMS_score"]), int(r["DMS_score_bin"])))
        hm.cleanup()
        if (n + 1) % 20 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(rows)}", flush=True)

    np.savez_compressed(
        args.out, ca=np.stack(CA), ca_wt=ca_wt, plddt=np.stack(PL), plddt_wt=pl_wt,
        cond_rel_diff=np.array(COND), cond_names=np.array(cond_names),
        score=np.array([m[2] for m in meta]), bin=np.array([m[3] for m in meta]),
        pos=np.array([m[1] for m in meta]), mutant=np.array([m[0] for m in meta]),
        wt_seq=wt, samples=args.samples, beta=args.beta,
    )
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}  n={len(meta)}", flush=True)


if __name__ == "__main__":
    main()
