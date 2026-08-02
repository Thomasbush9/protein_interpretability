"""Difference amplification: is the decoder's insensitivity a GAIN problem?

Established so far: the trunk represents mutational effect well (probe rho 0.548
on held-out positions), the diffusion conditioning still carries it
(||dq||/||q|| = 0.285), the sampled structure does not (TM-to-WT rho 0.214), and
the sampler ignores the conditioning entirely while the global fold is being
decided (divergence/floor ~0.6 for 80 % of the schedule).

Two explanations remain, and they call for different fixes:

  GAIN      the mutation-specific part of the trunk state is real but too small
            relative to everything else the decoder conditions on. Scale it up
            and the structure should start tracking stability.
  STRUCTURAL the decoder's early steps are governed by a prior that does not
            read this part of the conditioning at all. Scaling changes the
            magnitude of the perturbation but not what the decoder does with it,
            so the structure moves without becoming more informative.

The intervention separates them. For a variant we run the trunk twice and form

    s(gamma) = s_wt + gamma * (s_mut - s_wt)
    z(gamma) = z_wt + gamma * (z_mut - z_wt)

then decode from that state using the MUTANT's features (so the molecule is
right) and a diffusion key held fixed across gamma (so any difference between
gammas is the conditioning, not the noise draw). gamma = 1 reproduces the
ordinary mutant prediction, gamma = 0 decodes the mutant's atoms from the wild
type's trunk state, gamma > 1 amplifies.

This is deliberately NOT the earlier beta experiment. beta scaled the pair
representation in absolute terms, which widened the sampler and degraded the
signal (0.435 -> 0.018) -- it added noise because it scaled everything, mutation
and background alike. Here only the mutant-minus-wild-type DIFFERENCE is scaled;
the background is untouched.

**The control is the whole experiment.** Amplifying anything by 8x moves the
structure, and a structure that moves more will trivially have a different TM to
the wild type. So each gamma is paired with a magnitude-matched control that
carries no information about *this* variant: variant i is given variant j's
difference vector, rescaled to ||d_i||. If the true difference raises
rho(TM, dG) and the permuted one of identical norm does not, the effect is
mutation-specific. If both move together, we have only rediscovered that larger
perturbations give larger structural changes.

Saves coordinates rather than TM-scores: tmtools is not installed in the mosaic
container, and keeping the coordinates means any other geometry can be computed
later without the GPU. Analysis is analyze_amplify.py on a login node.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m  # noqa: E402


def pick(rows, n, seed=0):
    """n variants spread across the whole stability range, not the extremes."""
    rows = sorted(rows, key=lambda r: float(r["DMS_score"]))
    if len(rows) <= n:
        return rows
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    return [rows[i] for i in np.unique(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-variants", type=int, default=60)
    ap.add_argument("--gammas", default="0,1,2,4,8")
    ap.add_argument("--control-gammas", default="2,8",
                    help="gammas at which to also run the norm-matched permuted control")
    ap.add_argument("--sampling-steps", type=int, default=100)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from joltz import TrunkState
    from mosaic.losses.boltz2 import boltz2_forward_from_trunk

    gammas = [float(g) for g in args.gammas.split(",")]
    cgammas = [float(g) for g in args.control_gammas.split(",") if g]
    work = Path(args.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((Path(args.assay_dir) / f"{args.assay}.csv").open()))
    rows = [r for r in rows if ":" not in r["mutant"]]
    wt = None
    for r in rows:                       # reconstruct WT from any variant
        m = re.match(r"([A-Z])(\d+)([A-Z])", r["mutant"])
        if m:
            s = list(r["mutated_sequence"])
            s[int(m.group(2)) - 1] = m.group(1)
            wt = "".join(s)
            break
    picked = pick(rows, args.n_variants, args.seed)
    src = Path(args.a3m)

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    print(f"[{time.time()-t0:6.1f}s] {args.assay} len={len(wt)} n={len(picked)} "
          f"gammas={gammas} control@{cgammas} steps={args.sampling_steps}", flush=True)

    def featurise(seq, tag):
        a3m = work / "msa" / f"{tag}.a3m"
        graft_a3m(a3m, src, seq, wt, cap=args.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=a3m.resolve()))
        return pi.load_features(y.read_text())

    key = jax.random.key(args.seed)

    def trunk_of(feats):
        emb = model.embed_inputs(feats)
        tr = pi.run_trunk(model, emb, feats, recycling_steps=args.recycles,
                          key=key, deterministic=True, capture_last=False)
        return emb, tr["trunk_state"]

    def decode(feats, emb, s, z, k):
        """One structure from an arbitrary trunk state, at a fixed noise key."""
        out = boltz2_forward_from_trunk(
            model, feats, emb, TrunkState(s=s, z=z),
            num_sampling_steps=args.sampling_steps, deterministic=True, key=k)
        mask = np.asarray(feats["token_pad_mask"][0]).astype(bool)
        return (np.asarray(out.backbone_coordinates)[mask][:, 1].astype(np.float32),
                np.asarray(out.plddt)[mask].astype(np.float32))

    # ---- wild type: reference state and reference structure ----------------
    f_wt, h = featurise(wt, "wt")
    emb_wt, st_wt = trunk_of(f_wt)
    s_wt, z_wt = np.asarray(st_wt.s), np.asarray(st_wt.z)
    ca_wt, pl_wt = decode(f_wt, emb_wt, st_wt.s, st_wt.z, jax.random.key(777))
    print(f"[{time.time()-t0:6.1f}s] WT trunk + structure done, "
          f"pLDDT {pl_wt.mean():.3f}, N={len(ca_wt)}", flush=True)
    h.cleanup()

    # ---- pass 1: every variant's trunk state --------------------------------
    ds, dz, meta, feats_cache = [], [], [], []
    for n, r in enumerate(picked):
        f_m, hm = featurise(r["mutated_sequence"], f"mut{n}")
        emb_m, st_m = trunk_of(f_m)
        ds.append(np.asarray(st_m.s) - s_wt)
        dz.append(np.asarray(st_m.z) - z_wt)
        meta.append((r["mutant"], int(re.match(r"[A-Z](\d+)", r["mutant"]).group(1)) - 1,
                     float(r["DMS_score"])))
        feats_cache.append((f_m, emb_m, hm))
        if (n + 1) % 20 == 0:
            print(f"[{time.time()-t0:6.1f}s] trunks {n+1}/{len(picked)}", flush=True)
    ds, dz = np.stack(ds), np.stack(dz)
    nz = np.linalg.norm(dz.reshape(len(dz), -1), axis=1)
    print(f"[{time.time()-t0:6.1f}s] all trunks done; ||dz|| mean {nz.mean():.3f} "
          f"min {nz.min():.3f} max {nz.max():.3f}", flush=True)

    # norm-matched permutation: variant i receives variant (i+1)'s difference,
    # rescaled so the perturbation has exactly the magnitude variant i's has
    perm = (np.arange(len(picked)) + 1) % len(picked)

    # ---- pass 2: decode every (variant, gamma, condition) -------------------
    conds = [("true", g) for g in gammas] + [("perm", g) for g in cgammas]
    CA = np.zeros((len(picked), len(conds), len(ca_wt), 3), np.float32)
    PL = np.zeros((len(picked), len(conds), len(ca_wt)), np.float32)
    done = 0
    for i, (f_m, emb_m, hm) in enumerate(feats_cache):
        k_i = jax.random.key(9000 + i)          # fixed across gamma, by design
        for c, (kind, g) in enumerate(conds):
            if kind == "true":
                s_i, z_i = ds[i], dz[i]
            else:
                j = perm[i]
                scale = nz[i] / max(nz[j], 1e-9)
                s_i, z_i = ds[j] * scale, dz[j] * scale
            CA[i, c], PL[i, c] = decode(
                f_m, emb_m, st_wt.s + g * s_i, st_wt.z + g * z_i, k_i)
            done += 1
        hm.cleanup()
        if (i + 1) % 10 == 0:
            print(f"[{time.time()-t0:6.1f}s] decoded {i+1}/{len(picked)} variants "
                  f"({done} structures)", flush=True)

    np.savez_compressed(
        args.out, ca=CA, plddt=PL, ca_wt=ca_wt, plddt_wt=pl_wt,
        cond_kind=np.array([c[0] for c in conds]),
        cond_gamma=np.array([c[1] for c in conds], np.float32),
        mutant=np.array([m[0] for m in meta]),
        pos=np.array([m[1] for m in meta]),
        score=np.array([m[2] for m in meta], np.float32),
        dz_norm=nz.astype(np.float32), perm=perm,
        wt_seq=np.array(wt), assay=np.array(args.assay),
        sampling_steps=np.array(args.sampling_steps),
    )
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}  "
          f"({len(picked)} variants x {len(conds)} conditions)", flush=True)
    for c, (kind, g) in enumerate(conds):
        print(f"    {kind:5s} gamma={g:<4g} mean pLDDT {PL[:, c].mean():.3f}")


if __name__ == "__main__":
    main()
