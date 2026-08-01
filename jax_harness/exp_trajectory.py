"""Experiment 16 -- does the mutation ever exist in the denoising trajectory?

The chain so far: the Pairformer carries the mutation (rho ~0.55), the diffusion
conditioning still carries it, the final coordinates do not. Widening the
sampler (beta on the pair biases) does not release it -- it degrades the signal.
So the loss is not a dispersion problem. The remaining question is *when*, during
the reverse diffusion, the difference disappears:

  NEVER GROWN     mutant and wild-type trajectories are indistinguishable at
                  every denoising step. The conditioning difference simply never
                  moves the score function.
  GROWN THEN LOST the trajectories separate at high noise and re-converge as
                  sigma falls. The sampler pulls both into the same basin --
                  an attractor, and a much more interesting claim.

`AtomDiffusion2.sample` runs `jax.lax.scan` over the noise schedule and discards
`ys`, exactly like the Pairformer stack. So the trajectory is obtained by
re-running that scan with `ys` populated -- no patching of joltz, and bit-identical
dynamics.

**The pairing problem, and the fix.** A substitution changes the side chain, so
the all-atom count differs between wild type and mutant (492 vs 491 for RCRO).
`shape = (*atom_mask.shape, 3)` therefore differs, so
`jax.random.normal(shape=shape, key=k)` draws a *different noise realisation*
even from an identical key. Mutant and wild-type trajectories are consequently
independent samples, not a paired comparison, and raw mutant-vs-WT divergence is
dominated by that — it simply tracks the sigma envelope (measured: 2605 A at
sigma 2560 down to 8.75 A, against ~1 A implied by the ensemble runs' TM 0.985).

The fix is a **within-sequence noise floor**. The wild type is sampled with
several independent keys; the pairwise divergence among *those* runs is exactly
"two independent samples of the same sequence at this step", which is the only
correct reference for a mutant-vs-WT number. A mutation has a detectable effect
on the trajectory at step s only if

    divergence(mutant, WT)  >  divergence(WT_key_i, WT_key_j)

at that step. Reported as the excess over the floor and as the ratio, both with
the floor's own spread, so "no effect" and "not measurable" stay distinguishable.

Reported per step: mutant-vs-WT divergence, the WT-vs-WT floor (mean and sd over
key pairs), each trajectory's distance to its own endpoint, and sigma, so the
x-axis has physical meaning and step indices are never compared across runs with
different schedules.
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
import geom  # noqa: E402  (self-tests the superposition on import)
import joltz  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m  # noqa: E402


def sample_with_trajectory(diff, atom_mask, num_sampling_steps, *, key, **cond):
    """Mirror of joltz AtomDiffusion2.sample, with the scan's ys populated.

    Every line is the library's; the only change is returning atom_coords_next
    as the scan output instead of None.
    """
    shape = (*atom_mask.shape, 3)
    sigmas = diff.sample_schedule(num_sampling_steps)
    gammas = jnp.where(sigmas > diff.gamma_min, diff.gamma_0, 0.0)
    step_scale = diff.step_scale

    @jax.checkpoint
    def body(carry, inp):
        (sigma_tm, sigma_t, gamma) = inp
        atom_coords, k = carry
        random_R, random_tr = joltz.compute_random_augmentation(key=k)
        k = jax.random.fold_in(k, 1)
        atom_coords = atom_coords - atom_coords.mean(axis=-2, keepdims=True)
        atom_coords = jnp.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr

        t_hat = sigma_tm * (1 + gamma)
        noise_var = diff.noise_scale**2 * (t_hat**2 - sigma_tm**2)
        eps = jnp.sqrt(noise_var) * jax.random.normal(shape=shape, key=k)
        k = jax.random.fold_in(k, 1)
        atom_coords_noisy = atom_coords + eps
        atom_coords_denoised = diff.preconditioned_network_forward(
            atom_coords_noisy, t_hat,
            network_condition_kwargs=dict(**cond), key=k,
        )
        if diff.alignment_reverse_diff:
            atom_coords_noisy = joltz.weighted_rigid_align(
                atom_coords_noisy, atom_coords_denoised, atom_mask, atom_mask)
        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
        atom_coords_next = (
            atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma)
        return (atom_coords_next, jax.random.fold_in(k, 0)), atom_coords_next

    init = (sigmas[0] * jax.random.normal(shape=shape, key=key),
            jax.random.fold_in(key, 1))
    (final, _), traj = jax.lax.scan(body, init, (sigmas[:-1], sigmas[1:], gammas[1:]))
    return final, traj, np.asarray(sigmas)


# Superposition lives in geom.py, which self-tests on import. A local copy of
# this was transposed and silently wrong; see the history note in geom.py.
kabsch_rmsd = geom.kabsch_rmsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-variants", type=int, default=100)
    ap.add_argument("--wt-keys", type=int, default=4,
                    help="independent WT samples; their pairwise divergence is "
                         "the noise floor every mutant number is measured against")
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(Path(args.assay_dir) / f"{args.assay}.csv"))
            if ":" not in r["mutant"]]
    wt = list(rows[0]["mutated_sequence"])
    m0 = re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"])
    wt[int(m0.group(2)) - 1] = m0.group(1)
    wt = "".join(wt)

    # Sample across the whole stability range, not just the extremes. The
    # previous version took the 12 most and 12 least destabilising, which makes
    # a rank correlation over a bimodal set and inflates it; n=24 was also far
    # too small (SE on a Spearman ~0.21).
    rng0 = np.random.default_rng(args.seed)
    picked = [rows[i] for i in sorted(rng0.choice(len(rows),
              min(args.n_variants, len(rows)), replace=False))]

    work = Path(args.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)
    src = Path(args.a3m)

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] {args.assay} len={len(wt)} "
          f"n={len(picked)} (extremes) steps={args.sampling_steps}", flush=True)

    def featurise(seq, tag):
        a3m = work / "msa" / f"{tag}.a3m"
        graft_a3m(a3m, src, seq, wt, cap=args.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=a3m.resolve()))
        return pi.load_features(y.read_text())

    def trajectory(feats, key_idx=2):
        emb = model.embed_inputs(feats)
        tr = pi.run_trunk(model, emb, feats, recycling_steps=args.recycles,
                          key=key, deterministic=True, capture_last=False)
        st = tr["trunk_state"]
        q, c, to_keys, aeb, adb, ttb = model.diffusion_conditioning(
            st.s, st.z, emb.relative_position_encoding, feats)
        with jax.default_matmul_precision("float32"):
            final, traj, sigmas = sample_with_trajectory(
                model.structure_module, feats["atom_pad_mask"], args.sampling_steps,
                key=jax.random.fold_in(key, key_idx),
                s_trunk=st.s, s_inputs=emb.s_inputs, feats=feats, multiplicity=1,
                diffusion_conditioning={"q": q, "c": c, "to_keys": to_keys,
                                        "atom_enc_bias": aeb, "atom_dec_bias": adb,
                                        "token_trans_bias": ttb},
            )
        # Compare CA atoms only. A substitution changes the side chain, so the
        # ALL-ATOM count differs between wild type and mutant (e.g. 492 vs 491)
        # and atom-level arrays are not comparable. Per mosaic's own backbone
        # extraction, the first atom of each token is N and CA is that +1
        # (ref_atoms["UNK"][:4] == ["N","CA","C","O"]).
        a2t = np.asarray(feats["atom_to_token"][0])          # [N_atom, N_token]
        first = a2t.argmax(0)                                # first atom per token
        tok = np.asarray(feats["token_pad_mask"][0]).astype(bool)
        ca_idx = (first + 1)[tok]
        return np.asarray(traj)[:, 0][:, ca_idx], sigmas

    f_wt, h = featurise(wt, "wt")
    wt_trajs = []
    for j in range(args.wt_keys):
        tj, sigmas = trajectory(f_wt, key_idx=2 + 100 * j)
        wt_trajs.append(tj)
    traj_wt = wt_trajs[0]
    S = traj_wt.shape[0]
    # noise floor: pairwise divergence among independent WT samples, per step
    floor = np.array([[kabsch_rmsd(a[s], b[s]) for s in range(S)]
                      for i, a in enumerate(wt_trajs) for b in wt_trajs[i + 1:]])
    print(f"[{time.time()-t0:6.1f}s] WT: {args.wt_keys} independent samples, "
          f"traj {traj_wt.shape}; sigma {sigmas[0]:.1f} -> {sigmas[-2]:.3f}", flush=True)
    print(f"           noise floor final step: {floor[:, -1].mean():.3f} "
          f"+/- {floor[:, -1].std():.3f} A over {len(floor)} key pairs", flush=True)
    h.cleanup()

    div, conv_m, meta, end_mut = [], [], [], []
    conv_wt = np.array([kabsch_rmsd(traj_wt[s], traj_wt[-1]) for s in range(S)])
    for n, r in enumerate(picked):
        f_m, hm = featurise(r["mutated_sequence"], "mut")
        traj_m, _ = trajectory(f_m)
        div.append([kabsch_rmsd(traj_m[s], traj_wt[s]) for s in range(S)])
        conv_m.append([kabsch_rmsd(traj_m[s], traj_m[-1]) for s in range(S)])
        end_mut.append(traj_m[-1])
        meta.append((r["mutant"], int(re.match(r"[A-Z](\d+)", r["mutant"]).group(1)) - 1,
                     float(r["DMS_score"])))
        hm.cleanup()
        if (n + 1) % 20 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(picked)}", flush=True)

    div = np.array(div); conv_m = np.array(conv_m)
    np.savez_compressed(
        args.out, divergence=div, conv_wt=conv_wt, conv_mut=conv_m, sigmas=sigmas,
        floor=floor, n_wt_keys=args.wt_keys,
        # endpoints, so TM-scores and any other geometry can be recomputed on a
        # login node without re-running the sampler
        ca_wt=np.stack([t[-1] for t in wt_trajs]).astype(np.float32),
        ca_mut=np.stack(end_mut).astype(np.float32),
        score=np.array([m[2] for m in meta]), pos=np.array([m[1] for m in meta]),
        mutant=np.array([m[0] for m in meta]), wt_seq=wt, n_steps=S,
    )
    m = div.mean(0)
    fl = floor.mean(0)
    ratio = m / np.maximum(fl, 1e-9)
    print(f"\n  mutant-vs-WT divergence RELATIVE TO the same-sequence noise floor")
    print(f"  ({len(div)} variants; floor from {len(floor)} WT key pairs)\n")
    print(f"    {'step':>5s} {'sigma':>9s} {'mut-vs-WT':>10s} {'WT floor':>10s} "
          f"{'ratio':>7s} {'excess':>8s}")
    for s in list(range(0, S, 20)) + [S - 1]:
        print(f"    {s:5d} {sigmas[s]:9.3f} {m[s]:10.3f} {fl[s]:10.3f} "
              f"{ratio[s]:7.3f} {m[s]-fl[s]:+8.3f}")
    b = int(np.argmax(ratio))
    print(f"\n    ratio peaks at step {b} (sigma {sigmas[b]:.3f}): {ratio[b]:.3f}")
    print(f"    ratio at final step: {ratio[-1]:.3f}   "
          f"(1.0 = mutation indistinguishable from resampling the wild type)", flush=True)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
