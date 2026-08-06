"""Does the mutation subspace CAUSE anything downstream, or is it only present?

Everything in this project so far is correlational. The trunk carries mutation
severity (+0.72 held out), the emitted structure does not (+0.31 at any
description size), and PC2 of the shared basis is severity-and-uncertainty at
once. None of that shows the model USES the direction; it shows the direction is
there to be read.

The intervention is cheap because of where the direction lives. PC2 was derived
from `dz_site`, the pair row at the mutated residue AFTER all 64 Pairformer
layers -- which is exactly the tensor the structure module is conditioned on. So
there is no need to inject inside the `lax.scan`: run the trunk normally, add a
vector to the final z, and hand the modified TrunkState to
`boltz2_forward_from_trunk`. Injecting mid-stack would test something different
and harder to interpret (the perturbation would be reshaped by the remaining
layers); injecting at the end asks the precise question leg 3 needs answered --
the trunk says X, does the structure module do anything about it?

Scaling. Directions are unit L2 norm in RAW z space and alpha is in multiples of
the MEDIAN ||dz_site|| of real mutations in that assay. So alpha = 1 moves the
mutation-site pair row by as much as a typical real mutation moves it. It is NOT
"as large as a real mutation" overall -- a substitution perturbs the whole pair
tensor, not one row -- which is why the sweep runs to 30x rather than stopping
at 1. A null at alpha = 1 alone could not tell "the model ignores this
direction" apart from "the intervention was too small to be a fair test".

Controls, without which this measures nothing:

  random directions   Any perturbation of that size will move the outputs
                      somewhat. The question is whether PC2 moves them MORE, or
                      differently. Random unit directions are drawn in the same
                      space with the same norm, so the only difference is
                      orientation.
  PC1 and PC3         PC1 is substitution volume and PC3 hydropathy; neither is
                      the stability axis. If all components behave alike, the
                      effect is about perturbation size, not about PC2.
  sampler drift       alpha = 0 reproduces the baseline EXACTLY (deterministic
                      sampling, fixed key), so it is a determinism check and
                      NOT a noise floor. The floor comes from re-running the
                      unperturbed state under different diffusion keys, which
                      is what bounds how large a coordinate change has to be
                      before it means anything.

Readouts span the whole downstream path, because the claim is about where the
signal stops: the distogram (a head on z), pLDDT (the confidence head, per
residue) and the emitted CA coordinates (the diffusion sampler's product).
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
import geom  # noqa: E402
import pi_core as pi  # noqa: E402
import pi_conf  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m  # noqa: E402
from exp_gym2 import _softmax  # noqa: E402
from exp_gym3 import trunk_capture  # noqa: E402
from joltz import TrunkState  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--basis", required=True, help="pc2_v*.npz, holds V and orient")
    ap.add_argument("--features", required=True, help="gym2s_*.npz for this assay")
    ap.add_argument("--work", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-pairs", type=int, default=1500)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--n-sites", type=int, default=4)
    ap.add_argument("--symmetric", action="store_true", default=True,
                    help="also inject into the transposed row; a real mutation "
                         "perturbs z[i,r] as well as z[r,j]")
    ap.add_argument("--n-random", type=int, default=4)
    # Dose-response rather than a narrow bracket. alpha=1 matches how much a
    # typical mutation moves the AVERAGED row, but a real mutation perturbs the
    # whole tensor, so a null at alpha=1 alone would not distinguish "the model
    # ignores this direction" from "the intervention is too small to be a fair
    # test". Going to 30x settles that.
    # SIGNED, and centred on the effect-matched scale. Two corrections to an
    # earlier version. (1) alpha=1 matches the norm of a typical mutation's
    # dz_site but produces only ~0.1x its site-level distogram effect, because
    # a substitution perturbs the whole pair tensor and this perturbs one row;
    # alpha~10 is the effect-matched dose. (2) Positive-only alphas throw away
    # the sharpest specificity test there is: PC2 is the broadening axis, so
    # +alpha should broaden and -alpha should sharpen, roughly antisymmetrically,
    # whereas a random direction has no privileged sign and should respond to
    # |alpha| instead. Sign structure, not effect size, is what separates a
    # direction the model uses from one that merely perturbs it.
    ap.add_argument("--alphas", default="-30,-10,-3,0,3,10,30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mosaic.losses.boltz2 import boltz2_forward_from_trunk

    alphas = [float(x) for x in args.alphas.split(",")]
    rows = [r for r in csv.DictReader(open(Path(args.assay_dir) / f"{args.assay}.csv"))
            if ":" not in r["mutant"]]
    wt = list(rows[0]["mutated_sequence"])
    m0 = re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"])
    wt[int(m0.group(2)) - 1] = m0.group(1)
    wt = "".join(wt)

    # ---- the directions, in raw z units -----------------------------------
    B = np.load(args.basis)
    V = np.asarray(B["V"], np.float64)                    # (n_pc, 128)
    orient = np.asarray(B["orient"], np.float64)
    V = V * orient[:, None]                               # same convention as the report
    F = np.load(args.features, allow_pickle=True)
    dz = np.asarray(F["dz_site"], np.float64)[:, -1, :]   # (n, 128) last layer
    sd_ch = dz.std(0) + 1e-9
    scale = float(np.median(np.linalg.norm(dz, axis=1)))  # typical mutation size

    def unit_raw(v):
        raw = v * sd_ch
        return raw / (np.linalg.norm(raw) + 1e-12)

    rng = np.random.default_rng(args.seed)
    dirs = {f"PC{i+1}": unit_raw(V[i]) for i in range(min(3, len(V)))}
    for r in range(args.n_random):
        dirs[f"random{r+1}"] = unit_raw(rng.normal(size=V.shape[1]))
    print(f"typical ||dz_site|| for a real mutation: {scale:.4f}  "
          f"(alpha = 1 injects this much)", flush=True)
    for a_, b_ in ((x, y) for x in dirs for y in dirs if x < y):
        c = float(np.dot(dirs[a_], dirs[b_]))
        if abs(c) > 0.3:
            print(f"   note: {a_} and {b_} are not orthogonal (cos {c:+.2f})")

    work = Path(args.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    a3m = work / "msa" / "wt.a3m"
    graft_a3m(a3m, Path(args.a3m), wt, wt, cap=args.msa_cap)
    y = work / "yamls" / "wt.yaml"
    y.write_text(YAML_TMPL.format(seq=wt, msa=a3m.resolve()))
    feats, h = pi.load_features(y.read_text())
    mask = np.asarray(feats["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask)[0]
    rp = np.random.default_rng(args.seed + 1)
    a, b = rp.choice(valid, args.n_pairs), rp.choice(valid, args.n_pairs)
    keep = a != b
    ii_np, jj_np = a[keep], b[keep]
    ii, jj = jnp.asarray(ii_np), jnp.asarray(jj_np)

    # sites spread along the chain, so the answer is not about one position
    sites = np.linspace(0, len(valid) - 1, args.n_sites + 2)[1:-1].astype(int)
    print(f"[{time.time()-t0:6.1f}s] injecting at rows {sites.tolist()}", flush=True)

    emb, trunk, per = trunk_capture(model, feats, ii, jj, 0,
                                    recycles=args.recycles, key=key)
    lw = np.asarray(per["logits"])[-1]
    mu_w, sd_w = pi_conf.moments(_softmax(lw.astype(np.float32)))

    def readout(z_new, at, dkey=7):
        """`at` selects the sampled pairs that touch the injected residue.

        Averaging the distogram change over ALL ~1479 pairs is the wrong
        statistic for a perturbation applied to one row: only ~3% of the pairs
        involve that residue, so a real local effect is diluted roughly
        thirtyfold and cannot be compared against the per-mutation numbers the
        rest of the project reports. Both are recorded -- the site-restricted
        one is the comparable quantity, the global one shows how far the
        perturbation spreads.
        """
        st = TrunkState(s=trunk.s, z=z_new)
        logits = np.asarray(model.distogram_module(z_new)[0, :, :, 0, :][ii, jj])
        mu, sd = pi_conf.moments(_softmax(logits.astype(np.float32)))
        out = boltz2_forward_from_trunk(
            model, feats, emb, st, num_sampling_steps=args.sampling_steps,
            deterministic=True, key=jax.random.fold_in(key, dkey))
        p = np.asarray(out.plddt)[mask]
        ca = np.asarray(out.backbone_coordinates)[mask][:, 1].astype(np.float32)
        d_sd, d_mu = sd - sd_w, mu - mu_w
        return {"d_sd_site": float(d_sd[at].mean()) if at.any() else np.nan,
                "d_mu_site": float(d_mu[at].mean()) if at.any() else np.nan,
                "abs_d_sd_site": float(np.abs(d_sd[at]).mean()) if at.any() else np.nan,
                "d_sd": float(d_sd.mean()), "d_mu": float(d_mu.mean()),
                "abs_d_mu": float(np.abs(d_mu).mean()),
                "plddt": float(p.mean()), "plddt_res": p.astype(np.float32),
                "ca": ca}

    def rmsd_to_base(ca, base_ca):
        """Superposed RMSD. The sampler emits each structure in its own rigid
        frame, so raw coordinate differences measure global placement, not
        shape -- an earlier version of this script did exactly that and read a
        change of diffusion key as an 18 A conformational change."""
        A = np.asarray(ca, float) - np.asarray(ca, float).mean(0)
        B = np.asarray(base_ca, float) - np.asarray(base_ca, float).mean(0)
        d = np.linalg.norm(A @ geom.kabsch(A, B).T - B, axis=1)
        return float(np.sqrt((d ** 2).mean())), float(d.max())

    all_at = np.ones(len(ii_np), bool)
    base = readout(trunk.z, all_at)
    print(f"[{time.time()-t0:6.1f}s] unperturbed pLDDT {base['plddt']:.4f}",
          flush=True)

    # alpha = 0 reproduces the baseline EXACTLY -- deterministic=True with a
    # fixed key -- so it is a determinism check, not a noise floor. The floor
    # that matters comes from re-running the sampler with different diffusion
    # keys, and from the random directions.
    drift = []
    for dk in (11, 12):
        r = readout(trunk.z, all_at, dkey=dk)
        rm, _ = rmsd_to_base(r["ca"], base["ca"])
        drift.append({"d_plddt": r["plddt"] - base["plddt"], "ca_rmsd": rm})
        print(f"[{time.time()-t0:6.1f}s] sampler drift, key {dk}: "
              f"dplddt {drift[-1]['d_plddt']:+.5f}  "
              f"caRMSD {drift[-1]['ca_rmsd']:.4f}", flush=True)

    # "glob" adds the direction to EVERY pair, not one row. Without it the null
    # is ambiguous: a single row is a small lever, and "the coordinates did not
    # move" could mean the structure module ignores this channel or merely that
    # the perturbation was too local to matter. A real substitution changes the
    # whole pair tensor, so this is the intervention that matches it in extent.
    # Site is irrelevant for it, so it runs once rather than per site.
    modes = ("row", "sym", "glob") if args.symmetric else ("row", "glob")
    recs = []
    for site in sites:
        tok = int(valid[site])
        at = (ii_np == tok) | (jj_np == tok)
        for mode in modes:
            if mode == "glob" and site != sites[0]:
                continue
            for dn, dv in dirs.items():
                for al in alphas:
                    zc = trunk.z
                    if al != 0.0:
                        delta = jnp.asarray(al * scale * dv, zc.dtype)
                        if mode == "glob":
                            zc = zc.at[0].add(delta)
                        else:
                            zc = zc.at[0, tok, :, :].add(delta)
                            if mode == "sym":
                                zc = zc.at[0, :, tok, :].add(delta)
                    r = readout(zc, at)
                    rms, rmax = rmsd_to_base(r["ca"], base["ca"])
                    recs.append({
                        "site": int(site), "dir": dn, "alpha": al, "mode": mode,
                        "d_mu": r["d_mu"], "d_sd": r["d_sd"],
                        "abs_d_mu": r["abs_d_mu"],
                        "d_sd_site": r["d_sd_site"], "d_mu_site": r["d_mu_site"],
                        "abs_d_sd_site": r["abs_d_sd_site"],
                        "plddt": r["plddt"], "d_plddt": r["plddt"] - base["plddt"],
                        "d_plddt_site": float(r["plddt_res"][site]
                                              - base["plddt_res"][site]),
                        "ca_rmsd": rms, "ca_max": rmax,
                    })
                    print(f"[{time.time()-t0:6.1f}s] site {site:3d} {mode:3s} "
                          f"{dn:9s} a={al:+5.1f}  dsd@site "
                          f"{r['d_sd_site']:+.4f}  dsd@all {r['d_sd']:+.5f}  "
                          f"dplddt@site {recs[-1]['d_plddt_site']:+.4f}  "
                          f"caRMSD {recs[-1]['ca_rmsd']:.4f}", flush=True)
    h.cleanup()

    np.savez_compressed(
        args.out, base_plddt=base["plddt"], base_plddt_res=base["plddt_res"],
        base_ca=base["ca"], scale=scale, sites=sites, alphas=np.array(alphas),
        dirs=np.array(list(dirs)), dir_vectors=np.stack(list(dirs.values())),
        drift_plddt=np.array([d["d_plddt"] for d in drift]),
        drift_ca=np.array([d["ca_rmsd"] for d in drift]),
        **{k: np.array([r[k] for r in recs])
           for k in ("site", "alpha", "d_mu", "d_sd", "abs_d_mu", "d_sd_site",
                     "d_mu_site", "abs_d_sd_site", "plddt", "d_plddt",
                     "d_plddt_site", "ca_rmsd", "ca_max")},
        rec_dir=np.array([r["dir"] for r in recs]),
        rec_mode=np.array([r["mode"] for r in recs]), assay=args.assay)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}  ({len(recs)} runs)")


if __name__ == "__main__":
    main()
