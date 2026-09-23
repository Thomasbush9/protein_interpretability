"""Take a REAL mutation, delete one direction from what it did, and see what changes.

The steering experiment added a synthetic vector to z and found the model
responds to perturbation size rather than to direction. That is a fair test of
"is PC2 a lever", but a poor test of "does PC2 carry the phenotype", because an
injected vector is not something the model ever produces.

This is the better-posed version. Run the trunk on wild type and on an actual
variant, take the difference the model itself computed,

    Dz = z_mut - z_wt          (the whole pair tensor, not one row)

remove a single direction from it,

    Dz_ablated = Dz - <Dz, v> v      applied at every pair

and re-run the structure module on z_wt + Dz_ablated. Nothing synthetic is
added; one of 128 dimensions is deleted from a real mutational response. If PC2
is the phenotype channel, removing it should cost the model more than removing
an arbitrary direction.

Two readouts answer two different questions.

  probe    Does the ablation actually remove the decodable signal? The PC2
           score of the ablated Dz should collapse to ~0 by construction, and
           that is checked rather than assumed -- it is the positive control
           for the surgery having worked at all.

  output   Do the distogram, per-residue pLDDT and emitted coordinates care?
           This is the question. Ablating the direction that carries severity
           should, if the structure module reads it, change the structure more
           than ablating a random direction does.

Controls: PC1 (volume, not stability), and random unit directions drawn in the
same space. Ablating ANY single direction out of 128 removes some variance, so
the comparison is always PC2 against those, never PC2 against nothing.
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
    ap.add_argument("--basis", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-pairs", type=int, default=1500)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--n-variants", type=int, default=16)
    ap.add_argument("--n-random", type=int, default=3)
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

    # Variants spanning the DMS range, so the comparison is not confined to
    # near-neutral substitutions where there is little signal to remove.
    rows.sort(key=lambda r: float(r["DMS_score"]))
    pick = np.linspace(0, len(rows) - 1, args.n_variants).astype(int)
    rows = [rows[i] for i in pick]

    B = np.load(args.basis)
    V = np.asarray(B["V"], np.float64) * np.asarray(B["orient"], np.float64)[:, None]
    F = np.load(args.features, allow_pickle=True)
    dzf = np.asarray(F["dz_site"], np.float64)[:, -1, :]
    sd_ch = dzf.std(0) + 1e-9

    def unit_raw(v):
        raw = v * sd_ch
        return raw / (np.linalg.norm(raw) + 1e-12)

    rng = np.random.default_rng(args.seed)
    dirs = {"PC1": unit_raw(V[0]), "PC2": unit_raw(V[1])}
    for r in range(args.n_random):
        dirs[f"random{r+1}"] = unit_raw(rng.normal(size=V.shape[1]))

    work = Path(args.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)

    def featurise(seq, tag):
        a3m = work / "msa" / f"{tag}.a3m"
        graft_a3m(a3m, Path(args.a3m), seq, wt, cap=args.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=a3m.resolve()))
        return pi.load_features(y.read_text())

    f_wt, h = featurise(wt, "wt")
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask)[0]
    pos_of = {int(r): k for k, r in enumerate(valid)}
    rp = np.random.default_rng(args.seed + 1)
    a_, b_ = rp.choice(valid, args.n_pairs), rp.choice(valid, args.n_pairs)
    keep = a_ != b_
    ii_np, jj_np = a_[keep], b_[keep]
    ii, jj = jnp.asarray(ii_np), jnp.asarray(jj_np)

    emb_w, tr_w, ref_w = trunk_capture(model, f_wt, ii, jj, 0,
                                       recycles=args.recycles, key=key)
    lw = np.asarray(ref_w["logits"])[-1]
    mu_w, sd_w = pi_conf.moments(_softmax(lw.astype(np.float32)))
    print(f"[{time.time()-t0:6.1f}s] WT trunk done, {len(rows)} variants", flush=True)

    def readout(emb, z, feats, at):
        st = TrunkState(s=tr_w.s, z=z)
        logits = np.asarray(model.distogram_module(z)[0, :, :, 0, :][ii, jj])
        mu, sd = pi_conf.moments(_softmax(logits.astype(np.float32)))
        out = boltz2_forward_from_trunk(
            model, feats, emb, st, num_sampling_steps=args.sampling_steps,
            deterministic=True, key=jax.random.fold_in(key, 7))
        p = np.asarray(out.plddt)[mask]
        ca = np.asarray(out.backbone_coordinates)[mask][:, 1].astype(np.float32)
        return {"d_sd_site": float((sd - sd_w)[at].mean()) if at.any() else np.nan,
                "d_sd": float((sd - sd_w).mean()),
                "plddt": float(p.mean()),
                "plddt_res": p.astype(np.float32), "ca": ca}

    def rmsd(x, y):
        A = np.asarray(x, float) - np.asarray(x, float).mean(0)
        Bq = np.asarray(y, float) - np.asarray(y, float).mean(0)
        return float(np.sqrt((np.linalg.norm(A @ geom.kabsch(A, Bq).T - Bq,
                                             axis=1) ** 2).mean()))

    recs = []
    for n, r in enumerate(rows):
        mo = re.match(r"([A-Z])(\d+)([A-Z])", r["mutant"])
        p0 = int(mo.group(2)) - 1
        if p0 not in pos_of:
            continue
        row = pos_of[p0]
        at = (ii_np == p0) | (jj_np == p0)
        f_m, hm = featurise(r["mutated_sequence"], "mut")
        emb_m, tr_m, _ = trunk_capture(model, f_m, ii, jj, row,
                                       recycles=args.recycles, key=key)
        dz = tr_m.z - tr_w.z                       # what the mutation did
        base = readout(emb_m, tr_m.z, f_m, at)

        for dn, dv in dirs.items():
            v = jnp.asarray(dv, dz.dtype)
            proj = jnp.tensordot(dz, v, axes=([-1], [0]))[..., None] * v
            z_ab = tr_w.z + (dz - proj)
            # positive control: the component we removed should be gone
            resid = float(jnp.abs(jnp.tensordot(dz - proj, v,
                                                axes=([-1], [0]))).mean())
            before = float(jnp.abs(jnp.tensordot(dz, v, axes=([-1], [0]))).mean())
            r_ab = readout(emb_m, z_ab, f_m, at)
            recs.append({
                "mutant": r["mutant"], "dms": float(r["DMS_score"]), "dir": dn,
                "proj_before": before, "proj_after": resid,
                "d_sd_site_full": base["d_sd_site"],
                "d_sd_site_abl": r_ab["d_sd_site"],
                "plddt_full": base["plddt"], "plddt_abl": r_ab["plddt"],
                "ca_shift": rmsd(r_ab["ca"], base["ca"]),
                "ca_mut_vs_wt": np.nan,
            })
        hm.cleanup()
        if (n + 1) % 4 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(rows)}", flush=True)
    h.cleanup()

    np.savez_compressed(
        args.out, assay=args.assay,
        **{k: np.array([r[k] for r in recs])
           for k in ("dms", "proj_before", "proj_after", "d_sd_site_full",
                     "d_sd_site_abl", "plddt_full", "plddt_abl", "ca_shift")},
        rec_dir=np.array([r["dir"] for r in recs]),
        rec_mut=np.array([r["mutant"] for r in recs]))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}  ({len(recs)} ablations)")


if __name__ == "__main__":
    main()
