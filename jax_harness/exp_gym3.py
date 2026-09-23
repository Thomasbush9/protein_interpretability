"""Experiment 14c -- everything gym2s captured, plus the two things it did not.

`analyze_symmetry` showed that the emitted structure carries almost none of the
mutation signal the trunk carries, at any description size from ten dimensions
to 1741. That result has one hole in it, and this run exists to close it.

  PER-RESIDUE pLDDT.  The whole claim is that the internal response is
      predictive UNCERTAINTY. The model has an uncertainty output -- pLDDT --
      and the comparison so far gave it two numbers: the chain mean and the
      value at the mutated residue. Comparing a rich internal notion of
      certainty against an impoverished version of the model's own certainty
      head is exactly the objection a referee should raise. If per-residue
      pLDDT turns out to carry the signal, the finding is not "the model does
      not express it" but "the model expresses it in the confidence head and
      not in the coordinates" -- a different paper, and one worth knowing about
      before submission rather than after. The full vector is already computed
      inside `structure_of`; gym2 simply threw it away.

  THE FULL PAIR ROW.  `z_site` is averaged over partner residues, so no
      archived quantity can say where in the protein a component acts.
      `analyze_pc2` got at this indirectly through the distogram and found PC2
      to be an amplitude rather than a location, but the distogram is a
      projection of z through a head, and could discard structure z carries.
      Storing dz[r, j, :] settles it directly.

Deliberately NOT changed: the trunk is called exactly as gym2 called it, twice
per variant (once on wild type at the same row, once on the mutant), with the
same recycles, alignment cap, pair sample and keys. Caching the wild-type pass
would roughly halve the wall clock and is algebraically equivalent, but this run
has to be comparable to `gym2s_*` variant-for-variant, and a subtle divergence
in a rewritten hot path would be nearly impossible to detect after the fact. The
archived kl_glob is re-emitted so that comparability can be checked rather than
assumed.

Layer subsampling for the pair row only: a full (64, N, N, 128) tensor per
variant is ~170 MB, so the row is kept for a coarse sweep of early layers plus
every one of the last eight -- the window every downstream analysis reads.
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
import pi_conf  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m, skl  # noqa: E402
from exp_gym2 import _softmax, shape_features  # noqa: E402
from joltz import TrunkState  # noqa: E402


def keep_layers(n_layers, stride=8, tail=8):
    """Coarse sweep of the depth plus every layer of the tail window."""
    return sorted(set(list(range(0, n_layers, stride))
                      + list(range(max(0, n_layers - tail), n_layers))))


def trunk_capture(model, feats, ii, jj, row, *, recycles, key):
    """As exp_gym2.trunk_capture, plus the unaveraged pair row at `row`."""
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
            "z_row": z_[0][row],                      # [N_padded, 128]
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
    ap.add_argument("--row-stride", type=int, default=8)
    ap.add_argument("--row-tail", type=int, default=8)
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
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.n_variants,
                                                   replace=False))]

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
        """pLDDT per residue (not just two summaries of it), plus CA coords."""
        out = boltz2_forward_from_trunk(
            model, feats, emb, trunk, num_sampling_steps=args.sampling_steps,
            deterministic=True, key=jax.random.fold_in(key, 7),
        )
        p = np.asarray(out.plddt)[mask]
        ca = np.asarray(out.backbone_coordinates)[mask][:, 1].astype(np.float32)
        return p.astype(np.float32), float(p.mean()), float(p[row]), ca

    emb_w, tr_w, ref = trunk_capture(model, f_wt, ii, jj, 0, recycles=args.recycles,
                                     key=key)
    lw = np.asarray(ref["logits"])
    L = lw.shape[0]
    KEEP = keep_layers(L, args.row_stride, args.row_tail)
    mu_w, sd_w = pi_conf.moments(_softmax(lw.astype(np.float32)))
    pl_res_wt, pl_wt_mean, _, ca_wt = structure_of(emb_w, tr_w, f_wt, 0)
    print(f"[{time.time()-t0:6.1f}s] WT: {L} layers, pLDDT {pl_wt_mean:.3f}, "
          f"pair row kept for {len(KEEP)} layers {KEEP}", flush=True)
    h.cleanup()

    Z, ZR, S, KLg, KLs, PL, PLs, PLR, CA, DIS, meta = ([] for _ in range(11))
    SHP = []
    for n, r in enumerate(rows):
        mo = re.match(r"([A-Z])(\d+)([A-Z])", r["mutant"])
        p0 = int(mo.group(2)) - 1
        if p0 not in pos_of:
            continue
        row = pos_of[p0]
        f_m, hm = featurise(r["mutated_sequence"], "mut")
        _, _, refr = trunk_capture(model, f_wt, ii, jj, row, recycles=args.recycles,
                                   key=key)
        emb_m, tr_m, cur = trunk_capture(model, f_m, ii, jj, row,
                                         recycles=args.recycles, key=key)
        lm = np.asarray(cur["logits"])
        kl = skl(lm, lw)
        at = (ii_np == p0) | (jj_np == p0)
        SHP.append(shape_features(lw, mu_w, sd_w, lm, at))
        pl_res, pm, ps, ca = structure_of(emb_m, tr_m, f_m, row)
        dis = np.asarray(cur["logits"])[-1]

        Z.append((np.asarray(cur["z_site"]) - np.asarray(refr["z_site"])).astype(np.float32))
        # padded columns are dropped here, not inside the scan: the reduce_fn
        # has to stay a pure function of traced arrays
        zr = (np.asarray(cur["z_row"]) - np.asarray(refr["z_row"]))[:, mask, :]
        ZR.append(zr[KEEP].astype(np.float32))
        S.append((np.asarray(cur["s_site"]) - np.asarray(refr["s_site"])).astype(np.float32))
        KLg.append(kl.mean(1).astype(np.float32))
        KLs.append((kl[:, at].mean(1) if at.any() else np.zeros(L)).astype(np.float32))
        PL.append(pm); PLs.append(ps); PLR.append(pl_res)
        CA.append(ca); DIS.append(dis.astype(np.float32))
        meta.append((r["mutant"], p0, float(r["DMS_score"]), int(r["DMS_score_bin"])))
        hm.cleanup()
        if (n + 1) % 25 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(rows)}", flush=True)

    shape = {k: np.stack([s[k] for s in SHP]) for k in SHP[0]} if SHP else {}
    np.savez_compressed(
        args.out, dz_site=np.stack(Z), ds_site=np.stack(S),
        dz_row=np.stack(ZR), dz_row_layers=np.array(KEEP),
        kl_glob=np.stack(KLg), kl_site=np.stack(KLs), **shape,
        plddt=np.array(PL), plddt_site=np.array(PLs),
        plddt_res=np.stack(PLR), plddt_res_wt=pl_res_wt,
        plddt_wt=pl_wt_mean, n_layers=L,
        ca=np.stack(CA), ca_wt=ca_wt, disto=np.stack(DIS),
        disto_wt=lw[-1].astype(np.float32),
        score=np.array([m[2] for m in meta]), bin=np.array([m[3] for m in meta]),
        pos=np.array([m[1] for m in meta]), mutant=np.array([m[0] for m in meta]),
        wt_seq=wt,
    )
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}  n={len(meta)}  "
          f"dz_row {np.stack(ZR).shape}", flush=True)


if __name__ == "__main__":
    main()
