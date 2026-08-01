"""Experiment 10 -- the circuit as a matrix over layers.

Everything so far has collapsed each layer to one number. That hides *where*
the mutation acts. This resolves the per-layer divergence two ways and emits
matrices, not curves:

  A.  residue x layer      [N, 64]   per-residue KL(mutant || WT), where a
                                     residue's value is the mean over its
                                     sampled partners. Answers: does the signal
                                     start at the mutated sites and spread, or
                                     is it delocalised from the first layer?

  B.  separation x layer    [B, 64]  the same KL binned by sequence separation
                                     |i-j|. Answers: does the mutation act on
                                     local contacts first and long-range later
                                     (the order the Pairformer's triangle
                                     operations would imply), or all at once?

Sampling is *stratified by residue* -- every residue i gets the same number of
partners j -- so row A is an unbiased per-residue estimate rather than whatever
a uniform pair sample happened to cover. Capturing full N x N logits for 64
layers would be ~0.9 GB per run; this is ~0.3 GB and exact for what we plot.

Divergence is symmetric KL, per the units lesson: E[d] in Angstrom is not
comparable across depth once the distogram starts sharpening.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
from joltz import TrunkState  # noqa: E402


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
    s, z, per = pi.pairformer_capture(
        model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=True,
        reduce_fn=lambda s_, z_: model.distogram_module(z_)[0, :, :, 0, :][ii, jj],
    )
    return np.asarray(per)          # [L, P, bins]


def skl_pairwise(la, lb):
    """Symmetric KL per pair -> [L, P]."""
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
    ap.add_argument("--mutants", default="gfp_core_32,gfp_surface_32")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--partners", type=int, default=80,
                    help="sampled partners per residue (stratified)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    manifest = {r["id"]: r for r in csv.DictReader((data / "manifest.csv").open())}

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    f_wt, h = pi.load_features((data / "yamls" / f"{args.wt}.yaml").read_text())
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask)[0]
    N = len(valid)
    rng = np.random.default_rng(args.seed)

    # stratified: every residue gets `partners` sampled partners
    ii_l, jj_l = [], []
    for i in valid:
        others = valid[valid != i]
        js = rng.choice(others, size=min(args.partners, len(others)), replace=False)
        ii_l.append(np.full(len(js), i))
        jj_l.append(js)
    ii_np, jj_np = np.concatenate(ii_l), np.concatenate(jj_l)
    ii, jj = jnp.asarray(ii_np), jnp.asarray(jj_np)
    print(f"[{time.time()-t0:6.1f}s] N={N}, {len(ii_np)} stratified pairs "
          f"({args.partners} partners/residue)", flush=True)

    lw = capture_logits(model, f_wt, ii, jj, recycles=args.recycles, key=key)
    L = lw.shape[0]
    print(f"[{time.time()-t0:6.1f}s] WT logits {lw.shape}", flush=True)

    sep = np.abs(ii_np - jj_np)
    edges = [1, 3, 6, 12, 24, 48, 96, 10**6]
    sep_bin = np.digitize(sep, edges[:-1], right=True)
    sep_labels = ["1-2", "3-5", "6-11", "12-23", "24-47", "48-95", ">=96"]
    pos_of = {int(r): k for k, r in enumerate(valid)}
    row_of_pair = np.array([pos_of[int(i)] for i in ii_np])

    out = {"n_tokens": int(N), "n_layers": int(L), "sep_labels": sep_labels,
           "residues": [int(v) for v in valid], "mutants": {}}

    for mid in args.mutants.split(","):
        mid = mid.strip()
        f_m, hm = pi.load_features((data / "yamls" / f"{mid}.yaml").read_text())
        lm = capture_logits(model, f_m, ii, jj, recycles=args.recycles, key=key)
        kl = skl_pairwise(lm, lw)                       # [L, P]

        # A. residue x layer
        res_mat = np.zeros((N, L))
        for r in range(N):
            sel = row_of_pair == r
            res_mat[r] = kl[:, sel].mean(axis=1)

        # B. separation x layer
        sep_mat = np.stack(
            [kl[:, sep_bin == b].mean(axis=1) if (sep_bin == b).any() else np.zeros(L)
             for b in range(len(sep_labels))]
        )

        muts = manifest[mid]["mutations"]
        mut_pos = sorted({int(m) - 1 for m in re.findall(r"[A-Z](\d+)[A-Z]", muts)}) if muts else []
        mut_rows = [pos_of[p] for p in mut_pos if p in pos_of]

        out["mutants"][mid] = {
            "residue_by_layer": res_mat.tolist(),
            "separation_by_layer": sep_mat.tolist(),
            "mutated_rows": mut_rows,
        }

        # how concentrated is the signal on mutated residues, per layer?
        if mut_rows:
            m = np.zeros(N, bool); m[mut_rows] = True
            enrich = res_mat[m].mean(axis=0) / np.maximum(res_mat[~m].mean(axis=0), 1e-12)
            out["mutants"][mid]["enrichment_at_mutated"] = enrich.tolist()
            print(f"\n=== {mid} === {len(mut_rows)} mutated residues")
            print("  KL enrichment at mutated vs non-mutated residues, every 8th layer:")
            print("   " + " ".join(f"{v:.2f}" for v in enrich[::8]))
            print(f"   L0 {enrich[0]:.2f}  peak L{int(enrich.argmax())} {enrich.max():.2f}"
                  f"  final {enrich[-1]:.2f}", flush=True)
        hm.cleanup()

    Path(args.out).write_text(json.dumps(out))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    h.cleanup()


if __name__ == "__main__":
    main()
