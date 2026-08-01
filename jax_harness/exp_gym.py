"""Experiment 13 -- per-layer internal features for ProteinGym variants.

Goal: predict measured mutational effect (Tsuboyama 2023 folding stability)
from the model's *internal* state, and find which layer does it best -- rather
than assuming the last one, and rather than using only the model's own outputs.

For each single-mutant variant we run the trunk once and record, for all 64
Pairformer layers:

  kl_glob[L]   mean symmetric KL between mutant and wild-type distogram over
               sampled residue pairs
  kl_site[L]   the same, restricted to pairs involving the mutated position
  dz_site[L]   ||z_mut - z_wt|| at the mutated position's rows
  ds_site[L]   ||s_mut - s_wt|| at the mutated position

plus the model's own readouts (pLDDT, and mean |dE[d]|) as baselines. That is
4x64 + 2 features per variant -- enough for a layer sweep and a small MLP,
small enough to store for thousands of variants.

Cost control: these assays are 60-70 aa, the MSA is grafted from the wild type
(so featurisation is identical apart from row 0), and no diffusion is run. The
trunk is the only expense.
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

YAML_TMPL = """version: 1

sequences:
  - protein:
      id: "A"
      sequence: {seq}
      msa: {msa}
"""


def graft_a3m(dst: Path, src: Path, query: str, wt: str, cap: int | None = None):
    """Write the variant's a3m: its own query row plus the wild-type alignment.

    `cap` truncates the homolog rows. Boltz re-parses the whole alignment for
    every variant inside `process_inputs`, so with thousands of variants the
    alignment depth -- not the GPU -- sets the wall-clock (an 827-row assay runs
    ~2x faster per variant than a 14,243-row one). Capping keeps depth an
    exactly controlled quantity across variants, which the comparison needs
    anyway, and makes large sweeps affordable."""
    lines = src.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(">"))
    body = []
    for h, s in zip(lines[i + 2 :: 2], lines[i + 3 :: 2]):
        core = "".join(c for c in s if not c.islower()).replace("-", "")
        if core == wt:      # drop the wild-type self-hit; see build_dataset
            continue
        body += [h, s]
        if cap is not None and len(body) >= 2 * cap:
            break
    dst.write_text("\n".join([">A", query, *body]) + "\n")


def capture(model, feats, ii, jj, *, recycles, key):
    """Per-layer distogram logits at sampled pairs, plus per-layer s and z rows."""
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
            "s": s_[0],
            "zrow": z_[0].mean(axis=1),      # per-residue mean over partners
        }

    s, z, per = pi.pairformer_capture(
        model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=True, reduce_fn=reduce_fn,
    )
    return per


def skl(la, lb):
    def sm(x):
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)
    pa, pb = sm(la), sm(lb)
    return ((pa - pb) * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True, help="scratch dir for per-variant yamls")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-variants", type=int, default=400)
    ap.add_argument("--n-pairs", type=int, default=1500)
    ap.add_argument("--msa-cap", type=int, default=None,
                    help="truncate the grafted alignment to this many homologs; "
                         "the per-variant cost is dominated by re-parsing it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(Path(args.assay_dir) / f"{args.assay}.csv"))
            if ":" not in r["mutant"]]
    wt = rows[0]["mutated_sequence"]
    m = re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"])
    wt = list(wt); wt[int(m.group(2)) - 1] = m.group(1); wt = "".join(wt)

    rng = np.random.default_rng(args.seed)
    if len(rows) > args.n_variants:
        idx = rng.choice(len(rows), args.n_variants, replace=False)
        rows = [rows[i] for i in sorted(idx)]

    work = Path(args.work); (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)
    src = Path(args.a3m)

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] model loaded; assay {args.assay} "
          f"len={len(wt)} variants={len(rows)}", flush=True)

    def featurise(seq, tag):
        a3m = work / "msa" / f"{tag}.a3m"
        graft_a3m(a3m, src, seq, wt, cap=args.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=a3m.resolve()))
        return pi.load_features(y.read_text())

    f_wt, h = featurise(wt, "wt")
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask)[0]
    a, b = rng.choice(valid, args.n_pairs), rng.choice(valid, args.n_pairs)
    keep = a != b
    ii_np, jj_np = a[keep], b[keep]
    ii, jj = jnp.asarray(ii_np), jnp.asarray(jj_np)
    ref = capture(model, f_wt, ii, jj, recycles=args.recycles, key=key)
    lw = np.asarray(ref["logits"]); sw = np.asarray(ref["s"]); zw = np.asarray(ref["zrow"])
    L = lw.shape[0]
    pos_of = {int(r): k for k, r in enumerate(valid)}
    print(f"[{time.time()-t0:6.1f}s] WT captured, {L} layers, {len(ii_np)} pairs", flush=True)
    h.cleanup()

    feats_out, meta = [], []
    for n, r in enumerate(rows):
        mo = re.match(r"([A-Z])(\d+)([A-Z])", r["mutant"])
        p0 = int(mo.group(2)) - 1
        if p0 not in pos_of:
            continue
        f_m, hm = featurise(r["mutated_sequence"], "mut")
        cur = capture(model, f_m, ii, jj, recycles=args.recycles, key=key)
        lm = np.asarray(cur["logits"]); sm_ = np.asarray(cur["s"]); zm = np.asarray(cur["zrow"])

        kl = skl(lm, lw)                                   # [L, P]
        at_site = (ii_np == p0) | (jj_np == p0)
        row = pos_of[p0]
        feats_out.append(np.concatenate([
            kl.mean(1),                                                  # kl_glob [L]
            kl[:, at_site].mean(1) if at_site.any() else np.zeros(L),    # kl_site [L]
            np.linalg.norm(zm[:, row] - zw[:, row], axis=-1),            # dz_site [L]
            np.linalg.norm(sm_[:, row] - sw[:, row], axis=-1),           # ds_site [L]
        ]).astype(np.float32))
        meta.append({"mutant": r["mutant"], "pos": p0,
                     "score": float(r["DMS_score"]), "bin": int(r["DMS_score_bin"])})
        hm.cleanup()
        if (n + 1) % 50 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(rows)}", flush=True)

    X = np.stack(feats_out)
    np.savez_compressed(args.out, X=X, n_layers=L,
                        blocks=np.array(["kl_glob", "kl_site", "dz_site", "ds_site"]),
                        score=np.array([m["score"] for m in meta]),
                        bin=np.array([m["bin"] for m in meta]),
                        pos=np.array([m["pos"] for m in meta]),
                        mutant=np.array([m["mutant"] for m in meta]))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}  X={X.shape}", flush=True)


if __name__ == "__main__":
    main()
