"""Experiment 2 -- does mutation sensitivity decay with MSA depth, and how?

OuterProductMean averages over MSA rows: the query occupies one row out of S,
so its share of the MSA -> pair write scales like 1/S. If that dilution is what
makes the model ignore mutations, then mutation sensitivity should fall
systematically as the alignment deepens, and should be maximal at S=1 (single
sequence, no alignment to hide behind).

The competing possibility is that the query's *undiluted* routes -- the direct
s_inputs -> z_init path and the s_proj broadcast into every MSA row -- carry the
signal, in which case sensitivity should be roughly flat in S.

Reported per depth:
  sens_A     mean |E[d](mutant) - E[d](WT)| over off-diagonal pairs, both run
             at that depth (the quantity the 1/S argument predicts)
  contact_disagree  fraction of pairs whose contact call (p>0.5) flips
  z_opm / z_init    magnitude of the MSA write vs the direct query write

Caveat: rows are taken in file order. A ColabFold a3m is roughly ordered by
decreasing similarity to the query, so shallow subsets are also *closer*
subsets -- depth and diversity move together here. `--shuffle-rows` draws a
random subset instead, which breaks that confound at the cost of comparability
with how Boltz's own subsampling behaves.
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
import pi_paths as pp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--mutant", default="gfp_core_32")
    ap.add_argument("--depths", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--shuffle-rows", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    f_wt, h_wt = pi.load_features((data / "yamls" / f"{args.wt}.yaml").read_text())
    f_m, h_m = pi.load_features((data / "yamls" / f"{args.mutant}.yaml").read_text())
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    S_full = f_wt["msa"].shape[1]
    print(f"[{time.time()-t0:6.1f}s] N={mask.sum()} S_full={S_full}", flush=True)

    depths = [d for d in (int(x) for x in args.depths.split(",")) if d <= S_full]
    if S_full not in depths:
        depths.append(S_full)

    perm = None
    if args.shuffle_rows:
        rng = np.random.default_rng(args.seed)
        # row 0 (the query) must stay first; shuffle only the homologs
        perm = np.concatenate([[0], rng.permutation(np.arange(1, S_full))])

    def prep(feats, depth):
        f = dict(feats)
        if perm is not None:
            for k in ("msa", "msa_mask", "has_deletion", "deletion_value", "msa_paired"):
                if k in f:
                    f[k] = jnp.asarray(np.asarray(f[k])[:, perm])
        return pp.truncate_msa(f, depth)

    def emap(z):
        d = np.asarray(pi.expected_distance(pi.logit_lens(model, z)))
        return d[np.ix_(mask, mask)]

    def cmap(z):
        c = np.asarray(pi.contact_prob(pi.logit_lens(model, z)))
        return c[np.ix_(mask, mask)]

    key = jax.random.key(0)
    rows = []
    n = int(mask.sum())
    off = ~np.eye(n, dtype=bool)

    for S in depths:
        fw, fm = prep(f_wt, S), prep(f_m, S)
        ew, em = model.embed_inputs(fw), model.embed_inputs(fm)
        R = dict(recycling_steps=args.recycles, key=key, deterministic=True)

        ow = pp.run_trunk(model, ew, fw, capture_last=True, **R)
        om = pp.run_trunk(model, em, fm, capture_last=False, **R)

        Dw, Dm = emap(ow["trunk_state"].z), emap(om["trunk_state"].z)
        Cw, Cm = cmap(ow["trunk_state"].z), cmap(om["trunk_state"].z)

        zi = float(jnp.linalg.norm(ow["z_after_init"][0], axis=-1)[np.ix_(mask, mask)].mean())
        zm = float(jnp.linalg.norm(ow["z_after_msa"][0], axis=-1)[np.ix_(mask, mask)].mean())
        opm = np.asarray(ow["msa_layers"]["opm_norm"])[:, mask][:, :, mask].mean(axis=(1, 2))

        row = {
            "depth": S,
            "eff_depth": pp.effective_depth(fw),
            "sens_A": float(np.abs(Dm - Dw)[off].mean()),
            "sens_max_A": float(np.abs(Dm - Dw)[off].max()),
            "contact_disagree": float(((Cm > 0.5) != (Cw > 0.5))[off].mean()),
            "z_after_init": zi,
            "z_after_msa": zm,
            "opm_per_block": [float(x) for x in opm],
        }
        rows.append(row)
        print(
            f"  S={S:5d} (eff {row['eff_depth']:7.1f})  sens={row['sens_A']:.4f} A"
            f"  max={row['sens_max_A']:6.3f}  contact_flip={row['contact_disagree']:.4f}"
            f"  |z|init={zi:7.2f} |z|msa={zm:7.2f}",
            flush=True,
        )

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    h_wt.cleanup()
    h_m.cleanup()


if __name__ == "__main__":
    main()
