"""Experiment 12 -- structure vs confidence vs internal state, on the same variants.

The thesis is that the mutation is registered more strongly inside the model
than at its output. That is only worth asserting if all three are measured on
the same sequences, with the same settings, in one run:

  TM-score      (tmtools, the reference TM-align) -- the output geometry
  pLDDT         -- the model's own confidence
  KL            symmetric KL between the mutant and wild-type distogram at the
                 trunk output -- the internal state

Also runs a *random* mutation series at 5/10/20/40/70 %, matching the levels in
the adversarial-mutation literature, alongside the targeted buried-core series.
That matters because the core series is only ~13 % mutated but chosen to be
maximally destabilising, so any dose-response it shows cannot be compared to a
random-mutation dose-response without running both.

Coordinates are saved so TM can be recomputed without re-running the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
from joltz import TrunkState  # noqa: E402


def trunk_logits_and_structure(model, feats, ii, jj, *, recycles, sampling_steps, key,
                               n_samples):
    """Final-layer distogram logits at sampled pairs, plus structures + pLDDT."""
    from mosaic.losses.boltz2 import boltz2_forward_from_trunk

    emb = model.embed_inputs(feats)
    trunk = pi.run_trunk(model, emb, feats, recycling_steps=recycles, key=key,
                         deterministic=True, capture_last=False)
    z = trunk["trunk_state"].z
    logits = np.asarray(model.distogram_module(z)[0, :, :, 0, :][ii, jj])

    mask = np.asarray(feats["token_pad_mask"][0]).astype(bool)
    cas, plddts = [], []
    for i in range(n_samples):
        out = boltz2_forward_from_trunk(
            model, feats, emb, trunk["trunk_state"],
            num_sampling_steps=sampling_steps, deterministic=True,
            key=jax.random.fold_in(jax.random.key(2000 + i), i),
        )
        cas.append(np.asarray(out.backbone_coordinates)[mask][:, 1])
        plddts.append(float(np.asarray(out.plddt)[mask].mean()))
    return logits, np.stack(cas), np.array(plddts)


def skl(la, lb):
    def sm(x):
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)
    pa, pb = sm(la), sm(lb)
    return float(((pa - pb) * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--ids", default=None, help="comma list; default = all non-WT in manifest")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--n-pairs", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coords-out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    man = {r["id"]: r for r in csv.DictReader((data / "manifest.csv").open())}
    ids = ([i.strip() for i in args.ids.split(",")] if args.ids
           else [i for i in man if i != args.wt])

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    f_wt, h = pi.load_features((data / "yamls" / f"{args.wt}.yaml").read_text())
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask)[0]
    rng = np.random.default_rng(0)
    a, b = rng.choice(valid, args.n_pairs), rng.choice(valid, args.n_pairs)
    keep = a != b
    ii, jj = jnp.asarray(a[keep]), jnp.asarray(b[keep])

    R = dict(recycles=args.recycles, sampling_steps=args.sampling_steps, key=key,
             n_samples=args.samples)
    lw, ca_wt, pl_wt = trunk_logits_and_structure(model, f_wt, ii, jj, **R)
    coords = {args.wt: ca_wt}
    print(f"[{time.time()-t0:6.1f}s] {args.wt}: pLDDT {pl_wt.mean():.3f}", flush=True)

    rows = [{"id": args.wt, "mode": "wt", "n_mut": 0, "pct_mut": 0.0,
             "kl": 0.0, "plddt": float(pl_wt.mean()), "plddt_sd": float(pl_wt.std())}]

    for cid in ids:
        f_m, hm = pi.load_features((data / "yamls" / f"{cid}.yaml").read_text())
        lm, ca, pl = trunk_logits_and_structure(model, f_m, ii, jj, **R)
        coords[cid] = ca
        n = int(man[cid]["n_mut"])
        rows.append({
            "id": cid, "mode": man[cid]["mode"], "n_mut": n,
            "pct_mut": 100.0 * n / int(man[cid]["seq_len"]),
            "kl": skl(lm, lw), "plddt": float(pl.mean()), "plddt_sd": float(pl.std()),
        })
        print(f"[{time.time()-t0:6.1f}s] {cid:18s} KL {rows[-1]['kl']:.4f}  "
              f"pLDDT {pl.mean():.3f}", flush=True)
        hm.cleanup()

    np.savez_compressed(args.coords_out, **{k: v for k, v in coords.items()})
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out} and {args.coords_out}", flush=True)
    print("TM-scores are computed separately with tmtools (analysis venv) from the "
          "saved coordinates -- see score_bench.py", flush=True)
    h.cleanup()


if __name__ == "__main__":
    main()
