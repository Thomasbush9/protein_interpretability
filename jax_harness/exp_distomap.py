"""Full distogram maps and structures for the mechanism figure.

Everything else in this project reduces the distogram to a scalar. For the
figure that explains the mechanism, the N x N maps themselves are needed:
what the model believes about every residue pair for the wild type, for a
mutant, and the difference between them — alongside the predicted structures,
so "the belief changed / the structure did not" is visible rather than asserted.

Saved per sequence:
    ed        [N, N]      expected distance, E[d] = sum_b softmax(logits)_b * centre_b
    kl_vs_wt  [N, N]      symmetric KL against the wild type, same pairs
    entropy   [N, N]      entropy of each pair's distogram, in nats
    ca        [N, 3]      predicted CA coordinates
    plddt     [N]
    logits_row            full 64-bin distogram for a few chosen pairs, so the
                          histogram itself can be plotted rather than summarised
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids", required=True, help="comma list; first is the reference")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mosaic.losses.boltz2 import boltz2_forward_from_trunk

    data = Path(args.data)
    ids = [i.strip() for i in args.ids.split(",")]
    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] model loaded; {len(ids)} sequences", flush=True)

    centres = np.asarray(pi.BIN_CENTRES)
    out, ref_p = {}, None
    for cid in ids:
        feats, h = pi.load_features((data / "yamls" / f"{cid}.yaml").read_text())
        emb = model.embed_inputs(feats)
        tr = pi.run_trunk(model, emb, feats, recycling_steps=args.recycles,
                          key=key, deterministic=True, capture_last=False)
        st = tr["trunk_state"]
        mask = np.asarray(feats["token_pad_mask"][0]).astype(bool)
        logits = np.asarray(model.distogram_module(st.z)[0, :, :, 0, :])[np.ix_(mask, mask)]
        p = softmax(logits)
        ed = (p * centres).sum(-1)
        ent = -(p * np.log(p + 1e-12)).sum(-1)

        struct = boltz2_forward_from_trunk(
            model, feats, emb, st, num_sampling_steps=args.sampling_steps,
            deterministic=True, key=key)
        ca = np.asarray(struct.backbone_coordinates)[mask][:, 1]
        pl = np.asarray(struct.plddt)[mask]

        if ref_p is None:
            ref_p = p
            kl = np.zeros_like(ed)
        else:
            kl = ((p - ref_p) * (np.log(p + 1e-12) - np.log(ref_p + 1e-12))).sum(-1)

        out[f"{cid}__ed"] = ed.astype(np.float32)
        out[f"{cid}__kl"] = kl.astype(np.float32)
        out[f"{cid}__entropy"] = ent.astype(np.float32)
        out[f"{cid}__ca"] = ca.astype(np.float32)
        out[f"{cid}__plddt"] = pl.astype(np.float32)
        out[f"{cid}__logits"] = logits.astype(np.float32)
        print(f"[{time.time()-t0:6.1f}s] {cid:16s} N={mask.sum()} "
              f"meanE[d]={ed.mean():.2f}A  meanKL={kl.mean():.4f}  pLDDT={pl.mean():.3f}",
              flush=True)
        h.cleanup()

    man = {r["id"]: r for r in csv.DictReader((data / "manifest.csv").open())} \
        if (data / "manifest.csv").exists() else {}
    out["ids"] = np.array(ids)
    out["bin_centres"] = centres
    out["mutations"] = np.array([man.get(i, {}).get("mutations", "") for i in ids])
    np.savez_compressed(args.out, **out)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
