"""One extractor, every model, one on-disk schema.

Replaces the per-model `exp_distomap_*.py` scripts. Featurisation is
model-specific (and alignment-controlled -- see pi_models.features_for), but the
output is identical in shape and meaning for every model, so every analysis and
figure script runs unchanged on any of them.

Schema (matches exp_distomap.py, so fig_mechanism.py / fig_crossmodel.py work):
    <id>__ed  <id>__kl  <id>__entropy  <id>__ca  <id>__plddt  <id>__logits
    ids  bin_centres  n_bins  model  mutations  msa_depth

The MSA server is blocked before anything is featurised: if a wrapper tries to
fetch its own alignment the run dies instead of silently comparing models on
different inputs.
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import pi_models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=pi_models.available())
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids", required=True, help="comma list; first is the reference")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax
    blocked = pi_models.block_network()
    print(f"MSA server blocked at: {blocked}", flush=True)

    data, work = Path(args.data), Path(args.work)
    ids = [i.strip() for i in args.ids.split(",")]
    man = {r["id"]: r for r in csv.DictReader((data / "manifest.csv").open())}

    t0 = time.time()
    model = pi_models.load(args.model)
    print(f"[{time.time()-t0:6.1f}s] {args.model} loaded; {len(ids)} sequences", flush=True)

    out, ref_p, depths = {}, None, []
    for cid in ids:
        y = (data / "yamls" / f"{cid}.yaml").read_text()
        seq = [l.split(":", 1)[1].strip() for l in y.splitlines()
               if l.strip().startswith("sequence:")][0]
        a3m = data / "msa" / f"{cid}.a3m"
        feats, depth = pi_models.features_for(
            args.model, model, seq, str(a3m), work=work / cid)
        depths.append(depth)
        o = model.model_output(features=feats, recycling_steps=args.recycles,
                               sampling_steps=args.sampling_steps,
                               key=jax.random.key(0))
        e = pi_models.extraction_from(o, name=args.model)

        kl = (np.zeros_like(e.entropy) if ref_p is None
              else pi_models.sym_kl(e.p, ref_p))
        if ref_p is None:
            ref_p = e.p

        out[f"{cid}__ed"] = e.ed
        out[f"{cid}__kl"] = kl.astype(np.float32)
        out[f"{cid}__entropy"] = e.entropy
        out[f"{cid}__ca"] = e.ca
        out[f"{cid}__plddt"] = e.plddt
        out[f"{cid}__logits"] = e.logits
        print(f"[{time.time()-t0:6.1f}s] {cid:16s} N={e.ed.shape[0]} msa={depth} "
              f"meanKL={kl.mean():.4f} entropy={e.entropy.mean():.4f} "
              f"pLDDT={e.plddt.mean():.3f}", flush=True)

    out["ids"] = np.array(ids)
    out["model"] = np.array(args.model)
    out["n_bins"] = np.array(e.n_bins)
    out["bin_centres"] = e.centres
    out["msa_depth"] = np.array(depths)
    out["mutations"] = np.array([man.get(i, {}).get("mutations", "") for i in ids])
    np.savez_compressed(args.out, **out)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}  "
          f"(bins={e.n_bins}, grid {e.centres[0]:.3f}..{e.centres[-1]:.3f} A)", flush=True)


if __name__ == "__main__":
    main()
