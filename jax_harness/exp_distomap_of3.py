"""OpenFold3 counterpart of exp_distomap.py -- the phenomenon, on a second model.

Question: is "the trunk's belief moves, the decoded structure does not" a fact
about Boltz-2, or about this class of architecture? This produces, for the same
GFP cohort and the same alignments, the arrays the Boltz-2 mechanism figure was
built from, so the two can be put side by side.

Saved per sequence (schema matches exp_distomap.py so fig_mechanism.py works):
    ed        [N, N]   expected distance, only meaningful if the bin grid is known
    kl_vs_wt  [N, N]   symmetric KL against the wild type, same pairs
    entropy   [N, N]
    ca        [N, 3]   representative (CA) atom per token
    plddt     [N]
    logits    [N, N, B]

TWO PITFALLS, both load-bearing:

1. **MSA file naming.** `parse_msas_direct` silently SKIPS any alignment whose
   basename is not a key of `msa.max_seq_counts` -- it does not warn, it just
   returns an empty dict and the pipeline dies later with an unrelated
   IndexError. Our colabfold output must therefore be presented as
   `colabfold_main.a3m`. This is the single reason a naive port fails.

2. **Templates.** `InferenceDataset.create_all_features` calls the template
   pipeline unconditionally, and with no template alignment supplied it crashes
   inside `create_template_restype` (`vectorize` on size-0 input). Setting
   `n_templates = 0` does not help -- it produces the empty arrays that cause
   the crash. We therefore override the step with an explicit **empty template
   block**: one template slot whose masks are all zero, i.e. no template
   information. This also keeps the cross-model comparison honest, because the
   Boltz-2 runs carry no templates either.

3. **Bin grid.** Boltz-2 uses 64 bins over 2-22 A. OF3's distogram head is a
   bare linear layer and carries no bin metadata, so we record `n_bins` and
   refuse to fabricate `ed` unless the count matches a grid we have asserted.
   Symmetric KL and entropy are computed over whatever the bins are and are
   valid regardless, because both distributions share the binning -- which is
   why the cross-model figure leans on KL rather than Angstrom.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import jax
import numpy as np

# Boltz-2's grid, asserted -- see pitfall 2
BOLTZ_MIN, BOLTZ_MAX, BOLTZ_BINS = 2.0, 22.0, 64


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cohort dir with manifest.csv, msa/")
    ap.add_argument("--ids", required=True, help="comma list; first is the reference")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--weights",
                    default="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
                            "mosaic_setup/weights/openfold3/jax/of3")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from jopenfold3.model import OpenFold3
    from jopenfold3.batch import Batch
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.config \
        .inference_query_format import InferenceQuerySet
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.config \
        import dataset_configs as dc
    from jopenfold3._vendor.openfold3.core.data.framework.single_datasets \
        .inference import InferenceDataset

    class NoTemplateInferenceDataset(InferenceDataset):
        """InferenceDataset with the template stage replaced by an empty block.

        One template slot, every mask zero. The template embedder therefore
        receives no information, which is what we want and what Boltz-2 gets.
        """

        def create_template_features(self, query, atom_array, n_tokens,
                                     *a, **k) -> dict:
            ntp = 1
            return {
                "template_distogram": np.zeros((ntp, n_tokens, n_tokens, 39), np.float32),
                "template_unit_vector": np.zeros((ntp, n_tokens, n_tokens, 3), np.float32),
                "template_restype": np.zeros((ntp, n_tokens, 32), np.int64),
                "template_backbone_frame_mask": np.zeros((ntp, n_tokens), np.float32),
                "template_pseudo_beta_mask": np.zeros((ntp, n_tokens), np.float32),
            }

    data, work = Path(args.data), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ids = [i.strip() for i in args.ids.split(",")]
    man = {r["id"]: r for r in csv.DictReader((data / "manifest.csv").open())}

    t0 = time.time()
    model = OpenFold3.load(args.weights)
    print(f"[{time.time()-t0:6.1f}s] OF3 loaded; {len(ids)} sequences", flush=True)

    def featurise(cid, seq):
        """Query JSON -> InferenceDataset -> Batch, with the a3m renamed (pitfall 1)."""
        d = work / cid
        d.mkdir(parents=True, exist_ok=True)
        a3m = d / "colabfold_main.a3m"          # NOT cid.a3m -- see pitfall 1
        shutil.copyfile(data / "msa" / f"{cid}.a3m", a3m)
        qj = d / "query.json"
        qj.write_text(json.dumps({
            "seeds": [42],
            "queries": {cid: {
                "query_name": cid, "use_msas": True,
                "use_paired_msas": False, "use_main_msas": True,
                "chains": [{"molecule_type": "protein", "chain_ids": ["A"],
                            "sequence": seq,
                            "main_msa_file_paths": [str(a3m.resolve())]}],
            }},
        }, indent=1))
        qs = InferenceQuerySet.from_json(qj)
        tps_cls = dc.InferenceJobConfig.model_fields[
            "template_preprocessor_settings"].annotation
        cfg = dc.InferenceJobConfig(query_set=qs, template_preprocessor_settings=tps_cls())
        ds = NoTemplateInferenceDataset(cfg)
        item = ds[0]
        if "restype" not in item:
            raise RuntimeError(
                f"featurisation produced no model features for {cid} "
                f"(keys: {sorted(item)}). Almost always the a3m naming pitfall.")
        return Batch.from_torch_dict(item)

    out, ref_p, n_bins = {}, None, None
    for cid in ids:
        # the cohort's Boltz yaml is the single source of truth for the sequence,
        # so OF3 and Boltz-2 are guaranteed to be fed identical strings
        y = (data / "yamls" / f"{cid}.yaml").read_text()
        seq = [l.split(":", 1)[1].strip() for l in y.splitlines()
               if l.strip().startswith("sequence:")][0]
        batch = featurise(cid, seq)

        res = model(batch, num_recycles=args.recycles,
                    num_sampling_steps=args.sampling_steps, num_samples=1,
                    key=jax.random.key(0), deterministic=True)

        mask = np.asarray(batch.token_mask[0]).astype(bool)
        logits = np.asarray(model.aux_heads.distogram(z=res.zij_trunk))
        logits = logits.reshape((-1,) + logits.shape[-3:])[0][np.ix_(mask, mask)]
        if n_bins is None:
            n_bins = logits.shape[-1]
            print(f"  distogram bins = {n_bins}"
                  f"{'  (matches the Boltz-2 grid)' if n_bins == BOLTZ_BINS else ''}",
                  flush=True)
        p = softmax(logits)
        ent = -(p * np.log(p + 1e-12)).sum(-1)

        # representative atom per token IS the CA for a standard residue
        ri = np.asarray(batch.representative_atom_index[0])[mask]
        coords = np.asarray(res.coordinates)
        coords = coords.reshape((-1,) + coords.shape[-2:])[0]
        ca = coords[ri]
        # OF3 exposes plddt as LOGITS over bins, per ATOM (not per token, as in
        # Boltz-2). Expectation over bin centres on [0,1], then indexed at the
        # representative atom so the result is per-residue and comparable.
        pll = np.asarray(res.confidence.plddt_logits)
        pll = pll.reshape((-1,) + pll.shape[-2:])[0]          # [Na, n_pl_bins]
        nb = pll.shape[-1]
        pl_centres = (np.arange(nb) + 0.5) / nb
        pl = (softmax(pll) * pl_centres).sum(-1)[ri]

        if ref_p is None:
            ref_p, kl = p, np.zeros(ent.shape)
        else:
            kl = ((p - ref_p) * (np.log(p + 1e-12) - np.log(ref_p + 1e-12))).sum(-1)

        if n_bins == BOLTZ_BINS:
            w = (BOLTZ_MAX - BOLTZ_MIN) / BOLTZ_BINS
            centres = BOLTZ_MIN + w / 2 + w * np.arange(BOLTZ_BINS)
            ed = (p * centres).sum(-1)
        else:
            ed = np.full(ent.shape, np.nan, np.float32)   # refuse to invent a grid

        out[f"{cid}__ed"] = ed.astype(np.float32)
        out[f"{cid}__kl"] = kl.astype(np.float32)
        out[f"{cid}__entropy"] = ent.astype(np.float32)
        out[f"{cid}__ca"] = ca.astype(np.float32)
        out[f"{cid}__plddt"] = pl.astype(np.float32)
        out[f"{cid}__logits"] = logits.astype(np.float32)
        print(f"[{time.time()-t0:6.1f}s] {cid:16s} N={mask.sum()} "
              f"meanKL={kl.mean():.4f} entropy={ent.mean():.4f} pLDDT={pl.mean():.3f}",
              flush=True)

    out["ids"] = np.array(ids)
    out["n_bins"] = np.array(n_bins)
    out["model"] = np.array("openfold3")
    out["mutations"] = np.array([man.get(i, {}).get("mutations", "") for i in ids])
    if n_bins == BOLTZ_BINS:
        w = (BOLTZ_MAX - BOLTZ_MIN) / BOLTZ_BINS
        out["bin_centres"] = BOLTZ_MIN + w / 2 + w * np.arange(BOLTZ_BINS)
    np.savez_compressed(args.out, **out)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
