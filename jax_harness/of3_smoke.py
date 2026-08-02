"""Exploratory smoke test for OpenFold3: can we get a distogram and coordinates?

This is the first step of porting the harness to a second model. It answers, in
one GPU run rather than a dozen greps:

  1. does `OpenFold3.load()` work off the weights on disk?
  2. can the vendored torch data pipeline featurise one of OUR sequences with
     OUR a3m, and does `Batch.from_torch_dict` accept the result?
  3. does a forward pass give a distogram and coordinates?
  4. is the Pairformer really a `lax.scan` over stacked params, i.e. does the
     per-layer capture technique port?

Deliberately defensive: every stage is wrapped, prints what it found, and keeps
going, so a failure at stage 3 still teaches us about stage 4.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def stage(n, name):
    print(f"\n{'=' * 70}\n[{n}] {name}\n{'=' * 70}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--weights",
                    default="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
                            "mosaic_setup/weights/openfold3/jax/of3")
    ap.add_argument("--work", required=True)
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    # ---- 1. load the model ------------------------------------------------
    stage(1, "OpenFold3.load()")
    model = None
    try:
        from jopenfold3.model import OpenFold3
        model = OpenFold3.load(args.weights)
        print(f"loaded. type={type(model).__name__}")
        pf = model.pairformer_stack
        print(f"pairformer_stack: {type(pf).__name__}")
        import jax
        leaves = jax.tree.leaves(pf.stacked_params)
        print(f"  stacked_params leaves={len(leaves)}, "
              f"example shape={leaves[0].shape}  <- leading dim = n blocks")
        print(f"  num_recycles default: {model.num_recycles}")
        print(f"  msa_module: {type(model.msa_module).__name__}")
        print(f"  fields: {[f.name for f in model.__dataclass_fields__.values()][:14]}")
    except Exception:
        traceback.print_exc()

    # ---- 2. build a query and featurise -----------------------------------
    # NOT via InferenceExperimentRunner: that constructs InferenceExperimentConfig,
    # which writes a checkpoint-root marker into ~/.openfold3 -- bound READ-ONLY by
    # mosaic-exec.sh. We do not need the torch checkpoint at all (the JAX weights
    # load separately in stage 1), only the data pipeline, so we build
    # InferenceJobConfig -> InferenceDataset directly.
    stage(2, "query JSON -> InferenceDataset -> Batch")
    batch = None
    qjson = work / "query.json"
    qjson.write_text(json.dumps({
        "seeds": [42],
        "queries": {"probe": {
            "query_name": "probe",
            "use_msas": True, "use_paired_msas": False, "use_main_msas": True,
            "chains": [{"molecule_type": "protein", "chain_ids": ["A"],
                        "sequence": args.seq,
                        "main_msa_file_paths": [str(Path(args.a3m).resolve())]}],
        }},
    }, indent=1))
    print(f"wrote {qjson}")
    try:
        from jopenfold3._vendor.openfold3.projects.of3_all_atom.config \
            .inference_query_format import InferenceQuerySet
        from jopenfold3._vendor.openfold3.projects.of3_all_atom.config \
            import dataset_configs as dc
        from jopenfold3._vendor.openfold3.core.data.framework.single_datasets \
            .inference import InferenceDataset
        from jopenfold3.batch import Batch

        qs = InferenceQuerySet.from_json(qjson)
        print(f"query set ok: {list(qs.queries)}")

        tps_field = dc.InferenceJobConfig.model_fields["template_preprocessor_settings"]
        tps_cls = tps_field.annotation
        print(f"template_preprocessor_settings type: {tps_cls}")
        try:
            tps = tps_cls()
            print("  default-constructed ok")
        except Exception as e:
            print(f"  default construction failed: {e}")
            tps = tps_cls(preparse_structures=False)
            print("  constructed with preparse_structures=False")

        cfg = dc.InferenceJobConfig(query_set=qs, template_preprocessor_settings=tps)
        print(f"InferenceJobConfig ok; seeds={cfg.seeds}")
        # templates off: we have no template alignments and do not want them
        cfg.template.n_templates = 0
        ds = InferenceDataset(cfg)
        print(f"InferenceDataset ok, len={len(ds)}")
        item = ds[0]
        print(f"datapoint keys ({len(item)}):")
        for k in sorted(item):
            v = item[k]
            sh = getattr(v, "shape", None)
            print(f"    {k:34s} {str(sh) if sh is not None else type(v).__name__}")
        batch = Batch.from_torch_dict(item)
        print(f"\nBatch built. token_mask={batch.token_mask.shape} "
              f"msa={batch.msa.shape} atom_mask={batch.atom_mask.shape}")
    except Exception:
        traceback.print_exc()

    # ---- 4. forward pass, if we got a batch -------------------------------
    stage(4, "forward pass")
    if batch is not None and model is not None:
        try:
            import jax, numpy as np
            out = model(batch, num_recycles=3, num_sampling_steps=50, num_samples=1,
                        key=jax.random.key(0), deterministic=True)
            print(f"ModelOutput fields: {[f for f in dir(out) if not f.startswith('_')][:20]}")
            for nm in ("distogram_logits", "distogram", "coordinates",
                       "atom_coords", "plddt", "pae"):
                if hasattr(out, nm):
                    v = getattr(out, nm)
                    print(f"    {nm}: {getattr(v, 'shape', type(v).__name__)}")
        except Exception:
            traceback.print_exc()
    if model is not None:
        try:
            import inspect
            print("OpenFold3.__call__ signature:",
                  inspect.signature(model.__call__))
            print("embed_inputs signature:",
                  inspect.signature(model.embed_inputs))
            print("recycle signature:", inspect.signature(model.recycle))
            for h in ("distogram_head", "distogram", "heads", "confidence_head",
                      "diffusion_module", "structure_module"):
                if hasattr(model, h):
                    print(f"  has {h}: {type(getattr(model, h)).__name__}")
        except Exception:
            traceback.print_exc()

    print("\nSMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()
