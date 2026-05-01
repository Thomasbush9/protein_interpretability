"""Extract per-residue attention patterns from Boltz2 Pairformer during inference.

Uses forward hooks to capture softmaxed attention weights from each
AttentionPairBias layer in the Pairformer trunk, without modifying Boltz source code.

Usage:
    python -m protein_interpretability.extract_attention input.yaml \
        --out_dir ./attention_output \
        --layers all \
        --average_heads

The script piggybacks on the standard Boltz2 inference pipeline and saves
attention maps as .pt files alongside the normal predictions.
"""

import argparse
import gc
import os
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import Trainer, seed_everything
from rdkit import Chem

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzProcessedInput,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
    check_inputs,
    download_boltz2,
    filter_inputs_structure,
    process_inputs,
)
from boltz.model.models.boltz2 import Boltz2

from protein_interpretability.extractor_boltz import Boltz2Extractor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract attention patterns from Boltz2 Pairformer"
    )
    parser.add_argument("data", type=str, help="Input YAML file or directory")
    parser.add_argument("--out_dir", type=str, default="./", help="Output directory")
    parser.add_argument(
        "--cache", type=str, default="~/.boltz", help="Model cache directory"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint")
    parser.add_argument("--accelerator", type=str, default="gpu", help="gpu or cpu")
    parser.add_argument("--recycling_steps", type=int, default=3)
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--diffusion_samples", type=int, default=1)
    parser.add_argument("--step_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layer indices to capture, or 'all'",
    )
    parser.add_argument(
        "--average_heads",
        action="store_true",
        help="Average attention across heads before saving",
    )
    parser.add_argument(
        "--layer_sites",
        type=str,
        default="attention_weights",
        help=(
            "Comma-separated weight sites to capture. Supported: "
            "attention_weights (AttentionPairBias, (B,H,N,N)), "
            "tri_att_start_weights / tri_att_end_weights "
            "(TriangleAttention, (B,I,H,J,J) — forces eager path; "
            "pair with --no_kernels for a fully consistent run), "
            "pwa_weights (MSA-module PairWeightedAveraging, (B,H,N,N) — "
            "forces unchunked-heads path; selected MSA layers via "
            "--msa_layers). Default: attention_weights."
        ),
    )
    parser.add_argument(
        "--msa_layers",
        type=str,
        default="all",
        help=(
            "Comma-separated MSA-module layer indices to capture for "
            "pwa_weights, or 'all'. Boltz2 has 4 MSA layers."
        ),
    )
    parser.add_argument(
        "--recycling_steps_to_save",
        type=str,
        default="last",
        help=(
            "Which recycling steps to save. "
            "'last' = final step only (flat output, default), "
            "'all' = every step, "
            "'every:N' = every Nth step plus the last, "
            "or comma-separated indices e.g. '0,2' (supports negative indexing). "
            "For multi-step runs the output is nested as "
            "{token_mask, step_<N>: {layer_name: tensor, ...}, ...}."
        ),
    )
    parser.add_argument(
        "--save_format",
        choices=["pt", "npz"],
        default="pt",
        help="Output format: pt (PyTorch) or npz (NumPy)",
    )
    parser.add_argument("--use_msa_server", action="store_true")
    parser.add_argument(
        "--msa_server_url", type=str, default="https://api.colabfold.com"
    )
    parser.add_argument("--no_kernels", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    if args.seed is not None:
        seed_everything(args.seed)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        os.environ[key] = os.environ.get(key, "1")

    # Setup paths
    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    data = Path(args.data).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    attn_dir = out_dir / "attention"
    attn_dir.mkdir(parents=True, exist_ok=True)

    # Process inputs
    data = check_inputs(data)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=args.use_msa_server,
        msa_server_url=args.msa_server_url,
        msa_pairing_strategy="greedy",
        boltz2=True,
    )

    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    filtered_manifest = filter_inputs_structure(
        manifest=manifest, outdir=out_dir, override=True
    )

    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    if not filtered_manifest.records:
        print("No inputs to process.")
        return

    # Data module
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=args.num_workers,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )

    # Load model
    checkpoint = args.checkpoint or cache / "boltz2_conf.ckpt"
    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = args.step_scale
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=True, use_paired_feature=True)
    steering_args = BoltzSteeringParams()

    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=not args.no_kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()

    # Parse layer indices
    layer_indices = None
    if args.layers != "all":
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    msa_layer_indices = None
    if args.msa_layers != "all":
        msa_layer_indices = [int(x.strip()) for x in args.msa_layers.split(",")]

    # Parse & validate layer_sites (weight sites only)
    _SUPPORTED = (
        "attention_weights",
        "tri_att_start_weights",
        "tri_att_end_weights",
        "pwa_weights",
    )
    layer_sites = [s.strip() for s in args.layer_sites.split(",") if s.strip()]
    for s in layer_sites:
        if s not in _SUPPORTED:
            raise ValueError(
                f"Unsupported layer_site {s!r}. Supported by this script: {_SUPPORTED}"
            )
    if not layer_sites:
        raise ValueError("--layer_sites must include at least one site")

    tri_sites_requested = any(s.startswith("tri_att_") for s in layer_sites)
    if tri_sites_requested and not args.no_kernels:
        print(
            "[warn] tri_att_*_weights force the eager path on patched modules only; "
            "other modules still use kernels unless you pass --no_kernels. "
            "For a fully consistent run, add --no_kernels."
        )

    # Install attention hooks via Boltz2Extractor
    extractor = Boltz2Extractor(
        model,
        sites=[],
        layers=layer_indices,
        layer_sites=layer_sites,
        msa_layers=msa_layer_indices,
    )
    extractor.install()

    pairformer = Boltz2Extractor._unwrap(model.pairformer_module)
    num_layers = len(pairformer.layers)
    captured = len(layer_indices) if layer_indices else num_layers
    print(f"Hooking {captured}/{num_layers} Pairformer layers")

    # Setup data
    data_module.setup(stage="predict")
    dataloader = data_module.predict_dataloader()

    # Move model to device
    device = torch.device(
        "cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    # Run inference and capture attention
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        extractor.clear()

        # Run forward pass (just the trunk, not the full predict_step)
        _ = model(
            batch,
            recycling_steps=args.recycling_steps,
            num_sampling_steps=args.sampling_steps,
            diffusion_samples=args.diffusion_samples,
        )

        def _collect(step_dict: dict) -> dict:
            """Filter + rename + optional head-avg for one recycling step."""
            out: dict = {}
            for k, v in step_dict.items():
                # Pair attention (AttentionPairBias): (B, H, N, N) -> head dim 1
                # Triangle attention: (B, I, H, J, J) -> head dim 2
                # PWA (PairWeightedAveraging): (B, H, N, N) -> head dim 1
                if k.endswith("_attention_weights"):
                    head_dim = 1
                    layer_name = k.replace("_attention_weights", "").replace(
                        "pairformer_", "pairformer_layer_"
                    )
                elif k.endswith("_tri_att_start_weights") or k.endswith("_tri_att_end_weights"):
                    head_dim = 2
                    layer_name = k.replace("pairformer_", "pairformer_layer_")
                elif k.endswith("_pwa_weights"):
                    head_dim = 1
                    layer_name = k.replace("msa_", "msa_layer_", 1)
                else:
                    continue
                out[layer_name] = v.mean(dim=head_dim) if args.average_heads else v
            return out

        # Resolve which recycling steps to save (same semantics as extract_hidden_reps)
        n_steps = len(extractor.activations)
        save_arg = args.recycling_steps_to_save
        if save_arg == "all":
            step_indices = list(range(n_steps))
        elif save_arg == "last":
            step_indices = [-1]
        elif save_arg.startswith("every:"):
            stride = int(save_arg.split(":")[1])
            step_indices = list(range(0, n_steps, stride))
            if step_indices and step_indices[-1] != n_steps - 1:
                step_indices.append(n_steps - 1)
        else:
            step_indices = [int(x.strip()) for x in save_arg.split(",")]

        record_id = batch["record"][0].id
        token_mask = batch["token_pad_mask"].cpu()
        save_path = attn_dir / f"{record_id}_attention"

        if len(step_indices) == 1:
            attention = _collect(extractor.get_step(step_indices[0]))
            save_data = {"token_mask": token_mask, **attention}
        else:
            save_data = {"token_mask": token_mask}
            attention = {}
            for i in step_indices:
                step_out = _collect(extractor.get_step(i))
                save_data[f"step_{i % n_steps}"] = step_out
                attention.update(step_out)  # just for summary/shape print

        if args.save_format == "pt":
            torch.save(save_data, f"{save_path}.pt")
        else:
            import numpy as np

            # npz can't nest dicts — flatten multi-step keys as "step_X__name".
            flat: dict = {}
            for k, v in save_data.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"{k}__{kk}"] = vv.numpy()
                else:
                    flat[k] = v.numpy()
            np.savez_compressed(f"{save_path}.npz", **flat)

        if attention:
            shape_info = next(iter(attention.values())).shape
            print(
                f"[{batch_idx + 1}/{len(dataloader)}] Saved attention for '{record_id}' "
                f"| shape per layer: {list(shape_info)} "
                f"| {len(attention)} layers x {len(step_indices)} step(s) "
                f"-> {save_path}.{args.save_format}"
            )
        else:
            print(
                f"[{batch_idx + 1}/{len(dataloader)}] WARNING: no attention captured "
                f"for '{record_id}' — check --layers / --layer_sites / --recycling_steps_to_save"
            )

        # Free memory
        del attention
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    extractor.remove()
    print(f"\nDone. Attention maps saved to {attn_dir}")


if __name__ == "__main__":
    main()
