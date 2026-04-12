"""Extract hidden representations from all Boltz2 Pairformer layers during inference.

Runs a full Boltz2 forward pass, captures every available activation site via
Boltz2Extractor, and saves the model outputs (coords, confidence, s, z) plus
per-layer hidden reps to disk.

Usage (single sequence):
    python -m protein_interpretability.extract_hidden_reps input.yaml \
        --out_dir ./hidden_reps_output

Usage (directory of sequences):
    python -m protein_interpretability.extract_hidden_reps ./yaml_dir/ \
        --out_dir ./hidden_reps_output

The script saves:
    <out_dir>/<record_id>/
        model_outputs.pt      — coords, s, z, confidence metrics
        hidden_reps.pt        — all Boltz2Extractor activations (last recycling step)
        metadata.json         — sequence info, shapes, file sizes
"""

import argparse
import gc
import json
import os
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import seed_everything
from rdkit import Chem

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Coords, Interface, Manifest, StructureV2
from boltz.data.write.mmcif import to_mmcif
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
    p = argparse.ArgumentParser(
        description="Extract hidden representations from Boltz2 Pairformer"
    )
    p.add_argument("data", type=str, help="Input YAML/FASTA file or directory")
    p.add_argument("--out_dir", type=str, default="./", help="Output directory")
    p.add_argument(
        "--cache", type=str, default="~/.boltz", help="Model cache directory"
    )
    p.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint")
    p.add_argument("--accelerator", type=str, default="gpu", help="gpu or cpu")
    p.add_argument("--recycling_steps", type=int, default=3)
    p.add_argument("--sampling_steps", type=int, default=200)
    p.add_argument("--diffusion_samples", type=int, default=1)
    p.add_argument("--step_scale", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated pairformer layer indices, or 'all'",
    )
    p.add_argument(
        "--sites",
        type=str,
        default="all",
        help=(
            "Comma-separated top-level sites to capture, or 'all'. "
            f"Options: {', '.join(Boltz2Extractor.SITES)}"
        ),
    )
    p.add_argument(
        "--layer_sites",
        type=str,
        default="all",
        help=(
            "Comma-separated per-layer sites to capture, or 'all'. "
            f"Options: {', '.join(Boltz2Extractor.LAYER_SITES)}"
        ),
    )
    p.add_argument(
        "--recycling_steps_to_save",
        type=str,
        default="last",
        help=(
            "Which recycling steps to save. "
            "'last' = final step only, 'all' = every step, "
            "'every:N' = every Nth step (e.g. 'every:2' saves 0,2,4,...), "
            "or comma-separated indices e.g. '0,2' (supports negative indexing). "
            "Default: last."
        ),
    )
    p.add_argument("--use_msa_server", action="store_true")
    p.add_argument(
        "--msa_server_url", type=str, default="https://api.colabfold.com"
    )
    p.add_argument("--no_kernels", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--write_structures",
        action="store_true",
        help="Write predicted structures as .cif files (one per diffusion sample).",
    )
    return p.parse_args()


def _parse_list_arg(value: str, valid: tuple[str, ...]) -> list[str] | None:
    """Parse a comma-separated arg; return None for 'all'."""
    if value == "all":
        return None  # signals "use default = all"
    items = [x.strip() for x in value.split(",")]
    for item in items:
        if item not in valid:
            raise ValueError(f"Unknown site '{item}'. Valid: {valid}")
    return items


def _tensor_size_mb(t: torch.Tensor) -> float:
    return t.nelement() * t.element_size() / (1024 * 1024)


def _write_structures(
    record,
    output: dict,
    batch: dict,
    targets_dir: Path,
    record_dir: Path,
) -> list[Path]:
    """Write predicted structures as CIF files, mirroring Boltz's BoltzWriter."""
    structure = StructureV2.load(targets_dir / f"{record.id}.npz")

    chain_map = {}
    for i, mask in enumerate(structure.mask):
        if mask:
            chain_map[len(chain_map)] = i
    structure = structure.remove_invalid_chains()

    coords = output["sample_atom_coords"].detach().cpu()  # (n_samples, n_atoms, 3)
    pad_mask = batch["atom_pad_mask"][0].cpu()
    plddts_all = output.get("plddt")

    # Rank samples by confidence if available
    if "confidence_score" in output:
        argsort = torch.argsort(output["confidence_score"], descending=True)
        idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}
    else:
        idx_to_rank = {i: i for i in range(coords.shape[0])}

    written = []
    for model_idx in range(coords.shape[0]):
        coord_unpad = coords[model_idx][pad_mask.bool()].numpy()

        atoms = structure.atoms
        atoms["coords"] = coord_unpad
        atoms["is_present"] = True

        coord_structured = np.array([(x,) for x in coord_unpad], dtype=Coords)

        residues = structure.residues
        residues["is_present"] = True

        new_structure = replace(
            structure,
            atoms=atoms,
            residues=residues,
            interfaces=np.array([], dtype=Interface),
            coords=coord_structured,
        )

        plddts = None
        if plddts_all is not None:
            plddts = plddts_all[model_idx]

        rank = idx_to_rank[model_idx]
        outname = f"{record.id}_model_{rank}"
        path = record_dir / f"{outname}.cif"
        with path.open("w") as f:
            f.write(to_mmcif(new_structure, plddts=plddts, boltz2=True))
        written.append(path)

    return written


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

    # ---- Parse extraction config ----
    sites = _parse_list_arg(args.sites, Boltz2Extractor.SITES)
    if sites is None:
        sites = list(Boltz2Extractor.SITES)

    layer_sites = _parse_list_arg(args.layer_sites, Boltz2Extractor.LAYER_SITES)
    if layer_sites is None:
        layer_sites = list(Boltz2Extractor.LAYER_SITES)

    layer_indices = None
    if args.layers != "all":
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    # ---- Setup paths ----
    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    data = Path(args.data).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Process inputs (reuses Boltz pipeline) ----
    processing_dir = out_dir / f"_boltz_processing_{data.stem}"
    processing_dir.mkdir(parents=True, exist_ok=True)

    data = check_inputs(data)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    process_inputs(
        data=data,
        out_dir=processing_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=args.use_msa_server,
        msa_server_url=args.msa_server_url,
        msa_pairing_strategy="greedy",
        boltz2=True,
    )

    manifest = Manifest.load(processing_dir / "processed" / "manifest.json")
    filtered_manifest = filter_inputs_structure(
        manifest=manifest, outdir=processing_dir, override=True
    )

    processed_dir = processing_dir / "processed"
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
            (processed_dir / "mols")
            if (processed_dir / "mols").exists()
            else None
        ),
    )

    if not filtered_manifest.records:
        print("No inputs to process.")
        return

    # ---- Data module ----
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

    # ---- Load model ----
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

    # ---- Install extractor ----
    extractor = Boltz2Extractor(
        model,
        sites=sites,
        layers=layer_indices,
        layer_sites=layer_sites,
    )
    extractor.install()

    pairformer = Boltz2Extractor._unwrap(model.pairformer_module)
    num_layers = len(pairformer.layers)
    captured_layers = len(layer_indices) if layer_indices else num_layers
    print(f"Extractor installed: {len(sites)} top-level sites, "
          f"{len(layer_sites)} layer sites x {captured_layers}/{num_layers} layers")

    # ---- Setup data ----
    data_module.setup(stage="predict")
    dataloader = data_module.predict_dataloader()

    device = torch.device(
        "cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    # ---- Run inference ----
    for batch_idx, batch in enumerate(dataloader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        extractor.clear()

        output = model(
            batch,
            recycling_steps=args.recycling_steps,
            num_sampling_steps=args.sampling_steps,
            diffusion_samples=args.diffusion_samples,
        )

        record_id = batch["record"][0].id
        record_dir = out_dir / record_id
        record_dir.mkdir(parents=True, exist_ok=True)

        # ---- Save model outputs ----
        model_output_keys = [
            "sample_atom_coords", "s", "z", "pdistogram",
            "plddt", "pae", "ptm", "iptm", "pde",
            "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
        ]
        model_outputs = {}
        for k in model_output_keys:
            if k in output:
                model_outputs[k] = output[k].detach().cpu()

        # Include token mask for downstream use
        model_outputs["token_pad_mask"] = batch["token_pad_mask"].cpu()

        model_out_path = record_dir / "model_outputs.pt"
        torch.save(model_outputs, model_out_path)
        model_out_size = model_out_path.stat().st_size / (1024 * 1024)

        # ---- Write structure files ----
        if args.write_structures:
            record = batch["record"][0]
            cif_paths = _write_structures(
                record=record,
                output=output,
                batch=batch,
                targets_dir=processed.targets_dir,
                record_dir=record_dir,
            )
            print(f"  wrote {len(cif_paths)} structure(s): {[p.name for p in cif_paths]}")

        # ---- Save hidden reps ----
        n_steps = len(extractor.activations)
        save_arg = args.recycling_steps_to_save
        if save_arg == "all":
            step_indices = list(range(n_steps))
        elif save_arg == "last":
            step_indices = [-1]
        elif save_arg.startswith("every:"):
            stride = int(save_arg.split(":")[1])
            step_indices = list(range(0, n_steps, stride))
            # Always include the last step
            if step_indices[-1] != n_steps - 1:
                step_indices.append(n_steps - 1)
        else:
            step_indices = [int(x.strip()) for x in save_arg.split(",")]

        if len(step_indices) == 1:
            hidden_reps = extractor.get_step(step_indices[0])
        else:
            hidden_reps = {
                f"step_{i % n_steps}": extractor.get_step(i)
                for i in step_indices
            }

        hidden_reps_path = record_dir / "hidden_reps.pt"
        torch.save(hidden_reps, hidden_reps_path)
        hidden_reps_size = hidden_reps_path.stat().st_size / (1024 * 1024)

        # ---- Save metadata ----
        # Collect shape/size info for each saved tensor
        shapes_info = {}
        if isinstance(hidden_reps, dict) and not any(
            k.startswith("step_") for k in hidden_reps
        ):
            # Single step
            for k, v in hidden_reps.items():
                shapes_info[k] = {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "size_mb": round(_tensor_size_mb(v), 3),
                }
        else:
            # Multiple steps
            for step_key, step_dict in hidden_reps.items():
                for k, v in step_dict.items():
                    shapes_info[f"{step_key}/{k}"] = {
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                        "size_mb": round(_tensor_size_mb(v), 3),
                    }

        model_output_shapes = {
            k: {"shape": list(v.shape), "dtype": str(v.dtype), "size_mb": round(_tensor_size_mb(v), 3)}
            for k, v in model_outputs.items()
        }

        metadata = {
            "record_id": record_id,
            "config": {
                "recycling_steps": args.recycling_steps,
                "sampling_steps": args.sampling_steps,
                "diffusion_samples": args.diffusion_samples,
                "step_scale": args.step_scale,
                "layers": args.layers,
                "sites": args.sites,
                "layer_sites": args.layer_sites,
                "recycling_steps_saved": args.recycling_steps_to_save,
                "num_recycling_steps_captured": len(extractor.activations),
            },
            "file_sizes_mb": {
                "model_outputs.pt": round(model_out_size, 2),
                "hidden_reps.pt": round(hidden_reps_size, 2),
                "total": round(model_out_size + hidden_reps_size, 2),
            },
            "model_output_shapes": model_output_shapes,
            "hidden_rep_shapes": shapes_info,
        }

        metadata_path = record_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"[{batch_idx + 1}/{len(dataloader)}] '{record_id}' -> {record_dir}\n"
            f"  model_outputs.pt: {model_out_size:.1f} MB  |  "
            f"hidden_reps.pt: {hidden_reps_size:.1f} MB  |  "
            f"total: {model_out_size + hidden_reps_size:.1f} MB\n"
            f"  keys: {sorted(hidden_reps.keys()) if isinstance(hidden_reps, dict) else 'N/A'}"
        )

        # Free memory
        del output, model_outputs, hidden_reps
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    extractor.remove()
    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
