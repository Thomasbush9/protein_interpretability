"""Extract per-residue attention patterns from Boltz2 Pairformer during inference.

Uses forward hooks to capture softmaxed attention weights from each
AttentionPairBias layer in the Pairformer trunk, without modifying Boltz source code.

Usage:
    python scripts/extract_attention.py input.yaml \
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
from functools import wraps
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
from boltz.model.layers.attentionv2 import AttentionPairBias
from boltz.model.models.boltz2 import Boltz2


class AttentionStore:
    """Collects attention maps from hooked AttentionPairBias modules."""

    def __init__(self):
        self.maps: dict[str, torch.Tensor] = {}
        self._hooks = []

    def clear(self):
        self.maps.clear()

    def hook_model(self, model: Boltz2, layer_indices: Optional[list[int]] = None):
        """Install hooks on Pairformer AttentionPairBias layers.

        Parameters
        ----------
        model : Boltz2
            The model instance.
        layer_indices : list[int] | None
            Which Pairformer layer indices to hook. None means all layers.

        """
        # Get the uncompiled pairformer module
        pairformer = model.pairformer_module
        if hasattr(pairformer, "_orig_mod"):
            pairformer = pairformer._orig_mod  # noqa: SLF001

        for i, layer in enumerate(pairformer.layers):
            if layer_indices is not None and i not in layer_indices:
                continue

            attn_module = layer.attention
            self._patch_attention(attn_module, f"pairformer_layer_{i}")

    def _patch_attention(self, module: AttentionPairBias, name: str):
        """Monkey-patch forward to capture attention weights after softmax."""
        original_forward = module.forward
        store = self

        @wraps(original_forward)
        def patched_forward(s, z, mask, k_in, multiplicity=1):
            B = s.shape[0]

            q = module.proj_q(s).view(B, -1, module.num_heads, module.head_dim)
            k = module.proj_k(k_in).view(B, -1, module.num_heads, module.head_dim)
            v = module.proj_v(k_in).view(B, -1, module.num_heads, module.head_dim)

            bias = module.proj_z(z)
            bias = bias.repeat_interleave(multiplicity, 0)

            g = module.proj_g(s).sigmoid()

            with torch.autocast("cuda", enabled=False):
                attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
                attn = attn / (module.head_dim**0.5) + bias.float()
                attn = attn + (1 - mask[:, None, None].float()) * -module.inf
                attn = attn.softmax(dim=-1)

                # Store attention weights (detach + move to CPU to save GPU memory)
                store.maps[name] = attn.detach().cpu()

                o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
            o = o.reshape(B, -1, module.c_s)
            o = module.proj_o(g * o)
            return o

        module.forward = patched_forward

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_attention(
        self, average_heads: bool = False
    ) -> dict[str, torch.Tensor]:
        """Return collected attention maps.

        Parameters
        ----------
        average_heads : bool
            If True, average across attention heads, returning (B, N, N).
            Otherwise returns (B, H, N, N).

        Returns
        -------
        dict[str, Tensor]
            Mapping from layer name to attention tensor.

        """
        out = {}
        for name, attn in self.maps.items():
            if average_heads:
                out[name] = attn.mean(dim=1)  # (B, N, N)
            else:
                out[name] = attn
        return out


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

    # Install attention hooks
    store = AttentionStore()
    store.hook_model(model, layer_indices=layer_indices)

    num_layers = len(model.pairformer_module.layers)
    if hasattr(model.pairformer_module, "_orig_mod"):
        num_layers = len(model.pairformer_module._orig_mod.layers)  # noqa: SLF001
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

        store.clear()

        # Run forward pass (just the trunk, not the full predict_step)
        _ = model(
            batch,
            recycling_steps=args.recycling_steps,
            num_sampling_steps=args.sampling_steps,
            diffusion_samples=args.diffusion_samples,
        )

        # Collect attention maps
        attention = store.get_attention(average_heads=args.average_heads)

        # Get record ID for filename
        record_id = filtered_manifest.records[batch_idx].id

        # Get token mask for reference
        token_mask = batch["token_pad_mask"].cpu()

        # Save
        save_path = attn_dir / f"{record_id}_attention"
        save_data = {
            "token_mask": token_mask,
            **attention,
        }

        if args.save_format == "pt":
            torch.save(save_data, f"{save_path}.pt")
        else:
            import numpy as np

            np_data = {k: v.numpy() for k, v in save_data.items()}
            np.savez_compressed(f"{save_path}.npz", **np_data)

        shape_info = next(iter(attention.values())).shape
        print(
            f"[{batch_idx + 1}/{len(dataloader)}] Saved attention for '{record_id}' "
            f"| shape per layer: {list(shape_info)} "
            f"| {len(attention)} layers -> {save_path}.{args.save_format}"
        )

        # Free memory
        del attention
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone. Attention maps saved to {attn_dir}")


if __name__ == "__main__":
    main()
