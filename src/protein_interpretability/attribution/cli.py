"""Single-record CLI: forward + backward at chosen K values, save .pt per step.

Mirrors ``protein_interpretability.extract_attention`` but for gradient
attribution. Multi-GPU orchestration lives in ``scripts/run_boltz_gradients.py``
(chunk 2).

Usage::

    python -m protein_interpretability.attribution.cli input.yaml \
        --out_dir ./attribution_output \
        --target contact:128,142 \
        --recycling_steps 0,5,10 \
        --no_kernels
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from pathlib import Path

import torch
from pytorch_lightning import seed_everything

from .io import save_result
from .runner import run_per_step
from .targets import (
    DEFAULT_CONTACT_BIN_HI,
    AttributionTarget,
    ContactBinNLL,
    PairLogProb,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gradient attribution on Boltz2 distogram logits.",
    )
    p.add_argument("data", type=str, help="Input YAML file or directory")
    p.add_argument("--out_dir", type=str, default="./attribution_output")
    p.add_argument("--cache", type=str, default="~/.boltz")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    p.add_argument(
        "--recycling_steps",
        type=str,
        default="0,5,10",
        help="Comma-separated K values to evaluate.",
    )
    p.add_argument(
        "--target",
        type=str,
        required=True,
        help=(
            "Target spec. Supported forms: "
            "'contact:i,j' (ContactBinNLL on pair (i,j)); "
            "'pair_bin:i,j,b' (PairLogProb on bin b for pair (i,j))."
        ),
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--no_kernels",
        action="store_true",
        help="Disable Boltz custom kernels (recommended for grad runs).",
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--use_msa_server", action="store_true")
    p.add_argument(
        "--msa_server_url", type=str, default="https://api.colabfold.com"
    )
    return p.parse_args()


def parse_target(spec: str) -> AttributionTarget:
    kind, _, body = spec.partition(":")
    parts = [p.strip() for p in body.split(",") if p.strip()]
    if kind == "contact":
        if len(parts) != 2:
            raise ValueError("contact target needs 'i,j'")
        i, j = (int(x) for x in parts)
        return ContactBinNLL(
            pair_i=i, pair_j=j, contact_bins=tuple(range(DEFAULT_CONTACT_BIN_HI))
        )
    if kind == "pair_bin":
        if len(parts) != 3:
            raise ValueError("pair_bin target needs 'i,j,b'")
        i, j, b = (int(x) for x in parts)
        return PairLogProb(pair_i=i, pair_j=j, bin=b)
    raise ValueError(f"unknown target kind: {kind!r}")


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    # Lazy imports — Boltz is heavyweight and only needed at runtime.
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
        check_inputs,
        download_boltz2,
        filter_inputs_structure,
        process_inputs,
    )
    from boltz.model.models.boltz2 import Boltz2

    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    target = parse_target(args.target)
    recycling_steps = [int(x) for x in args.recycling_steps.split(",") if x.strip()]
    if not recycling_steps:
        raise SystemExit("--recycling_steps must list at least one K value")

    data = check_inputs(Path(args.data).expanduser())
    data = filter_inputs_structure(data, out_dir, override=False)
    if not data:
        print("No new inputs to process — all outputs already exist.")
        return
    processed = process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=cache / "mols",
        use_msa_server=args.use_msa_server,
        msa_server_url=args.msa_server_url,
    )
    mol_dir = cache / "mols"

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

    checkpoint = args.checkpoint or cache / "boltz2_conf.ckpt"
    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=True, use_paired_feature=True)
    steering_args = BoltzSteeringParams()

    predict_args = {
        "recycling_steps": max(recycling_steps),
        "sampling_steps": 1,
        "diffusion_samples": 1,
        "max_parallel_samples": None,
        "write_confidence_summary": False,
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

    device = torch.device(
        "cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    data_module.setup(stage="predict")
    dataloader = data_module.predict_dataloader()

    config_for_provenance = {
        "target": args.target,
        "recycling_steps": recycling_steps,
        "no_kernels": args.no_kernels,
        "checkpoint": str(checkpoint),
    }

    for batch_idx, batch in enumerate(dataloader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        results = run_per_step(
            model=model,
            batch=batch,
            target=target,
            recycling_steps=recycling_steps,
            extra_provenance={"config": config_for_provenance},
        )
        record_id = results[0].record_id
        for r in results:
            path = out_dir / f"{record_id}_attribution_R{r.recycling_steps}.pt"
            save_result(r, path)
            print(
                f"[{batch_idx + 1}] {record_id} R={r.recycling_steps} "
                f"loss={r.target_value:.4f} -> {path}"
            )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
