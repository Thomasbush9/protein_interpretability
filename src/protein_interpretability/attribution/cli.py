"""Single-record CLI: forward + backward at chosen K values, save .pt per step.

Mirrors :mod:`protein_interpretability.extract_attention` but for gradient
attribution. Multi-GPU orchestration lives in
``scripts/run_boltz_gradients.py`` (chunk 2).

Usage::

    # Activate the boltz env first (cluster: source scripts/prepare_env.sh)
    python -m protein_interpretability.attribution.cli input.yaml \\
        --out_dir ./attribution_output \\
        --target contact:65,202 \\
        --recycling_steps 0,5,10 \\
        --no_kernels
"""

from __future__ import annotations

import argparse
import gc
import os
import warnings
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
    MeanContactNLL,
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
        action="append",
        required=True,
        help=(
            "Target spec; may be repeated to run multiple targets per record. "
            "Supported forms: "
            "'contact:i,j' (ContactBinNLL on pair (i,j), 0-indexed); "
            "'mean_contact' (MeanContactNLL — whole-distogram contact NLL); "
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
    if kind == "mean_contact":
        return MeanContactNLL(contact_bins=tuple(range(DEFAULT_CONTACT_BIN_HI)))
    if kind == "pair_bin":
        if len(parts) != 3:
            raise ValueError("pair_bin target needs 'i,j,b'")
        i, j, b = (int(x) for x in parts)
        return PairLogProb(pair_i=i, pair_j=j, bin=b)
    raise ValueError(f"unknown target kind: {kind!r}")


def target_short_name(spec: str) -> str:
    """Filename-safe short name for a target spec."""
    kind, _, body = spec.partition(":")
    if not body:
        return kind
    return kind + "_" + body.replace(",", "_")


def main() -> None:
    args = parse_args()

    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )
    torch.set_grad_enabled(True)
    torch.set_float32_matmul_precision("highest")

    if args.seed is not None:
        seed_everything(args.seed)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        os.environ[key] = os.environ.get(key, "1")

    # Lazy imports — Boltz is heavyweight and only needed at runtime.
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
    from rdkit import Chem

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    data_path = Path(args.data).expanduser()
    out_dir = Path(args.out_dir).expanduser() / f"boltz_results_{data_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    grad_dir = out_dir / "gradients"
    grad_dir.mkdir(parents=True, exist_ok=True)

    target_specs: list[str] = list(args.target)
    targets: list[tuple[str, AttributionTarget]] = [
        (spec, parse_target(spec)) for spec in target_specs
    ]
    recycling_steps = [int(x) for x in args.recycling_steps.split(",") if x.strip()]
    if not recycling_steps:
        raise SystemExit("--recycling_steps must list at least one K value")

    data = check_inputs(data_path)
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

    print(
        f"[attribution] {len(filtered_manifest.records)} record(s) in manifest "
        f"({out_dir})"
    )
    if not filtered_manifest.records:
        print(
            "No inputs to process. The out_dir's processed/manifest.json may "
            "be empty — try a fresh --out_dir or delete the existing "
            f"{out_dir}/processed/ tree."
        )
        return

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

    base_provenance = {
        "recycling_steps": recycling_steps,
        "no_kernels": args.no_kernels,
        "checkpoint": str(checkpoint),
        "all_targets": target_specs,
    }

    # Match the forward-call pattern proven in extract_attention.py — Boltz's
    # forward takes num_sampling_steps + diffusion_samples and may take a
    # different code path without them. Sampling is downstream of distogram so
    # cost here is only the diffusion overhead; we keep it minimal (1, 1).
    forward_kwargs = {"num_sampling_steps": 1, "diffusion_samples": 1}

    for batch_idx, batch in enumerate(dataloader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        for spec, target in targets:
            short = target_short_name(spec)
            results = run_per_step(
                model=model,
                batch=batch,
                target=target,
                recycling_steps=recycling_steps,
                forward_kwargs=forward_kwargs,
                extra_provenance={"config": {**base_provenance, "target": spec}},
            )
            record_id = results[0].record_id
            for r in results:
                path = grad_dir / f"{record_id}_attribution_R{r.recycling_steps}_{short}.pt"
                save_result(r, path)
                print(
                    f"[{batch_idx + 1}/{len(dataloader)}] {record_id} "
                    f"target={spec} R={r.recycling_steps} loss={r.target_value:.4f} "
                    f"-> {path}"
                )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone. Gradients saved to {grad_dir}")


if __name__ == "__main__":
    main()
