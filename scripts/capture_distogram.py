#!/usr/bin/env python3
"""Capture and save a Boltz2 predicted distogram for one YAML.

Single no-grad forward pass with ``skip_run_structure=True`` (distogram is
computed before the structure module — `boltz2.py:491`). The distogram is
captured via a forward hook on ``model.distogram_module`` and saved to disk
along with the token mask, sequence, and provenance metadata.

This is Phase-0 of the chromophore-block attribution experiment
(`log/2026-05-06-chromophore-attribution-pivot.md`): we need the WT
distogram as a frozen reference for the gradient loss. Running the same
script on a mutant gives us a baseline distogram we can sanity-check before
the gradient pass.

Usage::

    python scripts/capture_distogram.py \\
        path/to/query.yaml \\
        --out_dir /path/to/distograms \\
        --cache ~/.boltz \\
        --recycling_steps 3
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

# Bootstrap so the script runs under the cluster boltz env without `python -m`.
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import torch  # noqa: E402
from pytorch_lightning import seed_everything  # noqa: E402
from rdkit import Chem  # noqa: E402

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule  # noqa: E402
from boltz.data.types import Manifest  # noqa: E402
from boltz.main import (  # noqa: E402
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
from boltz.model.models.boltz2 import Boltz2  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("yaml", type=Path, help="Boltz YAML for the query.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--cache", type=str, default="~/.boltz")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--recycling_steps", type=int, default=3)
    p.add_argument("--no_kernels", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    if args.seed is not None:
        seed_everything(args.seed)
    for k in ("CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"):
        os.environ.setdefault(k, "1")

    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir / f"_staging_{args.yaml.stem}"
    if staging_dir.exists():
        for f in staging_dir.iterdir():
            if f.is_file() or f.is_symlink():
                f.unlink()
    staging_dir.mkdir(parents=True, exist_ok=True)
    # Boltz's check_inputs() globs *.yaml in a directory, so we symlink the
    # single query into a clean staging dir.
    link = staging_dir / args.yaml.name
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(args.yaml.resolve(), link)

    print(f"[capture] yaml={args.yaml.name} stem={args.yaml.stem}")

    # ---- Process input through Boltz's standard pipeline
    data = check_inputs(staging_dir)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=False,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=True,
    )
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    filtered_manifest = filter_inputs_structure(
        manifest=manifest, outdir=out_dir, override=True
    )
    if not filtered_manifest.records:
        raise SystemExit("No inputs survived structure filtering.")

    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(processed_dir / "constraints")
        if (processed_dir / "constraints").exists()
        else None,
        template_dir=(processed_dir / "templates")
        if (processed_dir / "templates").exists()
        else None,
        extra_mols_dir=(processed_dir / "mols")
        if (processed_dir / "mols").exists()
        else None,
    )

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
    data_module.setup(stage="predict")
    dataloader = data_module.predict_dataloader()

    # ---- Load model
    checkpoint = args.checkpoint or cache / "boltz2_conf.ckpt"
    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=True, use_paired_feature=True)
    steering_args = BoltzSteeringParams()

    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": 200,
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

    # ---- Forward hook on distogram_module
    captured: dict[str, torch.Tensor] = {}

    def _hook(_mod, _inp, out):
        captured["d"] = out.detach()

    handle = model.distogram_module.register_forward_hook(_hook)

    # Skip diffusion sampling — distogram is computed before structure module.
    prev_skip = getattr(model, "skip_run_structure", False)
    model.skip_run_structure = True

    try:
        # Single batch (one query)
        batch = next(iter(dataloader))
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        record_id = batch["record"][0].id
        _ = model(
            batch,
            recycling_steps=args.recycling_steps,
            num_sampling_steps=200,
            diffusion_samples=1,
        )
        d = captured.get("d")
        if d is None:
            raise RuntimeError("distogram_module hook didn't fire")
        if d.ndim == 5:  # (B, N, N, num_distograms, num_bins)
            d = d[..., 0, :]
        token_mask = batch["token_pad_mask"].cpu().bool()

        print(f"[capture] record_id={record_id} distogram shape={tuple(d.shape)}")
    finally:
        handle.remove()
        model.skip_run_structure = prev_skip

    # ---- Save
    out_path = out_dir / f"{args.yaml.stem}_distogram.pt"
    torch.save(
        {
            "record_id": record_id,
            "yaml": str(args.yaml),
            "recycling_steps": args.recycling_steps,
            "distogram": d.cpu(),
            "token_mask": token_mask,
        },
        out_path,
    )
    print(f"[capture] wrote -> {out_path}")
    print(
        f"[capture] distogram.dtype={d.dtype} sum={float(d.sum()):.3f} "
        f"mean={float(d.mean()):.4f}"
    )


if __name__ == "__main__":
    main()
