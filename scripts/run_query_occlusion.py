#!/usr/bin/env python3
"""Per-position query occlusion for Boltz2.

For a given mutant sequence, revert each mutated position one at a time to
its WT residue, run Boltz2 forward on each variant, and record per-position
distogram divergence vs the un-reverted (mutant) baseline. Pure forward
passes — no autograd. The output answers: which mutations actually shift
the predicted contact map?

Usage::

    python scripts/run_query_occlusion.py \\
        path/to/mutant.yaml \\
        --wt_yaml path/to/original.yaml \\
        --out_dir /tmp/occ \\
        --cache ~/.boltz \\
        --recycling_steps 3
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

# Bootstrap so the script runs under the cluster boltz env (which doesn't
# install protein_interpretability) without needing python -m.
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import torch  # noqa: E402
import yaml  # noqa: E402
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


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


def load_yaml_seq(yaml_path: Path) -> tuple[str, str, str]:
    """Return ``(sequence, msa_path, protein_id)`` from a Boltz YAML."""
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)
    prot = cfg["sequences"][0]["protein"]
    return prot["sequence"], prot.get("msa", "empty"), str(prot["id"])


def write_variant_yaml(
    sequence: str, msa: str, protein_id: str, out_path: Path
) -> None:
    """Write a single-protein YAML with the given sequence."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": protein_id,
                    "sequence": sequence,
                    "msa": msa,
                }
            }
        ],
    }
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# Boltz parses ``record.id`` from the YAML filename stem (parse/yaml.py:67),
# so we encode the variant identity in the filename and recover it later.
BASELINE_SUFFIX = "__baseline"
REVERT_PREFIX = "__revert_"


def revert_stem(base_stem: str, pos: int, wt_aa: str) -> str:
    return f"{base_stem}{REVERT_PREFIX}{pos:04d}_{wt_aa}"


def parse_variant_id(record_id: str, base_stem: str) -> tuple[str, int | None, str | None]:
    """Decode a record_id back into (kind, position, wt_aa).

    Returns ``("baseline", None, None)`` or ``("revert", pos, wt_aa)``.
    """
    if record_id == f"{base_stem}{BASELINE_SUFFIX}":
        return "baseline", None, None
    prefix = f"{base_stem}{REVERT_PREFIX}"
    if record_id.startswith(prefix):
        rest = record_id[len(prefix) :]
        pos_str, wt_aa = rest.split("_", 1)
        return "revert", int(pos_str), wt_aa
    return "unknown", None, None


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------


def kl_per_pair(
    baseline_logits: torch.Tensor, variant_logits: torch.Tensor
) -> torch.Tensor:
    """KL(baseline || variant) per (i, j) on softmaxed distogram bins.

    Inputs are ``(B, N, N, K)`` logits over ``K`` distance bins. Output is
    ``(B, N, N)``.
    """
    log_p = torch.log_softmax(baseline_logits, dim=-1)
    log_q = torch.log_softmax(variant_logits, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def aggregate_divergence(
    kl_map: torch.Tensor, token_mask: torch.Tensor, exclude_diag: bool = True
) -> dict[str, float]:
    """Reduce a ``(B, N, N)`` KL map over valid token pairs."""
    B, N, _ = kl_map.shape
    valid = token_mask.bool()  # (B, N)
    pair_mask = valid[:, :, None] & valid[:, None, :]  # (B, N, N)
    if exclude_diag:
        eye = torch.eye(N, dtype=torch.bool, device=pair_mask.device)
        pair_mask = pair_mask & ~eye[None]
    vals = kl_map[pair_mask]
    if vals.numel() == 0:
        return {"kl_mean": float("nan"), "kl_max": float("nan"), "n_pairs": 0}
    return {
        "kl_mean": float(vals.mean()),
        "kl_max": float(vals.max()),
        "n_pairs": int(vals.numel()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mutant_yaml", type=Path, help="Mutant query YAML.")
    p.add_argument(
        "--wt_yaml",
        type=Path,
        required=True,
        help="WT (reference) query YAML — used to identify mutated positions.",
    )
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--cache", type=str, default="~/.boltz")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--recycling_steps", type=int, default=3)
    p.add_argument("--sampling_steps", type=int, default=200)
    p.add_argument("--diffusion_samples", type=int, default=1)
    p.add_argument("--no_kernels", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--keep_distograms",
        action="store_true",
        help="Save full per-variant distogram tensors (B, N, N, K) for re-analysis.",
    )
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

    out_dir = Path(args.out_dir).expanduser() / f"occlusion_{args.mutant_yaml.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir / "_variants"
    if staging_dir.exists():
        for f in staging_dir.iterdir():
            if f.is_file() or f.is_symlink():
                f.unlink()
    staging_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load sequences and identify mutated positions
    mut_seq, mut_msa, mut_id = load_yaml_seq(args.mutant_yaml)
    wt_seq, _, _ = load_yaml_seq(args.wt_yaml)
    if len(mut_seq) != len(wt_seq):
        raise SystemExit(
            f"sequence length mismatch: mutant={len(mut_seq)} wt={len(wt_seq)}"
        )

    mutated_positions = [
        i for i, (a, b) in enumerate(zip(mut_seq, wt_seq)) if a != b
    ]
    if not mutated_positions:
        raise SystemExit("Mutant and WT are identical — nothing to revert.")

    base_stem = args.mutant_yaml.stem
    if "__" in base_stem:
        raise SystemExit(
            f"mutant_yaml stem contains '__' which conflicts with the variant "
            f"naming scheme: {base_stem!r}. Rename the file."
        )
    print(
        f"[occ] mutant={mut_id} stem={base_stem} length={len(mut_seq)} "
        f"mutations={len(mutated_positions)}"
    )

    # ---- Generate variant YAMLs
    write_variant_yaml(
        mut_seq, mut_msa, mut_id, staging_dir / f"{base_stem}{BASELINE_SUFFIX}.yaml"
    )
    for pos in mutated_positions:
        wt_aa = wt_seq[pos]
        reverted = mut_seq[:pos] + wt_aa + mut_seq[pos + 1 :]
        write_variant_yaml(
            reverted,
            mut_msa,
            mut_id,
            staging_dir / f"{revert_stem(base_stem, pos, wt_aa)}.yaml",
        )
    n_variants = 1 + len(mutated_positions)
    print(f"[occ] wrote {n_variants} variant YAMLs to {staging_dir}")

    # ---- Process all through Boltz
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
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
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

    # ---- Distogram capture (output of distogram_module is the per-bin logits).
    captured: dict[str, torch.Tensor] = {}

    def _hook(_mod, _inp, out):
        captured["d"] = out.detach()

    handle = model.distogram_module.register_forward_hook(_hook)

    # Skip diffusion sampling — we only need the distogram, which is computed
    # before the structure module (boltz2.py:491). ~5-10x speedup per variant.
    prev_skip = getattr(model, "skip_run_structure", False)
    model.skip_run_structure = True

    # ---- Forward each variant
    distograms: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    try:
        for batch_idx, batch in enumerate(dataloader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            captured.clear()
            _ = model(
                batch,
                recycling_steps=args.recycling_steps,
                num_sampling_steps=args.sampling_steps,
                diffusion_samples=args.diffusion_samples,
            )
            d = captured.get("d")
            if d is None:
                raise RuntimeError("distogram_module hook didn't fire")
            if d.ndim == 5:  # (B, N, N, num_distograms, num_bins)
                d = d[..., 0, :]
            record_id = batch["record"][0].id
            distograms[record_id] = d.cpu()
            masks[record_id] = batch["token_pad_mask"].cpu().bool()
            print(
                f"[occ {batch_idx + 1}/{len(dataloader)}] {record_id} "
                f"distogram shape={tuple(d.shape)}"
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        handle.remove()
        model.skip_run_structure = prev_skip

    # ---- Identify baseline and compute divergences
    baseline_id = f"{base_stem}{BASELINE_SUFFIX}"
    if baseline_id not in distograms:
        raise SystemExit(
            f"baseline record {baseline_id!r} missing from forward outputs. "
            f"Got: {sorted(distograms)}"
        )

    baseline = distograms[baseline_id]
    baseline_mask = masks[baseline_id]

    rows = []
    for record_id, d in distograms.items():
        kind, pos, wt_aa = parse_variant_id(record_id, base_stem)
        if kind == "baseline":
            kl_map = torch.zeros(baseline.shape[:3])
        elif kind == "revert":
            if d.shape != baseline.shape:
                raise RuntimeError(
                    f"shape mismatch for {record_id}: "
                    f"variant={tuple(d.shape)} baseline={tuple(baseline.shape)}"
                )
            kl_map = kl_per_pair(baseline, d)
        else:
            print(f"[occ] skipping unrecognised record_id {record_id!r}")
            continue

        agg = aggregate_divergence(kl_map, baseline_mask)
        rows.append(
            {
                "record_id": record_id,
                "kind": kind,
                "position": pos,
                "wt_aa": wt_aa,
                "mut_aa": mut_seq[pos] if pos is not None else None,
                **agg,
            }
        )

    # ---- Save outputs
    summary_path = out_dir / f"{base_stem}_occlusion_summary.pt"
    torch.save(
        {
            "base_stem": base_stem,
            "mutant_id": mut_id,
            "mutant_sequence": mut_seq,
            "wt_sequence": wt_seq,
            "mutated_positions": mutated_positions,
            "recycling_steps": args.recycling_steps,
            "rows": rows,
            "token_mask": baseline_mask,
        },
        summary_path,
    )
    print(f"[occ] wrote summary -> {summary_path}")

    if args.keep_distograms:
        dist_path = out_dir / f"{base_stem}_occlusion_distograms.pt"
        torch.save(
            {"baseline_id": baseline_id, "distograms": distograms, "masks": masks},
            dist_path,
        )
        print(f"[occ] wrote raw distograms -> {dist_path}")

    # ---- Console summary
    revert_rows = [r for r in rows if r["kind"] == "revert"]
    revert_rows.sort(key=lambda r: r["kl_mean"], reverse=True)
    print("\n[occ] top reverted positions by KL(baseline || variant):")
    print(f"  {'pos':>5}  {'wt->mut':>8}  {'kl_mean':>10}  {'kl_max':>10}")
    for r in revert_rows[:15]:
        pos1 = r["position"] + 1  # 1-indexed for display
        sub = f"{r['wt_aa']}{pos1}{r['mut_aa']}"
        print(f"  {r['position']:>5}  {sub:>8}  {r['kl_mean']:>10.4f}  {r['kl_max']:>10.4f}")


if __name__ == "__main__":
    main()
