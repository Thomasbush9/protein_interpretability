#!/usr/bin/env python3
"""Run Boltz2 attention-weight extraction over a directory of sequences.

Thin orchestrator around :mod:`protein_interpretability.extract_attention`,
mirroring ``scripts/run_boltz_extract.py``.

Outputs land at::

    <out_dir>/boltz_results_<chunk_name>/attention/<record_id>_attention.pt

Usage::

    # Activate the boltz env first (e.g. source scripts/prepare_env.sh on the cluster)
    python scripts/run_boltz_attention.py --config scripts/boltz_attention_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from protein_interpretability.orchestrator import (
    JobSpec,
    discover_yamls,
    format_list_arg,
    load_config,
    load_subset,
    repo_src_from,
    required,
    run_chunked,
)


_SUPPORTED_SITES = (
    "attention_weights",
    "tri_att_start_weights",
    "tri_att_end_weights",
    "pwa_weights",
)


def validate_layer_sites(value) -> list[str]:
    """Reject ``"all"`` and unknown sites — this runner only handles weights."""
    raw = format_list_arg(value)
    if raw == "all":
        raise ValueError(
            "layer_sites='all' is not supported here — pick from "
            f"{_SUPPORTED_SITES} (this runner only handles weight sites)."
        )
    sites = [s.strip() for s in raw.split(",") if s.strip()]
    for s in sites:
        if s not in _SUPPORTED_SITES:
            raise ValueError(
                f"Unsupported layer_site {s!r}. Supported: {_SUPPORTED_SITES}"
            )
    return sites


def _build_attention_command(
    cfg: dict,
    layer_sites: list[str],
    chunk_dir: Path,
    out_dir: Path,
) -> list[str]:
    boltz = cfg["boltz"]
    extraction = cfg.get("extraction") or {}
    runtime = cfg.get("runtime") or {}

    cmd = [
        runtime.get("python", "python") or "python",
        "-m",
        "protein_interpretability.extract_attention",
        str(chunk_dir),
        "--out_dir", str(out_dir),
        "--cache", str(Path(boltz["cache"]).expanduser()),
        "--accelerator", runtime.get("accelerator", "gpu"),
        "--recycling_steps", str(boltz.get("recycling_steps", 3)),
        "--sampling_steps", str(boltz.get("sampling_steps", 200)),
        "--diffusion_samples", str(boltz.get("diffusion_samples", 1)),
        "--step_scale", str(boltz.get("step_scale", 1.5)),
        "--layers", format_list_arg(extraction.get("layers", "all")),
        "--msa_layers", format_list_arg(extraction.get("msa_layers", "all")),
        "--layer_sites", ",".join(layer_sites),
        "--recycling_steps_to_save",
        format_list_arg(extraction.get("recycling_steps_to_save", "last")),
        "--save_format", extraction.get("save_format", "pt"),
        "--num_workers", str(runtime.get("num_workers", 2)),
    ]
    if boltz.get("seed") is not None:
        cmd += ["--seed", str(boltz["seed"])]
    if boltz.get("no_kernels"):
        cmd += ["--no_kernels"]
    if extraction.get("average_heads", False):
        cmd += ["--average_heads"]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Boltz2 attention-weight extraction over a directory of sequences.",
    )
    ap.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML config (see scripts/boltz_attention_config.yaml).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    boltz_cfg = required(cfg, "boltz")
    extraction_cfg = cfg.get("extraction") or {}
    runtime_cfg = cfg.get("runtime") or {}

    sequences_dir = Path(required(cfg, "input", "sequences_dir")).expanduser()
    out_dir = Path(required(cfg, "output", "out_dir")).expanduser()

    subset_path = (cfg.get("input") or {}).get("subset_file")
    subset = load_subset(Path(subset_path).expanduser()) if subset_path else None

    staging_dir_cfg = (cfg.get("output") or {}).get("staging_dir")
    staging_dir = Path(staging_dir_cfg).expanduser() if staging_dir_cfg else None

    layer_sites = validate_layer_sites(extraction_cfg.get("layer_sites", ["attention_weights"]))
    tri_requested = any(s.startswith("tri_att_") for s in layer_sites)
    if tri_requested and not boltz_cfg.get("no_kernels"):
        print(
            "[warn] tri_att_*_weights requested without boltz.no_kernels=true — "
            "the patched triangular forward runs eager, but other modules "
            "(AttentionPairBias, tri_mul, ...) still use kernels. For a fully "
            "consistent run set boltz.no_kernels: true in the config.",
            file=sys.stderr,
        )

    spec = JobSpec(
        sequences_dir=sequences_dir,
        out_dir=out_dir,
        staging_dir=staging_dir,
        subset=subset,
        num_gpus=int(runtime_cfg.get("num_gpus", 1) or 1),
        accelerator=runtime_cfg.get("accelerator", "gpu"),
        python_exe=runtime_cfg.get("python", "python") or "python",
        env_overrides=runtime_cfg.get("env") or {},
        repo_src=repo_src_from(__file__),
        discover=discover_yamls,
        build_command=lambda c, o: _build_attention_command(cfg, layer_sites, c, o),
        job_title="Boltz2 attention-weight extraction",
        item_noun="yamls",
        summary_fields={
            "recycling": boltz_cfg.get("recycling_steps"),
            "layers": format_list_arg(extraction_cfg.get("layers", "all")),
            "msa_layers": format_list_arg(extraction_cfg.get("msa_layers", "all")),
            "layer_sites": ",".join(layer_sites),
            "recycling_save": format_list_arg(extraction_cfg.get("recycling_steps_to_save", "last")),
            "average_heads": extraction_cfg.get("average_heads", False),
            "save_format": extraction_cfg.get("save_format", "pt"),
            "no_kernels": bool(boltz_cfg.get("no_kernels")),
        },
    )

    rc = run_chunked(spec)
    if rc != 0:
        print(f"[error] at least one chunk failed (rc={rc})", file=sys.stderr)
        raise SystemExit(rc)
    print(f"[ok] all chunks complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
