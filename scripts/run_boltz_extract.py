#!/usr/bin/env python3
"""Run Boltz2 hidden-rep extraction over a directory of sequences.

Thin orchestrator around :mod:`protein_interpretability.extract_hidden_reps`.

Given a directory organised as::

    <sequences_dir>/
      seq_00132/
        msa/
        seq_00132.yaml
      seq_00318/
        ...

this script discovers every ``*.yaml``, splits them across N GPUs (one chunk
per GPU), and launches one ``python -m protein_interpretability.extract_hidden_reps``
subprocess per chunk in parallel. MSAs referenced inside the yamls must use
absolute paths.

Usage::

    python scripts/run_boltz_extract.py --config scripts/boltz_extract_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap sys.path so this script works even when the package isn't installed
# (e.g. inside the cluster boltz env).
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


def _resolve_recycling_save(extraction: dict) -> str:
    """Honour the legacy ``save_all_recycling_steps`` boolean if present."""
    if "recycling_steps_to_save" in extraction:
        return extraction["recycling_steps_to_save"]
    if "save_all_recycling_steps" in extraction:
        return "all" if extraction["save_all_recycling_steps"] else "last"
    return "last"


def _build_extract_command(
    cfg: dict,
    chunk_dir: Path,
    out_dir: Path,
) -> list[str]:
    boltz = cfg["boltz"]
    extraction = cfg.get("extraction") or {}
    runtime = cfg.get("runtime") or {}

    cmd = [
        runtime.get("python", "python") or "python",
        "-m",
        "protein_interpretability.extract_hidden_reps",
        str(chunk_dir),
        "--out_dir", str(out_dir),
        "--cache", str(Path(boltz["cache"]).expanduser()),
        "--accelerator", runtime.get("accelerator", "gpu"),
        "--recycling_steps", str(boltz.get("recycling_steps", 3)),
        "--sampling_steps", str(boltz.get("sampling_steps", 200)),
        "--diffusion_samples", str(boltz.get("diffusion_samples", 1)),
        "--step_scale", str(boltz.get("step_scale", 1.5)),
        "--layers", format_list_arg(extraction.get("layers", "all")),
        "--sites", format_list_arg(extraction.get("sites", "all")),
        "--layer_sites", format_list_arg(extraction.get("layer_sites", "all")),
        "--recycling_steps_to_save", format_list_arg(_resolve_recycling_save(extraction)),
        "--num_workers", str(runtime.get("num_workers", 2)),
    ]
    if boltz.get("seed") is not None:
        cmd += ["--seed", str(boltz["seed"])]
    if boltz.get("no_kernels"):
        cmd += ["--no_kernels"]
    if extraction.get("write_structures", False):
        cmd += ["--write_structures"]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Boltz2 hidden-rep extraction over a directory of sequences.",
    )
    ap.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML config (see scripts/boltz_extract_config.yaml).",
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

    recycling_save = _resolve_recycling_save(extraction_cfg)

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
        build_command=lambda c, o: _build_extract_command(cfg, c, o),
        job_title="Boltz2 hidden-rep extraction",
        item_noun="yamls",
        summary_fields={
            "recycling": boltz_cfg.get("recycling_steps"),
            "diffusion": boltz_cfg.get("diffusion_samples"),
            "recycling_save": recycling_save,
            "sites": format_list_arg(extraction_cfg.get("sites", "all")),
            "layer_sites": format_list_arg(extraction_cfg.get("layer_sites", "all")),
            "layers": format_list_arg(extraction_cfg.get("layers", "all")),
        },
    )

    rc = run_chunked(spec)
    if rc != 0:
        print(f"[error] at least one chunk failed (rc={rc})", file=sys.stderr)
        raise SystemExit(rc)
    print(f"[ok] all chunks complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
