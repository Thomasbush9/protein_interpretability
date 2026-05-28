#!/usr/bin/env python3
"""Run ESMFold hidden-rep extraction over a directory of sequences.

Thin orchestrator around :mod:`protein_interpretability.extract_hidden_reps_esmfold`.

Given a directory of FASTA files (flat or nested inside subdirs), this script
discovers every ``*.fasta`` / ``*.fa`` / ``*.faa``, splits them across N GPUs,
and launches one extraction subprocess per chunk in parallel.

Usage::

    python scripts/run_esmfold_extract.py --config scripts/esmfold_extract_config.yaml
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
    discover_fastas,
    format_list_arg,
    load_config,
    load_subset,
    repo_src_from,
    required,
    run_chunked,
)


def _build_esmfold_command(
    cfg: dict,
    chunk_dir: Path,
    out_dir: Path,
) -> list[str]:
    model = cfg.get("model") or {}
    extraction = cfg.get("extraction") or {}
    runtime = cfg.get("runtime") or {}

    cmd = [
        runtime.get("python", "python") or "python",
        "-m",
        "protein_interpretability.extract_hidden_reps_esmfold",
        str(chunk_dir),
        "--out_dir", str(out_dir),
        "--model_name", str(model.get("name", "facebook/esmfold_v1")),
        "--num_recycles", str(model.get("num_recycles", 4)),
        "--chunk_size", str(model.get("chunk_size", 64)),
        "--accelerator", runtime.get("accelerator", "gpu"),
        "--trunk_layers", format_list_arg(extraction.get("trunk_layers", "all")),
        "--esm_layers", format_list_arg(extraction.get("esm_layers", "all")),
        "--sites", format_list_arg(extraction.get("sites", "all")),
        "--layer_sites", format_list_arg(extraction.get("layer_sites", "all")),
        "--recycling_steps_to_save",
        format_list_arg(extraction.get("recycling_steps_to_save", "last")),
    ]
    cache_dir = model.get("cache_dir")
    if cache_dir:
        cmd += ["--cache_dir", str(Path(cache_dir).expanduser())]
    if model.get("fp16"):
        cmd += ["--fp16"]
    max_length = runtime.get("max_length")
    if max_length is not None:
        cmd += ["--max_length", str(max_length)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run ESMFold hidden-rep extraction over a directory of sequences.",
    )
    ap.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML config (see scripts/esmfold_extract_config.yaml).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model") or {}
    extraction_cfg = cfg.get("extraction") or {}
    runtime_cfg = cfg.get("runtime") or {}

    sequences_dir = Path(required(cfg, "input", "sequences_dir")).expanduser()
    out_dir = Path(required(cfg, "output", "out_dir")).expanduser()

    subset_path = (cfg.get("input") or {}).get("subset_file")
    subset = load_subset(Path(subset_path).expanduser()) if subset_path else None

    staging_dir_cfg = (cfg.get("output") or {}).get("staging_dir")
    staging_dir = Path(staging_dir_cfg).expanduser() if staging_dir_cfg else None

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
        discover=discover_fastas,
        build_command=lambda c, o: _build_esmfold_command(cfg, c, o),
        job_title="ESMFold hidden-rep extraction",
        item_noun="fastas",
        summary_fields={
            "model": model_cfg.get("name", "facebook/esmfold_v1"),
            "num_recycles": model_cfg.get("num_recycles", 4),
            "fp16": model_cfg.get("fp16", False),
            "recycling_save": extraction_cfg.get("recycling_steps_to_save", "last"),
            "sites": format_list_arg(extraction_cfg.get("sites", "all")),
            "layer_sites": format_list_arg(extraction_cfg.get("layer_sites", "all")),
            "trunk_layers": format_list_arg(extraction_cfg.get("trunk_layers", "all")),
            "esm_layers": format_list_arg(extraction_cfg.get("esm_layers", "all")),
        },
    )

    rc = run_chunked(spec)
    if rc != 0:
        print(f"[error] at least one chunk failed (rc={rc})", file=sys.stderr)
        raise SystemExit(rc)
    print(f"[ok] all chunks complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
