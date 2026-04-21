#!/usr/bin/env python3
"""Run ESMFold hidden-rep extraction over a directory of sequences.

Thin orchestrator around :mod:`protein_interpretability.extract_hidden_reps_esmfold`.

Given a directory of FASTA files (flat or nested inside subdirs), this script:

1. Recursively finds every ``*.fasta`` / ``*.fa`` / ``*.faa``.
2. Symlinks them into N per-GPU chunk directories under ``<staging_dir>``.
3. Launches one ``python -m protein_interpretability.extract_hidden_reps_esmfold``
   subprocess per GPU in parallel, with ``CUDA_VISIBLE_DEVICES`` pinned.

Usage::

    python scripts/run_esmfold_extract.py --config scripts/esmfold_extract_config.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def required(cfg: dict, *keys: str):
    """Pluck a nested key, raising a clear error if missing/empty."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node or node[k] in (None, ""):
            dotted = ".".join(keys)
            raise ValueError(f"Missing required config key: {dotted}")
        node = node[k]
    return node


def format_list_arg(value) -> str:
    """Normalise list-style config values into the CLI form expected by
    ``extract_hidden_reps_esmfold.py``."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "all"):
        return "all"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "none"
        return ",".join(str(x) for x in value)
    raise ValueError(f"Cannot parse list-style arg: {value!r}")


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

FASTA_EXTS = {".fasta", ".fa", ".faa"}


def discover_fastas(sequences_dir: Path, subset: set[str] | None) -> list[Path]:
    """Find every FASTA file under ``sequences_dir`` (recursively)."""
    fastas = sorted(
        p for p in sequences_dir.rglob("*")
        if p.suffix.lower() in FASTA_EXTS
    )
    if subset is not None:
        fastas = [f for f in fastas if f.stem in subset]
    return fastas


def load_subset(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with open(path) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def make_chunks(
    fastas: list[Path],
    staging_dir: Path,
    num_chunks: int,
) -> list[Path]:
    """Symlink ``fastas`` into ``num_chunks`` subdirs under ``staging_dir``.

    Returns the list of non-empty chunk directories.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    num_chunks = max(1, min(num_chunks, len(fastas)))

    chunk_dirs: list[Path] = []
    for i in range(num_chunks):
        d = staging_dir / f"chunk_{i}"
        d.mkdir(parents=True, exist_ok=True)
        # Clean stale symlinks from previous runs.
        for child in d.iterdir():
            if child.is_symlink():
                child.unlink()
        chunk_dirs.append(d)

    total = len(fastas)
    base, rem = divmod(total, num_chunks)
    cursor = 0
    for i, d in enumerate(chunk_dirs):
        size = base + (1 if i < rem else 0)
        for fa in fastas[cursor : cursor + size]:
            link = d / fa.name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(fa.resolve(), link)
        cursor += size

    return [d for d in chunk_dirs if any(d.iterdir())]


# ---------------------------------------------------------------------------
# Subprocess launch
# ---------------------------------------------------------------------------

def build_command(
    python_exe: str,
    chunk_dir: Path,
    out_dir: Path,
    model_cfg: dict,
    extraction_cfg: dict,
    runtime_cfg: dict,
) -> list[str]:
    """Build the extract_hidden_reps_esmfold.py command line for one chunk."""
    recycling_save = extraction_cfg.get("recycling_steps_to_save", "last")

    cmd = [
        python_exe,
        "-m",
        "protein_interpretability.extract_hidden_reps_esmfold",
        str(chunk_dir),
        "--out_dir", str(out_dir),
        "--model_name", str(model_cfg.get("name", "facebook/esmfold_v1")),
        "--num_recycles", str(model_cfg.get("num_recycles", 4)),
        "--chunk_size", str(model_cfg.get("chunk_size", 64)),
        "--accelerator", runtime_cfg.get("accelerator", "gpu"),
        "--trunk_layers", format_list_arg(extraction_cfg.get("trunk_layers", "all")),
        "--esm_layers", format_list_arg(extraction_cfg.get("esm_layers", "all")),
        "--sites", format_list_arg(extraction_cfg.get("sites", "all")),
        "--layer_sites", format_list_arg(extraction_cfg.get("layer_sites", "all")),
        "--recycling_steps_to_save", format_list_arg(recycling_save),
    ]
    if model_cfg.get("fp16"):
        cmd += ["--fp16"]
    max_length = runtime_cfg.get("max_length")
    if max_length is not None:
        cmd += ["--max_length", str(max_length)]
    return cmd


def launch_chunks(
    chunk_dirs: list[Path],
    out_dir: Path,
    model_cfg: dict,
    extraction_cfg: dict,
    runtime_cfg: dict,
    repo_src: Path,
) -> int:
    """Launch one subprocess per chunk in parallel. Returns non-zero exit on any failure."""
    python_exe = runtime_cfg.get("python", "python") or "python"
    extra_env = runtime_cfg.get("env") or {}

    processes: list[tuple[Path, subprocess.Popen]] = []
    for i, chunk_dir in enumerate(chunk_dirs):
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{repo_src}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(repo_src)
        )
        if runtime_cfg.get("accelerator", "gpu") == "gpu":
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        for k, v in extra_env.items():
            env[str(k)] = str(v)

        cmd = build_command(
            python_exe=python_exe,
            chunk_dir=chunk_dir,
            out_dir=out_dir,
            model_cfg=model_cfg,
            extraction_cfg=extraction_cfg,
            runtime_cfg=runtime_cfg,
        )

        log_path = out_dir / f"_log_{chunk_dir.name}.out"
        log_file = open(log_path, "w")
        log_file.write(f"# cmd: {' '.join(cmd)}\n")
        log_file.write(f"# CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'unset')}\n")
        log_file.flush()

        print(
            f"[launch] {chunk_dir.name}: "
            f"GPU={env.get('CUDA_VISIBLE_DEVICES', 'cpu')} log={log_path}"
        )
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((chunk_dir, proc))

    exit_code = 0
    for chunk_dir, proc in processes:
        rc = proc.wait()
        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[done]   {chunk_dir.name}: {status}")
        if rc != 0 and exit_code == 0:
            exit_code = rc
    return exit_code


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    sequences_dir = Path(required(cfg, "input", "sequences_dir")).expanduser()
    out_dir = Path(required(cfg, "output", "out_dir")).expanduser()
    model_cfg = cfg.get("model") or {}
    extraction_cfg = cfg.get("extraction") or {}
    runtime_cfg = cfg.get("runtime") or {}

    if not sequences_dir.is_dir():
        raise SystemExit(f"sequences_dir does not exist: {sequences_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_path = (cfg.get("input") or {}).get("subset_file")
    subset = load_subset(Path(subset_path).expanduser()) if subset_path else None

    fastas = discover_fastas(sequences_dir, subset)
    if not fastas:
        raise SystemExit(f"No FASTA files found under {sequences_dir}")

    num_gpus = int(runtime_cfg.get("num_gpus", 1) or 1)
    num_gpus = max(1, num_gpus)

    staging_dir_cfg = (cfg.get("output") or {}).get("staging_dir")
    staging_dir = (
        Path(staging_dir_cfg).expanduser() if staging_dir_cfg else out_dir / "_staging"
    )

    chunk_dirs = make_chunks(fastas, staging_dir, num_chunks=num_gpus)

    repo_src = Path(__file__).resolve().parents[1] / "src"

    recycling_save = extraction_cfg.get("recycling_steps_to_save", "last")

    print("=" * 60)
    print("ESMFold hidden-rep extraction")
    print("=" * 60)
    print(f"  sequences_dir : {sequences_dir}")
    print(f"  out_dir       : {out_dir}")
    print(f"  staging_dir   : {staging_dir}")
    print(f"  fastas found  : {len(fastas)}")
    print(f"  num_gpus      : {num_gpus}")
    print(f"  chunks        : {len(chunk_dirs)}")
    for d in chunk_dirs:
        n = sum(1 for _ in d.iterdir())
        print(f"    {d.name}: {n} fastas")
    print(f"  model         : {model_cfg.get('name', 'facebook/esmfold_v1')}")
    print(f"  num_recycles  : {model_cfg.get('num_recycles', 4)}")
    print(f"  fp16          : {model_cfg.get('fp16', False)}")
    print(f"  recycling_save: {recycling_save}")
    print(f"  sites         : {format_list_arg(extraction_cfg.get('sites', 'all'))}")
    print(f"  layer_sites   : {format_list_arg(extraction_cfg.get('layer_sites', 'all'))}")
    print(f"  trunk_layers  : {format_list_arg(extraction_cfg.get('trunk_layers', 'all'))}")
    print(f"  esm_layers    : {format_list_arg(extraction_cfg.get('esm_layers', 'all'))}")
    print("=" * 60)

    rc = launch_chunks(
        chunk_dirs=chunk_dirs,
        out_dir=out_dir,
        model_cfg=model_cfg,
        extraction_cfg=extraction_cfg,
        runtime_cfg=runtime_cfg,
        repo_src=repo_src,
    )

    if rc != 0:
        print(f"[error] at least one chunk failed (rc={rc})", file=sys.stderr)
        raise SystemExit(rc)

    print(f"[ok] all chunks complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
