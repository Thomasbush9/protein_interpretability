#!/usr/bin/env python3
"""Run Boltz2 attention-weight extraction over a directory of sequences.

Thin orchestrator around :mod:`protein_interpretability.extract_attention`,
mirroring ``scripts/run_boltz_extract.py``.

Given a directory organised as::

    <sequences_dir>/
      seq_00132/
        msa/
        seq_00132.yaml
      seq_00318/
        ...

this script:

1. Recursively finds every ``*.yaml``.
2. Symlinks them into N per-GPU chunk directories under ``<staging_dir>``.
3. Launches one ``python -m protein_interpretability.extract_attention``
   subprocess per GPU in parallel, with ``CUDA_VISIBLE_DEVICES`` pinned.

Outputs land at::

    <out_dir>/boltz_results_<chunk_name>/attention/<record_id>_attention.pt

Usage::

    # Activate the boltz env first (e.g. source scripts/prepare_env.sh on the cluster)
    python scripts/run_boltz_attention.py --config scripts/boltz_attention_config.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


_SUPPORTED_SITES = (
    "attention_weights",
    "tri_att_start_weights",
    "tri_att_end_weights",
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def required(cfg: dict, *keys: str):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node or node[k] in (None, ""):
            dotted = ".".join(keys)
            raise ValueError(f"Missing required config key: {dotted}")
        node = node[k]
    return node


def format_list_arg(value) -> str:
    """Normalise ``layers`` / ``layer_sites`` into CLI form."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "all"):
        return "all"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    raise ValueError(f"Cannot parse list-style arg: {value!r}")


def validate_layer_sites(value) -> list[str]:
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


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def discover_yamls(sequences_dir: Path, subset: set[str] | None) -> list[Path]:
    yamls = sorted(sequences_dir.rglob("*.yaml"))
    if subset is not None:
        yamls = [y for y in yamls if y.stem in subset]
    return yamls


def load_subset(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with open(path) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def make_chunks(
    yamls: list[Path],
    staging_dir: Path,
    num_chunks: int,
) -> list[Path]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    num_chunks = max(1, min(num_chunks, len(yamls)))

    chunk_dirs: list[Path] = []
    for i in range(num_chunks):
        d = staging_dir / f"chunk_{i}"
        d.mkdir(parents=True, exist_ok=True)
        for child in d.iterdir():
            if child.is_symlink():
                child.unlink()
        chunk_dirs.append(d)

    total = len(yamls)
    base, rem = divmod(total, num_chunks)
    cursor = 0
    for i, d in enumerate(chunk_dirs):
        size = base + (1 if i < rem else 0)
        for yml in yamls[cursor : cursor + size]:
            link = d / yml.name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(yml.resolve(), link)
        cursor += size

    return [d for d in chunk_dirs if any(d.iterdir())]


# ---------------------------------------------------------------------------
# Subprocess launch
# ---------------------------------------------------------------------------

def build_command(
    python_exe: str,
    chunk_dir: Path,
    out_dir: Path,
    boltz: dict,
    extraction: dict,
    runtime: dict,
    layer_sites: list[str],
) -> list[str]:
    cmd = [
        python_exe,
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


def launch_chunks(
    chunk_dirs: list[Path],
    out_dir: Path,
    boltz: dict,
    extraction: dict,
    runtime: dict,
    layer_sites: list[str],
    repo_src: Path,
) -> int:
    python_exe = runtime.get("python", "python") or "python"
    extra_env = runtime.get("env") or {}

    processes: list[tuple[Path, subprocess.Popen]] = []
    for i, chunk_dir in enumerate(chunk_dirs):
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{repo_src}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(repo_src)
        )
        if runtime.get("accelerator", "gpu") == "gpu":
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        for k, v in extra_env.items():
            env[str(k)] = str(v)

        cmd = build_command(
            python_exe=python_exe,
            chunk_dir=chunk_dir,
            out_dir=out_dir,
            boltz=boltz,
            extraction=extraction,
            runtime=runtime,
            layer_sites=layer_sites,
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

    sequences_dir = Path(required(cfg, "input", "sequences_dir")).expanduser()
    out_dir = Path(required(cfg, "output", "out_dir")).expanduser()
    boltz_cfg = required(cfg, "boltz")
    extraction_cfg = cfg.get("extraction") or {}
    runtime_cfg = cfg.get("runtime") or {}

    if not sequences_dir.is_dir():
        raise SystemExit(f"sequences_dir does not exist: {sequences_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_path = (cfg.get("input") or {}).get("subset_file")
    subset = load_subset(Path(subset_path).expanduser()) if subset_path else None

    yamls = discover_yamls(sequences_dir, subset)
    if not yamls:
        raise SystemExit(f"No .yaml files found under {sequences_dir}")

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

    num_gpus = max(1, int(runtime_cfg.get("num_gpus", 1) or 1))

    staging_dir_cfg = (cfg.get("output") or {}).get("staging_dir")
    staging_dir = (
        Path(staging_dir_cfg).expanduser() if staging_dir_cfg else out_dir / "_staging"
    )

    chunk_dirs = make_chunks(yamls, staging_dir, num_chunks=num_gpus)
    repo_src = Path(__file__).resolve().parents[1] / "src"

    print("=" * 60)
    print("Boltz2 attention-weight extraction")
    print("=" * 60)
    print(f"  sequences_dir : {sequences_dir}")
    print(f"  out_dir       : {out_dir}")
    print(f"  staging_dir   : {staging_dir}")
    print(f"  yamls found   : {len(yamls)}")
    print(f"  num_gpus      : {num_gpus}")
    print(f"  chunks        : {len(chunk_dirs)}")
    for d in chunk_dirs:
        n = sum(1 for _ in d.iterdir())
        print(f"    {d.name}: {n} yamls")
    print(f"  recycling     : {boltz_cfg.get('recycling_steps')}")
    print(f"  layers        : {format_list_arg(extraction_cfg.get('layers', 'all'))}")
    print(f"  layer_sites   : {','.join(layer_sites)}")
    print(f"  recycling_save: {format_list_arg(extraction_cfg.get('recycling_steps_to_save', 'last'))}")
    print(f"  average_heads : {extraction_cfg.get('average_heads', False)}")
    print(f"  save_format   : {extraction_cfg.get('save_format', 'pt')}")
    print(f"  no_kernels    : {bool(boltz_cfg.get('no_kernels'))}")
    print("=" * 60)

    rc = launch_chunks(
        chunk_dirs=chunk_dirs,
        out_dir=out_dir,
        boltz=boltz_cfg,
        extraction=extraction_cfg,
        runtime=runtime_cfg,
        layer_sites=layer_sites,
        repo_src=repo_src,
    )

    if rc != 0:
        print(f"[error] at least one chunk failed (rc={rc})", file=sys.stderr)
        raise SystemExit(rc)

    print(f"[ok] all chunks complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
