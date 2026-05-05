#!/usr/bin/env python3
"""Run Boltz2 gradient attribution over a directory of sequences.

Thin orchestrator around :mod:`protein_interpretability.attribution.cli`,
mirroring ``scripts/run_boltz_attention.py``.

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
3. Launches one ``python -m protein_interpretability.attribution.cli``
   subprocess per GPU in parallel, with ``CUDA_VISIBLE_DEVICES`` pinned.

Outputs land at::

    <out_dir>/boltz_results_<chunk_name>/<record_id>/gradients/
        <record_id>_attribution_R{0,5,10}.pt

Usage::

    # Activate the boltz env first (e.g. source scripts/prepare_env.sh on the cluster)
    python scripts/run_boltz_gradients.py --config scripts/boltz_gradients_config.yaml

The target spec is global (one ``contact:i,j`` for the whole batch). For
per-record targets, run multiple jobs with different subset_files.
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
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node or node[k] in (None, ""):
            dotted = ".".join(keys)
            raise ValueError(f"Missing required config key: {dotted}")
        node = node[k]
    return node


def format_recycling_steps(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(x)) for x in value)
    raise ValueError(f"Cannot parse recycling_steps: {value!r}")


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

def normalize_targets(value) -> list[str]:
    """Accept str (single target) or list[str]."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out = [str(v) for v in value if v]
        if not out:
            raise ValueError("attribution.targets is empty")
        return out
    raise ValueError(f"Cannot parse attribution.targets: {value!r}")


def build_command(
    python_exe: str,
    yaml_path: Path,
    out_dir: Path,
    boltz: dict,
    attribution: dict,
    runtime: dict,
) -> list[str]:
    targets = normalize_targets(attribution.get("targets") or attribution.get("target"))
    cmd = [
        python_exe,
        "-m",
        "protein_interpretability.attribution.cli",
        str(yaml_path),
        "--out_dir", str(out_dir),
        "--cache", str(Path(boltz["cache"]).expanduser()),
        "--accelerator", runtime.get("accelerator", "gpu"),
        "--recycling_steps",
        format_recycling_steps(attribution.get("recycling_steps", "0,5,10")),
        "--num_workers", str(runtime.get("num_workers", 2)),
    ]
    for t in targets:
        cmd += ["--target", t]
    if boltz.get("seed") is not None:
        cmd += ["--seed", str(boltz["seed"])]
    if boltz.get("no_kernels", True):
        cmd += ["--no_kernels"]
    if boltz.get("checkpoint"):
        cmd += ["--checkpoint", str(boltz["checkpoint"])]
    return cmd


def launch_chunks(
    chunk_dirs: list[Path],
    out_dir: Path,
    boltz: dict,
    attribution: dict,
    runtime: dict,
    repo_src: Path,
) -> int:
    """Launch one subprocess per chunk; each subprocess iterates its yamls."""
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

        # The CLI accepts a directory and the boltz pipeline globs *.yaml,
        # so we point each subprocess at its chunk dir.
        cmd = build_command(
            python_exe=python_exe,
            yaml_path=chunk_dir,
            out_dir=out_dir,
            boltz=boltz,
            attribution=attribution,
            runtime=runtime,
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
        description="Run Boltz2 gradient attribution over a directory of sequences.",
    )
    ap.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML config (see scripts/boltz_gradients_config.yaml).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    sequences_dir = Path(required(cfg, "input", "sequences_dir")).expanduser()
    out_dir = Path(required(cfg, "output", "out_dir")).expanduser()
    boltz_cfg = required(cfg, "boltz")
    attribution_cfg = required(cfg, "attribution")
    runtime_cfg = cfg.get("runtime") or {}

    if "targets" not in attribution_cfg and "target" not in attribution_cfg:
        raise SystemExit(
            "attribution.targets is required (list, e.g. ['contact:65,202', 'mean_contact'])"
        )

    if not sequences_dir.is_dir():
        raise SystemExit(f"sequences_dir does not exist: {sequences_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_path = (cfg.get("input") or {}).get("subset_file")
    if subset_path:
        sp = Path(subset_path).expanduser()
        if not sp.is_absolute():
            sp = (args.config.parent / sp).resolve()
        subset = load_subset(sp)
    else:
        subset = None

    yamls = discover_yamls(sequences_dir, subset)
    if not yamls:
        raise SystemExit(f"No .yaml files found under {sequences_dir}")

    num_gpus = max(1, int(runtime_cfg.get("num_gpus", 1) or 1))

    staging_dir_cfg = (cfg.get("output") or {}).get("staging_dir")
    staging_dir = (
        Path(staging_dir_cfg).expanduser() if staging_dir_cfg else out_dir / "_staging"
    )

    chunk_dirs = make_chunks(yamls, staging_dir, num_chunks=num_gpus)
    repo_src = Path(__file__).resolve().parents[1] / "src"

    print("=" * 60)
    print("Boltz2 gradient attribution")
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
    targets = normalize_targets(attribution_cfg.get("targets") or attribution_cfg.get("target"))
    print(f"  targets       : {targets}")
    print(f"  recycling     : {format_recycling_steps(attribution_cfg.get('recycling_steps', '0,5,10'))}")
    print(f"  no_kernels    : {bool(boltz_cfg.get('no_kernels', True))}")
    print("=" * 60)

    rc = launch_chunks(
        chunk_dirs=chunk_dirs,
        out_dir=out_dir,
        boltz=boltz_cfg,
        attribution=attribution_cfg,
        runtime=runtime_cfg,
        repo_src=repo_src,
    )

    if rc != 0:
        print(f"[error] at least one chunk failed (rc={rc})", file=sys.stderr)
        raise SystemExit(rc)

    print(f"[ok] all chunks complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
