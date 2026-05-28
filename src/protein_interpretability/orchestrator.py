"""Shared orchestration core for the ``scripts/run_*`` extraction launchers.

Every ``run_*`` script (Boltz2 hidden-rep, Boltz2 attention, Boltz2 gradient
attribution, ESMFold hidden-rep) does the same four mechanical things:

1. Discover the input files under ``input.sequences_dir``.
2. Symlink them into N per-GPU chunk directories under ``output.staging_dir``.
3. Launch one ``python -m <entrypoint>`` subprocess per chunk with
   ``CUDA_VISIBLE_DEVICES`` pinned and ``PYTHONPATH`` prepended.
4. Wait, log, propagate the worst exit code.

This module owns those steps. Each ``run_*`` script supplies a :class:`JobSpec`
and calls :func:`run_chunked`.

Config parsing intentionally stays dict-based — ``load_config`` /
``required`` / ``load_subset`` are kept here so a future pydantic schema can
replace them without touching the orchestrator's surface.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Config helpers (dict-based for now; future pydantic schema will swap these)
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    # yaml is imported lazily: the dev env doesn't install it, but the cluster
    # boltz env does, and that's the only place the run_* scripts execute.
    import yaml

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


def load_subset(path: Path | None) -> set[str] | None:
    """Read a newline-delimited file of stems; ``#`` lines and blanks ignored."""
    if path is None:
        return None
    with open(path) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def format_list_arg(value) -> str:
    """Normalise ``sites`` / ``layers`` / ``layer_sites`` values into CLI form.

    Accepts ``"all"``, a comma-separated string, a list, or ``None``. An empty
    list returns ``"none"`` (ESMFold's extractor uses this sentinel; the Boltz
    extractors never pass empty lists in practice).
    """
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
# Discovery
# ---------------------------------------------------------------------------

def discover_yamls(sequences_dir: Path, subset: set[str] | None) -> list[Path]:
    """Find every ``*.yaml`` under ``sequences_dir`` (recursively)."""
    yamls = sorted(sequences_dir.rglob("*.yaml"))
    if subset is not None:
        yamls = [y for y in yamls if y.stem in subset]
    return yamls


_FASTA_EXTS = frozenset({".fasta", ".fa", ".faa"})


def discover_fastas(sequences_dir: Path, subset: set[str] | None) -> list[Path]:
    """Find every FASTA file under ``sequences_dir`` (recursively)."""
    fastas = sorted(
        p for p in sequences_dir.rglob("*") if p.suffix.lower() in _FASTA_EXTS
    )
    if subset is not None:
        fastas = [f for f in fastas if f.stem in subset]
    return fastas


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def make_chunks(
    items: list[Path],
    staging_dir: Path,
    num_chunks: int,
) -> list[Path]:
    """Symlink ``items`` into ``num_chunks`` subdirs under ``staging_dir``.

    Stale symlinks from previous runs are cleared first; real files are left
    alone. Returns the list of non-empty chunk directories.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    num_chunks = max(1, min(num_chunks, len(items)))

    chunk_dirs: list[Path] = []
    for i in range(num_chunks):
        d = staging_dir / f"chunk_{i}"
        d.mkdir(parents=True, exist_ok=True)
        for child in d.iterdir():
            if child.is_symlink():
                child.unlink()
        chunk_dirs.append(d)

    total = len(items)
    base, rem = divmod(total, num_chunks)
    cursor = 0
    for i, d in enumerate(chunk_dirs):
        size = base + (1 if i < rem else 0)
        for item in items[cursor : cursor + size]:
            link = d / item.name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(item.resolve(), link)
        cursor += size

    return [d for d in chunk_dirs if any(d.iterdir())]


# ---------------------------------------------------------------------------
# JobSpec + pipeline
# ---------------------------------------------------------------------------

@dataclass
class JobSpec:
    """Everything a ``run_*`` script supplies to :func:`run_chunked`.

    The orchestrator owns the *mechanical* pipeline (discover → chunk → launch
    → wait). The script owns the *intentional* parts via the adapter slots:
    discovery, command building, and any banner fields the human cares about.

    Attributes
    ----------
    sequences_dir, out_dir, staging_dir
        Input/output/scratch paths. ``staging_dir=None`` resolves to
        ``out_dir/_staging``.
    subset
        Optional set of file stems to keep after discovery.
    num_gpus, accelerator
        Parallelism + device. ``accelerator='gpu'`` triggers per-chunk
        ``CUDA_VISIBLE_DEVICES`` pinning; anything else (``'cpu'``) leaves it
        unset.
    python_exe
        Interpreter for the subprocesses (e.g. ``"python"`` or a full path).
    env_overrides
        Extra env vars merged into each subprocess env.
    repo_src
        Path to the package ``src/`` dir, prepended to subprocess
        ``PYTHONPATH`` so ``python -m protein_interpretability.<x>`` works
        without an install.
    discover
        Closure that returns the list of input files. Use
        :func:`discover_yamls` / :func:`discover_fastas` or supply your own.
    build_command
        Closure that builds the subprocess argv for one chunk:
        ``(chunk_dir, out_dir) -> argv``. The script captures all
        model-specific knowledge (entrypoint module, CLI flags) inside.
    job_title
        Header line printed before the pipeline starts.
    item_noun
        Pluralised label for the discovered files (e.g. ``"yamls"``,
        ``"fastas"``). Used only in the banner.
    summary_fields
        Job-specific banner rows, in display order. Anything important to see
        before launching — recycling steps, targets, kernel flags.
    """

    sequences_dir: Path
    out_dir: Path
    staging_dir: Path | None
    subset: set[str] | None
    num_gpus: int
    accelerator: str
    python_exe: str
    env_overrides: dict[str, str]
    repo_src: Path
    discover: Callable[[Path, set[str] | None], list[Path]]
    build_command: Callable[[Path, Path], list[str]]
    job_title: str
    item_noun: str = "items"
    summary_fields: dict[str, Any] = field(default_factory=dict)


def _resolve_staging(spec: JobSpec) -> Path:
    return spec.staging_dir if spec.staging_dir is not None else spec.out_dir / "_staging"


def _print_banner(
    spec: JobSpec,
    items: list[Path],
    chunk_dirs: list[Path],
    staging: Path,
) -> None:
    bar = "=" * 60
    print(bar)
    print(spec.job_title)
    print(bar)
    print(f"  sequences_dir : {spec.sequences_dir}")
    print(f"  out_dir       : {spec.out_dir}")
    print(f"  staging_dir   : {staging}")
    print(f"  {spec.item_noun:<13} : {len(items)}")
    print(f"  num_gpus      : {spec.num_gpus}")
    print(f"  chunks        : {len(chunk_dirs)}")
    for d in chunk_dirs:
        n = sum(1 for _ in d.iterdir())
        print(f"    {d.name}: {n} {spec.item_noun}")
    for key, value in spec.summary_fields.items():
        print(f"  {key:<13} : {value}")
    print(bar)


def _build_subprocess_env(spec: JobSpec, gpu_index: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{spec.repo_src}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(spec.repo_src)
    )
    if spec.accelerator == "gpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    for k, v in spec.env_overrides.items():
        env[str(k)] = str(v)
    return env


def _launch(spec: JobSpec, chunk_dirs: list[Path]) -> int:
    """Spawn one subprocess per chunk; wait; return the worst exit code."""
    processes: list[tuple[Path, subprocess.Popen]] = []
    for i, chunk_dir in enumerate(chunk_dirs):
        env = _build_subprocess_env(spec, gpu_index=i)
        cmd = spec.build_command(chunk_dir, spec.out_dir)

        log_path = spec.out_dir / f"_log_{chunk_dir.name}.out"
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


def run_chunked(spec: JobSpec) -> int:
    """Run the discover → chunk → launch pipeline described by ``spec``.

    Returns the worst subprocess exit code (0 on full success). Raises
    ``SystemExit`` if discovery returns no files or ``sequences_dir`` is
    missing — these are user errors and should fail loudly before launch.
    """
    if not spec.sequences_dir.is_dir():
        raise SystemExit(f"sequences_dir does not exist: {spec.sequences_dir}")
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    items = spec.discover(spec.sequences_dir, spec.subset)
    if not items:
        raise SystemExit(f"No input files found under {spec.sequences_dir}")

    staging = _resolve_staging(spec)
    num_gpus = max(1, spec.num_gpus)
    chunk_dirs = make_chunks(items, staging, num_chunks=num_gpus)

    _print_banner(spec, items, chunk_dirs, staging)
    return _launch(spec, chunk_dirs)


# ---------------------------------------------------------------------------
# Convenience: derive the repo `src/` path from a script location
# ---------------------------------------------------------------------------

def repo_src_from(script_file: str) -> Path:
    """Return ``<repo>/src`` given ``__file__`` of a ``scripts/run_*.py``."""
    return Path(script_file).resolve().parents[1] / "src"
