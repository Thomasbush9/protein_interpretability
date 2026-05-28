#!/usr/bin/env python3
"""Run Boltz2 gradient attribution over a directory of sequences.

Thin orchestrator around :mod:`protein_interpretability.attribution.cli`.

Outputs land at::

    <out_dir>/boltz_results_<chunk_name>/<record_id>/gradients/
        <record_id>_attribution_R{0,5,10}.pt

The target spec is global (one ``contact:i,j`` for the whole batch). For
per-record targets, run multiple jobs with different subset_files.

Usage::

    # Activate the boltz env first (e.g. source scripts/prepare_env.sh on the cluster)
    python scripts/run_boltz_gradients.py --config scripts/boltz_gradients_config.yaml
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
    load_config,
    load_subset,
    repo_src_from,
    required,
    run_chunked,
)


def format_recycling_steps(value) -> str:
    """``attribution.cli`` takes a comma-separated list of integer depths."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(x)) for x in value)
    raise ValueError(f"Cannot parse recycling_steps: {value!r}")


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


def _build_gradients_command(
    cfg: dict,
    targets: list[str],
    chunk_dir: Path,
    out_dir: Path,
) -> list[str]:
    boltz = cfg["boltz"]
    attribution = cfg["attribution"]
    runtime = cfg.get("runtime") or {}

    cmd = [
        runtime.get("python", "python") or "python",
        "-m",
        "protein_interpretability.attribution.cli",
        str(chunk_dir),
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
    boltz_cfg = required(cfg, "boltz")
    attribution_cfg = required(cfg, "attribution")
    runtime_cfg = cfg.get("runtime") or {}

    if "targets" not in attribution_cfg and "target" not in attribution_cfg:
        raise SystemExit(
            "attribution.targets is required (list, e.g. ['contact:65,202', 'mean_contact'])"
        )

    sequences_dir = Path(required(cfg, "input", "sequences_dir")).expanduser()
    out_dir = Path(required(cfg, "output", "out_dir")).expanduser()

    # Subset paths may be relative to the config file (preserved from previous impl).
    subset_path = (cfg.get("input") or {}).get("subset_file")
    if subset_path:
        sp = Path(subset_path).expanduser()
        if not sp.is_absolute():
            sp = (args.config.parent / sp).resolve()
        subset = load_subset(sp)
    else:
        subset = None

    staging_dir_cfg = (cfg.get("output") or {}).get("staging_dir")
    staging_dir = Path(staging_dir_cfg).expanduser() if staging_dir_cfg else None

    targets = normalize_targets(attribution_cfg.get("targets") or attribution_cfg.get("target"))

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
        build_command=lambda c, o: _build_gradients_command(cfg, targets, c, o),
        job_title="Boltz2 gradient attribution",
        item_noun="yamls",
        summary_fields={
            "targets": targets,
            "recycling": format_recycling_steps(attribution_cfg.get("recycling_steps", "0,5,10")),
            "no_kernels": bool(boltz_cfg.get("no_kernels", True)),
        },
    )

    rc = run_chunked(spec)
    if rc != 0:
        print(f"[error] at least one chunk failed (rc={rc})", file=sys.stderr)
        raise SystemExit(rc)
    print(f"[ok] all chunks complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
