"""Render a SLURM script from the site profile, so a job can be read before it runs.

The point is not to replace `analysis.sbatch` — that file and its siblings have
produced every archived result and should keep doing so. The point is that
rendering is what makes `inspect` and `render` possible at all: the resolved
account, partition, memory, image and command, printed on a login node, before
anything is queued.

Two properties this has to keep, both from §9 of the plan:

  * it imports no backend and initialises no CUDA, so it runs where you type it;
  * it renders the SAME job the hand-written submitters do. `equivalent_to()`
    checks a rendered script against one of them field by field, because a
    renderer that silently drifts from the scripts in use is worse than no
    renderer -- it would look authoritative while describing a job nobody runs.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from protein_interpretability.experiments.site import Site, SiteError

# Which directory a job's script is executed from.
SOURCES = {
    "mirror": ("roots.work", "harness",
               "the deployed copy, which has produced every archived result"),
    "checkout": ("roots.repo", "jax_harness",
                 "the git checkout -- required for anything touching src/"),
}


@dataclass(frozen=True)
class JobSpec:
    """Everything that differs between one job and another."""

    script: str
    args: tuple[str, ...] = ()
    name: str = "pi"
    source: str = "mirror"
    mem_mb: int | None = None
    time_min: int | None = None
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    exclusive: bool = False

    def resolved(self, site: Site) -> dict:
        if self.source not in SOURCES:
            raise SiteError(
                f"source must be one of {sorted(SOURCES)}, got {self.source!r}")
        root_key, subdir, _ = SOURCES[self.source]
        return {
            "name": self.name,
            "account": site.get("scheduler.account"),
            "partition": site.get("scheduler.partition"),
            "gpus_per_node": self.gpus_per_node
            or site.get("scheduler.gpus_per_node", 1),
            "cpus_per_task": self.cpus_per_task
            or site.get("scheduler.cpus_per_task", 8),
            "mem_mb": self.mem_mb or site.get("scheduler.mem_mb", 180000),
            "time_min": self.time_min or site.get("scheduler.time_min", 60),
            "logs": site.get("roots.logs"),
            "exec": site.get("container.exec"),
            "image": site.get("container.image"),
            "weights": site.get("container.weights"),
            "container_scratch": site.get("container.scratch"),
            "script_dir": f"{site.get(root_key)}/{subdir}",
            "env": site.env,
            "script": self.script,
            "args": list(self.args),
            "exclusive": self.exclusive,
        }


def render(spec: JobSpec, site: Site | None = None) -> str:
    """The exact script that would be submitted."""
    site = site or Site.load()
    r = spec.resolved(site)
    # The script path is one quoted argument; the args are separate ones. The
    # first version quoted the whole line, so `python "…/analyze_svd.py --out
    # /tmp/x.json"` looked for a file with spaces in its name.
    args = " ".join(shlex.quote(a) for a in r["args"])

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={r['name']}",
        f"#SBATCH --partition={r['partition']}",
        f"#SBATCH --account={r['account']}",
        f"#SBATCH --gpus-per-node={r['gpus_per_node']}",
        f"#SBATCH --cpus-per-task={r['cpus_per_task']}",
        f"#SBATCH --mem={r['mem_mb']}",
        f"#SBATCH --time={r['time_min']}",
    ]
    if r["exclusive"]:
        lines.append("#SBATCH --exclusive")
    lines += [
        f"#SBATCH --output={r['logs']}/%x_%j.out",
        f"#SBATCH --error={r['logs']}/%x_%j.out",
        "#",
        f"# Rendered from the site profile: {', '.join(p.name for p in site.sources)}",
        f"# Runs {SOURCES[spec.source][2]}.",
        "set -euo pipefail",
        "",
        f"export MOSAIC_SIF={r['image']}",
        f"export MOSAIC_WEIGHTS={r['weights']}",
        f"export MOSAIC_SCRATCH={r['container_scratch']}",
    ]
    lines += [f"export {k}={v}" for k, v in sorted(r["env"].items())]
    lines += [
        "",
        '# Recorded so a log says what ran, not only that something did.',
        'echo "host=$(hostname) job=${SLURM_JOB_ID:-none} '
        f'script={r["script"]}"',
        f'echo "source={spec.source} dir={r["script_dir"]}"',
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true",
        "",
        f'exec "{r["exec"]}" python "{r["script_dir"]}/{r["script"]}"'
        + (f" {args}" if args else ""),
    ]
    return "\n".join(lines) + "\n"


# ---- checking the renderer against the scripts actually in use --------------

_SBATCH = re.compile(r"^#SBATCH --([a-z-]+)=(.*)$", re.M)

# Fields whose difference is the job's business, not the site's.
_PER_JOB = {"job-name", "mem", "time", "output", "error"}


def sbatch_fields(text: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in _SBATCH.finditer(text)}


def equivalent_to(rendered: str, existing: Path | str,
                  *, ignore: set[str] = frozenset()) -> list[str]:
    """Differences between a rendered script and a hand-written one.

    Returns the disagreements on the fields the SITE owns — account, partition,
    GPUs, CPUs. Memory, time and job name are per-job and excluded, as is
    anything named in `ignore`.

    Empty means the renderer agrees with the file that has been producing
    results, which is the only evidence that it describes a real job.
    """
    theirs = sbatch_fields(Path(existing).read_text()
                           if not isinstance(existing, str) or "\n" not in existing
                           else existing)
    ours = sbatch_fields(rendered)
    skip = _PER_JOB | set(ignore)
    out = []
    for key, value in theirs.items():
        if key in skip:
            continue
        if key not in ours:
            out.append(f"{key}: {value!r} in the existing script, absent here")
        elif ours[key] != value:
            out.append(f"{key}: rendered {ours[key]!r}, existing {value!r}")
    return out
