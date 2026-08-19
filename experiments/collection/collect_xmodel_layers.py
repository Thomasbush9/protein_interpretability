"""Per-layer trunk capture for three models over the held-out cohort.

The whole scientific declaration is the `task` below. Everything else in this
file is a way to run it or to look at it before running it.

    cohort    heldout_assays -- 16 assays, 40 to 118 residues, disjoint from
              the basis the shared PC directions were fitted on
    models    boltz2, of3, protenix -- the same three the cross-model section
              of the report compares, through the same capture kernel
    regime    subsample, matching the archived xm_* captures so tonight's runs
              extend them rather than starting a second incompatible family
    capture   the per-layer pair and single rows as VECTORS, both KL fields,
              and the structure module's own pLDDT and coordinates -- the
              internal side and the output side in one pass

WHY THE OUTPUT SIDE IS IN THE SAME CAPTURE. `internal versus output` is a PAIRED
comparison: the same variants, the same protocol, the internal features against
what the model actually emitted. Collecting the two in separate runs would make
them a comparison of two jobs. The sampler is most of the per-variant cost --
about 22 s against 9 s for the trunk alone -- and it buys the other half of the
headline claim.

    # login node: resolve, price, and check every input, loading no model
    uv run python experiments/collection/collect_xmodel_layers.py --inspect

    # one assay, one model, on a GPU
    sbatch jax_harness/checkout.sbatch \\
        ../experiments/collection/collect_xmodel_layers.py \\
        --model protenix --assay RCRO_LAMBD_Tsuboyama_2023_1ORC
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from protein_interpretability.collection import CaptureSpec, Cohort   # noqa: E402
from protein_interpretability.collection.task import (                # noqa: E402
    CollectionTask,
    ModelSpec,
)

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")

COHORT = "heldout_assays"
MODELS = ("boltz2", "of3", "protenix")
N_VARIANTS = 100          # spread across the sorted score range, as exp_gym_deep does

# Captured for every model. `dz_vec`/`ds_vec` are the DIRECTIONS -- the norms
# come free from them and the reverse is not true, which is the whole reason the
# cross-model analysis could be done offline at all.
FIELDS = ("dz_vec", "ds_vec", "dz_site", "ds_site", "kl_site", "kl_glob",
          "ca", "plddt_mean", "plddt_site", "score", "pos")


def task_for(model: str, *, layers="all", n_variants=N_VARIANTS,
             cohort=COHORT, output=None, run="r1") -> CollectionTask:
    """The declaration, for one model. Identical in every other respect.

    THE TASK NAME IS THE FILE PREFIX, and it is `xm_{model}_{run}` on purpose.
    `analyze_xmodel.py`, `analyze_xmodel_io.py` and `analyze_depth.py` all locate
    their inputs by that exact pattern, so captures written under it are
    readable by the existing cross-model analyses with nothing but a `--dir`
    pointed at them -- no producer had to be edited to accept a second naming
    convention. They land in their own directory rather than beside the archived
    `xm_*` captures, so the two sets are selected by path and never mixed.
    """
    return CollectionTask(
        name=f"xm_{model}_{run}",
        cohort=Cohort.load(cohort),
        model=ModelSpec(
            name=model,
            backend="mosaic",
            recycles=3,
            seed=0,
            msa="subsample",       # matches the archived xm_* captures
            msa_cap=2048,
            network="blocked",
            options={"sampling_steps": 200},
        ),
        capture=CaptureSpec(
            model=model,
            fields=FIELDS,
            layers=layers,
            reduction="vector",
            recycles=3,
            dtype="float32",
            notes="per-layer trunk capture plus the structure module's own "
                  "pLDDT and CA, so internal and output are the same run",
        ),
        output=str(output or W / "runs" / "xmodel_layers"),
        resume="resume",
        n_variants=n_variants,
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS,
                    help="omit with --inspect to price all three")
    ap.add_argument("--assay", action="append",
                    help="restrict to these assays; repeatable")
    ap.add_argument("--cohort", default=COHORT)
    ap.add_argument("--layers", default="all",
                    help="'all', 'final', or a comma-separated list")
    ap.add_argument("--n-variants", type=int, default=N_VARIANTS)
    ap.add_argument("--run", default="r1",
                    help="replicate tag; part of the filename and of the task "
                         "id, so a second replicate cannot overwrite the first")
    ap.add_argument("--output")
    ap.add_argument("--inspect", action="store_true",
                    help="resolve and price; loads no model")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def _layers(text):
    if text in ("all", "final"):
        return text
    return tuple(int(x) for x in text.split(",") if x.strip())


def main(argv=None) -> int:
    a = parse_args(argv)
    models = [a.model] if a.model else list(MODELS)

    total = 0
    for model in models:
        task = task_for(model, layers=_layers(a.layers),
                        n_variants=a.n_variants, cohort=a.cohort,
                        output=a.output, run=a.run)
        resolved = task.inspect()
        total += resolved.estimated_bytes
        print(resolved.describe())
        print()

    if a.inspect:
        print(f"total {total / 1e9:.2f} GB across {len(models)} model(s); "
              f"no model loaded")
        return 0

    if not a.model:
        raise SystemExit(
            "--model is required to run; a single job collects one model so "
            "that a failure loses one model's queue and not three")

    task = task_for(a.model, layers=_layers(a.layers), n_variants=a.n_variants,
                    cohort=a.cohort, output=a.output, run=a.run)
    written = task.run(assays=a.assay, dry_run=a.dry_run)
    print(f"\n{len(written)} artifact(s) under {task.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
