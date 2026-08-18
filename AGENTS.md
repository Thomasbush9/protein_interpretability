# Repository Guidelines

## Project Structure & Module Organization

Two directories hold the work, and the split between them is the one rule worth
knowing before changing anything.

`src/protein_interpretability/` is the library.

- `analysis/` — statistics, basis, chemistry, metrics. **Reads artifacts; must
  never import a model backend** (joltz, mosaic, boltz, torch, transformers) or
  load weights. It *may* use jax as a numerics library: three report producers
  run their linear algebra on the GPU, and `analysis/basis.py` needs a float64
  `jnp` SVD because float32 corrupted an archived basis once. The rule is about
  backends, not jax, and `tests/test_boundaries.py` enforces exactly that.
- `collection/` — `records.py` is the cross-model schema every adapter must
  satisfy. Model backends are imported inside the adapter factories, never at
  module level, so inspection stays runnable on a login node.
- `artifacts.py` — the only sanctioned way to write a result, and the guard that
  reads it back. Both sides use it, which is why it sits above the split.
- `experiments/protocol.py` — the block that says what a number is comparable to.

`jax_harness/` holds the experiment entry points: `exp_*` collect, `analyze_*`
and `probe_*` analyse, `fig_*` plot, `build_*_report.py` assemble, `launch_*.sh`
and `*.sbatch` submit. The `pi_*.py` files here are six-line aliases into the
package, kept so existing call sites work unchanged; import either name.

`configs/cohorts/` holds four checksummed cohort manifests (basis, heldout,
cross-model, intervention). A cohort is a named list of assays, not whatever a
glob matched; `Cohort.load(name).verify()` refuses if an input moved.

Do not add a `pi_*` module. New library code goes in the package.

**`docs/API.md` is the guide for writing scripts against this library**, with a
runnable example at `experiments/analysis/example_transfer_probe.py`. Read it
before adding an experiment.

## Build, Test, and Development Commands
This repository is configured as a Python 3.12 project with `uv`.

- `uv sync`: create or update the local environment from `pyproject.toml` and `uv.lock`.
- `uv run pytest tests/ -q`: the model-free tests. Under a second.

Nothing that loads a model runs on the login node. Jobs go through one of two
submitters, and the difference matters:

- `sbatch jax_harness/analysis.sbatch <script.py> ...` runs
  `prot_interp_files/harness/`, a **copy** of `jax_harness` that is not a
  checkout and does not carry `src/`. Every archived result was produced this way.
- `sbatch jax_harness/checkout.sbatch <script.py> ...` runs the repository
  itself. Use this to verify anything that touches the library, since the mirror
  will not have it.

Reproducing an archived result does not require guessing its command: every
result file carries the exact `argv` that produced it in its `provenance` block.
`prot_interp_files/runs/check_20260817/` holds a harness that reruns all 17
report producers from those records and diffs them.

## Coding Style & Naming Conventions
Follow standard Python conventions: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and concise docstrings for public functions. Add type hints where practical; the existing codebase already uses them. Keep plotting and extraction concerns separated by module. Use descriptive filenames like `plot_attention.py`, not generic names like `helpers2.py`.

## Testing Guidelines
`tests/` holds the model-free tests: `uv run pytest tests/ -q`, well under a
second. They used to appear to hang, which was recorded here as a pytest bug in
this environment. It was not: `torch` was declared as a dependency, the analysis
venv was 5 GB, and a cold `import pytest` off this filesystem took 40 s. Only
`pi_core` imports torch, for a checkpoint load that runs inside the container
which supplies its own, so the declaration was vestigial. Removing it took the
venv to 569 MB. **Do not add torch back.**

The harness also carries self-tests next to the code they cover —
`pi_basis_test.py`, `pi_archive_test.py` — run directly.

Write guards that assert the **refusal**, not the happy path. Both existing
suites are built that way, and it is the reason they are worth having: a schema
that accepts a per-atom pLDDT, or a boundary test that passes because its glob
matched nothing, costs more than no test at all.

The real regression test is scientific, not unit-level. Any change touching the
library must rerun the 17 report producers and diff them against their archives
(`prot_interp_files/runs/check_20260817/`). Expect exact zeros, with three known
exceptions — all measured by running the same code twice, which is the only way
to tell a real change from this:

- `analyze_svd` — prediction-ordered curves move up to ~1e-3 per assay between
  identical runs; its SVD also carries ~3e-14 of float noise.
- `probe_gate` — `live_by_layer` moves ~1e-6 between identical runs. It reduces
  `jax.vmap` over the Pairformer transitions in explicit float32, so the
  variation is the accelerator's, not the code's.

Never conclude a change caused a difference until the unchanged code has been
run twice. Two findings this session that looked like regressions were not.

## Commit & Pull Request Guidelines
Recent commits use short, imperative, lower-case summaries such as `plotting attention with rollout`. Keep commits focused and similarly concise. Pull requests should include:

- a brief description of the change and its motivation
- linked issue or experiment context, if applicable
- exact validation commands run
- screenshots or exported figures for visualization changes

## Data & Environment Notes
Do not commit large model outputs, cache directories, or generated attention tensors. Keep local datasets and Boltz/ESM artifacts outside the repository or ignored by Git.
