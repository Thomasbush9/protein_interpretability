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

Do not add a `pi_*` module. New library code goes in the package.

## Build, Test, and Development Commands
This repository is configured as a Python 3.12 project with `uv`.

- `uv sync`: create or update the local environment from `pyproject.toml` and `uv.lock`.
- `uv run pytest tests/test_records.py tests/test_boundaries.py -q`: the model-free
  tests. Name the files; collecting the directory has hung here.

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
`tests/` holds the model-free tests; run them by naming the files, because
collecting the directory has hung here (a cold `import pytest` off this
filesystem has taken 40 s, so give a run that looks stuck a few minutes before
concluding anything). The harness also carries its own self-tests next to the
code they cover — `pi_basis_test.py`, `pi_archive_test.py` — run directly.

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
