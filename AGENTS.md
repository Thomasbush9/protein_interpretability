# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/protein_interpretability/`. Use `utils.py` for shared helpers, `viz.py` and `plot_attention.py` for visualization, and `extract_attention.py` and `extractor.py` for attention extraction workflows. Exploratory notebooks such as `esm_interpretability.ipynb` and `attention_boltz.ipynb` sit alongside the package code and should stay analysis-focused. `main.py` is a minimal entrypoint. Keep new modules inside `src/protein_interpretability/`; avoid adding logic at the repository root.

## Build, Test, and Development Commands
This repository is configured as a Python 3.12 project with `uv`.

- `uv sync`: create or update the local environment from `pyproject.toml` and `uv.lock`.
- `uv run python main.py`: run the current top-level entrypoint.
- `uv run python -m protein_interpretability.extract_attention --help`: inspect CLI options for the main extraction script.
- `uv run pytest`: run tests when a test suite is present.

If you add a new script, prefer a module entrypoint under `src/protein_interpretability/` so it can be run with `uv run python -m ...`.

## Coding Style & Naming Conventions
Follow standard Python conventions: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and concise docstrings for public functions. Add type hints where practical; the existing codebase already uses them. Keep plotting and extraction concerns separated by module. Use descriptive filenames like `plot_attention.py`, not generic names like `helpers2.py`.

## Testing Guidelines
There is no dedicated `tests/` directory yet. Add tests under `tests/` with names like `test_utils.py`. Prefer `pytest` style tests that cover parsing, tensor-shape assumptions, and CLI argument handling. For notebook-heavy changes, include a reproducible script or command in the PR so reviewers can verify outputs without rerunning the notebook manually. Current known issue: `uv run pytest` can segfault during `_pytest/capture.py` startup in this environment, so validate new code with direct `uv run python ...` smoke tests until the runner issue is fixed.

## Commit & Pull Request Guidelines
Recent commits use short, imperative, lower-case summaries such as `plotting attention with rollout`. Keep commits focused and similarly concise. Pull requests should include:

- a brief description of the change and its motivation
- linked issue or experiment context, if applicable
- exact validation commands run
- screenshots or exported figures for visualization changes

## Data & Environment Notes
Do not commit large model outputs, cache directories, or generated attention tensors. Keep local datasets and Boltz/ESM artifacts outside the repository or ignored by Git.
