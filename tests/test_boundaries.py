"""The analysis layer must not reach for a model. Asserted, not documented.

The refactoring plan states this as "offline analysis installs and runs without
model-specific environments". Read literally that is a rule about JAX, and JAX
is not the problem: three report producers run their linear algebra on the
accelerator, and `analysis.basis` reaches for a float64 `jnp.linalg.svd`
specifically because JAX's float32 default corrupted an archived basis once.
Forbidding jax would force those to be rewritten rather than moved.

What actually has to hold is that analysis never loads a model. So the rule is
about BACKENDS and WEIGHTS, and that is what these tests check.

    uv run pytest tests/test_boundaries.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "protein_interpretability"

# Importing any of these means a model environment is required.
BACKENDS = {"joltz", "mosaic", "boltz", "torch", "transformers", "jopenfold3",
            "equinox", "esm"}

# Allowed in analysis: jax is a numerics library here, not a model runtime.
NUMERICS = {"jax", "jaxlib", "numpy", "scipy"}


def imported_modules(path: Path) -> set[str]:
    """Every top-level module name this file imports, nested imports included."""
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.add(node.module.split(".")[0])
    return out


def analysis_files():
    return sorted((SRC / "analysis").glob("*.py"))


def test_analysis_subpackage_is_not_empty():
    """Guard the guard: an empty glob would make every test below vacuous."""
    assert len(analysis_files()) >= 4


@pytest.mark.parametrize("path", analysis_files(), ids=lambda p: p.name)
def test_analysis_imports_no_model_backend(path):
    offending = imported_modules(path) & BACKENDS
    assert not offending, (
        f"{path.name} imports {sorted(offending)}. Analysis reads artifacts; "
        "anything needing a model belongs in collection/.")


@pytest.mark.parametrize("path", analysis_files(), ids=lambda p: p.name)
def test_analysis_loads_no_weights(path):
    """A checkpoint path is a backend import that has not happened yet."""
    text = path.read_text()
    for needle in ("load_from_checkpoint", "MOSAIC_WEIGHTS", ".ckpt"):
        assert needle not in text, f"{path.name} references {needle!r}"


def test_jax_stays_allowed_in_analysis():
    """The rule is about backends, not about jax -- pin that so it is not
    'tightened' later into something three report producers cannot satisfy."""
    assert not (NUMERICS & BACKENDS)


def test_package_import_is_cheap():
    """`import protein_interpretability` must not drag in numpy or a backend.

    `inspect` and `render` are supposed to run on a login node without
    initialising CUDA, and that only holds if the top-level package is inert.
    """
    top = imported_modules(SRC / "__init__.py")
    assert not (top & (BACKENDS | NUMERICS)), f"top-level package imports {top}"


def test_the_basis_fits_without_jax_installed():
    """`analysis.basis` reached for `jax.numpy` unguarded, so every basis fit
    required a model environment -- the one thing analysis is supposed not to
    need. jax lives in the mosaic container, not in the analysis venv, so this
    failed for anyone running the documented `uv sync`.

    numpy is float64 natively, so the fallback is the same precision by another
    route, not a degraded path.
    """
    import numpy as np

    from protein_interpretability.analysis.basis import _svd

    rng = np.random.default_rng(0)
    M = rng.normal(size=(40, 8))
    V, ev = _svd(M)
    assert V.shape == (8, 8)
    assert abs(float(ev.sum()) - 1.0) < 1e-12
    # Orthonormality at float64, which is what the precision guard is about:
    # float32 lands near 1e-06 on a matrix this size.
    assert np.abs(V @ V.T - np.eye(8)).max() < 1e-12


def test_artifacts_module_is_backend_free():
    """Both sides read artifacts, so the storage seam has to stay importable
    from an environment with no model in it."""
    offending = imported_modules(SRC / "artifacts.py") & BACKENDS
    assert not offending, f"artifacts.py imports {sorted(offending)}"
