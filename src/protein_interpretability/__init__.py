"""Protein interpretability: representation collection, prediction, and offline analysis.

The package is organised around one boundary, and it is the only one that is
enforced rather than documented:

    collection/   loads models, runs them, writes artifacts   -- needs a backend
    analysis/     reads artifacts and computes results        -- must NOT

`analysis` may import numpy, scipy, and jax-as-a-numerics-library. It may not
import a model backend (joltz, mosaic, boltz, torch, transformers) and it may
not load weights. That distinction is deliberate and narrower than "no jax":
three of the report producers -- the SVD study, the PC2 projection and the
layer-match curve -- run their linear algebra on the accelerator and would have
to be rewritten, not moved, under a jax-free rule. `tests/test_boundaries.py`
asserts the real rule.

Nothing here is imported at package level: `import protein_interpretability`
must stay free of numpy, jax and every model backend so that `inspect` and
`render` can run on a login node without initialising CUDA.
"""

__version__ = "0.2.0"
