"""Offline analysis: reads artifacts, computes results, imports no model.

The rule this subpackage must satisfy is narrower than "no jax", and the
difference is not cosmetic. Three report producers -- the SVD study, the PC2
projection and the layer-match curve -- run their linear algebra on the
accelerator through `jnp`, and `basis` reaches for `jax` to get a float64 SVD
because JAX's float32 default silently corrupted an archived basis once. Under a
jax-free rule those would have to be rewritten rather than moved.

So the rule is: no model backend (joltz, mosaic, boltz, torch, transformers),
and no weights. `tests/test_boundaries.py` asserts it.
"""
