"""Numerical equivalence test for the PairWeightedAveraging weight monkey-patch.

Scope (verified here):
    - Eager unchunked path only.
    - float32, CPU, small (S, N).
    Under these conditions the patched ``PairWeightedAveraging.forward``
    returns output ``atol=0, rtol=0`` against the unpatched forward — the
    monkey-patch reuses the same op sequence as the upstream non-chunked
    branch.

Forced-unchunked behaviour:
    The patch *always* runs the unchunked path so we can capture the full
    ``(B, H, N, N)`` softmax matrix. When the caller passes
    ``chunk_heads=True`` the math is still equivalent — only memory layout
    differs — so we additionally assert the patched output matches the
    unpatched ``chunk_heads=False`` reference.

Skips cleanly if the ``boltz`` package is not importable.
"""

from __future__ import annotations

import pytest
import torch

boltz = pytest.importorskip("boltz")
from boltz.model.layers.pair_averaging import PairWeightedAveraging  # noqa: E402

from protein_interpretability.extractor_boltz import Boltz2Extractor  # noqa: E402


class _StubExtractor:
    """Minimal Boltz2Extractor stand-in that exposes `_store` and `_patched`."""

    def __init__(self):
        self.activations: list[dict] = [{}]
        self._patched: list = []

    def _store(self, key: str, value: torch.Tensor) -> None:
        self.activations[-1][key] = value.detach().clone()


def _build() -> PairWeightedAveraging:
    # Small config: cheap but exercises every tensor op in the forward.
    return PairWeightedAveraging(c_m=8, c_z=8, c_h=4, num_heads=2).eval()


def _make_inputs(B: int = 1, S: int = 3, N: int = 5):
    torch.manual_seed(0)
    m = torch.randn(B, S, N, 8, dtype=torch.float32)
    z = torch.randn(B, N, N, 8, dtype=torch.float32)
    mask = torch.ones(B, N, N, dtype=torch.float32)
    # Pad one column to exercise the mask_bias path.
    mask[:, :, -1] = 0.0
    return m, z, mask


@pytest.mark.parametrize("chunk_heads", [False, True])
def test_patched_forward_matches_unchunked(chunk_heads):
    """Patched forward must reproduce the unchunked-path output bit-for-bit.

    The unchunked path is the canonical reference; the chunked path is
    mathematically equivalent but accumulates head outputs sequentially, so we
    only assert exact equality against ``chunk_heads=False``.
    """
    mod = _build()
    m, z, mask = _make_inputs()

    with torch.no_grad():
        ref = mod(m.clone(), z.clone(), mask.clone(), chunk_heads=False).clone()

    stub = _StubExtractor()
    Boltz2Extractor._patch_pwa_weights(stub, mod, "pwa")

    with torch.no_grad():
        out = mod(m.clone(), z.clone(), mask.clone(), chunk_heads=chunk_heads)

    assert torch.allclose(out, ref, atol=0, rtol=0), (
        f"patched output drifts from unchunked reference "
        f"(max abs diff = {(out - ref).abs().max().item():.2e})"
    )

    assert "pwa" in stub.activations[-1]
    w = stub.activations[-1]["pwa"]
    assert w.shape == (1, mod.num_heads, 5, 5), f"unexpected weight shape: {w.shape}"

    # Softmax rows sum to 1 along the attended dim (j).
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # Masked column (j = -1) must receive zero attention everywhere.
    assert torch.all(w[..., -1] == 0)

    for module, attr, original in stub._patched:
        setattr(module, attr, original)


def test_patch_capture_then_restore_roundtrip():
    """After restoring the original forward, behaviour must match a fresh module."""
    mod = _build()
    m, z, mask = _make_inputs()

    with torch.no_grad():
        before = mod(m.clone(), z.clone(), mask.clone(), chunk_heads=False).clone()

    stub = _StubExtractor()
    Boltz2Extractor._patch_pwa_weights(stub, mod, "pwa")
    with torch.no_grad():
        _ = mod(m.clone(), z.clone(), mask.clone(), chunk_heads=False)

    for module, attr, original in stub._patched:
        setattr(module, attr, original)

    with torch.no_grad():
        after = mod(m.clone(), z.clone(), mask.clone(), chunk_heads=False)

    assert torch.allclose(before, after, atol=0, rtol=0)
