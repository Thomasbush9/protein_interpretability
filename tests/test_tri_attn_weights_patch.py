"""Numerical equivalence test for the triangular-attention weight monkey-patch.

Scope (verified here):
    - Eager path only (``use_kernels=False``).
    - ``chunk_size=None``.
    - float32, CPU, small N.
    Under these conditions the patched ``TriangleAttention.forward`` returns
    output ``atol=0, rtol=0`` against the unpatched forward.

NOT covered here (and not claimed equivalent):
    - ``use_kernels=True`` — the fused kernels (trifast / cuequivariance)
      never materialize the softmax matrix; the patch forces the eager path,
      so output drifts at kernel-fusion precision.
    - CUDA + bf16 autocast — expected to match to bf16-add precision since
      the patch reuses ``softmax_no_cast`` and the same autocast context,
      but this is not numerically asserted by these tests.
    - ``chunk_size != None`` — the patch ignores ``chunk_size`` and always
      runs the non-chunked path; equivalent in math, different in memory.

Skips cleanly if the ``boltz`` package is not importable.
"""

from __future__ import annotations

import pytest
import torch

boltz = pytest.importorskip("boltz")
from boltz.model.layers.triangular_attention.attention import (  # noqa: E402
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)

from protein_interpretability.extractor_boltz import Boltz2Extractor  # noqa: E402


class _StubExtractor:
    """Minimal Boltz2Extractor stand-in that exposes `_store` and `_patched`."""

    def __init__(self):
        self.activations: list[dict] = [{}]
        self._patched: list = []

    def _store(self, key: str, value: torch.Tensor) -> None:
        self.activations[-1][key] = value.detach().clone()


def _build(starting: bool) -> torch.nn.Module:
    cls = TriangleAttentionStartingNode if starting else TriangleAttentionEndingNode
    # Small config so the test is cheap but exercises all tensor paths.
    return cls(c_in=16, c_hidden=8, no_heads=4).eval()


@pytest.mark.parametrize("starting", [True, False])
def test_patched_forward_matches_original_eager(starting):
    torch.manual_seed(0)
    mod = _build(starting)

    B, N = 1, 6
    x = torch.randn(B, N, N, 16, dtype=torch.float32)
    mask = torch.ones(B, N, N, dtype=torch.float32)
    # Pad one column to exercise mask_bias.
    mask[:, :, -1] = 0.0

    with torch.no_grad():
        ref = mod(x.clone(), mask=mask.clone(), use_kernels=False).clone()

    stub = _StubExtractor()
    # Bind the real method from Boltz2Extractor onto the stub.
    Boltz2Extractor._patch_tri_attention_weights(stub, mod, "tri_weights")

    with torch.no_grad():
        out = mod(x.clone(), mask=mask.clone(), use_kernels=False)

    # Output must match the unpatched eager forward exactly.
    assert torch.allclose(out, ref, atol=0, rtol=0), (
        f"patched output differs from original "
        f"(max abs diff = {(out - ref).abs().max().item():.2e})"
    )

    # Weights must have been captured with the expected shape/properties.
    assert "tri_weights" in stub.activations[-1]
    w = stub.activations[-1]["tri_weights"]
    H = mod.mha.no_heads
    assert w.shape == (B, N, H, N, N), f"unexpected weight shape: {w.shape}"
    # Softmax rows sum to 1 along the last (attended) dim.
    assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5)

    # Restore original forward so the module is reusable.
    for module, attr, original in stub._patched:
        setattr(module, attr, original)


@pytest.mark.parametrize("starting", [True, False])
def test_patched_forward_handles_default_mask(starting):
    """Path where ``mask=None`` should still produce original-matching output."""
    torch.manual_seed(1)
    mod = _build(starting)
    x = torch.randn(1, 5, 5, 16)

    with torch.no_grad():
        ref = mod(x.clone(), use_kernels=False).clone()

    stub = _StubExtractor()
    Boltz2Extractor._patch_tri_attention_weights(stub, mod, "w")
    with torch.no_grad():
        out = mod(x.clone(), use_kernels=False)

    assert torch.allclose(out, ref, atol=0, rtol=0)

    for module, attr, original in stub._patched:
        setattr(module, attr, original)
