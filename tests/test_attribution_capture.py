"""Capture + runner tests against a mock model that mimics Boltz2's interface.

The mock exposes ``input_embedder``, ``msa_module``, ``distogram_module`` so
``GradientCapture`` can attach its hooks. Calling ``forward(batch, recycling_steps=K)``
runs a tiny computation that depends on K, exercising the per-step backward
pass end-to-end without touching Boltz weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from protein_interpretability.attribution import (
    AttributionResult,
    ContactBinNLL,
    GradientCapture,
    run_per_step,
    save_result,
    load_result,
)


class _Embedder(nn.Module):
    def __init__(self, n: int, d: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.n = n
        self.d = d

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["s_init"]              # (B, N, D)
        return self.proj(x)


class _MSAModule(nn.Module):
    def __init__(self, d_msa: int, n: int, d_pair: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(d_msa, d_msa, bias=False)
        self.read_to_pair = nn.Linear(d_msa, d_pair, bias=False)
        self.n = n

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # m: (B, S, N, D_msa). Mean over S → contribution to pair representation
        m = self.proj_in(m)
        z = self.read_to_pair(m.mean(dim=1))     # (B, N, D_pair)
        return z[:, :, None, :] + z[:, None, :, :]   # (B, N, N, D_pair) crude outer add


class _DistogramHead(nn.Module):
    def __init__(self, d_pair: int, num_bins: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_pair, num_bins)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class _MockBoltz(nn.Module):
    """Minimal model with the three attributes ``GradientCapture`` looks for."""

    def __init__(
        self,
        n: int = 12,
        d_s: int = 16,
        d_msa: int = 16,
        d_pair: int = 8,
        num_bins: int = 32,
    ) -> None:
        super().__init__()
        self.input_embedder = _Embedder(n, d_s)
        self.msa_module = _MSAModule(d_msa, n, d_pair)
        self.distogram_module = _DistogramHead(d_pair, num_bins)
        self.s_to_pair = nn.Linear(d_s, d_pair, bias=False)
        self.recycle_mix = nn.Linear(d_pair, d_pair, bias=False)
        self.n = n
        self.num_bins = num_bins

    def forward(self, batch: dict, *, recycling_steps: int = 0, **_) -> torch.Tensor:
        s = self.input_embedder(batch)               # (B, N, D_s)
        z_seq = self.s_to_pair(s)
        z = z_seq[:, :, None, :] + z_seq[:, None, :, :]  # (B, N, N, D_pair)

        for _ in range(recycling_steps + 1):
            z_msa = self.msa_module(batch["msa"])    # fires every recycle
            z = self.recycle_mix(z + z_msa)

        return self.distogram_module(z)


def _make_batch(B: int = 1, N: int = 12, S: int = 4, d_s: int = 16, d_msa: int = 16):
    torch.manual_seed(0)
    return {
        "s_init": torch.randn(B, N, d_s),
        "msa": torch.randn(B, S, N, d_msa),
        "token_pad_mask": torch.ones(B, N, dtype=torch.bool),
    }


def test_capture_records_three_surfaces() -> None:
    model = _MockBoltz().eval()
    batch = _make_batch()
    cap = GradientCapture(model)
    with cap, torch.set_grad_enabled(True):
        _ = model(batch, recycling_steps=2)
        assert "query_emb" in cap.captured_keys()
        assert "msa_emb" in cap.captured_keys()
        assert "distogram" in cap.captured_keys()
        assert cap.distogram.shape[-1] == model.num_bins


def test_capture_grad_flows_to_query_and_msa() -> None:
    model = _MockBoltz().eval()
    batch = _make_batch()
    target = ContactBinNLL(pair_i=2, pair_j=7)

    cap = GradientCapture(model)
    with cap, torch.set_grad_enabled(True):
        _ = model(batch, recycling_steps=1)
        loss = target(cap.distogram, token_mask=batch["token_pad_mask"])
        loss.backward()

        assert cap.query_emb.grad is not None
        assert cap.msa_emb.grad is not None
        assert cap.query_emb.grad.abs().sum() > 0
        assert cap.msa_emb.grad.abs().sum() > 0


def test_pair_rep_pre_hook_replaces_input_with_leaf() -> None:
    model = _MockBoltz().eval()
    batch = _make_batch()
    cap = GradientCapture(model)
    cap.install_pair_rep_attribution()
    try:
        with torch.set_grad_enabled(True):
            _ = model(batch, recycling_steps=1)
        assert "pair_input" in cap.captured_keys()
        assert "distogram" in cap.captured_keys()
        assert cap.pair_input.is_leaf, "captured pair_input must be a leaf"
        assert cap.pair_input.requires_grad
        # distogram should sit on the autograd graph rooted at pair_input.
        assert cap.distogram.requires_grad
    finally:
        cap.remove()


def test_pair_rep_backward_populates_leaf_grad() -> None:
    model = _MockBoltz().eval()
    batch = _make_batch()
    target = ContactBinNLL(pair_i=2, pair_j=7)

    cap = GradientCapture(model)
    cap.install_pair_rep_attribution()
    try:
        with torch.set_grad_enabled(True):
            _ = model(batch, recycling_steps=1)
            loss = target(cap.distogram, token_mask=batch["token_pad_mask"])
            loss.backward()
        assert cap.pair_input.grad is not None
        assert cap.pair_input.grad.abs().sum() > 0
    finally:
        cap.remove()


def test_run_per_step_returns_one_pair_result_per_K() -> None:
    model = _MockBoltz().eval()
    batch = _make_batch()
    target = ContactBinNLL(pair_i=2, pair_j=7)

    results = run_per_step(
        model=model,
        batch=batch,
        target=target,
        recycling_steps=(0, 1, 2),
    )
    assert len(results) == 3
    for r, expected_k in zip(results, (0, 1, 2)):
        assert isinstance(r, AttributionResult)
        assert r.recycling_steps == expected_k
        # Pair-rep mode populates pair_*; legacy query_*/msa_* are None.
        assert r.pair_grad is not None
        assert r.pair_input is not None
        assert r.pair_grad.shape == r.pair_input.shape
        assert r.pair_grad.abs().sum() > 0
        assert r.query_grad is None
        assert r.msa_grad is None
        assert torch.isfinite(torch.tensor(r.target_value))


def test_run_per_step_zeros_pair_grad_at_padded_positions() -> None:
    model = _MockBoltz().eval()
    batch = _make_batch()
    batch["token_pad_mask"][:, -2:] = False  # mask last two tokens
    target = ContactBinNLL(pair_i=2, pair_j=7)

    [r] = run_per_step(model=model, batch=batch, target=target, recycling_steps=(0,))
    # pair_grad shape: (B, N, N, D_pair). Any pair touching the masked positions
    # at indices [-2:] on either axis must be zero.
    assert torch.all(r.pair_grad[:, -2:, :, :] == 0)
    assert torch.all(r.pair_grad[:, :, -2:, :] == 0)
    assert torch.all(r.pair_input[:, -2:, :, :] == 0)
    assert torch.all(r.pair_input[:, :, -2:, :] == 0)


def test_save_load_round_trip(tmp_path) -> None:
    model = _MockBoltz().eval()
    batch = _make_batch()
    target = ContactBinNLL(pair_i=2, pair_j=7)

    [r] = run_per_step(model=model, batch=batch, target=target, recycling_steps=(1,))
    path = save_result(r, tmp_path / "out.pt")
    loaded = load_result(path)

    assert loaded.recycling_steps == r.recycling_steps
    assert loaded.target_spec == r.target_spec
    assert torch.equal(loaded.pair_grad, r.pair_grad)
    assert torch.equal(loaded.pair_input, r.pair_input)
    assert torch.equal(loaded.token_mask, r.token_mask)
    assert loaded.query_grad is None
    assert loaded.msa_grad is None


def test_capture_fails_loudly_without_required_attrs() -> None:
    class _Bare(nn.Module):
        pass

    cap = GradientCapture(_Bare())
    try:
        cap.install_pair_rep_attribution()
    except AttributeError as e:
        assert "distogram_module" in str(e)
        return
    raise AssertionError("expected AttributeError on missing distogram_module")


def test_input_x_grad_helper() -> None:
    model = _MockBoltz().eval()
    batch = _make_batch()
    target = ContactBinNLL(pair_i=2, pair_j=7)
    [r] = run_per_step(model=model, batch=batch, target=target, recycling_steps=(0,))
    pxg = r.input_x_grad("pair")
    assert pxg.shape == r.pair_grad.shape
    assert torch.allclose(pxg, r.pair_input * r.pair_grad)
