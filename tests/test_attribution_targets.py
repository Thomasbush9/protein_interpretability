"""Unit tests for attribution targets — pure tensor math, no Boltz needed."""

from __future__ import annotations

import pytest
import torch

from protein_interpretability.attribution.targets import (
    DEFAULT_CONTACT_BIN_HI,
    DEFAULT_NUM_BINS,
    ContactBinNLL,
    DistogramKL,
    PairLogProb,
)


@pytest.fixture
def logits() -> torch.Tensor:
    torch.manual_seed(0)
    B, N, K = 1, 8, DEFAULT_NUM_BINS
    return torch.randn(B, N, N, K, requires_grad=True)


def test_contact_bin_nll_returns_scalar(logits: torch.Tensor) -> None:
    target = ContactBinNLL(pair_i=2, pair_j=5)
    loss = target(logits)
    assert loss.shape == ()
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_contact_bin_nll_grad_flows_to_target_pair_only(logits: torch.Tensor) -> None:
    target = ContactBinNLL(pair_i=2, pair_j=5)
    loss = target(logits)
    loss.backward()
    grad = logits.grad
    assert grad is not None

    g = grad[0]                         # (N, N, K)
    nz_pair = g.abs().sum(dim=-1) > 0   # (N, N)
    expected = torch.zeros_like(nz_pair)
    expected[2, 5] = True
    assert torch.equal(nz_pair, expected), (
        f"gradient leaked to non-target pairs:\n{nz_pair.int()}"
    )


def test_pair_log_prob_grad_concentrated_on_target_bin(logits: torch.Tensor) -> None:
    target = PairLogProb(pair_i=1, pair_j=4, bin=7)
    loss = target(logits)
    loss.backward()
    g_pair = logits.grad[0, 1, 4]   # (K,)
    target_bin = 7
    assert g_pair[target_bin].abs() > 0
    other = torch.cat([g_pair[:target_bin], g_pair[target_bin + 1 :]])
    assert g_pair[target_bin].abs() > other.abs().max()


def test_distogram_kl_zero_against_self() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 6, DEFAULT_NUM_BINS)
    target = DistogramKL(ref_logits=logits.clone())
    loss = target(logits)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_distogram_kl_positive_against_perturbation() -> None:
    torch.manual_seed(0)
    ref = torch.randn(1, 6, 6, DEFAULT_NUM_BINS)
    pred = ref + torch.randn_like(ref) * 0.5
    pred.requires_grad_(True)
    target = DistogramKL(ref_logits=ref)
    loss = target(pred)
    assert loss.item() > 0
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_token_mask_rejects_padded_target_pair(logits: torch.Tensor) -> None:
    mask = torch.ones(1, logits.shape[1], dtype=torch.bool)
    mask[0, 5] = False
    target = ContactBinNLL(pair_i=2, pair_j=5)
    with pytest.raises(ValueError, match="padded position"):
        target(logits, token_mask=mask)


def test_distogram_kl_token_mask_excludes_padded_pairs() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 6, DEFAULT_NUM_BINS, requires_grad=True)
    ref = torch.randn(1, 6, 6, DEFAULT_NUM_BINS)
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 5] = False  # mask last position
    target = DistogramKL(ref_logits=ref)
    loss_full = target(logits)
    loss_masked = target(logits, token_mask=mask)
    assert loss_masked.item() != pytest.approx(loss_full.item())
    assert torch.isfinite(loss_masked)


def test_target_spec_round_trip() -> None:
    spec = ContactBinNLL(pair_i=3, pair_j=10).spec()
    assert spec["kind"] == "ContactBinNLL"
    assert spec["pair_i"] == 3
    assert spec["pair_j"] == 10
    assert len(spec["contact_bins"]) == DEFAULT_CONTACT_BIN_HI
