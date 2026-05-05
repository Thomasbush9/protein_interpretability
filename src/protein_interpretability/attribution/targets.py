"""Scalar attribution targets defined on Boltz2 distogram logits.

Each target maps distogram logits ``(B, N, N, num_bins)`` to a scalar loss
suitable for backward(). Targets sit *before* the diffusion sampler so the
backward graph is deterministic.

Bin convention (Boltz2 default):
    64 bins from 2 Å to 22 Å, ~0.3125 Å step. Bin 0 = [2.0, 2.3125), bin 63 =
    [21.6875, 22.0]. ``contact`` typically ⇔ d < 8 Å ⇔ bins [0..18].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F

DEFAULT_NUM_BINS = 64
DEFAULT_CONTACT_BIN_HI = 19  # exclusive — bins [0..18] cover d < 8 Å in default scheme


class AttributionTarget(Protocol):
    """A scalar function of distogram logits.

    Implementations must:
      - Return a 0-dim differentiable tensor.
      - Honour ``token_mask`` (True = valid token); padded positions must not
        contribute to the loss.
      - Expose a ``spec()`` dict for provenance.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def spec(self) -> dict: ...


@dataclass
class ContactBinNLL:
    """Negative log-prob that pair (i, j) lies within the contact bins.

    Loss = -log Σ_{b ∈ contact_bins} softmax(logits[..., i, j, :])[b]

    Use case: "did the model commit to a contact at (i, j)?" — sign is such
    that low loss ⇔ high contact confidence, so ∂loss/∂input flowing positive
    means "increasing this input decreases contact confidence."
    """

    pair_i: int
    pair_j: int
    contact_bins: tuple[int, ...] = field(
        default_factory=lambda: tuple(range(DEFAULT_CONTACT_BIN_HI))
    )

    def __call__(
        self,
        logits: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_mask is not None:
            valid = bool(token_mask[..., self.pair_i].all() and token_mask[..., self.pair_j].all())
            if not valid:
                raise ValueError(
                    f"Target pair ({self.pair_i},{self.pair_j}) hits a padded position."
                )
        log_probs = F.log_softmax(logits[..., self.pair_i, self.pair_j, :], dim=-1)
        bin_idx = torch.tensor(self.contact_bins, device=logits.device, dtype=torch.long)
        log_p_contact = torch.logsumexp(log_probs.index_select(-1, bin_idx), dim=-1)
        return -log_p_contact.mean()

    def spec(self) -> dict:
        return {
            "kind": "ContactBinNLL",
            "pair_i": self.pair_i,
            "pair_j": self.pair_j,
            "contact_bins": list(self.contact_bins),
        }


@dataclass
class PairLogProb:
    """Negative log-prob of a specific bin for pair (i, j).

    Loss = -log_softmax(logits[..., i, j, :])[bin]

    Sharper than ContactBinNLL — useful when you have a reference distance
    (e.g. WT bin assignment) and want gradient flow specifically toward
    matching it.
    """

    pair_i: int
    pair_j: int
    bin: int

    def __call__(
        self,
        logits: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_mask is not None:
            valid = bool(token_mask[..., self.pair_i].all() and token_mask[..., self.pair_j].all())
            if not valid:
                raise ValueError(
                    f"Target pair ({self.pair_i},{self.pair_j}) hits a padded position."
                )
        log_probs = F.log_softmax(logits[..., self.pair_i, self.pair_j, :], dim=-1)
        return -log_probs[..., self.bin].mean()

    def spec(self) -> dict:
        return {
            "kind": "PairLogProb",
            "pair_i": self.pair_i,
            "pair_j": self.pair_j,
            "bin": self.bin,
        }


@dataclass
class DistogramKL:
    """KL(p_pred || p_ref) summed across all valid (i, j) pairs.

    ``ref_logits`` is treated as a frozen reference (e.g. WT prediction).
    Diagonal (i == j) is excluded; padded positions are masked out.
    """

    ref_logits: torch.Tensor

    def __call__(
        self,
        logits: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.shape != self.ref_logits.shape:
            raise ValueError(
                f"shape mismatch: logits {tuple(logits.shape)} vs "
                f"ref_logits {tuple(self.ref_logits.shape)}"
            )
        log_p = F.log_softmax(logits, dim=-1)
        log_q = F.log_softmax(self.ref_logits.to(logits.device), dim=-1)
        p = log_p.exp()
        kl = (p * (log_p - log_q)).sum(dim=-1)  # (..., N, N)

        n = kl.shape[-1]
        eye = torch.eye(n, dtype=torch.bool, device=kl.device)
        pair_mask = ~eye
        if token_mask is not None:
            valid = token_mask.bool()
            pair_mask = pair_mask & valid[..., None] & valid[..., None, :]
        kl_masked = kl * pair_mask
        denom = pair_mask.sum().clamp(min=1).to(kl.dtype)
        return kl_masked.sum() / denom

    def spec(self) -> dict:
        return {
            "kind": "DistogramKL",
            "ref_logits_shape": list(self.ref_logits.shape),
            "ref_logits_hash": _tensor_hash(self.ref_logits),
        }


def _tensor_hash(t: torch.Tensor) -> str:
    import hashlib

    arr = t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
    return hashlib.sha1(arr).hexdigest()[:12]
