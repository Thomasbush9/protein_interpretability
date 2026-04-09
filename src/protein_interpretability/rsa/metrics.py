"""Pure-tensor similarity and dissimilarity metrics.

Every function operates on :class:`torch.Tensor` inputs and returns
:class:`torch.Tensor` outputs so that all computation can stay on GPU.
No model- or domain-specific knowledge lives here — these are generic
building blocks.

Naming conventions
------------------
* **RDM** – Representational Dissimilarity Matrix, shape ``(N, N)``.
* **RSA** – Representational Similarity Analysis: Spearman correlation
  between the upper triangles of two RDMs.
* **CKA** – Centered Kernel Alignment (Kornblith et al. 2019): a
  similarity index between two representation matrices that is invariant
  to orthogonal transforms and isotropic scaling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# RDM construction
# ======================================================================

def cosine_rdm(X: Tensor) -> Tensor:
    """Cosine-distance RDM.

    Parameters
    ----------
    X : Tensor, shape ``(N, D)``
        Row-per-token representation matrix.

    Returns
    -------
    Tensor, shape ``(N, N)``
        Pairwise ``1 - cosine_similarity``.  Values in ``[0, 2]``.
    """
    X_norm = F.normalize(X, dim=-1)  # (N, D)
    return 1.0 - X_norm @ X_norm.T


def euclidean_rdm(X: Tensor) -> Tensor:
    """Euclidean-distance RDM.

    Parameters
    ----------
    X : Tensor, shape ``(N, D)``

    Returns
    -------
    Tensor, shape ``(N, N)``
    """
    return torch.cdist(X.unsqueeze(0), X.unsqueeze(0)).squeeze(0)


def correlation_rdm(X: Tensor) -> Tensor:
    """Correlation-distance RDM (1 - Pearson r).

    Each row of *X* is mean-centred before computing cosine distance,
    making this equivalent to ``1 - Pearson_correlation``.

    Parameters
    ----------
    X : Tensor, shape ``(N, D)``

    Returns
    -------
    Tensor, shape ``(N, N)``
    """
    return cosine_rdm(X - X.mean(dim=-1, keepdim=True))


# ======================================================================
# Upper-triangle extraction
# ======================================================================

def upper_tri(M: Tensor, offset: int = 1) -> Tensor:
    """Flatten the strict upper triangle of a square matrix.

    Parameters
    ----------
    M : Tensor, shape ``(N, N)``
    offset : int
        Diagonal offset (1 = exclude main diagonal).

    Returns
    -------
    Tensor, shape ``(K,)`` where ``K = N*(N-1)/2`` for offset=1.
    """
    idx = torch.triu_indices(M.shape[0], M.shape[1], offset=offset, device=M.device)
    return M[idx[0], idx[1]]


def upper_tri_masked(M: Tensor, mask: Tensor) -> Tensor:
    """Extract upper-triangle elements where *mask* is True.

    Parameters
    ----------
    M : Tensor, shape ``(N, N)``
    mask : Tensor, shape ``(N, N)``, dtype bool
        E.g. a sequence-separation mask from :func:`seq_sep_mask`.

    Returns
    -------
    Tensor, shape ``(K,)``
    """
    triu = torch.triu(torch.ones_like(M, dtype=torch.bool), diagonal=1)
    combined = triu & mask
    return M[combined]


# ======================================================================
# Sequence-separation masks
# ======================================================================

def seq_sep_mask(N: int, min_sep: int = 12, device: torch.device | None = None) -> Tensor:
    """Boolean mask selecting pairs with ``|i - j| >= min_sep``.

    Useful for isolating long-range contacts and removing the trivial
    chain-proximity signal.

    Parameters
    ----------
    N : int
        Number of tokens.
    min_sep : int
        Minimum sequence separation.
    device : torch.device, optional

    Returns
    -------
    Tensor, shape ``(N, N)``, dtype bool
    """
    idx = torch.arange(N, device=device)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= min_sep


# ======================================================================
# Correlation measures
# ======================================================================

def _rank(x: Tensor) -> Tensor:
    """Fractional ranks (GPU-friendly)."""
    return torch.argsort(torch.argsort(x)).float()


def pearson(x: Tensor, y: Tensor) -> Tensor:
    """Pearson correlation between two 1-D tensors.

    Returns
    -------
    Tensor, scalar
    """
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = x.norm() * y.norm() + 1e-12
    return num / den


def spearman(x: Tensor, y: Tensor) -> Tensor:
    """Spearman rank correlation between two 1-D tensors.

    Implemented as Pearson on ranks so that everything stays on GPU.

    Returns
    -------
    Tensor, scalar
    """
    return pearson(_rank(x), _rank(y))


def partial_spearman(x: Tensor, y: Tensor, confound: Tensor) -> Tensor:
    """Spearman partial correlation of *x* and *y* controlling for *confound*.

    Uses the standard first-order partial-correlation formula::

        r_xy|z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz²)(1 - r_yz²))

    All correlations are Spearman (rank-based).

    Returns
    -------
    Tensor, scalar
    """
    rx, ry, rz = _rank(x), _rank(y), _rank(confound)
    r_xy = pearson(rx, ry)
    r_xz = pearson(rx, rz)
    r_yz = pearson(ry, rz)
    num = r_xy - r_xz * r_yz
    den = ((1 - r_xz**2) * (1 - r_yz**2)).clamp(min=1e-12).sqrt()
    return num / den


# ======================================================================
# RSA
# ======================================================================

def rsa(
    rdm1: Tensor,
    rdm2: Tensor,
    mask: Tensor | None = None,
    method: str = "spearman",
) -> Tensor:
    """Representational Similarity Analysis between two RDMs.

    Correlates the upper triangles of two ``(N, N)`` dissimilarity
    matrices.  Optionally restricts to elements where *mask* is True
    (e.g. long-range contacts only).

    Parameters
    ----------
    rdm1, rdm2 : Tensor, shape ``(N, N)``
    mask : Tensor, shape ``(N, N)``, dtype bool, optional
    method : ``"spearman"`` | ``"pearson"``

    Returns
    -------
    Tensor, scalar — correlation coefficient.
    """
    if mask is not None:
        v1 = upper_tri_masked(rdm1, mask)
        v2 = upper_tri_masked(rdm2, mask)
    else:
        v1 = upper_tri(rdm1)
        v2 = upper_tri(rdm2)

    corr_fn = spearman if method == "spearman" else pearson
    return corr_fn(v1, v2)


def partial_rsa(
    rdm1: Tensor,
    rdm2: Tensor,
    confound_rdm: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """RSA controlling for a confound RDM (e.g. sequence separation).

    Parameters
    ----------
    rdm1, rdm2 : Tensor, shape ``(N, N)``
    confound_rdm : Tensor, shape ``(N, N)``
        Typically the sequence-separation matrix
        ``|i - j|`` cast to float.
    mask : Tensor, shape ``(N, N)``, dtype bool, optional

    Returns
    -------
    Tensor, scalar — partial Spearman correlation.
    """
    if mask is not None:
        v1 = upper_tri_masked(rdm1, mask)
        v2 = upper_tri_masked(rdm2, mask)
        vc = upper_tri_masked(confound_rdm, mask)
    else:
        v1 = upper_tri(rdm1)
        v2 = upper_tri(rdm2)
        vc = upper_tri(confound_rdm)

    return partial_spearman(v1, v2, vc)


# ======================================================================
# CKA
# ======================================================================

def linear_cka(X: Tensor, Y: Tensor) -> Tensor:
    """Linear Centered Kernel Alignment (Kornblith et al. 2019).

    Measures similarity between two representation matrices that is
    invariant to orthogonal transformation and isotropic scaling.

    Parameters
    ----------
    X : Tensor, shape ``(N, D1)``
    Y : Tensor, shape ``(N, D2)``

    Returns
    -------
    Tensor, scalar in ``[0, 1]``.
    """
    # Centre columns
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    # Efficient HSIC with linear kernel:
    #   HSIC(X, Y) ∝ ‖Yᵀ X‖²_F
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    hsic_xy = (YtX * YtX).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    return hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt() + 1e-12)
