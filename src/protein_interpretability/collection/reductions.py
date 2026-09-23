"""How a per-pair divergence becomes the two numbers a capture stores.

`kl_glob` and `kl_site` are the same divergence read over two different sets of
sampled pairs: every pair, and only the pairs that touch the mutated token. They
are not interchangeable -- the first says the whole map moved, the second says
the mutation's own neighbourhood moved -- and a probe cannot tell them apart
after the fact, because both arrive as one float per layer.

WHY THIS MODULE EXISTS. `collect_pairformer_layers.py` computed the global
reduction twice and stored it under both names:

    kl_site.append(_kl(lw, lm))
    kl_glob.append(_kl(lw, lm))

Nothing failed. The archive had both fields, both with the promised shape, and
`kl_site` was a plausible number -- it was simply the other measurement wearing
its name. That is the `deep2_*` failure again in a different field, so the fix
is the same one: compute both reductions in one place, from one tensor, and
return them together so wiring them to the same thing takes a deliberate edit
to a function this file's tests cover.

The divergence is the SYMMETRIC KL (Jeffreys) of `exp_gym.skl`, expression for
expression, because that is what produced every archived `kl_glob` and
`kl_site`. It is not cast to float64 on the way in for the same reason: the
archives were made at the dtype the caller passed, and changing precision here
would move the numbers a regression check is supposed to hold fixed.

Pure numpy, no backend: the reduction is checkable on a login node with
synthetic logits, and that is where its tests run.
"""

from __future__ import annotations

import numpy as np


class ReductionError(ValueError):
    """A reduction cannot be computed honestly from what was sampled."""


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def symmetric_kl(logits_a, logits_b):
    """Jeffreys divergence over the last axis. `exp_gym.skl`, unchanged.

    Inputs are distogram logits `[..., n_bins]`; the result drops the bin axis.
    Symmetric by construction, so mutant-versus-wild-type is the same number
    either way round -- which is why the archived field has no direction in its
    name.
    """
    pa, pb = _softmax(logits_a), _softmax(logits_b)
    return ((pa - pb) * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum(-1)


def site_mask(ii, jj, token) -> np.ndarray:
    """Which sampled pairs touch `token`. Boolean, one entry per pair.

    `token` must be an index into the SAME space `ii` and `jj` were drawn from.
    The residue number parsed out of a mutant name is not that space unless the
    token grid happens to be one-token-per-residue with no padding ahead of it,
    so the caller resolves it and passes the resolved value.
    """
    ii = np.asarray(ii)
    jj = np.asarray(jj)
    if ii.shape != jj.shape:
        raise ReductionError(
            f"the pair sample is not paired: ii is {ii.shape}, jj is {jj.shape}")
    return (ii == token) | (jj == token)


def kl_reductions(logits_mut, logits_wt, ii, jj, token) -> dict:
    """Both stored KL fields, from one divergence tensor.

        arrays = kl_reductions(lm, lw, ii, jj, tok)
        arrays["kl_glob"]   # [n_layers] over every sampled pair
        arrays["kl_site"]   # [n_layers] over the pairs touching `token`

    Logits are `[n_layers, n_pairs, n_bins]`. The divergence is computed once;
    the two fields differ only in which pairs they average over, which is the
    whole content of the distinction and the reason returning them separately
    invites them to drift apart.

    Raises rather than returning zeros when no sampled pair touches the site.
    The archived producer wrote `np.zeros(L)` there, and a zero in this field is
    indistinguishable from a mutation that changed nothing -- the strongest
    possible claim, written by accident, for a variant that was never measured.
    """
    at = site_mask(ii, jj, token)
    if not at.any():
        raise ReductionError(
            f"no sampled pair touches token {token}, so `kl_site` would be an "
            f"average over nothing. Writing 0.0 there says the mutation moved "
            f"the distogram not at all, which is a result rather than a gap. "
            f"Draw more pairs, or a sample that covers the mutated sites.")
    lm, lw = np.asarray(logits_mut), np.asarray(logits_wt)
    if lm.shape[-2] != at.shape[0] or lw.shape[-2] != at.shape[0]:
        raise ReductionError(
            f"the logits carry {lm.shape[-2]} (mutant) and {lw.shape[-2]} "
            f"(wild type) pairs but the sample has {at.shape[0]}; the logits "
            f"and the pair indices are not from the same capture")
    kl = symmetric_kl(lm, lw)
    return {"kl_glob": kl.mean(-1), "kl_site": kl[..., at].mean(-1)}


def uncovered_sites(ii, jj, tokens) -> list:
    """Which of `tokens` no sampled pair touches. Empty means all are covered.

    The pair sample is drawn once and every variant is measured against it, so
    this is answerable before the first trunk pass -- and a site with no pairs
    is a run that cannot produce `kl_site` for that variant, which is worth
    knowing at minute zero of a GPU job rather than at its end.
    """
    return [t for t in tokens if not site_mask(ii, jj, t).any()]
