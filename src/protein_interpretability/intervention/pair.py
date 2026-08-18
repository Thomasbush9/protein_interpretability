"""Adding a direction to the pair representation: the algebra, separated from the model.

`exp_steer` produced `steer_pooled` by running the trunk normally, adding a
vector to the final `z`, and handing the modified state to the structure module.
The injection itself is three lines of tensor arithmetic buried in a quadruple
loop over sites, modes, directions and doses — and those three lines carry
several decisions that change the result and are invisible unless written down.

They are extracted here so they can be tested without a GPU. Everything in this
module is numpy on an array; nothing imports a backend.

WHY THE INJECTION IS AT THE END, NOT MID-STACK. PC2 was derived from `dz_site`,
the pair row after all 64 Pairformer layers, which is exactly the tensor the
structure module is conditioned on. Injecting inside the stack would test a
different question — the perturbation would be reshaped by the remaining layers.
Injecting at the end asks the one the causal claim needs: the trunk says X, does
the structure module do anything about it?

THE THREE MODES ARE NOT VARIATIONS ON A THEME.

    row    one row of the pair tensor. A small lever.
    sym    the row AND the transposed column, because a real substitution is not
           directional and the pair tensor is not symmetric by construction.
    glob   every pair. Without it the null is ambiguous: "the coordinates did
           not move" could mean the structure module ignores this channel, or
           merely that one row is too local to matter. A real substitution
           changes the whole tensor, so this is the dose matched in extent.

AND THE DIAGONAL IS ADDED TWICE UNDER `sym`. Applying the row then the column
hits `z[tok, tok]` in both passes, so it receives 2*delta. That is what produced
the archived numbers. It looks like an off-by-one and a reimplementation would
"fix" it — which is exactly why it is stated here and pinned by a test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("row", "sym", "glob")


class InterventionError(ValueError):
    """An intervention is not well posed."""


def unit(direction) -> np.ndarray:
    """L2-normalise a direction in raw z space."""
    v = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise InterventionError(
            "a direction must have non-zero finite norm; a zero direction makes "
            "every dose the identity and the sweep silently measures nothing")
    return v / n


@dataclass(frozen=True)
class PairDirectionIntervention:
    """Add `alpha * scale * direction` to the final pair representation.

        iv = PairDirectionIntervention(direction=pc2, scale=median_dz_norm,
                                       mode="sym", alphas=(-30,-10,-3,0,3,10,30))
        z_perturbed = iv.apply(z, token=41, alpha=10)

    `scale` is the MEDIAN ||dz_site|| of real mutations in the assay, so
    alpha = 1 moves the mutation-site row by as much as a typical real mutation
    moves it. It is not "as large as a real mutation" overall — a substitution
    perturbs the whole tensor, not one row — which is why the archived sweep
    runs to 30x rather than stopping at 1. A null at alpha = 1 alone cannot
    separate "the model ignores this direction" from "the dose was too small to
    be a fair test".
    """

    direction: np.ndarray
    scale: float
    mode: str = "sym"
    alphas: tuple[float, ...] = (-30.0, -10.0, -3.0, 0.0, 3.0, 10.0, 30.0)
    name: str = "pc2"

    def __post_init__(self):
        if self.mode not in MODES:
            raise InterventionError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise InterventionError(
                f"scale must be positive and finite, got {self.scale!r}; it is "
                "the median ||dz_site|| of real mutations, so a non-positive "
                "value means the doses have no physical meaning")
        v = np.asarray(self.direction, dtype=float)
        if v.ndim != 1:
            raise InterventionError(
                f"direction must be a vector over channels, got shape {v.shape}")
        if abs(float(np.linalg.norm(v)) - 1.0) > 1e-6:
            raise InterventionError(
                "direction must be unit L2 norm -- doses are expressed in "
                "multiples of `scale`, so a direction of any other length "
                "silently rescales every dose. Use unit().")
        if 0.0 not in tuple(float(a) for a in self.alphas):
            raise InterventionError(
                "alphas must include 0. It is the determinism check: with a "
                "fixed key and deterministic sampling it must reproduce the "
                "baseline EXACTLY, which is what distinguishes a real effect "
                "from sampler drift.")

    # ---- the algebra ------------------------------------------------------
    def delta(self, alpha: float) -> np.ndarray:
        return float(alpha) * float(self.scale) * np.asarray(self.direction, float)

    def apply(self, z, *, token: int | None = None, alpha: float) -> np.ndarray:
        """Return a copy of `z` with the direction added. `z` is [N, N, C].

        `token` is required for `row`/`sym` and ignored by `glob`, which is why
        the archived sweep runs `glob` once rather than once per site.
        """
        z = np.array(z, dtype=float, copy=True)
        if z.ndim != 3 or z.shape[0] != z.shape[1]:
            raise InterventionError(
                f"z must be [N, N, C], got {z.shape}")
        if z.shape[-1] != len(self.direction):
            raise InterventionError(
                f"direction has {len(self.direction)} channels but z has "
                f"{z.shape[-1]}")
        if alpha == 0.0:
            return z                       # exactly the baseline, by construction
        d = self.delta(alpha)
        if self.mode == "glob":
            return z + d
        if token is None:
            raise InterventionError(f"mode {self.mode!r} needs a token index")
        if not 0 <= token < z.shape[0]:
            raise InterventionError(
                f"token {token} is outside the {z.shape[0]} tokens")
        z[token, :, :] += d
        if self.mode == "sym":
            # The diagonal receives it twice. That is what produced the archive.
            z[:, token, :] += d
        return z

    def protocol(self) -> dict:
        return {
            "intervention": "pair-direction addition at the final z",
            "direction": self.name,
            "mode": self.mode,
            "alphas": [float(a) for a in self.alphas],
            "scale": float(self.scale),
            "scale_meaning": "median ||dz_site|| of real mutations in the assay",
            "injection_point": "final pair representation, before the structure "
                               "module -- not inside the Pairformer stack",
            "diagonal_note": ("under `sym` the (t, t) entry receives the delta "
                              "twice, once from the row pass and once from the "
                              "column pass; this matches steer_pooled"),
        }


def random_directions(n: int, width: int, *, seed: int = 0) -> list[np.ndarray]:
    """Unit directions in the same space, for the control arm.

    Any perturbation of that size moves the outputs somewhat; the question is
    whether PC2 moves them more, or differently. Same space, same norm, so the
    only difference is orientation.
    """
    rng = np.random.default_rng(seed)
    return [unit(rng.normal(size=width)) for _ in range(n)]
