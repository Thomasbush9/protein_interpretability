"""Structure comparison. Pure numpy + tmtools, no jax, so it runs on a login node.

HISTORY -- READ BEFORE CHANGING. This module used to contain a hand-rolled
TM-score with an iterative superposition. It was badly wrong and it cost two
false conclusions:

  * it scored two Boltz-2 predictions of the *same* sequence at TM 0.70
    (tmtools: 0.978), which I misread as "the diffusion sampler is
    irreproducible";
  * it scored mutant-vs-wild-type at TM 0.11-0.53 (tmtools: 0.93-0.98), which
    I misread as "single-sample structure comparison is noise-limited".

Neither was true. Both were this function. Do not reintroduce a hand-rolled
TM-score: tmtools wraps the reference TM-align implementation and is available
in the analysis venv.
"""

from __future__ import annotations

import numpy as np
from tmtools import tm_align


def kabsch(P, Q):
    """Rotation aligning centred P onto centred Q."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def tm_score(a, b, seq_a=None, seq_b=None):
    """TM-score of `a` onto `b`, normalised by the length of `b` (the reference).

    Sequences are only used by TM-align's alignment step; for equal-length,
    residue-aligned coordinate sets any consistent placeholder gives the same
    answer, but pass the real sequences when you have them.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = seq_a if seq_a is not None else "A" * len(a)
    sb = seq_b if seq_b is not None else "A" * len(b)
    return float(tm_align(a, b, sa, sb).tm_norm_chain2)


def tm_and_rmsd(a, b, seq_a=None, seq_b=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = seq_a if seq_a is not None else "A" * len(a)
    sb = seq_b if seq_b is not None else "A" * len(b)
    r = tm_align(a, b, sa, sb)
    return float(r.tm_norm_chain2), float(r.rmsd)
