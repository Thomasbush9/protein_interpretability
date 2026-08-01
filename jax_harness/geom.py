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


def _tm_align(*a, **k):
    """tmtools is present in the analysis venv but NOT in the mosaic container.

    Imported lazily so that `kabsch_rmsd` -- which the in-container experiments
    need -- does not drag in a dependency that only the login-node analysis has.
    """
    from tmtools import tm_align
    return tm_align(*a, **k)


def kabsch(P, Q):
    """Rotation R aligning centred P onto centred Q, i.e. `P @ R.T` approximates Q.

    Both inputs must already be centred. The `d` term forces a proper rotation:
    without it the SVD is free to return a reflection, which would happily
    superimpose a structure onto its own mirror image.
    """
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def kabsch_rmsd(P, Q):
    """RMSD between residue-corresponded coordinate sets after optimal superposition.

    Unlike `tm_score`, this is over ALL rows -- there is no alignment step and no
    partial credit -- so P and Q must be the same length and already in
    correspondence (row i of each is the same residue). For CA traces of a
    wild type and a point mutant that holds by construction.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    R = kabsch(Pc, Qc)
    return float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1).mean()))


def self_test():
    """Cases with known answers. Called at import of anything that superimposes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3)) * 10.0
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    # a rigid motion must be undone exactly
    assert kabsch_rmsd(X, X @ q.T + np.array([5.0, -3.0, 2.0])) < 1e-8
    # a reflection must NOT be undone
    assert kabsch_rmsd(X, X * np.array([1.0, 1.0, -1.0])) > 1.0
    # translation alone is free
    assert kabsch_rmsd(X, X + 100.0) < 1e-8
    return True


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
    return float(_tm_align(a, b, sa, sb).tm_norm_chain2)


def tm_and_rmsd(a, b, seq_a=None, seq_b=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = seq_a if seq_a is not None else "A" * len(a)
    sb = seq_b if seq_b is not None else "A" * len(b)
    r = _tm_align(a, b, sa, sb)
    return float(r.tm_norm_chain2), float(r.rmsd)


def tm_align_result(a, b, seq_a=None, seq_b=None):
    """Raw TM-align result, for when the superposition itself is needed.

    `.u` and `.t` place chain 1 onto chain 2: `a @ u.T + t` is `a` in `b`'s frame.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = seq_a if seq_a is not None else "A" * len(a)
    sb = seq_b if seq_b is not None else "A" * len(b)
    return _tm_align(a, b, sa, sb)


self_test()
