"""The injection algebra, pinned where it can be checked without a GPU.

These are the decisions buried in three lines of `exp_steer` that produced
`steer_pooled`. Each one changes the result, and each is the kind of thing a
reimplementation would quietly "improve".

    uv run pytest tests/test_intervention.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from protein_interpretability.intervention import (
    InterventionError,
    PairDirectionIntervention,
    random_directions,
    unit,
)

C = 8


def iv(**kw) -> PairDirectionIntervention:
    base = dict(direction=unit(np.arange(1.0, C + 1)), scale=2.0, mode="sym")
    base.update(kw)
    return PairDirectionIntervention(**base)


def z_zeros(n=5):
    return np.zeros((n, n, C))


# ---- what it refuses -------------------------------------------------------

def test_a_non_unit_direction_is_refused():
    """Doses are multiples of `scale`, so any other length silently rescales
    every one of them."""
    with pytest.raises(InterventionError, match="unit L2"):
        PairDirectionIntervention(direction=np.ones(C) * 3, scale=1.0)


def test_a_zero_direction_is_refused():
    with pytest.raises(InterventionError, match="non-zero finite"):
        unit(np.zeros(C))


def test_a_non_positive_scale_is_refused():
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(InterventionError, match="scale must be positive"):
            iv(scale=bad)


def test_alphas_must_include_zero():
    """alpha=0 is the determinism check that separates a real effect from
    sampler drift; a sweep without it cannot make that distinction."""
    with pytest.raises(InterventionError, match="determinism check"):
        iv(alphas=(-10.0, 10.0))


def test_an_unknown_mode_is_refused():
    with pytest.raises(InterventionError, match="mode must be"):
        iv(mode="column")


def test_a_channel_width_mismatch_is_refused():
    with pytest.raises(InterventionError, match="channels but z has"):
        iv().apply(np.zeros((4, 4, C + 1)), token=0, alpha=1.0)


def test_a_token_outside_the_sequence_is_refused():
    with pytest.raises(InterventionError, match="outside the 5 tokens"):
        iv().apply(z_zeros(), token=99, alpha=1.0)


def test_row_and_sym_need_a_token():
    with pytest.raises(InterventionError, match="needs a token"):
        iv(mode="row").apply(z_zeros(), alpha=1.0)


# ---- the algebra -----------------------------------------------------------

def test_alpha_zero_is_exactly_the_identity():
    """Not approximately. The archived sweep uses it as a determinism check, so
    any drift here would be read as an effect of the intervention."""
    z = np.random.default_rng(0).normal(size=(5, 5, C))
    assert np.array_equal(iv().apply(z, token=2, alpha=0.0), z)


def test_row_touches_one_row_only():
    out = iv(mode="row").apply(z_zeros(), token=2, alpha=1.0)
    touched = np.abs(out).sum(-1) > 0
    assert touched[2, :].all()
    assert not touched[np.arange(5) != 2][:, np.arange(5) != 2].any()
    assert not touched[:, 2][np.arange(5) != 2].any(), "row must not touch the column"


def test_sym_touches_the_row_and_the_column():
    out = iv(mode="sym").apply(z_zeros(), token=2, alpha=1.0)
    touched = np.abs(out).sum(-1) > 0
    assert touched[2, :].all() and touched[:, 2].all()
    assert not touched[np.arange(5) != 2][:, np.arange(5) != 2].any()


def test_sym_adds_the_diagonal_twice():
    """The detail that looks like a bug and is not: the (t, t) entry is hit by
    the row pass and again by the column pass. This is what produced
    steer_pooled, so a reimplementation that "fixes" it changes the archive."""
    d = iv().delta(1.0)
    out = iv(mode="sym").apply(z_zeros(), token=2, alpha=1.0)
    assert np.allclose(out[2, 2], 2 * d)
    assert np.allclose(out[2, 0], d)
    assert np.allclose(out[0, 2], d)


def test_glob_touches_every_pair():
    out = iv(mode="glob").apply(z_zeros(), alpha=1.0)
    assert (np.abs(out).sum(-1) > 0).all()
    assert np.allclose(out[0, 0], out[4, 4])


def test_glob_ignores_the_token_because_extent_is_the_point():
    """`glob` matches a real substitution in EXTENT, so it is site-independent;
    the archived sweep runs it once rather than once per site."""
    a = iv(mode="glob").apply(z_zeros(), token=0, alpha=3.0)
    b = iv(mode="glob").apply(z_zeros(), token=4, alpha=3.0)
    assert np.array_equal(a, b)


def test_the_dose_scales_linearly_in_alpha_and_scale():
    one = iv(scale=1.0).apply(z_zeros(), token=1, alpha=10.0)
    ten = iv(scale=10.0).apply(z_zeros(), token=1, alpha=1.0)
    assert np.allclose(one, ten), "alpha and scale enter only as their product"


def test_opposite_doses_are_antisymmetric():
    """Positive and negative alphas are not redundant: +alpha should broaden and
    -alpha sharpen. Only signed doses can show that."""
    plus = iv().apply(z_zeros(), token=1, alpha=7.0)
    minus = iv().apply(z_zeros(), token=1, alpha=-7.0)
    assert np.allclose(plus, -minus)


def test_the_input_is_not_mutated():
    z = np.ones((4, 4, C))
    iv().apply(z, token=1, alpha=5.0)
    assert np.array_equal(z, np.ones((4, 4, C))), (
        "a sweep reuses the same baseline z across every dose")


# ---- controls --------------------------------------------------------------

def test_random_controls_share_the_norm_and_differ_only_in_orientation():
    dirs = random_directions(8, C, seed=0)
    assert len(dirs) == 8
    for d in dirs:
        assert abs(np.linalg.norm(d) - 1.0) < 1e-12
    assert not np.allclose(dirs[0], dirs[1])


def test_random_controls_are_reproducible_from_the_seed():
    assert np.allclose(random_directions(3, C, seed=4)[0],
                       random_directions(3, C, seed=4)[0])


# ---- equivalence with the code that produced steer_pooled ------------------

def steer_reference(z4, delta, mode, tok):
    """`exp_steer`'s injection, transcribed index for index.

        zc = trunk.z                                  # [1, N, N, C]
        if mode == "glob": zc = zc.at[0].add(delta)
        else:
            zc = zc.at[0, tok, :, :].add(delta)
            if mode == "sym": zc = zc.at[0, :, tok, :].add(delta)

    Kept as a literal transcription rather than a tidied one: its value is in
    being the same expressions, so a difference here is a real difference.
    """
    zc = np.array(z4, copy=True)
    if mode == "glob":
        zc[0] += delta
    else:
        zc[0, tok, :, :] += delta
        if mode == "sym":
            zc[0, :, tok, :] += delta
    return zc


@pytest.mark.parametrize("mode", ["row", "sym", "glob"])
@pytest.mark.parametrize("alpha", [-30.0, -3.0, 0.0, 3.0, 30.0])
def test_matches_exp_steer_index_for_index(mode, alpha):
    """The module reproduces the archived mechanics on every mode and dose."""
    rng = np.random.default_rng(7)
    z = rng.normal(size=(6, 6, C))
    tok, scale = 3, 2.5
    d = unit(rng.normal(size=C))
    mine = PairDirectionIntervention(direction=d, scale=scale, mode=mode
                                     ).apply(z, token=tok, alpha=alpha)
    theirs = steer_reference(z[None], alpha * scale * d, mode, tok)[0]
    assert np.allclose(mine, theirs, rtol=0, atol=0), (
        f"{mode} at alpha={alpha} differs from exp_steer's injection")


def test_the_reference_transcription_can_disagree():
    """Guard the guard: if the two were equal for trivial reasons the test
    above would prove nothing."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(6, 6, C))
    d = unit(rng.normal(size=C))
    row = steer_reference(z[None], d, "row", 3)[0]
    sym = steer_reference(z[None], d, "sym", 3)[0]
    assert not np.allclose(row, sym)


# ---- provenance ------------------------------------------------------------

def test_the_protocol_records_the_diagonal_and_the_injection_point():
    p = iv().protocol()
    assert "final pair representation" in p["injection_point"]
    assert "twice" in p["diagonal_note"]
    assert p["mode"] == "sym" and 0.0 in p["alphas"]
