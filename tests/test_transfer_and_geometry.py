"""The two new probe protocols, and the richer emitted-geometry block.

Both exist to close a specific objection to the cross-model result, so what is
tested is exactly the property the objection turns on:

    the reduced probe must not be a different probe. At full rank it has to
    reproduce the frozen `leave_one_group_out` number, or a rho-versus-d curve
    cannot be read against the archived point at all.

    the cross-cohort probe must never see its test assays, including through
    the PCA basis, which is a fitted object and the easy place for a leak.

    the geometry block must agree with the harness `kabsch` it reimplements,
    and must report no deformation when there is none.

    uv run pytest tests/test_transfer_and_geometry.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from protein_interpretability.analysis.emitted_geometry import (
    GEOMETRY_FEATURES,
    _kabsch,
    geometry_matrix,
)
from protein_interpretability.analysis.probes import leave_one_group_out
from protein_interpretability.analysis.transfer import (
    fit_groups_predict_groups,
    leave_one_group_out_reduced,
    pca_apply,
    pca_basis,
)


def _blocks(n_groups=4, n=40, p=6, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for g in range(n_groups):
        X = rng.normal(size=(n, p))
        y = X @ rng.normal(size=p) + 0.5 * rng.normal(size=n)
        out[f"A{g}"] = {"X": X, "y": y}
    return out


def test_full_rank_reduction_reproduces_the_frozen_probe():
    """PCA at full rank is an orthogonal change of basis, and ridge with an
    unpenalised intercept is invariant under one. If this drifts, every point
    on a rho-versus-d curve is on a different scale from the archived number."""
    b = _blocks()
    plain = leave_one_group_out(b, lam=10.0)
    reduced = leave_one_group_out_reduced(b, d=6, lam=10.0)
    for k in plain:
        assert reduced[k] == pytest.approx(plain[k], abs=1e-9)


def test_reduction_below_full_rank_actually_reduces():
    b = _blocks()
    assert leave_one_group_out_reduced(b, d=2) != leave_one_group_out(b, lam=10.0)


def test_reduction_refuses_more_components_than_columns():
    with pytest.raises(ValueError, match="exceeds the block width"):
        leave_one_group_out_reduced(_blocks(p=4), d=5)


def test_pca_basis_is_orthonormal_and_reconstructs_at_full_rank():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 5))
    mu, comp = pca_basis(X, 5)
    assert comp @ comp.T == pytest.approx(np.eye(5), abs=1e-10)
    assert pca_apply(X, mu, comp) @ comp + mu == pytest.approx(X, abs=1e-10)


def test_cross_cohort_refuses_overlapping_assays():
    """The failure this guards is silent: an overlap turns a transfer number
    into a within-cohort one that still prints."""
    b = _blocks()
    with pytest.raises(ValueError, match="share 1 assay"):
        fit_groups_predict_groups({k: b[k] for k in ["A0", "A1"]},
                                  {k: b[k] for k in ["A1", "A2"]})


def test_cross_cohort_fit_ignores_the_test_assays_entirely():
    """Changing a test assay's FEATURES must not change any other test assay's
    prediction -- the fit is on the training set alone. Under a within-cohort
    protocol it would, which is exactly the confusion being ruled out."""
    b = _blocks(n_groups=6)
    train = {k: b[k] for k in ["A0", "A1", "A2"]}
    test = {k: b[k] for k in ["A3", "A4", "A5"]}
    before = fit_groups_predict_groups(train, test, d=3)

    rng = np.random.default_rng(99)
    test["A5"] = {"X": rng.normal(size=b["A5"]["X"].shape), "y": b["A5"]["y"]}
    after = fit_groups_predict_groups(train, test, d=3)
    assert after["A3"] == pytest.approx(before["A3"], abs=1e-12)
    assert after["A4"] == pytest.approx(before["A4"], abs=1e-12)


def test_cross_cohort_refuses_mismatched_widths():
    b, c = _blocks(p=6), _blocks(n_groups=2, p=4, seed=7)
    with pytest.raises(ValueError, match="must mean the same thing"):
        fit_groups_predict_groups({k: b[k] for k in ["A0", "A1"]},
                                  {f"B{k}": v for k, v in c.items()})


def _fake_structure(n_res=30, n_var=4, seed=0):
    rng = np.random.default_rng(seed)
    ca_wt = np.cumsum(rng.normal(scale=3.0, size=(n_res, 3)), axis=0)
    ca = np.stack([ca_wt + rng.normal(scale=0.2, size=ca_wt.shape)
                   for _ in range(n_var)])
    return ca, ca_wt


def test_geometry_matrix_has_one_column_per_named_feature():
    ca, ca_wt = _fake_structure()
    G = geometry_matrix(ca, ca_wt, np.full(len(ca), 0.9), np.full(len(ca), 0.8),
                        np.full(len(ca), 0.7), np.arange(len(ca)))
    assert G.shape == (len(ca), len(GEOMETRY_FEATURES))
    assert np.isfinite(G).all()


def test_geometry_reports_no_deformation_for_an_unchanged_structure():
    """Every deformation feature must be zero when the variant IS the wild type,
    including under an arbitrary rigid rotation and translation -- otherwise the
    block is reading the reference frame rather than the structure."""
    _, ca_wt = _fake_structure()
    R = _kabsch(np.random.default_rng(3).normal(size=(8, 3)),
                np.random.default_rng(4).normal(size=(8, 3)))
    ca = np.stack([ca_wt, ca_wt @ R.T + np.array([10.0, -4.0, 7.0])])

    G = geometry_matrix(ca, ca_wt, np.ones(2), np.full(2, 0.8), np.full(2, 0.6),
                        np.array([5, 5]))
    named = dict(zip(GEOMETRY_FEATURES, G[1]))
    for k in [n for n in GEOMETRY_FEATURES
              if n.startswith(("rms_disp", "abs_dD_site", "d_gyr", "contacts_g",
                               "contacts_l", "contacts_n", "site_contacts", "dD_"))
              or n in ("d_radius_gyration", "rmsd_global", "disp_at_site",
                       "disp_max")]:
        if k == "dD_frac_within_12A":
            continue          # 0/0 by construction; defined as 0.0, checked below
        assert named[k] == pytest.approx(0.0, abs=1e-8), k
    assert named["dD_frac_within_12A"] == 0.0
    assert named["contacts_jaccard"] == pytest.approx(1.0)


def test_kabsch_agrees_with_the_harness_implementation():
    """This module reimplements `geom.kabsch` to stay importable without
    tmtools. The two must not drift apart."""
    geom = pytest.importorskip("geom", reason="harness geom not on the path")
    rng = np.random.default_rng(11)
    P, Q = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
    P, Q = P - P.mean(0), Q - Q.mean(0)
    assert _kabsch(P, Q) == pytest.approx(geom.kabsch(P, Q), abs=1e-10)


def test_principal_angles_of_a_subspace_with_itself_are_all_zero():
    rng = np.random.default_rng(5)
    from protein_interpretability.analysis.transfer import (
        orthonormal, principal_angles, random_subspace, subspace_overlap,
    )
    Q = random_subspace(20, 4, rng)
    assert principal_angles(Q, Q) == pytest.approx(np.ones(4), abs=1e-10)
    assert subspace_overlap(Q, Q) == pytest.approx(1.0, abs=1e-10)


def test_principal_angles_are_invariant_to_how_a_subspace_is_written():
    """The property the raw weight-vector cosine lacks: rotating the basis
    within a subspace must not change its relationship to another subspace."""
    from protein_interpretability.analysis.transfer import (
        principal_angles, random_subspace,
    )
    rng = np.random.default_rng(6)
    Q1, Q2 = random_subspace(20, 3, rng), random_subspace(20, 3, rng)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    assert principal_angles(Q1, Q2) == pytest.approx(
        principal_angles(Q1 @ R, Q2), abs=1e-10)


def test_orthogonal_subspaces_have_zero_overlap():
    from protein_interpretability.analysis.transfer import (
        principal_angles, subspace_overlap,
    )
    I = np.eye(10)
    assert subspace_overlap(I[:, :3], I[:, 3:6]) == pytest.approx(0.0, abs=1e-12)
    assert principal_angles(I[:, :3], I[:, 3:6]) == pytest.approx(np.zeros(3),
                                                                 abs=1e-12)


def test_random_subspace_overlap_sits_near_d_over_p():
    """The floor a real overlap has to be read against. d/p = 4/128 = 0.031."""
    from protein_interpretability.analysis.transfer import (
        random_subspace, subspace_overlap,
    )
    rng = np.random.default_rng(7)
    got = np.mean([subspace_overlap(random_subspace(128, 4, rng),
                                    random_subspace(128, 4, rng))
                   for _ in range(200)])
    assert got == pytest.approx(4 / 128, abs=0.01)


def test_orthonormal_spans_the_same_directions_as_pca_basis():
    from protein_interpretability.analysis.transfer import orthonormal, pca_basis
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 12))
    _, comp = pca_basis(X, 4)
    assert orthonormal(X, 4) == pytest.approx(comp.T, abs=1e-12)
