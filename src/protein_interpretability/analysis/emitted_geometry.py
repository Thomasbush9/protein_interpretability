"""The richest description of the EMITTED structure the saved coordinates allow.

`output_rich` (ten features, in `compare_internal_output.output_matrix`) is the
baseline the cross-model result is reported against. It is a fair baseline in
the sense that it is fitted through the identical ridge protocol, but it is ten
numbers against a 128-channel internal block, and a reader is entitled to ask
whether the gap measures "the trunk knows more than the structure module said"
or merely "128 > 10".

This module answers the first half of that question by making the emitted side
as rich as the coordinates permit. It is NOT a tuned baseline: every shell edge,
contact cutoff and summary below is prespecified here, computed identically for
every assay and every model, and fitted with the same lambda on the same rows.
Nothing in it is selected against the outcome.

    shell deformation  RMS displacement after Kabsch superposition, in eight
                       shells of WILD-TYPE distance from the mutated site --
                       the radial profile of the deformation, where
                       `rmsd_local_8A` / `rmsd_local_12A` are two nested
                       cumulative slices of the same thing.
    shell distortion   mean |change in distance FROM the mutated site| in each
                       of the same shells. Displacement after superposition and
                       change in internal distance are different quantities: a
                       rigid-body hinge moves a domain far without altering any
                       distance within it.
    contact change     contacts gained, lost, net and Jaccard against wild type
                       at an 8 A CA cutoff with |i-j| > 2, globally and at the
                       mutated site. This is what a structural biologist would
                       read off the two structures.
    distance matrix    Frobenius norm of the change globally and in the site's
                       own row, and the fraction of the squared change that
                       falls within 12 A of the site -- local versus distal.
    shape              radius of gyration and the three gyration-tensor
                       eigenvalues, which separate "swollen" from "elongated"
                       in a way a single Rg cannot.
    global + confidence  TM, global RMSD, displacement at the site and its
                       maximum, pLDDT chain mean, pLDDT at the site and their
                       difference -- the seven of the original ten that are not
                       already subsumed above.

DELIBERATELY EXCLUDED, for the same reason `output_matrix` excludes it: the
trunk distogram. It is a head on the Pairformer, not a product of the structure
module. Everything here is read from `ca` and pLDDT, which is exactly what the
model emitted.

Numpy only, like the rest of analysis.
"""

from __future__ import annotations

import numpy as np

# Wild-type distance from the mutated site, in Angstroms. The first edge is the
# CA contact distance and the last is open, so every residue lands in exactly
# one shell for any protein length.
SHELLS = ((0.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0),
          (10.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, np.inf))

CONTACT_CUTOFF = 8.0     # CA-CA, the conventional coarse contact
SEQ_SEPARATION = 2       # |i - j| > 2, so the backbone is not counted
LOCAL_RADIUS = 12.0      # for the local/distal split of the distance change

_SHELL_NAMES = ["0_4", "4_6", "6_8", "8_10", "10_12", "12_16", "16_20", "20_inf"]

GEOMETRY_FEATURES = (
    [f"rms_disp_{s}A" for s in _SHELL_NAMES]
    + [f"abs_dD_site_{s}A" for s in _SHELL_NAMES]
    + ["contacts_gained", "contacts_lost", "contacts_net", "contacts_jaccard"]
    + ["site_contacts_gained", "site_contacts_lost", "site_contacts_net"]
    + ["dD_frobenius", "dD_site_row", "dD_frac_within_12A"]
    + ["d_radius_gyration", "d_gyr_eig1", "d_gyr_eig2", "d_gyr_eig3"]
    + ["tm_to_wt", "rmsd_global", "disp_at_site", "disp_max"]
    + ["plddt_mean", "plddt_site", "plddt_mean_minus_site"]
)


# RMS change in CA-CA distance, in Angstroms, below which a variant is treated
# as not having deformed at all. `dD_frac_within_12A` is a ratio of two sums of
# squared distance changes, so for a variant that barely moves it is a ratio of
# two quantities that are pure floating-point residue -- a `sq > 0` guard passes
# and returns noise with a physical-looking name. An identical structure under a
# rigid rotation reaches ~1e-3 through that guard, which is the whole class of
# bug this project keeps finding: a plausible number with a false label.
RIGID_TOL = 1e-6


def _local_fraction(dD_site, near12, sq, n_res):
    """Share of the squared distance change that falls within 12 A of the site.

    Undefined when nothing moved; reported as 0.0 rather than as noise.
    """
    if np.sqrt(sq) / n_res <= RIGID_TOL:
        return 0.0
    return float((dD_site[near12] ** 2).sum() / sq)


def _kabsch(P, Q):
    """Rotation taking centred P onto centred Q. Same convention as geom.kabsch.

    Reimplemented here rather than imported because `geom` lives in the harness
    directory and pulls in tmtools; this module is meant to be importable from
    the package alone. The two are checked against each other in the tests.
    """
    V, _, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    return (V @ D @ Wt).T


def _gyration_eigenvalues(c):
    """Eigenvalues of the gyration tensor, descending. Rotation invariant."""
    x = c - c.mean(0)
    return np.linalg.eigvalsh(x.T @ x / len(x))[::-1]


def geometry_matrix(ca, ca_wt, tm_wt, plddt, plddt_site, pos):
    """(n_variants, 37) description of what the structure module emitted.

    Arguments are exactly those of `output_matrix`, so the two baselines are
    computed from identical inputs and differ only in what they extract.

    An empty shell contributes 0.0, matching `output_matrix`'s convention for
    an empty 8 A neighbourhood. A constant column is harmless: the probe
    z-scores with `sd + 1e-9`.
    """
    ca_wt = np.asarray(ca_wt, dtype=float)
    n_res = len(ca_wt)
    dwt = np.linalg.norm(ca_wt[:, None, :] - ca_wt[None, :, :], axis=-1)
    rg_wt = float(np.sqrt(((ca_wt - ca_wt.mean(0)) ** 2).sum(1).mean()))
    eig_wt = _gyration_eigenvalues(ca_wt)

    sep = np.abs(np.arange(n_res)[:, None] - np.arange(n_res)[None, :])
    eligible = sep > SEQ_SEPARATION
    c_wt = (dwt < CONTACT_CUTOFF) & eligible

    B = ca_wt - ca_wt.mean(0)
    rows = []
    for i in range(len(ca)):
        c = np.asarray(ca[i], dtype=float)
        # `pos` is a 0-based residue index; the guard mirrors output_matrix,
        # which falls back to 0 rather than raising on an out-of-range site.
        p = int(pos[i]) if int(pos[i]) < n_res else 0

        A = c - c.mean(0)
        disp = np.linalg.norm(A @ _kabsch(A, B).T - B, axis=1)

        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        dD = d - dwt
        near12 = dwt[p] <= LOCAL_RADIUS

        shell_disp, shell_dist = [], []
        for lo, hi in SHELLS:
            m = (dwt[p] >= lo) & (dwt[p] < hi)
            shell_disp.append(float(np.sqrt((disp[m] ** 2).mean())) if m.any() else 0.0)
            shell_dist.append(float(np.abs(dD[p][m]).mean()) if m.any() else 0.0)

        con = (d < CONTACT_CUTOFF) & eligible
        gained = float((con & ~c_wt).sum()) / 2.0
        lost = float((~con & c_wt).sum()) / 2.0
        union = float((con | c_wt).sum())
        jaccard = float((con & c_wt).sum()) / union if union else 0.0

        sq = float((dD ** 2).sum())
        eig = _gyration_eigenvalues(c)
        rg = float(np.sqrt((A ** 2).sum(1).mean()))

        rows.append(
            shell_disp + shell_dist
            + [gained, lost, gained - lost, jaccard]
            + [float((con[p] & ~c_wt[p]).sum()),
               float((~con[p] & c_wt[p]).sum()),
               float(con[p].sum() - c_wt[p].sum())]
            + [float(np.sqrt(sq)) / n_res,
               float(np.linalg.norm(dD[p])) / np.sqrt(n_res),
               _local_fraction(dD[p], near12, sq, n_res)]
            + [rg - rg_wt] + list(eig - eig_wt)
            + [float(tm_wt[i]), float(np.sqrt((disp ** 2).mean())),
               float(disp[p]), float(disp.max())]
            + [float(plddt[i]), float(plddt_site[i]),
               float(plddt[i]) - float(plddt_site[i])]
        )
    out = np.asarray(rows, dtype=float)
    assert out.shape[1] == len(GEOMETRY_FEATURES), (
        f"{out.shape[1]} columns for {len(GEOMETRY_FEATURES)} names")
    return out
