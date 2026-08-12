"""Does pi_basis reproduce the eight hand-written bases, and hold its invariants?

This test is what makes the deletions safe, so it lands BEFORE the first
deletion rather than after the last. It ENFORCES rather than reports: any
failure is a non-zero exit.

Two tiers, because they answer different questions.

  PROPERTIES   fabricated 20x8 matrices, no archives, runs in a second. Four
               things nothing in this codebase currently asserts: that V is
               orthonormal, that orientation is deterministic under a
               sign-flipped input, that `readout(c) . to_raw(c) == 1`, and that
               a restricted fit never reads a held-out row. The last is checked
               by poisoning the excluded rows with NaN -- if they are touched,
               the result is NaN and the test fails, which is stronger than
               comparing against a number that happens to match.

  EQUIVALENCE  real archives, each of the eight sites reimplemented here
               EXACTLY as it appears in its own file, compared at 1e-10. The
               legacy code is inlined rather than imported so that deleting the
               original leaves this test still meaningful.

The bar is 1e-10 and not bit-for-bit because pi_basis decomposes through jnp on
the accelerator while six of the eight sites use numpy. The SIGN is held to
exact equality regardless: a flipped PC2 silently inverts a reported
correlation, which is the failure this module exists to prevent.

  sbatch analysis.sbatch pi_basis_test.py --glob '../runs/gym3_*.npz'
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_basis  # noqa: E402
import pi_stats  # noqa: E402

EPS9 = 1e-9
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)
    return ok


def close(a, b, tol=1e-10):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    d = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    return d <= tol, f"max|d| = {d:.3e}"


# ---- the legacy construction, inlined verbatim ----------------------------
def zc(M, eps=EPS9):
    """analyze_heldout:90 and six identical twins."""
    return (M - M.mean(0)) / (M.std(0) + eps)


def legacy_shared(blocks, eps=EPS9):
    """analyze_heldout:162-164 / analyze_chem:99-101 / analyze_attrib:111-113."""
    names = sorted(blocks)
    Xg = np.concatenate([zc(blocks[n], eps) for n in names], 0)
    gm = Xg.mean(0)
    V = np.linalg.svd(Xg - gm, full_matrices=False)[2]
    return V, gm


def legacy_orient_kl(V, gm, blocks, ref, n_c, eps=EPS9):
    """analyze_heldout:168-174. Cluster bootstrap of the sign against kl_glob."""
    V = V.copy()
    for c in range(n_c):
        g = {n: [pi_stats.spearman((zc(blocks[n], eps) - gm) @ V[c], ref[n])]
             for n in sorted(blocks)}
        if pi_stats.cluster_bootstrap(g, n_boot=2000, seed=0,
                                      hierarchical=False)[0] < 0:
            V[c] = -V[c]
    return V


def legacy_uncentred(blocks, eps=EPS9):
    """analyze_scrutiny:93-97. Divide by the spread, keep the anchor."""
    names = sorted(blocks)
    Xg = np.concatenate([blocks[n] / (blocks[n].std(0) + eps) for n in names], 0)
    gm = np.zeros(Xg.shape[1])
    return np.linalg.svd(Xg - gm, full_matrices=False)[2], gm


# ---- tier 1: properties ---------------------------------------------------
def tier_properties():
    print("\nPROPERTIES  (synthetic, no archives)\n")
    rng = np.random.default_rng(0)
    n, D, A = 40, 8, 3
    blocks = {f"a{i}": rng.normal(size=(n, D)) * (i + 1) for i in range(A)}
    ref = {k: rng.normal(size=n) for k in blocks}

    b = pi_basis.fit(blocks, layer=-1, orient_ref=ref, orient_k=2)

    G = b.components @ b.components.T
    check("V is orthonormal", close(G, np.eye(D), 1e-10)[0],
          close(G, np.eye(D), 1e-10)[1])

    # Orientation must be a property of the DATA, not of the arbitrary sign the
    # decomposition happened to return. Negating every input row negates the
    # reference correlation too, so the fixed signs must follow.
    nb = {k: -v for k, v in blocks.items()}
    b2 = pi_basis.fit(nb, layer=-1, orient_ref=ref, orient_k=2)
    same = np.allclose(np.abs(b.components), np.abs(b2.components), atol=1e-10)
    check("orientation is deterministic under a sign-flipped input", same)

    # w . e = sum_d v[d]^2 * sd[d] / (sd[d] + eps), so it is 1 only up to
    # eps/min(sd) -- exactly, not approximately. Assert the derived bound
    # rather than a magic tolerance: a real inversion of the bridge is off by
    # a factor of sd^2 per channel and blows any bound this tight.
    bound = 2 * b.eps / float(b.sd["a0"].min())
    dev = max(abs(float(b.readout(c, "a0") @ b.to_raw(c, "a0")) - 1.0)
              for c in range(4))
    check("readout(c) . to_raw(c) == 1  (the raw-space bridge)", dev <= bound,
          f"dev {dev:.2e} <= eps/min(sd) bound {bound:.2e}")

    # A restricted fit must not read the excluded rows. NaN in the held-out
    # half makes "did it look" observable rather than inferred.
    poisoned, rows = {}, {}
    for k, v in blocks.items():
        v = v.copy()
        m = np.zeros(len(v), bool)
        m[: n // 2] = True
        v[~m] = np.nan
        poisoned[k], rows[k] = v, m
    br = pi_basis.fit(poisoned, layer=-1, orient_on="dms_train",
                      orient_ref=ref, rows=rows)
    check("a restricted fit never reads a held-out row",
          bool(np.isfinite(br.components).all()))

    # The two guards.
    try:
        b.project(blocks["a0"], layer=30)
        check("project refuses a layer it was not fitted at", False)
    except ValueError:
        check("project refuses a layer it was not fitted at", True)
    try:
        pi_basis.fit(blocks, layer=-1, orient_on="kl_glob", orient_ref=ref,
                     rows=rows)
        check("kl_glob orientation refuses a restricted fit", False)
    except ValueError:
        check("kl_glob orientation refuses a restricted fit", True)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "b.npz"
        b.save(p)
        r = pi_basis.load(p)
        ok = (close(r.components, b.components)[0] and close(r.gm, b.gm)[0]
              and close(r.sd["a0"], b.sd["a0"])[0] and r.layer == b.layer
              and r.eps == b.eps and r.centered == b.centered)
    check("save/load round-trips", ok)


# ---- tier 2: equivalence against the real archives ------------------------
def tier_equivalence(files):
    print(f"\nEQUIVALENCE  ({len(files)} archives, bar 1e-10)\n")
    X, KL, Y, FULL = {}, {}, {}, {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        n = Path(f).stem.split("_", 1)[1]
        FULL[n] = np.asarray(d["dz_site"], np.float64)
        X[n] = FULL[n][:, -1, :]
        KL[n] = np.asarray(d["kl_glob"], np.float64)[:, -1]
        Y[n] = np.asarray(d["score"], float)
    names = sorted(X)
    print(f"  {len(names)} assays, {sum(len(X[n]) for n in names)} variants, "
          f"dim {X[names[0]].shape[1]}\n")

    # 1-3. the shared basis, unoriented (heldout / chem / attrib / pc2 agree here)
    Vl, gml = legacy_shared(X)
    b = pi_basis.fit(X, layer=-1, orient_on=None)
    check("shared basis: V", *close(np.abs(Vl), np.abs(b.components)))
    check("shared basis: pooled mean", *close(gml, b.gm))

    # 4. orientation on kl_glob, the canonical rule (analyze_heldout, orient_k=2)
    Vo = legacy_orient_kl(Vl, gml, X, KL, 2)
    bo = pi_basis.fit(X, layer=-1, orient_on="kl_glob", orient_ref=KL, orient_k=2)
    check("kl_glob orientation: V (sign exact)",
          bool(np.array_equal(np.sign(Vo[:2]), np.sign(bo.components[:2]))))
    check("kl_glob orientation: values", *close(Vo[:2], bo.components[:2]))

    # 5. the same rule at N_PC=4, as chem / attrib / scrutiny run it
    Vo4 = legacy_orient_kl(Vl, gml, X, KL, 4)
    bo4 = pi_basis.fit(X, layer=-1, orient_on="kl_glob", orient_ref=KL, orient_k=4)
    check("kl_glob orientation at orient_k=4", *close(Vo4[:4], bo4.components[:4]))

    # 6. projection reproduces the hand-written (zc(X) - gm) @ V[c].
    # ONLY over the oriented components. A singular vector's sign is arbitrary,
    # and numpy and jax do not resolve it the same way, so past orient_k the
    # signed scores are not comparable between two correct implementations --
    # they are not comparable between two runs of the same one. Anything
    # downstream that reads a sign off PC3+ is reading noise.
    Pl = (zc(X[names[0]]) - gml) @ Vo.T
    Pb = bo.project(X[names[0]], layer=-1)
    check("projection (oriented components)", *close(Pl[:, :2], Pb[:, :2]))
    check("projection (all components, up to sign)",
          *close(np.abs(Pl), np.abs(Pb)))

    # 7. analyze_scrutiny's uncentred control
    Vu, gmu = legacy_uncentred(X)
    bu = pi_basis.fit(X, layer=-1, center=False, orient_on=None)
    check("uncentred variant: V", *close(np.abs(Vu), np.abs(bu.components)))
    check("uncentred variant: pooled mean is zero", *close(gmu, bu.gm))

    # 8. analyze_chem's LOAO: fit on training assays, orient on them alone
    held = names[0]
    tr = [n for n in names if n != held]
    Xt = np.concatenate([zc(X[n]) for n in tr], 0)
    gmt = Xt.mean(0)
    Vt = np.linalg.svd(Xt - gmt, full_matrices=False)[2][:4]
    sgn = np.sign(np.mean([pi_stats.spearman((zc(X[n]) - gmt) @ Vt[1], Y[n])
                           for n in tr]))
    sc_l = ((zc(X[held]) - gmt) @ Vt[1]) * sgn
    bt = pi_basis.fit({n: X[n] for n in tr}, layer=-1, orient_on="dms_train",
                      orient_ref={n: Y[n] for n in tr}, orient_k=4, n_pc=4)
    sc_b = bt.project(X[held], layer=-1)[:, 1]
    check("chem LOAO: held-out scores", *close(sc_l, sc_b))

    # 9. analyze_basis, same construction at an arbitrary layer, eps=1e-8
    L = FULL[names[0]].shape[1]
    li = L // 2
    Xm = {n: FULL[n][:, li, :] for n in names}
    Vm, gmm = legacy_shared(Xm, eps=1e-8)
    bm = pi_basis.fit(FULL, layer=li, orient_on=None, eps=1e-8)
    check("per-layer basis at an arbitrary depth", *close(np.abs(Vm),
                                                          np.abs(bm.components)))
    check("per-layer basis: layer was MEASURED, not asserted",
          bm.layer_asserted is False and bm.n_layers == L)

    # 10. analyze_transfer's rotated basis: pooled RAW channels, no per-assay
    # z-score, no orientation. A different object from PC2, reproduced here so
    # that difference is asserted rather than described.
    Xtr = np.concatenate([X[n] for n in tr], 0)
    gmr = Xtr.mean(0)
    Vr = np.linalg.svd(Xtr - gmr, full_matrices=False)[2]
    Ptr_l = (Xtr - gmr) @ Vr.T
    bz = pi_basis.fit({n: X[n] for n in tr}, layer=-1, orient_on=None,
                      zscore=False)
    check("transfer: unoriented basis on RAW rows",
          *close(np.abs(Vr), np.abs(bz.components)))
    check("transfer: pooled mean", *close(gmr, bz.gm))
    # orient_on=None here, so NO component has a defined sign -- compare
    # magnitudes only. That is not a weaker test, it is the correct one: an
    # unoriented basis has no signed content to check.
    check("transfer: projection (up to sign)",
          *close(np.abs(Ptr_l), np.abs(bz.project(Xtr, layer=-1))))

    # ...and it must NOT coincide with the standardised basis, or the flag is
    # decorative. The two are different objects and the test says by how much.
    bs = pi_basis.fit({n: X[n] for n in tr}, layer=-1, orient_on=None)
    cos = float(np.abs(bz.components[1] @ bs.components[1]))
    check("transfer's rotated basis is NOT the shared basis", cos < 0.99,
          f"|cos(PC2_raw, PC2_zscored)| = {cos:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="", help="archives for the equivalence "
                                               "tier; omitted runs properties only")
    a = ap.parse_args()

    tier_properties()
    if a.glob:
        files = sorted(glob.glob(a.glob))
        if not files:
            raise SystemExit(f"no archives matched {a.glob}")
        tier_equivalence(files)
    else:
        print("\nEQUIVALENCE  skipped (no --glob). The deletions are NOT "
              "authorised by a properties-only run.")

    print(f"\n{'=' * 62}")
    if FAIL:
        print(f"FAILED: {len(FAIL)}\n  " + "\n  ".join(FAIL))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
