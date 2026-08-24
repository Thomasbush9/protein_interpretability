"""Is it the same direction? Asked where the signal actually lives.

`cross_phenotype_transfer.py` compares the two cohorts' fitted probes by cosine
between 128-dimensional ridge weight vectors. That test is weak, and its own
control says so: two probes fitted on disjoint halves of the SAME cohort agree
only +0.15 to +0.37. Ridge coefficients on 128 correlated pair channels are
loosely identified, so a low cosine is consistent with "different directions"
and equally with "the same subspace, differently parameterised".

`geometry_baseline.py` showed where the signal is: the rho-versus-dimension
curve saturates by four principal components, and two of them already beat all
37 emitted-geometry features. So the direction question should be asked in that
four-dimensional subspace, where a probe is fitted from ~1200-2500 rows onto 4
parameters instead of 128 and is actually identified.

THREE TESTS, ANSWERING DIFFERENT THINGS.

    shared-basis cosine  One basis, built by PCA on both cohorts' features
        pooled -- unsupervised, no label is touched, so this is a descriptive
        statistic about directions and NOT a transfer claim. A probe is fitted
        on each cohort inside it and the two d-vectors are compared. This is
        the well-conditioned version of the original test: same question, same
        split-half null, enough rows per parameter for the answer to mean
        something. The transfer rho inside the basis is reported alongside,
        because comparing directions in a subspace that does not predict would
        be meaningless.

    feature subspace     Principal angles between each cohort's OWN top-d
        principal subspace. Unsupervised: does the trunk's mutation response
        even span the same directions in the two cohorts, before any label is
        involved?

    direction subspace   Principal angles between the spans of the PER-ASSAY
        fitted directions. Supervised, and the closest thing to the question as
        asked: not "do two pooled fits agree" but "do the directions that
        predict in each assay lie in a common subspace".

Principal angles are invariant to how each subspace is written, which is
exactly the defect of the weight-vector cosine. Every number is read against
two scales: a split-half null within each cohort, and the floor for two random
d-dimensional subspaces of R^128, which is d/128 -- 0.031 at d=4. Without the
floor an overlap of 0.4 has no meaning.

Chemistry is carried as a positive control: it is model-independent and its
direction genuinely is shared, so it is what agreement looks like when there
is some.

    uv run python experiments/analysis/direction_agreement.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_blocks as fb                                          # noqa: E402

from protein_interpretability import artifacts                       # noqa: E402
from protein_interpretability.analysis import statistics as st       # noqa: E402
from protein_interpretability.analysis.probes import (               # noqa: E402
    ridge_fit, zscore,
)
from protein_interpretability.analysis.transfer import (             # noqa: E402
    fit_groups_predict_groups, orthonormal, pca_apply, pca_basis,
    principal_angles, random_subspace, subspace_overlap,
)
from protein_interpretability.collection import Cohort               # noqa: E402

W = fb.W
LAM = 10.0
DIMS = (2, 4, 8)
N_DRAW = 200
BLOCKS = ("internal", "chem")
STABILITY_MARK = "Tsuboyama_2023"


def stacked(X_by_assay, target, names=None):
    """Within-assay z-scored features and target, pooled. The frozen scaling."""
    names = sorted(X_by_assay) if names is None else sorted(names)
    return (np.concatenate([zscore(X_by_assay[n]) for n in names], 0),
            np.concatenate([zscore(target[n]) for n in names], 0))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def summarise(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()),
            "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)), "n": int(v.size)}


def halves(names, rng):
    perm = rng.permutation(sorted(names))
    h = len(perm) // 2
    return perm[:h], perm[h:2 * h]


def probe_in_basis(X_by_assay, target, mu, comp, lam, names=None):
    """Ridge weights inside a fixed subspace, intercept dropped."""
    X, y = stacked(X_by_assay, target, names)
    return ridge_fit(pca_apply(X, mu, comp), y, lam)[:-1]


def assay_directions(X_by_assay, target, lam, names=None):
    """One unit-norm fitted direction per assay, stacked (n_assays, n_features).

    Normalised because a cohort's most decodable assay would otherwise dominate
    the span, and the question is which directions are used, not how strongly.
    """
    names = sorted(X_by_assay) if names is None else sorted(names)
    rows = []
    for n in names:
        w = ridge_fit(zscore(X_by_assay[n]), zscore(target[n]), lam)[:-1]
        nrm = np.linalg.norm(w)
        if nrm > 0:
            rows.append(w / nrm)
    return np.asarray(rows)


def span_uncentred(D, d):
    """Top-d right singular vectors of D, WITHOUT centring, as columns.

    Centring is wrong here and would quietly invert the question: the mean of a
    set of fitted directions is the shared direction, so removing it discards
    precisely what is being tested. `orthonormal` centres, correctly, because
    it is applied to features rather than to directions.
    """
    return np.linalg.svd(np.asarray(D, float), full_matrices=False)[2][:d].T


def run_block(block, Xs, ys, Xp, yp, d, lam, rng):
    p = next(iter(Xs.values())).shape[1]
    if d > p:
        return None
    ns, np_ = sorted(ys), sorted(yp)

    # --- 1. shared basis, unsupervised, built on both cohorts' features -----
    Xboth = np.concatenate([stacked(Xs, ys)[0], stacked(Xp, yp)[0]], 0)
    mu, comp = pca_basis(Xboth, d)
    ws = probe_in_basis(Xs, ys, mu, comp, lam)
    wp = probe_in_basis(Xp, yp, mu, comp, lam)
    across = cosine(ws, wp)

    null = {}
    for tag, (X, y, names) in {"stability": (Xs, ys, ns),
                               "panel5": (Xp, yp, np_)}.items():
        draws = []
        for _ in range(N_DRAW):
            a, b = halves(names, rng)
            draws.append(cosine(probe_in_basis(X, y, mu, comp, lam, a),
                                probe_in_basis(X, y, mu, comp, lam, b)))
        null[tag] = summarise(draws)

    # Does the subspace still predict? Fit on one cohort inside the basis,
    # score each test assay. If this collapses, the cosine above is a
    # comparison of two directions that do not matter.
    rho = {}
    for tag, (Xtr, ytr, Xte, yte) in {
            "stability_to_panel5": (Xs, ys, Xp, yp),
            "panel5_to_stability": (Xp, yp, Xs, ys)}.items():
        X, y = stacked(Xtr, ytr)
        w = ridge_fit(pca_apply(X, mu, comp), y, lam)
        per = {n: float(st.spearman(
            np.column_stack([pca_apply(zscore(Xte[n]), mu, comp),
                             np.ones(len(Xte[n]))]) @ w, yte[n]))
            for n in sorted(yte)}
        rho[tag] = {"mean": float(np.mean(list(per.values()))), "per_assay": per}

    # --- 2. each cohort's own top-d FEATURE subspace ------------------------
    Qs, Qp = orthonormal(stacked(Xs, ys)[0], d), orthonormal(stacked(Xp, yp)[0], d)
    feat = {"overlap": subspace_overlap(Qs, Qp),
            "cosines": principal_angles(Qs, Qp).tolist()}
    for tag, (X, y, names) in {"stability": (Xs, ys, ns),
                               "panel5": (Xp, yp, np_)}.items():
        draws = []
        for _ in range(N_DRAW):
            a, b = halves(names, rng)
            draws.append(subspace_overlap(orthonormal(stacked(X, y, a)[0], d),
                                          orthonormal(stacked(X, y, b)[0], d)))
        feat[f"null_{tag}"] = summarise(draws)

    # --- 3. span of the PER-ASSAY fitted directions -------------------------
    Ds, Dp = assay_directions(Xs, ys, lam), assay_directions(Xp, yp, lam)
    Rs, Rp = span_uncentred(Ds, d), span_uncentred(Dp, d)
    direc = {"overlap": subspace_overlap(Rs, Rp),
             "cosines": principal_angles(Rs, Rp).tolist()}
    for tag, (X, y, names) in {"stability": (Xs, ys, ns),
                               "panel5": (Xp, yp, np_)}.items():
        draws = []
        for _ in range(N_DRAW):
            a, b = halves(names, rng)
            if min(len(a), len(b)) <= d:
                continue        # a d-dim span needs more than d directions
            draws.append(subspace_overlap(
                span_uncentred(assay_directions(X, y, lam, a), d),
                span_uncentred(assay_directions(X, y, lam, b), d)))
        direc[f"null_{tag}"] = summarise(draws) if draws else None

    floor = summarise([subspace_overlap(random_subspace(p, d, rng),
                                        random_subspace(p, d, rng))
                       for _ in range(N_DRAW)])

    # The cosine's OWN floor, which matters most where d is smallest: two random
    # unit vectors in R^2 are within 13 degrees of each other about 7% of the
    # time, so a shared-basis cosine of +0.97 at d=2 is not by itself strong
    # evidence of anything. The informative comparison is always against the
    # split-half null -- how much agreement this many assays can produce with
    # THEMSELVES -- and that comparison is unaffected by this floor. It is
    # reported so the weakness is visible rather than implied.
    rc = np.abs([cosine(rng.normal(size=d), rng.normal(size=d))
                 for _ in range(2000)])
    cos_floor = {"abs_mean": float(rc.mean()),
                 "abs_p95": float(np.percentile(rc, 95)),
                 "p_at_least_across": float((rc >= abs(across)).mean())}
    # The blind version of the transfer number above. The basis here is fitted
    # on the TRAINING cohort's features alone, so nothing about the test cohort
    # -- not even its unlabelled feature covariance -- touches the fit. The
    # shared-basis rho is a diagnostic that the subspace carries signal; this
    # one is a result.
    blind = {}
    for tag, (Xtr, ytr, Xte, yte) in {
            "stability_to_panel5": (Xs, ys, Xp, yp),
            "panel5_to_stability": (Xp, yp, Xs, ys)}.items():
        per = fit_groups_predict_groups(fb.as_probe_blocks(Xtr, ytr),
                                        fb.as_probe_blocks(Xte, yte),
                                        lam=lam, d=d)
        blind[tag] = {"mean": float(np.mean(list(per.values()))),
                      "per_assay": per}

    return {"d": d, "width": p, "blind_transfer_train_only_basis": blind,
            "shared_basis": {"cosine_across": across, "null": null,
                             "random_cosine_floor": cos_floor,
                             "transfer_rho_in_basis": rho},
            "feature_subspace": feat, "direction_subspace": direc,
            "random_floor": floor}


def run_model(model, a, stab, panel):
    Xs, ys = fb.load_blocks(model, stab, a.heldout_captures,
                            Path(a.heldout_tm.format(model=model)), blocks=BLOCKS)
    Xp, yp = fb.load_blocks(model, panel, a.panel_captures,
                            Path(a.panel_tm.format(model=model)), blocks=BLOCKS)
    shared = set(ys) & set(yp)
    if shared:
        raise SystemExit(f"cohorts share short key(s) {sorted(shared)}")

    print(f"\n=== {model} ===")
    out = {}
    for b in BLOCKS:
        out[b] = {}
        print(f"\n  {b}  ({next(iter(Xs[b].values())).shape[1]} channels)")
        print(f"  {'d':>2s} {'cos across':>11s} {'null stab':>11s} "
              f"{'null panel':>11s} {'feat ovl':>9s} {'dir ovl':>9s} "
              f"{'floor':>7s} {'p_rand':>7s} {'rho s>p':>8s} "
              f"{'rho p>s':>8s} {'blind s>p':>10s} {'blind p>s':>10s}")
        for d in DIMS:
            rng = np.random.default_rng(0)
            r = run_block(b, Xs[b], ys, Xp[b], yp, d, a.lam, rng)
            if r is None:
                continue
            out[b][str(d)] = r
            sb, fs, ds = (r["shared_basis"], r["feature_subspace"],
                          r["direction_subspace"])
            dn = ds["null_stability"]
            print(f"  {d:>2d} {sb['cosine_across']:>+11.3f} "
                  f"{sb['null']['stability']['mean']:>+11.3f} "
                  f"{sb['null']['panel5']['mean']:>+11.3f} "
                  f"{fs['overlap']:>9.3f} {ds['overlap']:>9.3f} "
                  f"{r['random_floor']['mean']:>7.3f} "
                  f"{sb['random_cosine_floor']['p_at_least_across']:>7.3f} "
                  f"{sb['transfer_rho_in_basis']['stability_to_panel5']['mean']:>+8.3f} "
                  f"{sb['transfer_rho_in_basis']['panel5_to_stability']['mean']:>+8.3f} "
                  f"{r['blind_transfer_train_only_basis']['stability_to_panel5']['mean']:>+10.3f} "
                  f"{r['blind_transfer_train_only_basis']['panel5_to_stability']['mean']:>+10.3f}")
            if d == 4:
                def _n(x):
                    return f"{x['mean']:.3f}" if x else "n/a"
                print(f"       cosines  features "
                      f"{np.round(fs['cosines'], 3).tolist()}   directions "
                      f"{np.round(ds['cosines'], 3).tolist()}")
                print(f"       nulls    features [stab "
                      f"{_n(fs['null_stability'])}, panel "
                      f"{_n(fs['null_panel5'])}]   directions [stab "
                      f"{_n(ds.get('null_stability'))}, panel "
                      f"{_n(ds.get('null_panel5'))}]")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=fb.MODELS, default="boltz2")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--heldout-captures", default=str(W / "runs" / "xmodel_layers"))
    ap.add_argument("--panel-captures", default=str(W / "runs" / "xmodel_panel5"))
    ap.add_argument("--heldout-tm",
                    default=str(W / "runs" / "tm_heldout16_{model}.npz"))
    ap.add_argument("--panel-tm", default=str(W / "runs" / "tm_panel5_{model}.npz"))
    ap.add_argument("--out", default=str(W / "runs" / "direction_agreement.json"))
    a = ap.parse_args()

    held, panel = Cohort.load("heldout_assays"), Cohort.load("panel5_assays")
    held.verify()
    panel.verify()
    h_ok, _ = fb.complete_assays(held, a.heldout_captures, fb.MODELS)
    p_ok, _ = fb.complete_assays(panel, a.panel_captures, fb.MODELS)
    stab = [x for x in h_ok if STABILITY_MARK in x.id]

    print(f"stability {len(stab)} assays vs panel5 {len(p_ok)} assays, "
          f"d in {DIMS}, {N_DRAW} null draws")
    models = fb.MODELS if a.all else (a.model,)
    payload = {"cohorts": {"stability": [x.id for x in stab],
                           "panel5": [x.id for x in p_ok]},
               "models": {m: run_model(m, a, stab, p_ok) for m in models}}

    artifacts.write_result(Path(a.out), payload, protocol={
        "question": "is the mutation direction shared across phenotypes, asked "
                    "in the subspace where the signal lives",
        "why_d": "the rho-versus-dimension curve in geometry_baseline.py "
                 "saturates by four principal components",
        "shared_basis": "PCA on both cohorts' features pooled -- UNSUPERVISED, "
                        "no label used; a descriptive statistic about "
                        "directions, not a transfer claim",
        "blind_transfer": "the same reduction with the basis fitted on the "
                          "TRAINING cohort alone, which is a transfer claim",
        "feature_subspace": "principal angles between each cohort's own top-d "
                            "principal subspace",
        "direction_subspace": "principal angles between the uncentred spans of "
                              "the per-assay fitted directions",
        "scales": "split-half null within each cohort, and the floor for two "
                  "random d-dimensional subspaces of R^128, which is d/128",
        "lam": a.lam, "dims": list(DIMS), "n_draw": N_DRAW,
        "scaling": "features and target z-scored within each assay",
    })
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
