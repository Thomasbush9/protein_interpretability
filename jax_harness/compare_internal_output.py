"""Internal state vs the model's own outputs, same mutants, same protocol.

gym2 captures per-layer internal features, the predicted structure, and pLDDT
for every variant in one run, so this is the only fully like-for-like comparison
available. Everything is scored by Spearman on held-out *positions* (no residue
in both train and test); single-number predictors need no fitting and are scored
directly on the same test rows.

What the number means. This is assay-specific SUPERVISED DECODABILITY, not a
stability predictor:

    one variant     -> 4 feature families x 64 layers = 256 internal features
    one assay       -> X: 250 x 256, y: 250 ProteinGym DMS_score values
    one outer split -> train positions / unseen test positions
    one fitted model-> select k train-correlated features, then ridge
    repeats         -> correlated estimates within that assay, not replicates

Three things the August 2026 audit required, all implemented here.

**Tie-aware ranks.** Every correlation comes from pi_stats (scipy under the
hood). The previous `np.argsort(np.argsort(x))` broke ties by array order, which
mattered most for the baseline that needed it least: the position-only
predictor emits the global training mean for every held-out row -- it is
constant, its Spearman is undefined -- and the old code scored it +0.069 out of
pure tie-ordering artefact. It is reported as NaN here, with a defined
nearest-position variant alongside it so the row is not simply empty.

**Assay-level uncertainty.** Repeated splits inside one assay are five
correlated looks at the same 250 variants. Intervals come from a hierarchical
bootstrap over ASSAYS (pi_stats.cluster_bootstrap); the spread across splits is
reported separately and labelled SD, because it is a description of split
heterogeneity and not an uncertainty in the mean.

**Chemistry baselines.** Position grouping stops site memorisation but not the
possibility that the internal features merely re-encode what the substitution
is. `chemistry` (BLOSUM, hydropathy, volume, charge, special residues) and
`identity` (one-hot of wild-type and mutant residue) are fitted the same way on
the same rows, and `internal+chem` measures whether the trunk adds anything on
top. If internal does not beat chemistry, the headline is not a claim about the
model.

One protocol, frozen. k and lambda are chosen on an INNER position-grouped
split of the training rows only; nothing is tuned on test. The old code path
that hard-coded k=16, lambda=1 while the report described inner tuning is kept
only as `internal_fixed16`, reported for continuity with the previous number.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_chem  # noqa: E402
import pi_stats  # noqa: E402
from geom import tm_score  # noqa: E402

BLOCKS = ["kl_glob", "kl_site", "dz_site", "ds_site"]


def grouped_split(pos, frac, rng):
    """Held-out RESIDUE POSITIONS, so no site appears on both sides."""
    up = np.unique(pos)
    rng.shuffle(up)
    held = set(up[: max(1, int(round(len(up) * frac)))].tolist())
    m = np.array([p in held for p in pos])
    return ~m, m


def ridge_fit(X, y, lam):
    Xb = np.column_stack([X, np.ones(len(X))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam                       # never penalise the intercept
    return np.linalg.solve(A, Xb.T @ y)


def ridge_pred(w, X):
    return np.column_stack([X, np.ones(len(X))]) @ w


def select_k(X, y, k):
    """Top-k features by |Spearman| on the TRAINING rows only."""
    return np.argsort(-pi_stats.rank_corr_columns(X, y))[:k]


def fit_ridge_block(X, y, pos, tr, te, rng, ks=(8, 16, 32, 64),
                    lams=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Standardise on train, select on train, tune on an inner grouped split.

    Returns (rho_on_test, chosen_k, chosen_lambda, selected_feature_indices).
    """
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd
    ytr_m, ytr_s = y[tr].mean(), y[tr].std() + 1e-9
    yz = (y - ytr_m) / ytr_s

    itr, iva = grouped_split(pos[tr], 0.25, rng)
    best = (-np.inf, min(ks[0], X.shape[1]), lams[0])
    if iva.sum() >= 4 and itr.sum() >= 10:
        Xi, yi = Xs[tr][itr], yz[tr][itr]
        Xv, yv = Xs[tr][iva], yz[tr][iva]
        for k in ks:
            k = min(k, X.shape[1])
            idx = select_k(Xi, yi, k)
            for lam in lams:
                r = pi_stats.spearman(ridge_pred(ridge_fit(Xi[:, idx], yi, lam),
                                                 Xv[:, idx]), yv)
                if np.isfinite(r) and abs(r) > best[0]:
                    best = (abs(r), k, lam)
    _, k, lam = best
    idx = select_k(Xs[tr], yz[tr], k)
    w = ridge_fit(Xs[tr][:, idx], yz[tr], lam)
    return pi_stats.spearman(ridge_pred(w, Xs[te][:, idx]), y[te]), k, lam, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # per-predictor, per-assay lists of split-level Spearman values
    per = defaultdict(lambda: defaultdict(list))
    sel_counter = defaultdict(Counter)
    res = {}

    for f in a.features:
        d = np.load(f, allow_pickle=True)
        name = Path(f).stem.replace("gym2_", "").split("_")[0]
        y, pos, seq = d["score"], d["pos"], str(d["wt_seq"])
        ca, ca_wt = d["ca"], d["ca_wt"]
        mutants = [str(m) for m in d["mutant"]]
        nL = int(d["n_layers"])

        X_int = np.concatenate(
            [d["kl_glob"], d["kl_site"],
             np.linalg.norm(d["dz_site"], axis=-1),
             np.linalg.norm(d["ds_site"], axis=-1)], axis=1)
        X_chem = pi_chem.chem_matrix(mutants)
        X_ident = pi_chem.identity_matrix(mutants)
        tm_wt = np.array([tm_score(ca[i], ca_wt, seq, seq) for i in range(len(y))])
        pl, pls = d["plddt"], d["plddt_site"]
        X_out = np.column_stack([tm_wt, pl, pls])
        X_both = np.column_stack([X_int, X_chem])

        for s in range(a.seeds):
            rng = np.random.default_rng(s)
            tr, te = grouped_split(pos, 0.25, rng)
            if te.sum() < 8 or tr.sum() < 20:
                continue
            tune = np.random.default_rng(1000 + s)

            rho_i, k_i, lam_i, idx_i = fit_ridge_block(X_int, y, pos, tr, te, tune)
            per["internal"][name].append(rho_i)
            for j in idx_i:
                sel_counter[name][f"{BLOCKS[j // nL]}[L{j % nL}]"] += 1

            # the previous headline protocol, kept only for continuity
            mu, sd = X_int[tr].mean(0), X_int[tr].std(0) + 1e-8
            Xs = (X_int - mu) / sd
            idx16 = select_k(Xs[tr], y[tr], 16)
            w16 = ridge_fit(Xs[tr][:, idx16],
                            (y[tr] - y[tr].mean()) / (y[tr].std() + 1e-8), 1.0)
            per["internal_fixed16"][name].append(
                pi_stats.spearman(ridge_pred(w16, Xs[te][:, idx16]), y[te]))

            per["chemistry"][name].append(
                fit_ridge_block(X_chem, y, pos, tr, te, tune)[0])
            per["identity"][name].append(
                fit_ridge_block(X_ident, y, pos, tr, te, tune)[0])
            per["internal+chem"][name].append(
                fit_ridge_block(X_both, y, pos, tr, te, tune)[0])
            per["output_ridge"][name].append(
                fit_ridge_block(X_out, y, pos, tr, te, tune,
                                ks=(3,), lams=(0.1, 1.0, 10.0))[0])

            # unfitted single-number predictors, identical rows
            per["TM_to_WT"][name].append(pi_stats.spearman(tm_wt[te], y[te]))
            per["pLDDT_mean"][name].append(pi_stats.spearman(pl[te], y[te]))
            per["pLDDT_site"][name].append(pi_stats.spearman(pls[te], y[te]))

            # position-only. By construction NO test position occurs in train,
            # so every test row receives the same global training mean: the
            # predictor is constant and its Spearman is undefined. Reported as
            # NaN rather than as the tie-ordering artefact it used to produce.
            g = y[tr].mean()
            pm = {p: y[tr][pos[tr] == p].mean() for p in np.unique(pos[tr])}
            per["position_only"][name].append(
                pi_stats.spearman(np.array([pm.get(p, g) for p in pos[te]]), y[te]))
            # a defined stand-in: the training score at the NEAREST train position
            tp, tv = pos[tr], y[tr]
            per["nearest_position"][name].append(pi_stats.spearman(
                np.array([tv[np.argmin(np.abs(tp - p))] for p in pos[te]]), y[te]))

        res[name] = {"n": int(len(y)), "n_layers": nL,
                     "mean_tm_to_wt": float(tm_wt.mean()),
                     "plddt_wt": float(d["plddt_wt"]),
                     "ridge_k": int(k_i), "ridge_lambda": float(lam_i)}

    ORDER = ["internal", "internal+chem", "internal_fixed16", "chemistry",
             "identity", "output_ridge", "TM_to_WT", "pLDDT_site", "pLDDT_mean",
             "nearest_position", "position_only"]
    assays = sorted(res)

    # ---- per-assay table -------------------------------------------------
    print(f"\nSpearman(prediction, ProteinGym DMS_score) on held-out positions, "
          f"mean over {a.seeds} splits\n")
    print(f"{'assay':10s}{'n':>5s}" + "".join(f"{k:>18s}" for k in ORDER))
    print("-" * (15 + 18 * len(ORDER)))
    for nm in assays:
        row = "".join(
            f"{np.nanmean(per[k][nm]):>+18.3f}" if len(per[k][nm]) and
            np.isfinite(np.nanmean(per[k][nm])) else f"{'n/a':>18s}"
            for k in ORDER)
        print(f"{nm:10s}{res[nm]['n']:>5d}{row}")

    # ---- assay-level intervals -------------------------------------------
    print(f"\nAssay-level hierarchical bootstrap ({len(assays)} assays are the "
          f"independent unit; {a.seeds} splits within each are repeats)\n")
    print(f"{'predictor':18s}{'mean':>9s}{'95% CI':>20s}{'SD across splits':>20s}")
    print("-" * 67)
    summary = {}
    for k in ORDER:
        pt, lo, hi, nk = pi_stats.cluster_bootstrap(
            dict(per[k]), n_boot=a.n_boot, seed=0)
        flat = np.concatenate([np.asarray(v) for v in per[k].values()]) \
            if per[k] else np.array([np.nan])
        sd = float(np.nanstd(flat))
        summary[k] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                      "sd_across_splits": sd, "n_assays": nk,
                      "per_assay": {nm: [float(x) for x in per[k][nm]]
                                    for nm in per[k]}}
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else "n/a"
        print(f"{k:18s}{pt:>+9.3f}{ci:>20s}{sd:>20.3f}"
              if np.isfinite(pt) else f"{k:18s}{'undefined':>9s}{'n/a':>20s}{'n/a':>20s}")

    # ---- the gaps that carry the claim -----------------------------------
    print("\nPaired gaps (same rows, same splits; assay-level CI)\n")
    gaps = {}
    for lab, kb in (("internal - TM_to_WT", "TM_to_WT"),
                    ("internal - pLDDT_site", "pLDDT_site"),
                    ("internal - output_ridge", "output_ridge"),
                    ("internal - chemistry", "chemistry"),
                    ("internal - identity", "identity"),
                    ("internal+chem - chemistry", "chemistry")):
        ka = "internal+chem" if lab.startswith("internal+chem") else "internal"
        pt, lo, hi, nk = pi_stats.paired_cluster_bootstrap(
            dict(per[ka]), dict(per[kb]), n_boot=a.n_boot, seed=0)
        wins = sum(1 for nm in per[ka] for x, yv in zip(per[ka][nm], per[kb][nm])
                   if np.isfinite(x) and np.isfinite(yv) and x > yv)
        tot = sum(len(per[ka][nm]) for nm in per[ka])
        gaps[lab] = {"gap": pt, "ci_lo": lo, "ci_hi": hi, "n_assays": nk,
                     "wins": wins, "splits": tot,
                     "excludes_zero": bool(np.isfinite(lo) and lo > 0)}
        flag = "" if (np.isfinite(lo) and lo > 0) else "   <- includes zero"
        print(f"  {lab:28s} {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"wins {wins}/{tot}{flag}")

    # ---- where the selected features live --------------------------------
    print("\nMost-selected internal features (across all assays x splits)\n")
    total = Counter()
    for nm in sel_counter:
        total.update(sel_counter[nm])
    for feat, c in total.most_common(15):
        print(f"    {feat:22s} {c:3d}")
    by_block = Counter()
    for feat, c in total.items():
        by_block[feat.split("[")[0]] += c
    print("  by block: " + ", ".join(f"{b} {c}" for b, c in by_block.most_common()))

    out = {"per_assay": res, "predictors": summary, "gaps": gaps,
           "selection_frequency": {nm: dict(sel_counter[nm]) for nm in sel_counter},
           "selection_frequency_total": dict(total),
           "protocol": {"splits_per_assay": a.seeds, "n_assays": len(assays),
                        "target": "ProteinGym DMS_score (Tsuboyama 2023 "
                                  "stability phenotype, ddG-like)",
                        "split": "position-grouped 75/25, held-out residues",
                        "tuning": "k and lambda on an inner position-grouped "
                                  "split of training rows only",
                        "uncertainty": "hierarchical bootstrap over assays"}}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    print("\nNOTE: 'position_only' is undefined by construction -- with held-out "
          "positions\n  every test row gets the same global training mean. The "
          "previous +0.069 was an\n  artefact of breaking ties by array order. "
          "'nearest_position' is the defined stand-in.")


if __name__ == "__main__":
    main()
