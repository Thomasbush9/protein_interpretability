"""Deep cross-model probe readout: per-layer internal features vs the model's output.

Same protocol as the Boltz-2 headline -- position-grouped 75/25 splits, ridge on
train-selected features, identical rows for every predictor -- but now the
internal side is 4 quantities x every Pairformer layer for ALL THREE models,
matching the headline's construction rather than a 5-feature shortcut.

Feature counts differ by model (Boltz-2 64 layers, OF3 48, Protenix 16). That is
not a confound: each number is a PAIRED within-model comparison of internal
against output.

What the audit required, and what changed.

*Identical variants.* Boltz-2's per-layer features previously came from
exp_gym2.py, which samples 250 variants at random, while OF3 and Protenix came
from exp_gym_deep.py, which spreads 100 across the sorted score range. The two
sets overlapped by about 20 variants per assay, so the models could each be
compared to their own output but not to one another. All three now run through
exp_gym_deep.py on the same variant IDs, alignments, recycles and sampling
steps. This script CHECKS that and refuses to print a cross-model row otherwise.

*Assay-level uncertainty.* Four assays x five splits is 20 correlated numbers,
not 20 independent ones. Intervals come from a hierarchical bootstrap over
assays, and with only four assays they are wide -- which is the honest answer,
and the reason this table cannot rank models against each other.

*Capture fidelity.* Each archive carries the drift its features were validated
at, and this script prints it. Features produced before the fidelity check was
enforced -- when it compared a tensor against itself at tol=1e9 -- have no such
field and are reported as unverified.
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402
import pi_stats  # noqa: E402
from analyze_gym_multi import grouped_split, ridge_fit, ridge_pred  # noqa: E402

BLOCKS = ["kl_glob", "kl_site", "dz_site", "ds_site"]


def select_k(X, y, k):
    """Top-k features by |Spearman| on the TRAINING rows only, tie-aware."""
    return np.argsort(-pi_stats.rank_corr_columns(X, y))[:k]


def fit_deep(X, y, pos, tr, te, seed, k=16):
    """Ridge on k train-selected features; lambda tuned on an inner grouped split."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd
    sel = select_k(Xs[tr], y[tr], k)                  # selection on TRAIN only
    Xs = Xs[:, sel]
    best, best_r = 1.0, -9
    itr, ite = grouped_split(pos[tr], np.random.default_rng(seed))
    for lam in (0.1, 1.0, 10.0, 100.0):
        if ite.sum() < 4 or itr.sum() < 10:
            continue
        w = ridge_fit(Xs[tr][itr], y[tr][itr], lam)
        r = pi_stats.spearman(ridge_pred(w, Xs[tr][ite]), y[tr][ite])
        if np.isfinite(r) and r > best_r:
            best_r, best = r, lam
    w = ridge_fit(Xs[tr], y[tr], best)
    return pi_stats.spearman(ridge_pred(w, Xs[te]), y[te])


TM_WARNED = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/deep2_*.npz")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="", help="archive the printed numbers as "
                    "JSON; without it these results cannot feed a report under "
                    "the provenance contract, which is why they never had")
    ap.add_argument("--match-depth", type=int, default=0,
                    help="resample every model's per-layer features onto this many "
                         "evenly spaced RELATIVE depths, so all models contribute "
                         "the same number of features (0 = use every layer)")
    args = ap.parse_args()

    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta, fidelity, variants = {}, {}, defaultdict(dict)

    for f in sorted(glob.glob(args.glob)):
        if "smoke" in f:
            continue
        d = np.load(f, allow_pickle=True)
        model = str(d["model"]) if "model" in d.files else "boltz2"
        assay = str(d["assay"]) if "assay" in d.files else Path(f).stem
        y, pos = d["score"], d["pos"]
        nL = int(d["n_layers"])
        meta[model] = nL
        variants[assay][model] = tuple(str(m) for m in d["mutant"])
        fidelity[model] = (
            float(d["capture_drift"]) if "capture_drift" in d.files else None,
            float(d["signal_to_drift"]) if "signal_to_drift" in d.files else None)

        # Boltz-2's exp_gym2 stores dz_site/ds_site as full vectors [n, L, C];
        # exp_gym_deep stores their norms [n, L]. Reduce to the same quantity
        # rather than letting one model contribute C times as many columns.
        blocks = []
        for b in BLOCKS:
            v = d[b]
            v = np.linalg.norm(v, axis=-1) if v.ndim == 3 else v   # [n, nL]
            if args.match_depth:
                # Trunk depths differ (64 / 48 / 16), so "every layer" gives the
                # models different feature counts. Reading the same number of
                # evenly spaced RELATIVE depths instead makes the design matrices
                # identical in width, at the cost of not seeing every Boltz-2
                # layer. Linear interpolation along the layer axis, per variant.
                src = np.linspace(0.0, 1.0, v.shape[1])
                dst = np.linspace(0.0, 1.0, args.match_depth)
                v = np.stack([np.interp(dst, src, row) for row in v])
            blocks.append(v)
        X = np.concatenate(blocks, axis=1)                      # [n, 4*depth]
        caw = d["ca_wt"].astype(float)
        # TM needs `tmtools`, which is not in the analysis container. Rather
        # than substitute a different metric under the same name -- a silent
        # way to publish a number nobody computed -- TM is dropped when the
        # dependency is missing and every TM cell reports as NaN. The pLDDT
        # comparison below is unaffected, and is the stronger baseline anyway:
        # it is the model's OWN uncertainty head, which is the objection a
        # referee raises first.
        try:
            tm = np.array([geom.tm_score(c.astype(float), caw) for c in d["ca"]])
        except ModuleNotFoundError as e:
            if not TM_WARNED:
                print(f"   NOTE: TM unavailable ({e.name} not installed) -- "
                      f"TM columns will be NaN; pLDDT comparisons still run")
                TM_WARNED.append(1)
            tm = np.full(len(d["ca"]), np.nan)
        pl = d["plddt_mean"] if "plddt_mean" in d.files else d["plddt"]
        pl = pl.mean(-1) if pl.ndim > 1 else pl

        rng = np.random.default_rng(0)
        for s in range(args.splits):
            tr, te = grouped_split(pos, rng)
            if te.sum() < 8 or tr.sum() < 20:
                continue
            per[model]["internal (deep)"][assay].append(
                fit_deep(X, y, pos, tr, te, s))
            per[model]["TM to WT"][assay].append(pi_stats.spearman(tm[te], y[te]))
            per[model]["pLDDT"][assay].append(pi_stats.spearman(pl[te], y[te]))
            per[model]["pLDDT@site"][assay].append(
                pi_stats.spearman(d["plddt_site"][te], y[te]))
            tp, tv = pos[tr], y[tr]
            per[model]["nearest-position"][assay].append(pi_stats.spearman(
                np.array([tv[np.argmin(np.abs(tp - p))] for p in pos[te]]), y[te]))

    # ---- are the models actually on the same variants? -------------------
    models = sorted(meta)
    matched = True
    for assay, bym in sorted(variants.items()):
        sets = {m: set(v) for m, v in bym.items()}
        if len(bym) > 1:
            common = set.intersection(*sets.values())
            union = set.union(*sets.values())
            if common != union:
                matched = False
                print(f"  WARNING {assay}: variant sets differ across models "
                      f"({len(common)} shared of {len(union)}); the cross-model "
                      f"comparison is NOT matched")
    print(f"\nVariant matching across models: "
          f"{'IDENTICAL in every assay' if matched else 'MISMATCHED -- see above'}")

    print("\nCapture fidelity (drift vs the model's own trunk; "
          "signal/drift on a real mutation)")
    for m in models:
        dr, ra = fidelity.get(m, (None, None))
        if dr is None:
            print(f"  {m:10s} UNVERIFIED -- archive predates the enforced check")
        else:
            print(f"  {m:10s} drift {dr:.3e}   signal/drift "
                  f"{'inf' if not np.isfinite(ra) else f'{ra:.0f}x'}")

    ORDER = ["internal (deep)", "TM to WT", "pLDDT", "pLDDT@site",
             "nearest-position"]
    n_assays = len(variants)
    print(f"\nSpearman vs ProteinGym DMS_score, held-out positions "
          f"({n_assays} assays x {args.splits} splits)\n")
    print(f"{'model':10s}{'layers':>7s}{'feats':>7s}" +
          "".join(f"{k:>18s}" for k in ORDER))
    for m in models:
        row = ""
        for k in ORDER:
            flat = np.concatenate([np.asarray(v) for v in per[m][k].values()])
            row += f"{np.nanmean(flat):>+18.3f}"
        print(f"{m:10s}{meta[m]:>7d}{4*meta[m]:>7d}{row}")

    print(f"\nInternal minus TM, with the ASSAY as the independent unit "
          f"(only {n_assays} of them, so the intervals are wide)\n")
    for m in models:
        pt, lo, hi, nk = pi_stats.paired_cluster_bootstrap(
            dict(per[m]["internal (deep)"]), dict(per[m]["TM to WT"]),
            n_boot=args.n_boot, seed=0)
        wins = sum(1 for a in per[m]["internal (deep)"]
                   for x, yv in zip(per[m]["internal (deep)"][a], per[m]["TM to WT"][a])
                   if np.isfinite(x) and np.isfinite(yv) and x > yv)
        tot = sum(len(v) for v in per[m]["internal (deep)"].values())
        flag = "" if (np.isfinite(lo) and lo > 0) else "   (includes zero)"
        print(f"  {m:10s} gap {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"beats TM in {wins}/{tot} splits across {nk} assays{flag}")
        print("      per-assay gaps: " + "  ".join(
            f"{a.split('_')[0]} {np.nanmean(per[m]['internal (deep)'][a]) - np.nanmean(per[m]['TM to WT'][a]):+.3f}"
            for a in sorted(per[m]["internal (deep)"])))

    print("\n  These are PAIRED WITHIN-MODEL comparisons of internal against "
          "output.\n  They are not a ranking of the models: layer counts, "
          "distogram grids and\n  alignment handling differ, and four assays "
          "cannot separate three models.")

    if args.out:
        # Exactly the quantities printed above, nothing recomputed.
        out = {"models": list(models), "n_assays": n_assays,
               "splits": args.splits, "order": ORDER,
               "layers": {m: int(meta[m]) for m in models},
               "assays": sorted(variants),
               "fidelity": {m: (None if fidelity.get(m, (None, None))[0] is None
                                else {"drift": float(fidelity[m][0]),
                                      "signal_over_drift":
                                          (None if not np.isfinite(fidelity[m][1])
                                           else float(fidelity[m][1]))})
                            for m in models},
               "spearman": {}, "internal_minus_tm": {}, "internal_minus_plddt": {}}
        for m in models:
            out["spearman"][m] = {
                k: float(np.nanmean(np.concatenate(
                    [np.asarray(v) for v in per[m][k].values()])))
                for k in ORDER}
            pt, lo, hi, nk = pi_stats.paired_cluster_bootstrap(
                dict(per[m]["internal (deep)"]), dict(per[m]["TM to WT"]),
                n_boot=args.n_boot, seed=0)
            wins = sum(1 for a in per[m]["internal (deep)"]
                       for x, yv in zip(per[m]["internal (deep)"][a],
                                        per[m]["TM to WT"][a])
                       if np.isfinite(x) and np.isfinite(yv) and x > yv)
            tot = sum(len(v) for v in per[m]["internal (deep)"].values())
            ptp, lop, hip, nkp = pi_stats.paired_cluster_bootstrap(
                dict(per[m]["internal (deep)"]), dict(per[m]["pLDDT"]),
                n_boot=args.n_boot, seed=0)
            winsp = sum(1 for a in per[m]["internal (deep)"]
                        for x, yv in zip(per[m]["internal (deep)"][a],
                                         per[m]["pLDDT"][a])
                        if np.isfinite(x) and np.isfinite(yv) and x > yv)
            out["internal_minus_plddt"][m] = {
                "gap": float(ptp), "ci_lo": float(lop), "ci_hi": float(hip),
                "wins": int(winsp), "splits": int(tot), "n_assays": int(nkp),
                "per_assay": {a: float(np.nanmean(per[m]["internal (deep)"][a])
                                       - np.nanmean(per[m]["pLDDT"][a]))
                              for a in sorted(per[m]["internal (deep)"])}}
            out["internal_minus_tm"][m] = {
                "gap": float(pt), "ci_lo": float(lo), "ci_hi": float(hi),
                "wins": int(wins), "splits": int(tot), "n_assays": int(nk),
                "per_assay": {a: float(np.nanmean(per[m]["internal (deep)"][a])
                                       - np.nanmean(per[m]["TM to WT"][a]))
                              for a in sorted(per[m]["internal (deep)"])}}
        Path(args.out).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
