"""One shared mutation direction, or one per phenotype?

Every archived number trains and tests inside a cohort. That establishes the
gap replicates; it cannot say whether the direction the probe finds in folding
stability is the SAME direction that matters for fitness, abundance and
activity, or merely an analogous one found separately in each. The captures to
settle it are already on disk -- the two cohorts are disjoint and share a
feature definition -- so this needs no GPU.

The design is deliberately blunt. Fit on one cohort, predict the other, never
touching the test assays:

    stability -> panel5     12 Tsuboyama mini-domains, 40-118 aa, predicting
                            25 fitness/abundance/activity assays, 101-553 aa.
                            Phenotype, length and organism all change at once.
    panel5 -> stability     the same in reverse, which is the harder direction
                            to fake: panel5 is the larger and more varied
                            training set.
    stability -> other4     the four non-stability assays inside the held-out
                            cohort, as a shorter-range version of the same step.

TWO THINGS ARE READ, AND THEY ANSWER DIFFERENT QUESTIONS.

    transfer    rho of a probe fitted on the other cohort, against the
                within-cohort leave-one-assay-out rho on the SAME test assays.
                The second is the ceiling: same phenotype, unseen protein. How
                much of it survives the phenotype switch is the answer.

    agreement   cosine between the two fitted weight vectors. Transfer rho can
                stay high because two different directions both correlate with
                something easy; the cosine asks about the direction itself. It
                is read against a split-half null -- two probes fitted on
                disjoint halves of the SAME cohort -- because 128 coefficients
                fitted on twelve assays do not agree perfectly with themselves,
                and a raw cosine has no scale without that.

CHEMISTRY IS NOT OPTIONAL HERE. It is model-independent, so it transfers
between cohorts for free. A trunk probe that transfers no better than
substitution chemistry has shown nothing about the model, and in a cross-cohort
design that is the easy way to over-read a positive number.

    uv run python experiments/analysis/cross_phenotype_transfer.py --all
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
    leave_one_group_out, ridge_fit, zscore,
)
from protein_interpretability.analysis.transfer import (             # noqa: E402
    fit_groups_predict_groups,
)
from protein_interpretability.collection import Cohort               # noqa: E402

W = fb.W
LAM = 10.0
N_BOOT = 10000
STABILITY_MARK = "Tsuboyama_2023"


def boot(per_assay):
    p, lo, hi, k = st.cluster_bootstrap({k: [v] for k, v in per_assay.items()},
                                        n_boot=N_BOOT, hierarchical=False)
    return {"mean": p, "lo": lo, "hi": hi, "n_assays": k}


def boot_gap(a, b):
    p, lo, hi, k = st.paired_cluster_bootstrap(
        {k: [v] for k, v in a.items()}, {k: [v] for k, v in b.items()},
        n_boot=N_BOOT, hierarchical=False)
    return {"mean": p, "lo": lo, "hi": hi, "n_assays": k}


def pooled_weights(blocks, lam):
    """Ridge coefficients from every assay pooled, minus the intercept.

    The same standardisation as every probe in this project: within assay,
    then pooled. Two weight vectors are comparable across cohorts only because
    of that -- the columns are in units of each assay's own spread.
    """
    names = sorted(blocks)
    X = np.concatenate([zscore(blocks[n]["X"]) for n in names], 0)
    y = np.concatenate([zscore(blocks[n]["y"]) for n in names], 0)
    return ridge_fit(X, y, lam)[:-1]


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def split_half_null(blocks, lam, n_draw=200, seed=0):
    """Cosine between probes fitted on disjoint halves of the SAME cohort.

    The scale the cross-cohort cosine has to be read against. Without it a
    cosine of 0.4 is uninterpretable: it could be most of what this many assays
    can agree on, or very little.
    """
    names = sorted(blocks)
    rng = np.random.default_rng(seed)
    half = len(names) // 2
    out = []
    for _ in range(n_draw):
        perm = rng.permutation(names)
        a, b = perm[:half], perm[half:2 * half]
        out.append(cosine(pooled_weights({n: blocks[n] for n in a}, lam),
                          pooled_weights({n: blocks[n] for n in b}, lam)))
    return {"mean": float(np.mean(out)),
            "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)),
            "n_draws": n_draw, "half_size": half}


def one_direction(name, train_X, train_y, test_X, test_y, within, lam):
    """Fit on `train`, predict `test`, against the within-cohort ceiling."""
    res = {"train_assays": sorted(train_y), "test_assays": sorted(test_y)}
    for b in fb.BLOCKS:
        rho = fit_groups_predict_groups(
            fb.as_probe_blocks(train_X[b], train_y),
            fb.as_probe_blocks(test_X[b], test_y), lam=lam)
        ceil = {k: within[b][k] for k in rho}
        res[b] = {
            "transfer": boot(rho),
            "within_cohort_ceiling": boot(ceil),
            "retained": boot_gap(rho, ceil),      # transfer minus ceiling
            "per_assay": rho,
        }
    res["transfer_gaps"] = {
        "internal_minus_rich": boot_gap(res["internal"]["per_assay"],
                                        res["rich"]["per_assay"]),
        "internal_minus_geometry": boot_gap(res["internal"]["per_assay"],
                                            res["geometry"]["per_assay"]),
        "internal_minus_chem": boot_gap(res["internal"]["per_assay"],
                                        res["chem"]["per_assay"]),
    }

    print(f"\n  --- {name}: {len(train_y)} train -> {len(test_y)} test ---")
    print(f"  {'block':10s} {'transfer':>9s} {'ceiling':>9s} {'retained':>9s}  "
          f"{'transfer 95% CI':>18s}")
    for b in fb.BLOCKS:
        t, c, r = (res[b]["transfer"], res[b]["within_cohort_ceiling"],
                   res[b]["retained"])
        print(f"  {b:10s} {t['mean']:>+9.3f} {c['mean']:>+9.3f} "
              f"{r['mean']:>+9.3f}  [{t['lo']:+.3f}, {t['hi']:+.3f}]")
    for k, v in res["transfer_gaps"].items():
        print(f"    {k:26s} {v['mean']:>+8.3f}  [{v['lo']:+.3f}, {v['hi']:+.3f}]"
              f"{'' if v['lo'] * v['hi'] > 0 else '   SPANS ZERO'}")
    return res


def run_model(model, a, stab, other, panel):
    """stab / other / panel are (assays, captures dir, tm cache) triples."""
    loaded = {}
    for tag, (assays, caps, tm) in {"stab": stab, "other": other,
                                    "panel": panel}.items():
        if assays:
            loaded[tag] = fb.load_blocks(model, assays, caps,
                                         Path(tm.format(model=model)))

    (Xs, ys), (Xp, yp) = loaded["stab"], loaded["panel"]
    shared = set(ys) & set(yp)
    if shared:
        raise SystemExit(
            f"cohorts share short key(s) {sorted(shared)}; a cross-cohort "
            f"number over overlapping sets is not a transfer number")

    # The within-cohort ceilings: same phenotype, unseen protein.
    within_s = {b: leave_one_group_out(fb.as_probe_blocks(Xs[b], ys), lam=a.lam)
                for b in fb.BLOCKS}
    within_p = {b: leave_one_group_out(fb.as_probe_blocks(Xp[b], yp), lam=a.lam)
                for b in fb.BLOCKS}

    print(f"\n=== {model} ===")
    res = {"stability_to_panel5":
           one_direction("stability -> panel5", Xs, ys, Xp, yp, within_p, a.lam),
           "panel5_to_stability":
           one_direction("panel5 -> stability", Xp, yp, Xs, ys, within_s, a.lam)}

    if "other" in loaded:
        Xo, yo = loaded["other"]
        # The four non-stability held-out assays have no cohort of their own to
        # provide a ceiling, so the held-out-cohort LOAO number is used: it is
        # trained on the other fifteen, twelve of which are stability. That is
        # a mixed-phenotype ceiling and is labelled as one.
        within_all = {b: leave_one_group_out(
            fb.as_probe_blocks({**Xs[b], **Xo[b]}, {**ys, **yo}), lam=a.lam)
            for b in fb.BLOCKS}
        res["stability_to_heldout_other"] = one_direction(
            "stability -> other4", Xs, ys, Xo, yo, within_all, a.lam)
        res["stability_to_heldout_other"]["ceiling_note"] = (
            "leave-one-assay-out within the full 16-assay held-out cohort, so "
            "the ceiling itself is trained mostly on stability")

    # Is it the same direction, or two directions that both work?
    agree = {}
    for b in fb.BLOCKS:
        ws = pooled_weights(fb.as_probe_blocks(Xs[b], ys), a.lam)
        wp = pooled_weights(fb.as_probe_blocks(Xp[b], yp), a.lam)
        agree[b] = {
            "cosine_stability_vs_panel5": cosine(ws, wp),
            "split_half_null_stability": split_half_null(
                fb.as_probe_blocks(Xs[b], ys), a.lam),
            "split_half_null_panel5": split_half_null(
                fb.as_probe_blocks(Xp[b], yp), a.lam),
        }
    res["direction_agreement"] = agree

    print(f"\n  --- direction agreement (cosine between fitted weights) ---")
    print(f"  {'block':10s} {'across':>8s}  {'within stab':>22s}  "
          f"{'within panel5':>22s}")
    for b in fb.BLOCKS:
        g = agree[b]
        s, p = g["split_half_null_stability"], g["split_half_null_panel5"]
        print(f"  {b:10s} {g['cosine_stability_vs_panel5']:>+8.3f}  "
              f"{s['mean']:>+8.3f} [{s['lo']:+.2f},{s['hi']:+.2f}]  "
              f"{p['mean']:>+8.3f} [{p['lo']:+.2f},{p['hi']:+.2f}]")
    return res


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
    ap.add_argument("--out", default=str(W / "runs" / "cross_phenotype.json"))
    a = ap.parse_args()

    held = Cohort.load("heldout_assays")
    panel = Cohort.load("panel5_assays")
    held.verify()
    panel.verify()

    h_ok, h_missing = fb.complete_assays(held, a.heldout_captures, fb.MODELS)
    p_ok, p_missing = fb.complete_assays(panel, a.panel_captures, fb.MODELS)
    stab = [x for x in h_ok if STABILITY_MARK in x.id]
    other = [x for x in h_ok if STABILITY_MARK not in x.id]

    print(f"stability   {len(stab):3d} assays  (Tsuboyama cDNA-display proteolysis)")
    print(f"other       {len(other):3d} assays  (non-stability, held-out cohort)")
    print(f"panel5      {len(p_ok):3d} assays  (no stability assay by construction)")
    for tag, miss in (("heldout", h_missing), ("panel5", p_missing)):
        if miss:
            print(f"  EXCLUDED from {tag}: {len(miss)} not captured in every model")
            for k, v in sorted(miss.items()):
                print(f"    {k:44s} missing {', '.join(v)}")

    models = fb.MODELS if a.all else (a.model,)
    payload = {
        "cohorts": {"stability": [x.id for x in stab],
                    "heldout_other": [x.id for x in other],
                    "panel5": [x.id for x in p_ok]},
        "excluded": {"heldout": h_missing, "panel5": p_missing},
        "models": {m: run_model(
            m, a,
            (stab, a.heldout_captures, a.heldout_tm),
            (other, a.heldout_captures, a.heldout_tm),
            (p_ok, a.panel_captures, a.panel_tm)) for m in models},
    }

    artifacts.write_result(Path(a.out), payload, protocol={
        "question": "is there one shared mutation direction across phenotypes",
        "design": "fit on one cohort, predict a disjoint cohort; no GPU, the "
                  "captures already exist",
        "layer": "final trunk layer",
        "blocks": {b: fb.FEATURE_NAMES.get(b, "dz_vec, 128 pair channels")
                   for b in fb.BLOCKS},
        "lam": a.lam,
        "scaling": "features and target z-scored within each assay",
        "statistic": "Spearman on each test assay, meaned equally over assays",
        "ceiling": "leave-one-assay-out WITHIN the test cohort, same assays",
        "agreement": "cosine between pooled ridge weights, read against a "
                     "split-half null within each cohort",
        "interval": f"cluster bootstrap over assays, {N_BOOT} draws",
    })
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
