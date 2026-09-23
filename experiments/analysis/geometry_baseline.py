"""Is the trunk-over-emitted gap a fact about information, or about width?

The archived result compares a 128-channel internal block against ten emitted
features. Two objections follow, and they are different:

    RICHNESS   ten hand-chosen summaries of CA geometry are not "everything the
               model emits". A shell-resolved deformation profile, contact-map
               change and distance-matrix change might carry what they miss.
    CAPACITY   128 features fitted through ridge will beat 10 on many problems
               for reasons that have nothing to do with what a trunk knows.

This answers both on the captures already on disk. Richness: `geometry`, 37
prespecified features from the SAME coordinates (`analysis.emitted_geometry`),
fitted through the identical protocol. Capacity: every block is also read at
matched width, by projecting it onto its first d principal components with the
basis fitted on the TRAINING assays only -- so `internal` at d=10 meets
`output_rich` at its own full width of 10, and the whole rho-versus-d curve is
reported rather than one point.

WHAT WOULD FALSIFY THE HEADLINE. If `geometry` closes the gap, the claim was
about the poverty of the ten features. If `internal` at d=10 falls to the
emitted level, the claim was about capacity. Either outcome is reported here
in the same table as the archived number, which is recomputed alongside as a
check that nothing else moved.

Chemistry is carried throughout because it is model-independent: an internal
block that does not beat it is not evidence about the model.

    uv run python experiments/analysis/geometry_baseline.py --all
    uv run python experiments/analysis/geometry_baseline.py --all \
        --cohort panel5_assays --captures $W/runs/xmodel_panel5 \
        --tm-cache "$W/runs/tm_panel5_{model}.npz" \
        --out $W/runs/geometry_panel5.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_blocks as fb                                          # noqa: E402

from protein_interpretability import artifacts                       # noqa: E402
from protein_interpretability.analysis import statistics as st       # noqa: E402
from protein_interpretability.analysis.probes import (               # noqa: E402
    leave_one_group_out,
)
from protein_interpretability.analysis.transfer import (             # noqa: E402
    leave_one_group_out_reduced,
)
from protein_interpretability.collection import Cohort               # noqa: E402

W = fb.W
LAM = 10.0
DIMS = (1, 2, 4, 8, 10, 16, 32, 37, 64, 128)
N_BOOT = 10000


def boot(per_assay):
    """Interval over ASSAYS, the independent unit. One rho per assay, so the
    hierarchical stage has nothing to resample and is switched off rather than
    left on to do nothing."""
    p, lo, hi, k = st.cluster_bootstrap(
        {k: [v] for k, v in per_assay.items()}, n_boot=N_BOOT,
        hierarchical=False)
    return {"mean": p, "lo": lo, "hi": hi, "n_assays": k}


def boot_gap(a, b):
    p, lo, hi, k = st.paired_cluster_bootstrap(
        {k: [v] for k, v in a.items()}, {k: [v] for k, v in b.items()},
        n_boot=N_BOOT, hierarchical=False)
    return {"mean": p, "lo": lo, "hi": hi, "n_assays": k}


def run_model(model, a, assays):
    X, y = fb.load_blocks(model, assays, a.captures,
                          Path(a.tm_cache.format(model=model)))
    widths = {b: next(iter(X[b].values())).shape[1] for b in fb.BLOCKS}

    full = {b: leave_one_group_out(fb.as_probe_blocks(X[b], y), lam=a.lam)
            for b in fb.BLOCKS}

    # The capacity curve. Each point refits the basis inside every fold, so a
    # held-out assay never contributes to the directions it is scored on.
    curve = {b: {str(d): leave_one_group_out_reduced(
                     fb.as_probe_blocks(X[b], y), d=d, lam=a.lam)
                 for d in DIMS if d <= widths[b]}
             for b in fb.BLOCKS}

    names = sorted(full["internal"])
    res = {
        "widths": widths,
        "full": {b: boot(full[b]) for b in fb.BLOCKS},
        "per_assay": {b: full[b] for b in fb.BLOCKS},
        "gaps": {
            "internal_minus_rich": boot_gap(full["internal"], full["rich"]),
            "internal_minus_geometry": boot_gap(full["internal"], full["geometry"]),
            "geometry_minus_rich": boot_gap(full["geometry"], full["rich"]),
            "internal_minus_chem": boot_gap(full["internal"], full["chem"]),
        },
        "curve": {b: {d: boot(v) for d, v in curve[b].items()} for b in fb.BLOCKS},
        # Per-assay values for every curve point, so a paired gap can be taken
        # at ANY d after the fact. Without these the rho-versus-d curve carries
        # its own interval but no paired interval against the emitted blocks,
        # and the headline claim -- that a two-dimensional trunk probe beats all
        # 37 emitted features -- would be a comparison of two point estimates.
        "curve_per_assay": {b: dict(curve[b].items()) for b in fb.BLOCKS},
        # The two comparisons the capacity objection actually asks for: the
        # trunk read at the emitted blocks' own widths.
        "matched": {
            "at_10_internal_minus_rich": boot_gap(
                curve["internal"]["10"], full["rich"]),
            "at_37_internal_minus_geometry": boot_gap(
                curve["internal"]["37"], full["geometry"]),
            # The headline: the trunk at TWO components against the emitted
            # blocks at their own full width.
            "at_2_internal_minus_geometry": boot_gap(
                curve["internal"]["2"], full["geometry"]),
            "at_2_internal_minus_rich": boot_gap(
                curve["internal"]["2"], full["rich"]),
        },
        "wins": {
            "internal_over_rich": sum(full["internal"][k] > full["rich"][k]
                                      for k in names),
            "internal_over_geometry": sum(full["internal"][k] > full["geometry"][k]
                                          for k in names),
            "internal_at_10_over_rich": sum(
                curve["internal"]["10"][k] > full["rich"][k] for k in names),
            "internal_at_2_over_geometry": sum(
                curve["internal"]["2"][k] > full["geometry"][k] for k in names),
            "n": len(names),
        },
    }

    print(f"\n=== {model}  ({len(names)} assays, lam={a.lam:g}) ===")
    print(f"{'block':10s} {'width':>6s} {'rho':>8s}  {'95% CI':>18s}")
    for b in fb.BLOCKS:
        f = res["full"][b]
        print(f"{b:10s} {widths[b]:>6d} {f['mean']:>+8.3f}  "
              f"[{f['lo']:+.3f}, {f['hi']:+.3f}]")
    print("  -- the trunk read at the emitted blocks' own widths --")
    for d, b in (("10", "rich"), ("37", "geometry")):
        c = res["curve"]["internal"][d]
        print(f"{'internal@'+d:10s} {int(d):>6d} {c['mean']:>+8.3f}  "
              f"[{c['lo']:+.3f}, {c['hi']:+.3f}]   vs {b}")
    print("  -- paired gaps --")
    for k, v in {**res["gaps"], **res["matched"]}.items():
        print(f"  {k:34s} {v['mean']:>+8.3f}  [{v['lo']:+.3f}, {v['hi']:+.3f}]"
              f"{'' if v['lo'] * v['hi'] > 0 else '   SPANS ZERO'}")
    w = res["wins"]
    print(f"  wins  internal>rich {w['internal_over_rich']}/{w['n']}   "
          f"internal>geometry {w['internal_over_geometry']}/{w['n']}   "
          f"internal@10>rich {w['internal_at_10_over_rich']}/{w['n']}")

    # Nothing else moved: the unreduced internal and rich numbers must still be
    # the archived ones. A drift here invalidates the comparison above, because
    # it would mean the blocks are not the blocks the result was reported on.
    ref_path = Path(a.reference.format(model=model)) if a.reference else None
    if ref_path is not None and ref_path.exists():
        ref = json.loads(ref_path.read_text())["predictors"]
        # Compared over the assays ACTUALLY analysed. Under --limit that is a
        # subset, and the check is reported as partial rather than quietly
        # passing on three of sixteen.
        pairs = [(full[b][k], ref[r]["per_assay"][k])
                 for b, r in (("internal", "internal_vec"), ("rich", "output_rich"))
                 for k in names if k in ref[r]["per_assay"]]
        worst = max(abs(x - y) for x, y in pairs) if pairs else float("nan")
        complete = len(pairs) == 2 * len(names)
        res["archive_check"] = {"reference": ref_path.name,
                                "worst_abs_diff": worst,
                                "assays_checked": len(pairs) // 2,
                                "complete": complete}
        print(f"  vs {ref_path.name}: worst per-assay difference {worst:.2e}"
              f"{'' if complete else f'  (PARTIAL: {len(pairs)//2}/{len(names)})'}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=fb.MODELS, default="boltz2")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cohort", default="heldout_assays")
    ap.add_argument("--captures", default=str(W / "runs" / "xmodel_layers"))
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N assays only -- a smoke test, never a result")
    ap.add_argument("--reference",
                    default=str(W / "runs" / "transfer_heldout16_{model}.json"))
    ap.add_argument("--tm-cache",
                    default=str(W / "runs" / "tm_heldout16_{model}.npz"))
    ap.add_argument("--out", default=str(W / "runs" / "geometry_heldout16.json"))
    a = ap.parse_args()

    cohort = Cohort.load(a.cohort)
    cohort.verify()
    assays, missing = fb.complete_assays(cohort, a.captures, fb.MODELS)
    print(f"{cohort.name}: {len(cohort)} assays, inputs verified")
    if missing:
        print(f"  EXCLUDED {len(missing)} not captured in every model:")
        for k, v in sorted(missing.items()):
            print(f"    {k:44s} missing {', '.join(v)}")
    if a.limit:
        assays = assays[:a.limit]
        print(f"  SMOKE TEST: {len(assays)} assays only, not a result")

    models = fb.MODELS if a.all else (a.model,)
    payload = {"cohort": cohort.name,
               "n_assays": len(assays),
               "assays": [x.id for x in assays],
               "excluded": missing,
               "models": {m: run_model(m, a, assays) for m in models}}

    if not a.limit:
        artifacts.write_result(Path(a.out), payload, protocol={
            "question": "is the trunk-over-emitted gap richness or capacity",
            "design": "leave-one-assay-out, assay is the independent unit",
            "layer": "final trunk layer",
            "blocks": {b: fb.FEATURE_NAMES.get(b, "dz_vec, 128 pair channels")
                       for b in fb.BLOCKS},
            "reduction": "PCA fitted on TRAINING assays only, per fold",
            "dims": list(DIMS),
            "lam": a.lam,
            "scaling": "features and target z-scored within each assay",
            "statistic": "Spearman on the held-out assay, meaned over assays",
            "interval": f"cluster bootstrap over assays, {N_BOOT} draws",
            "excluded_by_design": "trunk distogram -- a Pairformer head, not a "
                                  "product of the structure module",
        })
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
