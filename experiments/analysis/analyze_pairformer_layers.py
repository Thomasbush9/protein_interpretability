"""The layer trace: where in the trunk the mutation signal appears.

Reads a capture written by `collect_pairformer_layers.py` and asks, at each
Pairformer layer, how well the pair row's movement tracks the assay score. That
curve is the "derived layer trace" the plan's reference slice is meant to
reproduce, and it is the shape behind the depth results in the report.

Model-free by construction: it reads an artifact and imports no backend, so it
runs on a login node. `require_meta=True` because a capture that cannot state
its own protocol is one whose reduction is unknown, and a norm read as a vector
is the failure that returned +0.468.

    uv run python experiments/analysis/analyze_pairformer_layers.py \\
        --capture $W/runs/slice_pairformer.npz --out $W/runs/slice_layers.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

# The container's interpreter does not have this package installed -- jobs exec
# a bare `python` inside the mosaic image -- so the checkout's src/ is located
# the same way the pi_* shims locate it. Must precede the package imports.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from protein_interpretability import artifacts
from protein_interpretability.analysis import statistics as st
from protein_interpretability.experiments import protocol as P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-with", help="an archived capture to diff against")
    ap.add_argument("--band", type=float, default=3.9e-2,
                    help="mean |dz| difference between two runs of the SAME "
                         "collection, measured 2026-08-18 on this assay. "
                         "Comparing against zero would call the capture path "
                         "broken; comparing against this says whether it "
                         "reproduces as well as the harness reproduces itself.")
    ap.add_argument("--outlier-factor", type=float, default=3.0)
    a = ap.parse_args()

    cap = artifacts.load_capture(a.capture, require_meta=True,
                                 require_vectors=True)
    dz = np.asarray(cap.field("dz_site"), float)        # [n, n_layers, 128]
    score = np.asarray(cap.field("score"), float)
    mutants = [str(m) for m in cap.field("mutant")]
    n, n_layers, width = dz.shape
    print(f"{Path(a.capture).name}: {n} variants x {n_layers} layers x {width}")

    # The trace. |dz| at each layer against the assay score -- a magnitude,
    # deliberately: this asks WHERE the signal is, not which direction carries
    # it, and the direction is what the shared basis is for.
    per_layer = [float(st.spearman(np.linalg.norm(dz[:, l, :], axis=-1), score))
                 for l in range(n_layers)]
    finite = [v for v in per_layer if np.isfinite(v)]
    peak = int(np.nanargmax(np.abs(per_layer))) if finite else -1

    result = {
        "n_variants": n, "n_layers": n_layers, "width": width,
        "mutants": mutants,
        "layer_trace": per_layer,
        "peak_layer": peak,
        "peak_spearman": per_layer[peak] if peak >= 0 else None,
        "final_layer_spearman": per_layer[-1],
    }
    print(f"  peak at layer {peak}: rho={per_layer[peak]:+.3f}"
          f"   final layer: {per_layer[-1]:+.3f}")

    # ---- does this reproduce the harness? ---------------------------------
    if a.compare_with:
        ref = artifacts.load_capture(a.compare_with, require_vectors=True)
        ref_mut = {str(m): i for i, m in enumerate(ref.field("mutant"))}
        rows = [ref_mut[m] for m in mutants if m in ref_mut]
        if len(rows) != len(mutants):
            missing = [m for m in mutants if m not in ref_mut]
            raise SystemExit(f"{missing} are not in {a.compare_with}; the two "
                             f"captures do not describe the same variants")
        ref_dz = np.asarray(ref.field("dz_site"), float)[rows]
        delta = np.abs(ref_dz - dz)
        per_variant = delta.mean(axis=(1, 2))

        # Compared against a BAND, not against zero. Two runs of this same
        # collection differ by mean 3.9e-02 on dz_site -- a trunk in float32
        # over 64 layers and 4 recycles is not reproducible across jobs, and
        # measuring that was the only way to know what "reproduces" can mean
        # here. Exactness is not available; agreement within the band is.
        outliers = {m: float(v) for m, v in zip(mutants, per_variant)
                    if v > a.band * a.outlier_factor}
        result["vs_archive"] = {
            "archive": str(a.compare_with),
            "n_matched": len(rows),
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "corr": float(np.corrcoef(ref_dz.ravel(), dz.ravel())[0, 1]),
            "run_to_run_band": a.band,
            "per_variant_mean_abs": {m: float(v)
                                     for m, v in zip(mutants, per_variant)},
            "outliers": outliers,
            "verdict": ("within the collection's own run-to-run band"
                        if not outliers else
                        f"{len(outliers)} variant(s) beyond "
                        f"{a.outlier_factor}x the band"),
        }
        print(f"  vs archive: mean {delta.mean():.3e}  max {delta.max():.3e}  "
              f"corr {result['vs_archive']['corr']:.6f}")
        print(f"  run-to-run band {a.band:.1e}: "
              f"{result['vs_archive']['verdict']}")
        for m, v in outliers.items():
            print(f"    OUTLIER {m}: {v:.4f}, {v / a.band:.1f}x the band")

    artifacts.write_result(
        Path(a.out), result,
        protocol=P.protocol(
            script=Path(__file__).name,
            design="per-layer Spearman of |dz_site| against the assay score",
            layer=P.layers("all", n_layers=n_layers),
            features=P.features("dz_site row magnitude", width, kept=1),
            source=str(a.capture),
            n_assays=1,
            note="a magnitude, so this locates the signal in depth rather than "
                 "identifying its direction; the direction is the basis's job",
        ))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
