"""Pool the composition test: does the linearisation predict the real response?

`exp_compose` produces, per assay, two predictions of the archived `dz_site`
trajectory -- one re-seeded from the archive at every layer, one propagated
freely from layer 0 -- and scores both against what the model actually did.
This pools them across assays and states the result in the terms the rest of the
study uses.

The two curves answer different questions and the gap between them is the
informative part:

  one-step   high cosine means each individual layer's Jacobian describes that
             layer's action on a real mutation difference. Errors cannot
             accumulate here, so this is a test of the linearisation alone.

  free       high cosine means 63 chained Jacobians reproduce the whole
             trajectory. This is the composition claim, and it also carries the
             uniform-row reconstruction of the initial tangent, so it degrades
             for two reasons at once.

If one-step is high and free decays, the per-layer operators are right and
something is lost in chaining or in the row reconstruction -- which bounds how
far the composed picture can be pushed without saying the per-layer numbers are
wrong. If one-step is also low, the linearisation itself does not describe the
response and the descriptive results need re-reading.

  sbatch analysis.sbatch analyze_compose.py --glob '../runs/comp_*.npz' \
      --out ../runs/comp_pooled.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"no files matched {a.glob}")
    names, S = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        names.append(str(d["assay"]).split("_")[0])
        S.append(json.loads(str(d["stats"])))
    L = len(S[0]["one_step"]["cosine"])
    print(f"{len(names)} assays: {', '.join(names)}\n{L} layers\n")

    out = {"assays": names, "layers": L}
    for tag in ("one_step", "free"):
        cos = np.array([s[tag]["cosine"] for s in S])          # [A, L]
        err = np.array([s[tag]["rel_err"] for s in S])
        out[tag] = {
            "cosine": cos.mean(0).tolist(),
            "cosine_min_assay": cos.min(0).tolist(),
            "rel_err": err.mean(0).tolist(),
            "pc": {},
        }
        for c in S[0][tag]["pc"]:
            r = np.array([s[tag]["pc"][c]["r"] for s in S])
            sl = np.array([s[tag]["pc"][c]["slope"] for s in S])
            out[tag]["pc"][c] = {"r": r.mean(0).tolist(),
                                 "slope": sl.mean(0).tolist()}

    print("full dz_site vector, mean over assays\n")
    print(f"  {'layer':>5s} {'one-step cos':>13s} {'relerr':>8s} "
          f"{'free cos':>10s} {'relerr':>8s}")
    for li in list(range(1, L, 8)) + [L - 1]:
        print(f"  {li:5d} {out['one_step']['cosine'][li]:13.3f} "
              f"{out['one_step']['rel_err'][li]:8.3f} "
              f"{out['free']['cosine'][li]:10.3f} "
              f"{out['free']['rel_err'][li]:8.3f}")

    print("\nPC2 coordinate: correlation of predicted with actual, and the "
          "slope\n(1.0 would be a perfectly calibrated magnitude)\n")
    print(f"  {'layer':>5s} {'one-step r':>11s} {'slope':>8s} "
          f"{'free r':>9s} {'slope':>8s}")
    for li in list(range(1, L, 8)) + [L - 1]:
        print(f"  {li:5d} {out['one_step']['pc']['PC2']['r'][li]:11.3f} "
              f"{out['one_step']['pc']['PC2']['slope'][li]:8.3f} "
              f"{out['free']['pc']['PC2']['r'][li]:9.3f} "
              f"{out['free']['pc']['PC2']['slope'][li]:8.3f}")

    o1 = float(np.mean(out["one_step"]["cosine"][1:]))
    fr = float(np.mean(out["free"]["cosine"][1:]))
    out["one_step_mean_cosine"] = o1
    out["free_mean_cosine"] = fr
    out["free_final_cosine"] = float(out["free"]["cosine"][-1])
    print(f"\nmean cosine over layers 1-{L-1}: one-step {o1:.3f}, free {fr:.3f}")
    print(f"free-running cosine at the last layer: "
          f"{out['free_final_cosine']:.3f}")

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
