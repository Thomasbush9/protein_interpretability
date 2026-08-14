"""Internal versus output in three architectures, using the FULL pair vector.

This replaces the `deep2_*` route, which was structurally incapable of the
comparison it appeared to make. Those archives store `dz_site` as a per-layer
NORM -- shape (n, L), not (n, L, 128) -- so any probe built on them sees how far
the pair row moved and never which way. Run that way, Boltz-2's internal side
scores +0.468, well under the +0.63 the same model reaches when the direction is
available, and the deficit is an artefact of the archive rather than a property
of the model.

The `xm_*` archives carry `dz_vec` at (n, L, 128) for all three models, plus
each model's own pLDDT, so the whole comparison closes inside one family.

  internal   ridge on the 128-dim pair-row difference at the FINAL trunk layer.
             The final layer is not chosen for being best -- it is the tensor
             the structure module is conditioned on, which is what makes the
             comparison against that module's output meaningful.
  output     the model's own pLDDT, chain mean and at the mutated residue. This
             is the strongest form of the objection: not "the coordinates do not
             show it" but "your own confidence head already tells you".

Every comparison is PAIRED WITHIN MODEL on identical rows and identical
variants, with residue positions held out so no site appears on both sides.
Layer counts, distogram grids and alignment handling differ between models, so
nothing here ranks them; the question is only whether the internal-over-output
gap survives a change of architecture.

The independent unit is the assay and there are four, so intervals are wide.
That is the honest resolution and the reason this cannot carry the paper on its
own -- the twelve-protein Boltz-2 result does that.

  sbatch analysis.sbatch analyze_xmodel_io.py --dir $R/runs/ --out $R/runs/xmodel_io_vec.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402
import pi_stats  # noqa: E402
from compare_internal_output import fit_ridge_block, grouped_split  # noqa: E402

MODELS = ("boltz2", "of3", "protenix")
NICE = {"boltz2": "Boltz-2", "of3": "OpenFold3", "protenix": "Protenix"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--run", default="r1", help="xm replicate to use")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assays = sorted({os.path.basename(f).split(f"xm_boltz2_{a.run}_")[1][:-4]
                     for f in glob.glob(f"{a.dir}/xm_boltz2_{a.run}_*.npz")})
    print(f"{len(assays)} assays x {len(MODELS)} models, replicate {a.run}\n")

    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nlay, drift, ref_mut = {}, {}, {}
    for asy in assays:
        for m in MODELS:
            f = Path(a.dir) / f"xm_{m}_{a.run}_{asy}.npz"
            if not f.exists():
                raise SystemExit(f"missing {f}")
            d = pi_archive.load_capture(f)
            mut = [str(x) for x in d["mutant"]]
            # Identical variants across models, or the comparison is between
            # different experiments wearing the same label.
            if asy in ref_mut and ref_mut[asy] != mut:
                raise SystemExit(
                    f"{asy}: {m} variant list differs from the first model's. "
                    f"A cross-model row here would compare different mutations.")
            ref_mut.setdefault(asy, mut)

            z = np.asarray(d.field("dz_vec"), float)              # (n, L, 128)
            nlay[m] = z.shape[1]
            drift[m] = (float(d["capture_drift"])
                        if "capture_drift" in d.files else None)
            y = np.asarray(d["score"], float)
            pos = np.asarray(d["pos"])
            pl = np.asarray(d["plddt_mean"], float)
            pls = np.asarray(d["plddt_site"], float)
            pl = pl.mean(-1) if pl.ndim > 1 else pl
            pls = pls.mean(-1) if pls.ndim > 1 else pls

            rng = np.random.default_rng(0)
            for s in range(a.splits):
                tr, te = grouped_split(pos, 0.25, rng)
                if te.sum() < 8 or tr.sum() < 20:
                    continue
                rho = fit_ridge_block(z[:, -1, :], y, pos, tr, te, rng)[0]
                per[m]["internal (128-dim, final layer)"][asy].append(rho)
                per[m]["pLDDT"][asy].append(pi_stats.spearman(pl[te], y[te]))
                per[m]["pLDDT@site"][asy].append(pi_stats.spearman(pls[te], y[te]))

    ORDER = ["internal (128-dim, final layer)", "pLDDT", "pLDDT@site"]
    print(f"Spearman vs measured stability, held-out positions "
          f"({len(assays)} assays x {a.splits} splits)\n")
    print(f"{'model':10s}{'layers':>7s}" + "".join(f"{k:>34s}" for k in ORDER))
    out = {"models": list(MODELS), "assays": assays, "splits": a.splits,
           "run": a.run, "layers": {m: int(nlay[m]) for m in MODELS},
           "capture_drift": drift, "order": ORDER,
           "spearman": {}, "internal_minus": {}}
    for m in MODELS:
        row = ""
        out["spearman"][m] = {}
        for k in ORDER:
            flat = np.concatenate([np.asarray(v) for v in per[m][k].values()])
            out["spearman"][m][k] = float(np.nanmean(flat))
            row += f"{np.nanmean(flat):>+34.3f}"
        print(f"{m:10s}{nlay[m]:>7d}{row}")

    print(f"\nInternal minus each output, ASSAY as the independent unit "
          f"({len(assays)} assays, so intervals are wide)\n")
    for base in ("pLDDT", "pLDDT@site"):
        out["internal_minus"][base] = {}
        print(f"  vs {base}")
        for m in MODELS:
            pt, lo, hi, nk = pi_stats.paired_cluster_bootstrap(
                dict(per[m][ORDER[0]]), dict(per[m][base]),
                n_boot=a.n_boot, seed=0)
            wins = sum(1 for asy in per[m][ORDER[0]]
                       for x, yv in zip(per[m][ORDER[0]][asy], per[m][base][asy])
                       if np.isfinite(x) and np.isfinite(yv) and x > yv)
            tot = sum(len(v) for v in per[m][ORDER[0]].values())
            flag = "" if (np.isfinite(lo) and lo > 0) else "   (includes zero)"
            out["internal_minus"][base][m] = {
                "gap": float(pt), "ci_lo": float(lo), "ci_hi": float(hi),
                "wins": int(wins), "splits": int(tot), "n_assays": int(nk)}
            print(f"    {NICE[m]:10s} gap {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
                  f"  wins {wins}/{tot}{flag}")
        print()

    print("  PAIRED WITHIN-MODEL comparisons. Not a ranking: layer counts,\n"
          "  distogram grids and alignment handling all differ.")
    out["protocol"] = pi_protocol.protocol(
        script="analyze_xmodel_io.py",
        design="within-assay, position-grouped splits; paired WITHIN model",
        layer=pi_protocol.layers("final"),
        features=pi_protocol.features("dz_vec, final-layer pair row", 128),
        source=f"{a.dir}/xm_<model>_{a.run}_<assay>.npz", n_assays=len(assays),
        note="TM omitted: tmtools is not installed and substituting another "
             "metric under that name would fabricate a number")
    pi_archive.write_result(a.out, out, protocol=out.pop("protocol"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
