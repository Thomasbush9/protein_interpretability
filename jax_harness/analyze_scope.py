"""Is one averaged pair row too myopic a view of what a mutation does?

`dz_site` is the mutated residue's pair row averaged over partners: one
128-vector. That is not a single residue's activation -- it is that residue's
relationship to the entire chain -- but it discards two things, and the fair
question is whether either of them carries stability information.

  the profile within the row   Averaging over partner j collapses WHICH
                               partners moved. `gym3` archives the unaveraged
                               row `dz_row[r, j, :]`, so this is testable
                               directly: does the (partners x channels) object
                               beat its own average?

  everything off the row       Pairs not involving the mutated residue are
                               dropped entirely. No archived tensor holds them,
                               but the divergence features do: `kl_site` is
                               averaged over sampled pairs that TOUCH the
                               mutated residue while `kl_glob` is averaged over
                               ALL sampled pairs, most of which do not. Their
                               relative predictive value is a proxy for how much
                               lives off the row.

Three views of the row, all under the same estimator and splits so the
comparison is about information rather than protocol:

  averaged      dz_site, 128 dims -- what every other result on this page uses
  profile       per-partner norms ||dz_row[r, j, :]||, one number per residue:
                how much each partner moved, with the channel detail thrown away
  full          the whole (partners x channels) row, flattened

If `averaged` is not beaten, the collapse costs nothing and the myopia is
apparent rather than real. If `full` wins clearly, the spatial detail matters and
every component in this report is an average over something informative.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_stats  # noqa: E402
from compare_internal_output import fit_ridge_block, grouped_split  # noqa: E402

EPS = 1e-9


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/runs/"
    ap.add_argument("--glob", default=R + "gym3_*.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res, dims = {}, {}
    for f in sorted(glob.glob(a.glob)):
        d = np.load(f, allow_pickle=True)
        name = Path(f).stem.split("_", 1)[1].split("_")[0]
        y, pos = np.asarray(d["score"], float), np.asarray(d["pos"])
        row = np.asarray(d["dz_row"], float)[:, -1, :, :]      # (n, N, C) last kept layer
        blocks = {
            "row, averaged over partners (dz_site)":
                np.asarray(d["dz_site"], float)[:, -1, :],
            "row, per-partner magnitude only":
                np.linalg.norm(row, axis=-1),
            "row, full (partners x channels)":
                row.reshape(len(row), -1),
            "KL at the mutated site only (64 layers)":
                np.asarray(d["kl_site"], float),
            "KL over the whole protein (64 layers)":
                np.asarray(d["kl_glob"], float),
            "both KL views (128)":
                np.concatenate([np.asarray(d["kl_site"], float),
                                np.asarray(d["kl_glob"], float)], axis=1),
        }
        dims[name] = {k: int(v.shape[1]) for k, v in blocks.items()}
        for bn, X in blocks.items():
            vals = []
            for s in range(a.seeds):
                rng = np.random.default_rng(s)
                tr, te = grouped_split(pos, a.frac, rng)
                vals.append(fit_ridge_block(X, y, pos, tr, te, rng)[0])
            res.setdefault(bn, {})[name] = float(np.nanmean(vals))
        print(f"   {name:8s} N={row.shape[1]:3d}  " +
              "  ".join(f"{v}" for v in dims[name].values()), flush=True)

    names = sorted(res["row, averaged over partners (dz_site)"])
    print(f"\nHeld-out Spearman, identical protocol ({len(names)} assays)\n")
    out = {"dims": dims, "blocks": {}}
    for bn, per in res.items():
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(
            {n: [per[n]] for n in names}, n_boot=10000, seed=0, hierarchical=False)
        out["blocks"][bn] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                             "per_assay": per}
        print(f"   {bn:42s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\nPaired against the averaged row -- does the discarded detail matter?\n")
    ref = "row, averaged over partners (dz_site)"
    gaps = {}
    for bn in res:
        if bn == ref:
            continue
        pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(
            {n: [res[bn][n]] for n in names}, {n: [res[ref][n]] for n in names},
            n_boot=10000, seed=0, hierarchical=False)
        wins = sum(1 for n in names if res[bn][n] > res[ref][n])
        gaps[bn] = {"gap": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins,
                    "n": len(names)}
        flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
        print(f"   {bn:42s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"{wins}/{len(names)}{flag}")
    out["vs_averaged_row"] = gaps

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
