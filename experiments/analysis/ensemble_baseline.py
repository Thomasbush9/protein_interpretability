"""Does an ENSEMBLE output baseline close the internal-minus-output gap?

The analysis half of Gate 2. `exp_ensemble.py` collected, for a prespecified
balanced subset of heldout16, four diffusion draws per sequence with the MSA and
the trunk state held fixed. This turns those draws into output feature blocks
and re-runs the identical leave-one-assay-out protocol against the internal
probe on the same rows.

  single      draw 0 only -- the existing protocol, recomputed here so the
              comparison is like-for-like rather than read from an archive that
              used slightly different rows.
  ens_mean    each draw's 37 geometry features computed against THAT draw's
              wild type, then averaged over draws. The primary ensemble block.
  ens_mean_sd the same, concatenated with the across-draw standard deviation
              (74 features). Secondary: it can only help, so if it does not,
              feature count is not what was limiting the baseline.

FEATURES ARE AVERAGED, NEVER COORDINATES. Two draws land in different frames;
averaging coordinates would shrink every structure toward its own mean and
manufacture agreement. Every geometry feature is computed inside one draw, from
superposed structures, before anything is averaged.

The internal side is not re-collected: the trunk is deterministic given the MSA
and recycle count, so the archived `xm_boltz2_r1_*` pair rows ARE the internal
representation for these same variants.

    uv run python experiments/analysis/ensemble_baseline.py \
        --out $W/runs/ensemble_baseline.json
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax_harness"))

import geom                                                    # noqa: E402
import pi_archive                                              # noqa: E402
import pi_protocol                                             # noqa: E402
import pi_stats                                                # noqa: E402
from protein_interpretability import artifacts                 # noqa: E402
from protein_interpretability.analysis.emitted_geometry import (  # noqa: E402
    GEOMETRY_FEATURES, geometry_matrix,
)
from protein_interpretability.analysis.probes import (         # noqa: E402
    leave_one_group_out,
)

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
LAM = 10.0


def rmsd(x, y):
    A = np.asarray(x, float) - np.asarray(x, float).mean(0)
    B = np.asarray(y, float) - np.asarray(y, float).mean(0)
    return float(np.sqrt(
        (np.linalg.norm(A @ geom.kabsch(A, B).T - B, axis=1) ** 2).mean()))


def draw_block(ca_d, ca_wt_d, plddt_d, plddt_site_d, pos):
    """The 37 geometry features for ONE draw, against that draw's wild type."""
    tm = np.array([geom.tm_score(np.asarray(c, float),
                                 np.asarray(ca_wt_d, float)) for c in ca_d])
    return geometry_matrix(ca_d, ca_wt_d, tm, plddt_d, plddt_site_d, pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(W / "runs/ensemble/ens_boltz2_*.npz"))
    ap.add_argument("--captures", default=str(W / "runs/xmodel_layers"))
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"no ensemble archives matched {a.glob}")

    internal, blocks, target, diag = {}, {"single": {}, "ens_mean": {},
                                          "ens_mean_sd": {}}, {}, {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        assay = str(d["assay"])
        key = assay.split("_")[0]
        ca, ca_wt = d["ca"], d["ca_wt"]              # (n,D,R,3), (D,R,3)
        pl, pls = d["plddt"], d["plddt_site"]        # (n,D)
        pos, y = np.asarray(d["pos"]), np.asarray(d["score"], float)
        n, D = ca.shape[0], ca.shape[1]

        per_draw = [draw_block(ca[:, k], ca_wt[k], pl[:, k], pls[:, k], pos)
                    for k in range(D)]
        S = np.stack(per_draw)                       # (D, n, 37)
        blocks["single"][key] = S[0]
        blocks["ens_mean"][key] = S.mean(0)
        blocks["ens_mean_sd"][key] = np.concatenate([S.mean(0), S.std(0)], 1)

        cap = artifacts.load_capture(
            Path(a.captures) / f"xm_boltz2_r1_{assay}.npz",
            require_vectors=True)
        X = cap.pair_row(-1)
        m = [list(cap.field("mutant")).index(mm) for mm in
             [str(x) for x in d["mutant"]]]
        internal[key] = X[m]
        target[key] = y

        # The scale everything here has to clear: how far apart two draws of
        # the SAME sequence are, superposed.
        wt_spread = float(np.mean([rmsd(ca_wt[i], ca_wt[j])
                                   for i in range(D) for j in range(i + 1, D)]))
        mut_wt = float(np.mean([rmsd(ca[v, 0], ca_wt[0]) for v in range(n)]))
        atoms_differ = int((np.asarray(d["atoms_mut"])
                            != int(d["atoms_wt"])).sum())
        diag[key] = {
            "n_variants": n, "draws": D,
            "wt_draw_rmsd": wt_spread,
            "mut_vs_wt_rmsd_draw0": mut_wt,
            "signal_over_sampler_noise": mut_wt / (wt_spread + 1e-9),
            "variants_with_atom_count_differing_from_wt": atoms_differ,
        }
        print(f"{key:8s} n={n:4d} draws={D}  WT draw-to-draw {wt_spread:.3f} A"
              f"   mut vs WT {mut_wt:.3f} A"
              f"   ratio {mut_wt / (wt_spread + 1e-9):.2f}"
              f"   atoms differ {atoms_differ}/{n}")

    names = sorted(internal)
    print(f"\n{len(names)} assays, leave-one-assay-out, lam={a.lam:g}\n")

    rho = {"internal": leave_one_group_out(
        {k: {"X": internal[k], "y": target[k]} for k in names}, lam=a.lam)}
    for b in ("single", "ens_mean", "ens_mean_sd"):
        rho[b] = leave_one_group_out(
            {k: {"X": blocks[b][k], "y": target[k]} for k in names}, lam=a.lam)

    ORDER = ["internal", "single", "ens_mean", "ens_mean_sd"]
    print(f"{'assay':9s}" + "".join(f"{k:>14s}" for k in ORDER))
    for k in names:
        print(f"{k:9s}" + "".join(f"{rho[b][k]:>+14.3f}" for b in ORDER))
    print(f"{'mean':9s}" + "".join(
        f"{np.mean([rho[b][k] for k in names]):>+14.3f}" for b in ORDER))

    summary, gaps = {}, {}
    for b in ORDER:
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(
            {k: [rho[b][k]] for k in names}, n_boot=10000, seed=0,
            hierarchical=False)
        summary[b] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                      "per_assay": rho[b]}
    print("\npaired gaps (assay is the unit)\n")
    for aa, bb in (("internal", "single"), ("internal", "ens_mean"),
                   ("internal", "ens_mean_sd"), ("ens_mean", "single")):
        pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(
            {k: [rho[aa][k]] for k in names},
            {k: [rho[bb][k]] for k in names},
            n_boot=10000, seed=0, hierarchical=False)
        wins = sum(rho[aa][k] > rho[bb][k] for k in names)
        gaps[f"{aa} - {bb}"] = {"gap": pt, "ci_lo": lo, "ci_hi": hi,
                                "wins": wins, "n": len(names)}
        flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
        print(f"   {aa:10s} - {bb:12s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"{wins}/{len(names)}{flag}")

    proto = pi_protocol.protocol(
        script="ensemble_baseline.py",
        design="leave-one-assay-out on a prespecified balanced subset of "
               "heldout16; internal (archived pair rows, deterministic trunk) "
               "against single-draw and diffusion-ensemble output geometry on "
               "identical rows",
        layer=pi_protocol.layers("final"),
        features={"internal": pi_protocol.features("dz_vec final pair row", 128),
                  "geometry": pi_protocol.features(
                      "emitted geometry", len(GEOMETRY_FEATURES))},
        source=a.glob, n_assays=len(names), lam=a.lam,
        aggregation="features per draw against that draw's WT, then averaged; "
                    "coordinates never averaged across draws",
        common_random_numbers="not claimed; atom counts differ between WT and "
                              "mutant, recorded per assay in diagnostics")
    pi_archive.write_result(a.out, {"summary": summary, "gaps": gaps,
                                    "diagnostics": diag, "assays": names},
                            protocol=proto, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
