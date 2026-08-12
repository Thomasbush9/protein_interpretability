"""Three challenges to the PC2 result, answered with numbers.

1. ONE RANKING ACROSS PROTEINS, NOT TWELVE WITHIN THEM. Every correlation in
   this project is computed inside an assay and then averaged over assays. That
   answers "can the score rank the variants of this protein", which is the
   ProteinGym convention, but it is silent on whether one PC2 score means the
   same thing in two different proteins. The stronger question is whether a
   SINGLE ranking over all ~3,000 variants at once holds up.

   The complication is that DMS scores are not comparable across assays -- they
   differ in units and dynamic range -- so a raw pooled correlation partly
   measures which protein a row came from. Both are computed: pooled on raw
   targets, and pooled after z-scoring the target within each assay, which
   removes the assay offset without touching the within-assay ordering.

2. IS IT A SUBSPACE OF THE MODEL AT ALL? The basis is built from differences
   z_mut - z_wt, which the model never computes; we do. It is a genuine linear
   subspace of the model's pair-channel space, but it is an observer's
   construct on those coordinates, not a module the model allocates. One thing
   that CAN be tested is whether it is an artifact of using wild type as the
   reference: since z_a - z_b = dz_a - dz_b, the variant-to-variant difference
   space is exactly the mean-centred version of the dz space. If the centred and
   uncentred bases agree, the wild-type anchor is not what creates the
   direction.

3. PC2 IS A MIXTURE. It correlates with DMS at -0.65 but also with substitution
   volume at -0.53 and with the width change at +0.54. Calling it "the stability
   axis" overstates a direction that is entangled with chemistry. Reported here:
   the full correlation profile rather than the flattering rows, and the partial
   correlation with DMS holding all seventeen chemistry descriptors fixed.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_chem  # noqa: E402
import pi_basis  # noqa: E402
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402
import pi_stats  # noqa: E402

EPS = 1e-9
N_PC = 4


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/runs/"
    ap.add_argument("--glob", default=R + "gym2s_*.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = {}
    for f in sorted(glob.glob(a.glob)):
        d = np.load(f, allow_pickle=True)
        A[Path(f).stem.split("_", 1)[1].split("_")[0]] = {
            "X": np.asarray(d["dz_site"], float)[:, -1, :],
            "y": np.asarray(d["score"], float),
            "C": pi_chem.chem_matrix([str(m) for m in d["mutant"]]),
            "kl": np.asarray(d["kl_glob"], float)[:, -1],
            "dsd": np.asarray(d["dsd_glob"], float)[:, -1],
            "spread": np.asarray(d["spread_glob"], float)[:, -1],
            "shift": np.asarray(d["shift_glob"], float)[:, -1]}
    names = sorted(A)
    print(f"{len(names)} assays, "
          f"{sum(len(A[n]['y']) for n in names)} variants total\n")
    res = {"assays": names}

    def basis(center):
        """center=False keeps the wild-type offset in the data.

        The naive version of this test -- z-score within assay, then compare a
        mean-removed SVD against a non-mean-removed one -- is VACUOUS, because
        z-scoring already subtracts each assay's mean and the pooled matrix is
        therefore centred before the flag is consulted. Both branches return the
        same basis and the test reports cos^2 = 1 while testing nothing. To keep
        the wild-type anchor in the data the standardisation here has to be
        SCALE ONLY, so the mean of dz -- which is what carries the reference --
        survives into the decomposition.

        Scale-only standardisation is `center=False` in pi_basis, which is one
        flag rather than a second construction -- the control and the thing it
        controls for have to be provably the same code or the check has no
        force.

        The previous version oriented the uncentred branch on `zc(X)` while
        fitting it on `X/std`. Those differ by a per-assay constant and
        Spearman does not see a constant shift, so the sign decision was
        unaffected -- but the basis and its orientation were standardised
        differently, which nothing said.
        """
        return pi_basis.fit({n: A[n]["X"] for n in names}, layer=-1,
                            center=center, orient_on="kl_glob",
                            orient_ref={n: A[n]["kl"] for n in names},
                            orient_k=N_PC, n_pc=N_PC, eps=EPS)

    B = basis(True)
    gm, V = B.gm, B.components
    res["protocol"] = dict(B.protocol)
    for n in names:
        A[n]["P"] = B.project(A[n]["X"], layer=-1)

    # =================================================================== 1
    print("1. One ranking across proteins vs twelve rankings within them\n")
    pc2 = np.concatenate([A[n]["P"][:, 1] for n in names])
    y_raw = np.concatenate([A[n]["y"] for n in names])
    y_z = np.concatenate([(A[n]["y"] - A[n]["y"].mean()) / (A[n]["y"].std() + EPS)
                          for n in names])
    within = {n: [pi_stats.spearman(A[n]["P"][:, 1], A[n]["y"])] for n in names}
    wm, wlo, whi, _ = pi_stats.cluster_bootstrap(within, n_boot=10000, seed=0,
                                                 hierarchical=False)
    pooled_raw = pi_stats.spearman(pc2, y_raw)
    pooled_z = pi_stats.spearman(pc2, y_z)
    # how much of a raw pooled correlation is just "which protein is this"?
    assay_id = np.concatenate([np.full(len(A[n]["y"]), i)
                               for i, n in enumerate(names)])
    conf = pi_stats.spearman(np.concatenate([np.full(len(A[n]["y"]),
                                                     A[n]["y"].mean())
                                             for n in names]), y_raw)
    print(f"   mean WITHIN assay (what the report shows)   {wm:+.3f} "
          f"[{wlo:+.3f}, {whi:+.3f}]")
    print(f"   ONE pooled ranking, target z-scored/assay   {pooled_z:+.3f}")
    print(f"   ONE pooled ranking, raw targets             {pooled_raw:+.3f}")
    print(f"   (for reference: assay mean DMS alone vs raw targets {conf:+.3f} "
          f"-- how much a raw pooled number reflects assay identity)\n")
    res["ranking"] = {"within_assay_mean": {"mean": wm, "ci_lo": wlo,
                                            "ci_hi": whi},
                      "pooled_target_zscored": pooled_z,
                      "pooled_raw_target": pooled_raw,
                      "assay_offset_confound": conf,
                      "n_variants": int(len(pc2))}

    # =================================================================== 2
    print("2. Is the direction an artifact of anchoring on wild type?\n")
    B0 = basis(False)
    gm0, V0 = B0.gm, B0.components
    s = np.linalg.svd(V @ V0.T, compute_uv=False)
    ang = float((s ** 2).mean())
    pc2_unc = np.concatenate([((A[n]["X"] / (A[n]["X"].std(0) + EPS)) @ V0.T)[:, 1]
                              for n in names])
    print(f"   centred (variant-to-variant) vs uncentred (wild-type-anchored)")
    print(f"      top-{N_PC} subspace agreement  cos^2 = {ang:.4f} "
          f"(chance {N_PC/V.shape[1]:.3f})")
    print(f"      PC2 scores correlate           rho   = "
          f"{pi_stats.spearman(pc2, pc2_unc):+.4f}")
    print("   z_a - z_b = dz_a - dz_b, so the centred basis IS the")
    print("   variant-to-variant difference basis -- the wild-type term cancels.")
    print("   The uncentred branch keeps that term by standardising on scale")
    print("   only. Agreement between the two means the anchor is not what")
    print("   creates the direction; disagreement would mean it is.\n")
    res["wt_anchor"] = {"subspace_cos2": ang, "chance": N_PC / V.shape[1],
                        "score_rho": pi_stats.spearman(pc2, pc2_unc)}

    # =================================================================== 3
    print("3. What else is PC2 correlated with?\n")
    QUANT = {"DMS": "y", "symmetric KL": "kl", "width change (d sigma)": "dsd",
             "broadening (spread)": "spread", "relocation (shift)": "shift"}
    prof = {}
    for lab, k in QUANT.items():
        g = {n: [pi_stats.spearman(A[n]["P"][:, 1], A[n][k])] for n in names}
        prof[lab] = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                               hierarchical=False)[:3]
    for j, nm in enumerate(pi_chem.CHEM_FEATURES):
        g = {n: [pi_stats.spearman(A[n]["P"][:, 1], A[n]["C"][:, j])]
             for n in names}
        prof["chem: " + nm] = pi_stats.cluster_bootstrap(
            g, n_boot=10000, seed=0, hierarchical=False)[:3]
    for lab, v in sorted(prof.items(), key=lambda kv: -abs(kv[1][0])):
        if abs(v[0]) >= 0.15:
            print(f"   {lab:26s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
    res["pc2_profile"] = {k: {"mean": v[0], "ci_lo": v[1], "ci_hi": v[2]}
                          for k, v in prof.items()}

    g = {n: [pi_stats.partial_spearman(A[n]["P"][:, 1], A[n]["y"],
                                       [A[n]["C"][:, j]
                                        for j in range(A[n]["C"].shape[1])])]
         for n in names}
    pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                               hierarchical=False)
    raw = prof["DMS"][0]
    print(f"\n   PC2 vs DMS, raw                       {raw:+.3f}")
    print(f"   PC2 vs DMS, holding all 17 chemistry  {pt:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]")
    print(f"   -> {100*abs(pt/raw):.0f}% of the association survives removing "
          f"substitution chemistry")
    res["pc2_vs_dms_partial_on_chemistry"] = {"raw": raw, "partial": pt,
                                              "ci_lo": lo, "ci_hi": hi}

    pi_archive.write_result(a.out, res, protocol=pi_protocol.protocol(
        script="analyze_scrutiny.py",
        design="one pooled ranking against twelve within-assay ones; the "
               "wild-type-anchoring control refits the basis scale-only",
        layer=pi_protocol.layers("final"),
        features=pi_protocol.features("dz_site, final-layer pair row", 128,
                                      kept=N_PC),
        source=a.glob, n_assays=len(names)))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
