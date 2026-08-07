"""Summarise the projection-ablation runs into the form the report reads.

The experiment deletes one direction from a REAL mutation's dz = z_mut - z_wt
and re-runs the structure module. Two things are reported and the first is not
optional: the positive control, i.e. whether the component was actually removed.
Without it a null result is uninterpretable, because "nothing changed" and "the
surgery did nothing" look identical.

`recovery` is the fraction of the mutation's own distogram-width change that
removing the direction undoes -- 1.0 would mean the output reverted to wild
type. Deleting any single direction out of 128 removes some variance, so the
comparison is always PC2 against PC1 and random directions, never against
nothing.
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


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/runs/"
    ap.add_argument("--glob", default=R + "ablpc_*.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    per = {}
    for f in sorted(glob.glob(a.glob)):
        d = np.load(f, allow_pickle=True)
        per[Path(f).stem.split("_", 1)[1].split("_")[0]] = {
            "d": d, "dirs": np.array([str(x) for x in d["rec_dir"]])}
    names = sorted(per)
    dirs = sorted({x for p in per.values() for x in p["dirs"]})
    print(f"{len(names)} assays, directions: {dirs}\n")

    out = {"assays": names, "directions": {}}
    print("Positive control (residual projection after removing the direction)\n")
    ctrl = {}
    for dn in dirs:
        r = []
        for n in names:
            d, m = per[n]["d"], per[n]["dirs"] == dn
            r.append(float(d["proj_after"][m].mean()
                           / (d["proj_before"][m].mean() + 1e-12)))
        ctrl[dn] = float(np.mean(r))
        print(f"   {dn:10s} {100*ctrl[dn]:.6f}% of the component survives")
    out["positive_control_residual_fraction"] = ctrl

    print("\nEffect of the deletion on the model's own outputs\n")
    print(f"   {'direction':10s} {'distogram recovery':>22s} {'d pLDDT':>12s} "
          f"{'CA shift (A)':>13s}")
    for dn in dirs:
        rec, dpl, cas = {}, {}, {}
        for n in names:
            d, m = per[n]["d"], per[n]["dirs"] == dn
            full, abl = d["d_sd_site_full"][m], d["d_sd_site_abl"][m]
            rec[n] = [float(np.nanmean(1 - np.abs(abl) / (np.abs(full) + 1e-12)))]
            dpl[n] = [float(np.nanmean(d["plddt_abl"][m] - d["plddt_full"][m]))]
            cas[n] = [float(np.nanmean(d["ca_shift"][m]))]
        b = {k: pi_stats.cluster_bootstrap(v, n_boot=10000, seed=0,
                                           hierarchical=False)[:3]
             for k, v in (("recovery", rec), ("d_plddt", dpl), ("ca", cas))}
        out["directions"][dn] = {k: {"mean": v[0], "ci_lo": v[1], "ci_hi": v[2]}
                                 for k, v in b.items()}
        print(f"   {dn:10s} {b['recovery'][0]:+8.3f} [{b['recovery'][1]:+.3f},"
              f"{b['recovery'][2]:+.3f}] {b['d_plddt'][0]:+12.5f} "
              f"{b['ca'][0]:13.4f}")

    print("\nPaired: PC2 minus each random direction\n")
    gaps = {}
    for other in [d for d in dirs if d.startswith("random")]:
        A, B = {}, {}
        for n in names:
            d = per[n]["d"]
            for tag, dn in (("A", "PC2"), ("B", other)):
                m = per[n]["dirs"] == dn
                full, abl = d["d_sd_site_full"][m], d["d_sd_site_abl"][m]
                v = float(np.nanmean(1 - np.abs(abl) / (np.abs(full) + 1e-12)))
                (A if tag == "A" else B)[n] = [v]
        pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(
            A, B, n_boot=10000, seed=0, hierarchical=False)
        wins = sum(1 for n in names if A[n][0] > B[n][0])
        gaps[other] = {"gap": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins,
                       "n": len(names)}
        flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
        print(f"   PC2 - {other:9s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"{wins}/{len(names)}{flag}")
    out["pc2_minus_random"] = gaps

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
