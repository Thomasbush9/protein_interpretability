"""Are relocation and broadening two channels, or one thing measured twice?

`analyze_shape_probe` shows the broadening half alone reproduces the probe and
the relocation half alone does not. That result has two very different readings
and this script separates them.

  * If the two are strongly correlated across variants, they are one signal in
    two coordinates and "spread alone works" is arithmetic, not biology.
  * If they are weakly correlated and EACH retains predictive power after
    controlling for the other, they are distinct channels, and it becomes
    reasonable to ask what kind of mutation each one responds to.

Partial Spearman is the test for the second clause: shift against DMS holding
spread fixed, and vice versa. `pi_stats.partial_spearman` residualises the ranks
and correlates with Pearson, which is the standard construction -- re-ranking the
residuals would be a different and wrong statistic.

If they do separate, three biologically interpretable questions follow, all
answerable from what is already archived:

  chemistry   does one channel track volume and packing while the other tracks
              polarity and charge? (the 17 `pi_chem` descriptors)
  burial      does the balance change between buried and exposed sites?
              Burial is a CA neighbour count from the wild-type structure.
  direction   sigma is SIGNED. Mutants are broader on average, but some sharpen.
              Do the sharpening ones behave differently -- are they the
              stabilising mutations, where the model becomes MORE certain?

Channels are read as the mean over the last eight Pairformer layers, which is
where the probe's feature selection concentrates (layers 55-63 in every block).
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
import pi_stats  # noqa: E402

LAYERS = slice(-8, None)


def burial(ca_wt, pos, cutoff=10.0):
    """CA neighbours within `cutoff` of each mutated site: a packing proxy."""
    d = np.linalg.norm(ca_wt[:, None] - ca_wt[None, :], axis=-1)
    n = ((d < cutoff).sum(1) - 1).astype(float)
    return np.array([n[p] if 0 <= p < len(n) else np.nan for p in pos])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/"
                                      "tbush/prot_interp_files/runs/gym2s_*.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    per = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        name = Path(f).stem.replace("gym2s_", "").split("_")[0]
        y = d["score"]
        sh = d["shift_glob"][:, LAYERS].mean(1)
        sp = d["spread_glob"][:, LAYERS].mean(1)
        dmu = d["dmu_glob"][:, LAYERS].mean(1)
        dsd = d["dsd_glob"][:, LAYERS].mean(1)          # SIGNED
        muts = [str(m) for m in d["mutant"]]
        per[name] = {"y": y, "shift": sh, "spread": sp, "dmu": dmu, "dsd": dsd,
                     "chem": pi_chem.chem_matrix(muts),
                     "bur": burial(np.asarray(d["ca_wt"], float), d["pos"])}

    R = {}

    # ---- 1. are they redundant? -----------------------------------------
    print("1. Redundancy and unique contribution\n")
    print(f"   {'assay':8s} {'rho(shift,spread)':>18s} {'shift|spread':>13s} "
          f"{'spread|shift':>13s}")
    red, ps_sh, ps_sp = {}, {}, {}
    for k, v in per.items():
        r = pi_stats.spearman(v["shift"], v["spread"])
        a1 = pi_stats.partial_spearman(v["shift"], v["y"], [v["spread"]])
        a2 = pi_stats.partial_spearman(v["spread"], v["y"], [v["shift"]])
        red[k], ps_sh[k], ps_sp[k] = [r], [a1], [a2]
        print(f"   {k:8s} {r:18.3f} {a1:13.3f} {a2:13.3f}")
    for lab, dd in (("rho(shift, spread)", red),
                    ("shift vs DMS | spread", ps_sh),
                    ("spread vs DMS | shift", ps_sp)):
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(dd, n_boot=10000, seed=0,
                                                   hierarchical=False)
        R[lab] = {"mean": pt, "ci_lo": lo, "ci_hi": hi}
        print(f"   pooled {lab:26s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    # ---- 2. which chemistry does each channel respond to? ---------------
    print("\n2. Chemistry each channel responds to (pooled |rho|, z-scored per assay)\n")
    chem = {}
    for j, nm in enumerate(pi_chem.CHEM_FEATURES):
        g_sh = {k: [pi_stats.spearman(v["chem"][:, j], v["shift"])] for k, v in per.items()}
        g_sp = {k: [pi_stats.spearman(v["chem"][:, j], v["spread"])] for k, v in per.items()}
        s1 = pi_stats.cluster_bootstrap(g_sh, n_boot=2000, seed=0, hierarchical=False)[0]
        s2 = pi_stats.cluster_bootstrap(g_sp, n_boot=2000, seed=0, hierarchical=False)[0]
        chem[nm] = {"shift": s1, "spread": s2, "diff": s1 - s2}
    print(f"   {'feature':20s} {'shift':>8s} {'spread':>8s} {'difference':>11s}")
    for nm, v in sorted(chem.items(), key=lambda kv: -abs(kv[1]["diff"])):
        flag = "  <-- separates" if abs(v["diff"]) > 0.06 else ""
        print(f"   {nm:20s} {v['shift']:+8.3f} {v['spread']:+8.3f} "
              f"{v['diff']:+11.3f}{flag}")
    R["chemistry"] = chem

    # ---- 3. burial ------------------------------------------------------
    print("\n3. Buried versus exposed sites (CA neighbours within 10 A)\n")
    bur = {}
    for k, v in per.items():
        b = v["bur"]
        if not np.isfinite(b).any():
            continue
        hi_ = b >= np.nanmedian(b)
        share = v["shift"] / (v["shift"] + v["spread"] + 1e-12)
        bur[k] = {"buried_shift_share": float(np.nanmean(share[hi_])),
                  "exposed_shift_share": float(np.nanmean(share[~hi_]))}
    for lab, key in (("buried", "buried_shift_share"), ("exposed", "exposed_shift_share")):
        g = {k: [v[key]] for k, v in bur.items()}
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                   hierarchical=False)
        R[f"shift_share_{lab}"] = {"mean": pt, "ci_lo": lo, "ci_hi": hi}
        print(f"   relocation share at {lab:8s} sites  {pt:.3f} [{lo:.3f}, {hi:.3f}]")
    g = {k: [v["buried_shift_share"] - v["exposed_shift_share"]] for k, v in bur.items()}
    pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0, hierarchical=False)
    R["shift_share_buried_minus_exposed"] = {"mean": pt, "ci_lo": lo, "ci_hi": hi}
    print(f"   buried minus exposed              {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
          f"{'' if (lo > 0 or hi < 0) else '   <- includes zero'}")

    # ---- 4. who sharpens? ----------------------------------------------
    print("\n4. Mutations where the model gets MORE certain (d sigma < 0)\n")
    sharp = {}
    for k, v in per.items():
        m = v["dsd"] < 0
        if m.sum() < 5 or (~m).sum() < 5:
            continue
        sharp[k] = {"frac": float(m.mean()),
                    "dms_sharpen": float(np.nanmean(v["y"][m])),
                    "dms_broaden": float(np.nanmean(v["y"][~m]))}
        print(f"   {k:8s} {100*m.mean():5.1f}% sharpen   mean DMS "
              f"sharpen {sharp[k]['dms_sharpen']:+.3f}  vs broaden "
              f"{sharp[k]['dms_broaden']:+.3f}")
    if sharp:
        g = {k: [v["dms_sharpen"] - v["dms_broaden"]] for k, v in sharp.items()}
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                   hierarchical=False)
        R["dms_sharpen_minus_broaden"] = {"mean": pt, "ci_lo": lo, "ci_hi": hi}
        R["frac_sharpen"] = float(np.mean([v["frac"] for v in sharp.values()]))
        print(f"\n   pooled DMS(sharpen) - DMS(broaden)  {pt:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]"
              f"{'' if (lo > 0 or hi < 0) else '   <- includes zero'}")
        print(f"   {100*R['frac_sharpen']:.1f}% of variants sharpen on average")

    Path(a.out).write_text(json.dumps(R, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
