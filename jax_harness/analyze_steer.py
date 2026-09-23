"""Is PC2 a direction the model USES, or just a direction that perturbs it?

Effect size cannot answer this. Any vector added to z will move the distogram,
and a random vector of the same norm moves it about as much -- so "PC2 changed
the output" is not evidence of anything. What distinguishes a used direction is
SIGN STRUCTURE.

PC2 is the broadening axis: variants with a high score have wider predicted
distance distributions. If the model represents that as a signed quantity, then
injecting +alpha should broaden and -alpha should sharpen, and the response
should be odd in alpha. A direction that merely disturbs the computation has no
privileged sign; its response should depend on |alpha| and therefore be even.

So each response curve is split into its odd and even parts,

    odd(a)  = [f(+a) - f(-a)] / 2      signed, direction-specific
    even(a) = [f(+a) + f(-a)] / 2      magnitude-driven, orientation-blind

and PC2's odd component is compared against the random directions', which is
the null this needs. With only a handful of random draws the comparison is
descriptive, not a p-value, and is reported that way.

Coordinates are judged against the sampler's own drift: the same trunk state
re-sampled under a different diffusion key. Anything smaller than that is not a
structural effect no matter how it is plotted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_stats  # noqa: E402

METRICS = [("d_sd_site", "distogram width at the injected site"),
           ("d_plddt_site", "pLDDT at the injected residue"),
           ("ca_rmsd", "superposed CA RMSD")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = np.load(a.run, allow_pickle=True)
    dirs = [str(x) for x in d["rec_dir"]]
    modes = [str(x) for x in d["rec_mode"]]
    alpha, site = d["alpha"], d["site"]
    names = sorted(set(dirs))
    drift_ca = float(np.mean(np.abs(d["drift_ca"])))
    drift_pl = float(np.mean(np.abs(d["drift_plddt"])))
    print(f"assay {str(d['assay'])}\n")
    print(f"sampler drift floor (different diffusion key, same trunk state):")
    print(f"   CA RMSD {drift_ca:.4f} A     pLDDT {drift_pl:.5f}\n")

    res = {"drift": {"ca_rmsd": drift_ca, "d_plddt": drift_pl},
           "scale": float(d["scale"])}
    for mode in sorted(set(modes)):
        print(f"=== injection mode: {mode} ===\n")
        res[mode] = {}
        for key, label in METRICS:
            v = d[key]
            print(f"  {label}")
            print(f"     {'direction':10s} {'odd slope':>11s} {'even mean':>11s} "
                  f"{'odd/|even|':>11s}")
            row = {}
            for nm in names:
                odd, even = [], []
                for s in sorted(set(site)):
                    for al in sorted({abs(x) for x in alpha if x > 0}):
                        m_p = (np.array(dirs) == nm) & (np.array(modes) == mode) \
                            & (site == s) & (alpha == al)
                        m_n = (np.array(dirs) == nm) & (np.array(modes) == mode) \
                            & (site == s) & (alpha == -al)
                        if m_p.sum() != 1 or m_n.sum() != 1:
                            continue
                        fp, fn = float(v[m_p][0]), float(v[m_n][0])
                        odd.append((fp - fn) / 2 / al)      # per unit alpha
                        even.append((fp + fn) / 2)
                if not odd:
                    continue
                o, e = float(np.mean(odd)), float(np.mean(even))
                row[nm] = {"odd_per_alpha": o, "even_mean": e,
                           "ratio": o / (abs(e) + 1e-12)}
                print(f"     {nm:10s} {o:+11.5f} {e:+11.5f} {o/(abs(e)+1e-12):+11.2f}")
            rnd = [row[n]["odd_per_alpha"] for n in row if n.startswith("random")]
            if rnd and "PC2" in row:
                lo, hi = min(rnd), max(rnd)
                pc2 = row["PC2"]["odd_per_alpha"]
                out = "OUTSIDE" if (pc2 < lo or pc2 > hi) else "inside"
                print(f"     -> PC2 odd component is {out} the random range "
                      f"[{lo:+.5f}, {hi:+.5f}]")
            res[mode][key] = row
            print()

    # Coordinates against the sampler's own noise -- SPLIT BY MODE. Pooling the
    # modes hides the only thing this comparison is for: a one-row injection
    # leaves the structure untouched while a global one moves it by angstroms,
    # and a pooled median is dominated by whichever mode has more rows.
    print("Coordinate effect against the drift floor, by injection mode\n")
    ca, dirs_a, modes_a = d["ca_rmsd"], np.array(dirs), np.array(modes)
    res["coordinates_vs_drift"] = {}
    for mode in sorted(set(modes)):
        print(f"   mode {mode}")
        res["coordinates_vs_drift"][mode] = {}
        for nm in names:
            m = (dirs_a == nm) & (modes_a == mode) & (alpha != 0)
            if not m.any():
                continue
            med, mx = float(np.median(ca[m])), float(ca[m].max())
            res["coordinates_vs_drift"][mode][nm] = {"median": med, "max": mx}
            print(f"      {nm:10s} median {med:8.4f} A   max {mx:8.4f} A   "
                  f"{'EXCEEDS' if med > drift_ca else 'below  '} the "
                  f"{drift_ca:.3f} A drift floor")
        print()

    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
