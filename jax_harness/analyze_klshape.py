"""Does the mutant distogram MOVE, or does it just get less certain?

The probe's headline features are symmetric KL divergences between a mutant's
distogram and wild type's. A symmetric KL is large in two very different
situations, and reports the same number for both:

  * the distribution RELOCATES -- the model predicts a different distance. This
    is a geometric claim, and it is the reading the report currently gives.
  * the distribution BROADENS -- the model predicts the same distance with less
    confidence. This is a claim about certainty, much closer to a local pLDDT
    than to structural knowledge.

If the second dominates, section 3's wording has to change: the probe would be
decoding an uncertainty channel that happens to correlate with stability, not a
picture of geometry. So separate them.

The separation is exact for Gaussians, which is why it is done this way rather
than by correlating KL against |dmu| and hoping. For two one-dimensional normals
the Jeffreys divergence splits additively with no cross term:

    J = [ s1^2/(2 s2^2) + s2^2/(2 s1^2) - 1 ]      <- SPREAD, zero iff s1 == s2
      + [ d^2 (1/(2 s1^2) + 1/(2 s2^2)) ]          <- SHIFT,  zero iff mu1 == mu2

Each term is non-negative and vanishes exactly when its own effect is absent, so
"share of the divergence due to shift" is well defined rather than a regression
artefact. The distograms are not Gaussian, so the approximation is checked
against the true Jeffreys divergence on the same pairs and reported, not assumed.

This script uses the FINAL layer only, because that is what `exp_gym2.py`
archives per pair (`disto`, `disto_wt`); it needs no GPU. The per-layer version
requires the rerun that adds moment features at every layer -- the probe reads
64 layers and its feature selection concentrates in layers 45-63, so this is a
partial answer, and a positive result here does not license dropping the rerun.

Pair indices are not archived, so only the global (all-pairs) quantity is
available offline; `kl_site` needs the rerun too.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_conf  # noqa: E402
import pi_stats  # noqa: E402


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def jeffreys(p, q):
    return ((p - q) * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1)


decompose = pi_conf.jeffreys_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/"
                                      "tbush/prot_interp_files/runs/gym2_*.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    c = pi_conf.bin_centers()
    per_assay, rows = {}, []
    files = sorted(glob.glob(a.glob))
    print(f"{len(files)} assays, final Pairformer layer, all archived pairs\n")
    hdr = (f"{'assay':10s} {'n':>4s} {'pairs':>6s} {'shift%':>7s} {'spread%':>8s} "
           f"{'d.sigma':>8s} {'gauss~true':>11s} {'rho|dmu|':>9s} {'rho dsd':>8s} "
           f"{'rho KL':>7s}")
    print(hdr + "\n" + "-" * len(hdr))

    for f in files:
        d = np.load(f, allow_pickle=True)
        name = Path(f).stem.replace("gym2_", "").split("_")[0]
        y = d["score"]
        p_w = softmax(d["disto_wt"].astype(np.float64))          # [P,64]
        p_m = softmax(d["disto"].astype(np.float64))             # [V,P,64]
        mu_w, sd_w = pi_conf.moments(p_w, c)
        mu_m, sd_m = pi_conf.moments(p_m, c)

        kl_true = jeffreys(p_m, p_w)                             # [V,P]
        shift, spread = decompose(mu_w, sd_w, mu_m, sd_m)        # [V,P]
        kl_gauss = shift + spread

        # per variant: the features the probe would see
        v_shift, v_spread = shift.mean(1), spread.mean(1)
        v_kl = kl_true.mean(1)
        v_dmu = np.abs(mu_m - mu_w).mean(1)
        v_dsd = (sd_m - sd_w).mean(1)                            # SIGNED

        share = v_shift / (v_shift + v_spread + 1e-12)
        fid = pi_stats.spearman(kl_gauss.ravel()[::37], kl_true.ravel()[::37])

        r = {"assay": name, "n": int(len(y)), "pairs": int(p_w.shape[0]),
             "shift_share": float(np.mean(share)),
             "spread_share": float(1 - np.mean(share)),
             "d_sigma_mean": float(np.mean(v_dsd)),
             "gauss_vs_true_rho": float(fid),
             "rho_absdmu": float(pi_stats.spearman(v_dmu, y)),
             "rho_dsigma": float(pi_stats.spearman(v_dsd, y)),
             "rho_kl": float(pi_stats.spearman(v_kl, y)),
             "rho_shift": float(pi_stats.spearman(v_shift, y)),
             "rho_spread": float(pi_stats.spearman(v_spread, y))}
        per_assay[name] = r
        rows.append(r)
        print(f"{name:10s} {r['n']:4d} {r['pairs']:6d} {100*r['shift_share']:6.1f}% "
              f"{100*r['spread_share']:7.1f}% {r['d_sigma_mean']:+8.3f} "
              f"{fid:11.3f} {r['rho_absdmu']:+9.3f} {r['rho_dsigma']:+8.3f} "
              f"{r['rho_kl']:+7.3f}")

    print("-" * len(hdr))
    agg = {}
    for k in ("shift_share", "spread_share", "d_sigma_mean", "gauss_vs_true_rho",
              "rho_absdmu", "rho_dsigma", "rho_kl", "rho_shift", "rho_spread"):
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(
            {r["assay"]: [r[k]] for r in rows}, n_boot=10000, seed=0,
            hierarchical=False)
        agg[k] = {"mean": pt, "ci_lo": lo, "ci_hi": hi}

    print("\nPooled over assays (95% CI, bootstrap over assays)\n")
    LAB = {"shift_share": "divergence from SHIFT",
           "spread_share": "divergence from SPREAD",
           "d_sigma_mean": "mean sigma(mut) - sigma(wt), A",
           "gauss_vs_true_rho": "Gaussian split vs true KL (rho)",
           "rho_absdmu": "rho(|d mu|, DMS)",
           "rho_dsigma": "rho(d sigma, DMS)",
           "rho_kl": "rho(true KL, DMS)",
           "rho_shift": "rho(shift term, DMS)",
           "rho_spread": "rho(spread term, DMS)"}
    for k, v in agg.items():
        print(f"  {LAB[k]:34s} {v['mean']:+.3f}  [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]")

    Path(a.out).write_text(json.dumps(
        {"scope": {"layer": "final", "note": "single-layer, global pairs only; "
                                             "per-layer needs the rerun"},
         "pooled": agg, "per_assay": per_assay}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
