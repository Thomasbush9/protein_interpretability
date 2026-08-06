"""How much of the sharpening result survives run-to-run drift?

The audit asks for the sharpening claim to be calibrated against inference
noise, and proposes repeating wild type and a stratified mutant subset across
seeds to get the noise interval. That run is not necessary: the repeat already
exists.

`gym2_*.npz` and `gym2s_*.npz` are two independent executions of the same 250
variants per assay -- same `--seed 0`, same variant subsample, same alignment
cap -- separated by four days and a code change that only ADDED derived
features. The `mutant` lists are checked here and must match exactly before an
assay is used, because everything below treats row i of one archive and row i
of the other as the same variant measured twice.

Boltz-2's trunk is not bit-reproducible (`pi_capture` carries DRIFT_TOL = 2e-3
for this reason), so the difference between the two runs IS the noise
distribution, at no GPU cost. For a quantity measured twice as d1 and d2 with
independent errors of equal variance, sd(d1 - d2) = sqrt(2) * sigma, which is
where the noise estimate below comes from.

Three questions, in order:

  reproducibility   How large is the drift on each quantity the mechanism
                    report uses -- the symmetric KL, the signed width change,
                    and the raw representation difference `dz_site`?

  classification    `dsd < 0` is a threshold on a noisy quantity. How often do
                    the two runs disagree about whether a variant sharpened,
                    and how many variants sit far enough from zero for the call
                    to be safe?

  the claim         Does DMS(sharpen) - DMS(broaden) hold up when the
                    ambiguous band is excluded rather than silently split
                    between the two groups?

One limitation to state plainly: `disto` archives only the FINAL layer, so the
width-change drift here is measured at layer 63, whereas `analyze_channels`
averages the last eight layers. Averaging eight layers can only reduce the
noise, so the numbers below are conservative for that use.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import pi_conf  # noqa: E402
import pi_stats  # noqa: E402

CENTERS = jnp.asarray(pi_conf.bin_centers())


def sigma_of(logits):
    """Per-pair mean and standard deviation of the distogram, in angstroms.

    Same construction as `pi_conf.moments`, including the centred form of the
    variance and its floor, so a quantity compared here is the same quantity
    the mechanism report computed -- not an algebraically equivalent rewrite
    that could disagree in the last digits and be mistaken for drift.
    """
    p = jax.nn.softmax(jnp.asarray(logits, jnp.float64), axis=-1)
    mu = (p * CENTERS).sum(-1)
    var = (p * (CENTERS - mu[..., None]) ** 2).sum(-1)
    return mu, jnp.sqrt(jnp.maximum(var, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/runs/"
    ap.add_argument("--a-glob", default=R + "gym2s_*.npz")
    ap.add_argument("--b-glob", default=R + "gym2_*.npz")
    ap.add_argument("--z", type=float, default=2.0,
                    help="how many noise sigmas a variant must clear to be called")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print(f"jax devices: {jax.devices()}\n")
    B = {Path(f).stem.split("_", 1)[1]: f for f in glob.glob(a.b_glob)}
    rows, per = {}, {}

    print(f"{'assay':8s} {'KL drift':>9s} {'cos(dz)':>9s} {'sd(noise)':>10s} "
          f"{'sharp r1':>9s} {'sharp r2':>9s} {'sign agr':>9s} {'confident':>10s}")
    for fa in sorted(glob.glob(a.a_glob)):
        stem = Path(fa).stem.split("_", 1)[1]
        if stem not in B:
            continue
        A_, B_ = np.load(fa, allow_pickle=True), np.load(B[stem], allow_pickle=True)
        name = stem.split("_")[0]
        if [str(m) for m in A_["mutant"]] != [str(m) for m in B_["mutant"]]:
            print(f"{name:8s} SKIPPED: variant lists differ between the two runs")
            continue

        # --- drift on each quantity -------------------------------------
        k1, k2 = A_["kl_glob"][:, -1], B_["kl_glob"][:, -1]
        kl_drift = float(np.abs(k1 - k2).mean() / (np.abs(k1).mean() + 1e-12))
        z1 = np.asarray(A_["dz_site"], float)
        z2 = np.asarray(B_["dz_site"], float)
        cos = (z1 * z2).sum(-1) / (np.linalg.norm(z1, axis=-1)
                                   * np.linalg.norm(z2, axis=-1) + 1e-12)
        cos_layer = np.median(cos, axis=0)                  # (L,)

        # --- signed width change, recomputed identically in both runs ----
        d = {}
        for tag, DD in (("1", A_), ("2", B_)):
            _, sw = sigma_of(DD["disto_wt"])
            _, sm = sigma_of(DD["disto"])
            d[tag] = np.asarray((sm - sw).mean(-1))         # (N,) mean over pairs
        d1, d2 = d["1"], d["2"]
        noise = float(np.std(d1 - d2) / np.sqrt(2))
        dm = 0.5 * (d1 + d2)
        conf = np.abs(dm) > a.z * noise
        agree = float((np.sign(d1) == np.sign(d2)).mean())

        y = np.asarray(A_["score"], float)
        sharp = conf & (dm < 0)
        broad = conf & (dm > 0)
        gap = (float(np.nanmean(y[sharp]) - np.nanmean(y[broad]))
               if sharp.sum() >= 5 and broad.sum() >= 5 else np.nan)
        gap_all = float(np.nanmean(y[dm < 0]) - np.nanmean(y[dm > 0]))

        per[name] = {
            "kl_rel_drift": kl_drift,
            "cos_dz_last": float(cos_layer[-1]),
            "cos_dz_first": float(cos_layer[0]),
            "cos_dz_min": float(cos_layer.min()),
            "dsd_noise_sd": noise,
            "frac_sharpen_run1": float((d1 < 0).mean()),
            "frac_sharpen_run2": float((d2 < 0).mean()),
            "sign_agreement": agree,
            "frac_confident": float(conf.mean()),
            "frac_ambiguous": float(1 - conf.mean()),
            "n_sharpen_conf": int(sharp.sum()), "n_broaden_conf": int(broad.sum()),
            "dms_gap_confident": gap, "dms_gap_all": gap_all,
        }
        rows[name] = per[name]
        print(f"{name:8s} {100 * kl_drift:8.2f}% {float(cos_layer.min()):9.4f} "
              f"{noise:10.5f} {100 * (d1 < 0).mean():8.1f}% "
              f"{100 * (d2 < 0).mean():8.1f}% {100 * agree:8.1f}% "
              f"{100 * conf.mean():9.1f}%")

    if not rows:
        raise SystemExit("no assay had a usable repeat")

    print()
    out = {"protocol": {"z": a.z, "n_assays": len(rows),
                        "note": "drift measured at the final layer only; "
                                "disto archives layer 63"},
           "per_assay": per}
    for lab, key in (("sign agreement on dsd", "sign_agreement"),
                     ("fraction clearing the noise band", "frac_confident"),
                     ("DMS(sharpen) - DMS(broaden), all variants", "dms_gap_all"),
                     ("DMS(sharpen) - DMS(broaden), confident only",
                      "dms_gap_confident")):
        g = {k: [v[key]] for k, v in rows.items() if np.isfinite(v[key])}
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                   hierarchical=False)
        out[lab] = {"mean": pt, "ci_lo": lo, "ci_hi": hi, "n_assays": len(g)}
        flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
        print(f"   {lab:44s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]{flag}")

    amb = np.mean([v["frac_ambiguous"] for v in rows.values()])
    print(f"\n   {100 * amb:.1f}% of variants sit inside the +-{a.z}-sigma band and")
    print("   cannot be called either way; they are excluded from the confident")
    print("   comparison rather than assigned to whichever side their noise put")
    print("   them on.")

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
