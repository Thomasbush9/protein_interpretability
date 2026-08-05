"""Where do the mutations that make the model MORE certain sit?

About 22% of variants SHARPEN the distogram -- the model's predicted distance
distribution gets narrower, not wider -- and those variants are substantially
more tolerated experimentally (mean DMS higher by +0.535). The question here is
whether that is a positional fact about the protein rather than a fact about the
substitution: do sharpening mutations land on structurally or functionally
distinguishable sites?

Three independent measures of "this position matters", none of which is derived
from the model's internal state, so none can be circular:

  conservation   Shannon entropy of the alignment column, computed from the
                 protein's own a3m. Low entropy means evolution has not
                 tolerated change here. This is the measure of biological
                 importance least contaminated by anything else in this project.
  burial         CA neighbours within 10 A in the predicted wild-type structure.
                 A packing proxy: buried positions are where substitutions
                 disrupt the core.
  sensitivity    mean DMS across all variants AT that position. Purely
                 experimental, and the most direct definition of "key" -- a
                 position where every substitution is costly.

The unit of analysis is the POSITION, not the variant. Variants at the same site
share a residue environment and are not independent, so per-variant correlations
would report an effective sample size several times larger than the data
supports. Positions are aggregated first and the bootstrap clusters on assay.

Note the sign convention: entropy is LOW at conserved positions, so a negative
correlation between sharpening and entropy means sharpening happens at conserved
sites.
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

AA = "ACDEFGHIKLMNPQRSTVWY"
LAYERS = slice(-8, None)
GYM = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/data/gym")


def find_a3m(assay):
    for p in ("panel2", "panel"):
        f = GYM / p / "colabfold_output" / f"{assay}.a3m"
        if f.exists():
            return f
    return None


def column_entropy(a3m, length, cap=20000):
    """Per-column Shannon entropy (bits) over the 20 amino acids, gaps ignored."""
    counts = np.zeros((length, 20))
    idx = {c: i for i, c in enumerate(AA)}
    lines = a3m.read_text().splitlines()
    n = 0
    for ln in lines:
        if ln.startswith(">") or not ln.strip():
            continue
        s = "".join(c for c in ln if not c.islower())   # drop insertions
        if len(s) != length:
            continue
        for j, c in enumerate(s):
            k = idx.get(c.upper())
            if k is not None:
                counts[j, k] += 1
        n += 1
        if n >= cap:
            break
    if n == 0:
        return None, 0
    tot = counts.sum(1, keepdims=True)
    p = counts / np.maximum(tot, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(p > 0, p * np.log2(p), 0.0)).sum(1)
    h[tot[:, 0] == 0] = np.nan
    return h, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/"
                                      "tbush/prot_interp_files/runs/gym2s_*.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, per_assay = {}, {}
    print("Per-position: does sharpening track biological importance?\n")
    print(f"   {'assay':8s} {'sites':>6s} {'msa':>7s} "
          f"{'rho(sharp,entropy)':>19s} {'rho(sharp,burial)':>18s} "
          f"{'rho(sharp,sensitivity)':>23s}")
    for f in sorted(glob.glob(a.glob)):
        d = np.load(f, allow_pickle=True)
        assay = Path(f).stem.replace("gym2s_", "")
        name = assay.split("_")[0]
        wt = str(d["wt_seq"])
        dsd = d["dsd_glob"][:, LAYERS].mean(1)
        pos, y = d["pos"], d["score"]
        ca = np.asarray(d["ca_wt"], float)
        dist = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
        nb = (dist < 10.0).sum(1) - 1

        ent, ndep = (None, 0)
        p3 = find_a3m(assay)
        if p3 is not None:
            ent, ndep = column_entropy(p3, len(wt))

        sites = sorted(set(int(p) for p in pos))
        sf, bu, en, se = [], [], [], []
        for p in sites:
            m = pos == p
            if m.sum() < 2:
                continue
            sf.append(float((dsd[m] < 0).mean()))
            bu.append(float(nb[p]) if 0 <= p < len(nb) else np.nan)
            en.append(float(ent[p]) if ent is not None and p < len(ent) else np.nan)
            se.append(float(np.nanmean(y[m])))
        sf, bu, en, se = map(np.asarray, (sf, bu, en, se))
        r_e = pi_stats.spearman(sf, en) if np.isfinite(en).sum() > 5 else np.nan
        r_b = pi_stats.spearman(sf, bu)
        r_s = pi_stats.spearman(sf, se)
        per_assay[name] = {"n_sites": int(len(sf)), "msa_depth": ndep,
                           "rho_entropy": r_e, "rho_burial": r_b,
                           "rho_sensitivity": r_s,
                           "mean_sharpen_frac": float(np.nanmean(sf))}
        rows[name] = (r_e, r_b, r_s)
        print(f"   {name:8s} {len(sf):6d} {ndep:7d} {r_e:19.3f} {r_b:18.3f} "
              f"{r_s:23.3f}")

    print()
    out = {"per_assay": per_assay}
    for lab, i in (("sharpening vs alignment entropy", 0),
                   ("sharpening vs burial", 1),
                   ("sharpening vs position sensitivity", 2)):
        g = {k: [v[i]] for k, v in rows.items() if np.isfinite(v[i])}
        if not g:
            continue
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                   hierarchical=False)
        out[lab] = {"mean": pt, "ci_lo": lo, "ci_hi": hi, "n_assays": len(g)}
        sig = "" if (lo > 0 or hi < 0) else "   <- includes zero"
        print(f"   {lab:36s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}")

    print("\n   Entropy is LOW at conserved sites, so a NEGATIVE correlation with")
    print("   entropy means sharpening concentrates where evolution is strict.")
    print("   Sensitivity is mean DMS, so a POSITIVE correlation means sharpening")
    print("   concentrates at tolerant positions.")

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
