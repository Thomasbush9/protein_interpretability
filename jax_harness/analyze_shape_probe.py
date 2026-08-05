"""Which half of the divergence is the probe actually using?

`analyze_klshape` splits the final layer's divergence into relocation and
broadening and finds both carry signal. That is one layer of sixty-four, and the
probe's feature selection concentrates in layers 45-63, so it cannot settle the
question. This does.

The probe is re-fit, under the identical protocol, on feature blocks of MATCHED
dimensionality:

    kl_only      kl_glob + kl_site                 128   what the report uses
    shift        shift_glob + shift_site           128   relocation only
    spread       spread_glob + spread_site         128   broadening only
    dmu          |dmu_glob| + |dmu_site|           128   raw distance change
    dsd          dsd_glob + dsd_site               128   raw certainty change
    internal     the published 256                 256   reference
    shift+spread both halves                       256   reference

Matching dimensionality matters: 128 against 256 would confound "which signal"
with "how many features to select from", and the k=16 selection is done on
training rows inside each block.

Reading the result. If `shift` reproduces `kl_only` and `spread` does not, the
divergence features are geometric and section 3 stands as written. If `spread`
reproduces it, the probe is reading a confidence channel and the wording has to
change. If both do -- which the final-layer analysis suggests -- then the honest
statement is that the two are entangled, and the paper should say so rather than
claim the geometric reading by default.

The rerun that produced these features also recomputes kl_glob and kl_site. Those
are checked against the original archives first: if the KL features do not come
back identical, the rerun changed something and the new features cannot be
trusted either.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_stats  # noqa: E402
from compare_internal_output import fit_ridge_block, grouped_split  # noqa: E402

BLOCKS = {
    "internal":     ("kl_glob", "kl_site", "|dz_site|", "|ds_site|"),
    "kl_only":      ("kl_glob", "kl_site"),
    "shift":        ("shift_glob", "shift_site"),
    "spread":       ("spread_glob", "spread_site"),
    "dmu":          ("dmu_glob", "dmu_site"),
    "dsd":          ("dsd_glob", "dsd_site"),
    "shift+spread": ("shift_glob", "shift_site", "spread_glob", "spread_site"),
}
GAPS = [("kl_only - shift", "kl_only", "shift"),
        ("kl_only - spread", "kl_only", "spread"),
        ("shift - spread", "shift", "spread"),
        ("internal - shift+spread", "internal", "shift+spread")]


def build(d, names):
    cols = []
    for n in names:
        if n.startswith("|"):
            cols.append(np.linalg.norm(d[n.strip("|")], axis=-1))
        else:
            cols.append(d[n])
    return np.concatenate(cols, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/"
                                      "tbush/prot_interp_files/runs/gym2s_*.npz")
    ap.add_argument("--orig-dir", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/"
                                          "tbush/prot_interp_files/runs")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"no files match {a.glob}")
    per = defaultdict(lambda: defaultdict(list))
    sel = defaultdict(Counter)
    fidelity = {}

    print("Reproduction check against the original archives\n")
    for f in files:
        d = np.load(f, allow_pickle=True)
        assay = Path(f).stem.replace("gym2s_", "")
        name = assay.split("_")[0]
        orig = Path(a.orig_dir) / f"gym2_{assay}.npz"
        if orig.exists():
            o = np.load(orig, allow_pickle=True)
            same_rows = (len(o["score"]) == len(d["score"]) and
                         np.array_equal(o["mutant"], d["mutant"]))
            if same_rows:
                # Bit-equality is the WRONG bar. Boltz-2's trunk is not
                # reproducible to machine precision across runs -- that is why
                # pi_capture carries DRIFT_TOL = 2e-3 for this model rather than
                # a tolerance near zero -- and pLDDT moves by ~2e-2 between runs
                # of the identical command. What the probe consumes is the RANK
                # ordering of each feature, so that is what has to be preserved.
                rho = min(pi_stats.spearman(o[k].ravel(), d[k].ravel())
                          for k in ("kl_glob", "kl_site"))
                med = float(np.median(np.abs(o["kl_glob"] - d["kl_glob"]) /
                                      (np.abs(o["kl_glob"]) + 1e-9)))
                ok = bool(rho >= 0.99 and med < 1e-2)
            else:
                rho, med, ok = np.nan, np.nan, False
            fidelity[name] = {"rows_match": bool(same_rows), "kl_rank_rho": rho,
                              "kl_median_rel_err": med, "pass": ok}
            print(f"  {name:8s} rows {'match' if same_rows else 'DIFFER'}   "
                  f"kl rank rho {rho:.5f}   median rel err {med:.2e}   "
                  f"{'ok' if ok else 'FAIL'}")
        else:
            fidelity[name] = {"rows_match": None, "pass": None}
            print(f"  {name:8s} no original to compare")

    print("\nProbe re-fit on each half (position-grouped splits)\n")
    hdr = f"{'assay':9s}" + "".join(f"{b[:12]:>13s}" for b in BLOCKS)
    print(hdr + "\n" + "-" * len(hdr))
    for f in files:
        d = np.load(f, allow_pickle=True)
        name = Path(f).stem.replace("gym2s_", "").split("_")[0]
        y, pos = d["score"], d["pos"]
        nL = int(d["n_layers"])
        line = f"{name:9s}"
        for b, names in BLOCKS.items():
            X = build(d, names)
            vals = []
            for s in range(a.seeds):
                rng = np.random.default_rng(s)
                tr, te = grouped_split(pos, 0.25, rng)
                if te.sum() < 5 or tr.sum() < 20:
                    continue
                rho, k, lam, idx = fit_ridge_block(X, y, pos, tr, te, rng)
                if np.isfinite(rho):
                    # SIGNED, matching compare_internal_output. Taking |rho|
                    # here would give a pure-noise block a positive mean (~0.17
                    # measured on a synthetic null) and make an uninformative
                    # half look like it carried signal.
                    vals.append(rho)
                    for i in idx:
                        sel[b][(names[i // nL], int(i % nL))] += 1
            per[b][name] = vals
            line += f"{np.mean(vals):13.3f}" if vals else f"{'-':>13s}"
        print(line)

    print("\nPooled (95% CI, hierarchical bootstrap over assays)\n")
    summary = {}
    for b in BLOCKS:
        pt, lo, hi, n = pi_stats.cluster_bootstrap(per[b], n_boot=10000, seed=0)
        summary[b] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                      "per_assay": {k: float(np.mean(v)) for k, v in per[b].items() if v}}
        print(f"  {b:14s} {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    print("\nPaired differences\n")
    gaps = {}
    for lab, ka, kb in GAPS:
        pt, lo, hi, n = pi_stats.paired_cluster_bootstrap(per[ka], per[kb],
                                                          n_boot=10000, seed=0)
        wins = sum(1 for as_ in per[ka]
                   for x, z in zip(per[ka][as_], per[kb][as_]) if x > z)
        tot = sum(len(v) for v in per[ka].values())
        gaps[lab] = {"gap": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins, "splits": tot}
        flag = "" if (np.isfinite(lo) and lo > 0) else "   <- includes zero"
        print(f"  {lab:26s} {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {wins}/{tot}{flag}")

    print("\nWhere each block selects its features (top layers)\n")
    top = {}
    for b in ("kl_only", "shift", "spread"):
        c = sel[b]
        fam = Counter()
        for (f_, L), n in c.items():
            fam[f_] += n
        lay = Counter()
        for (f_, L), n in c.items():
            lay[L] += n
        top[b] = {"by_family": dict(fam),
                  "top_layers": [int(l) for l, _ in lay.most_common(8)]}
        print(f"  {b:10s} {dict(fam)}  top layers {sorted(l for l, _ in lay.most_common(8))}")

    Path(a.out).write_text(json.dumps(
        {"fidelity": fidelity, "blocks": summary, "gaps": gaps,
         "selection": top,
         "protocol": {"seeds": a.seeds, "split": "position-grouped 25% held out",
                      "note": "blocks matched at 128 features except the two "
                              "256-feature references"}}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
