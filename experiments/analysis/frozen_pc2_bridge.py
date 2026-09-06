"""Does the ORIGINAL frozen PC2 carry the newer cross-cohort gap, or only dz?

The 2026-09-06 audit's Gate 1A. The newer output comparisons (geometry_heldout16,
geometry_panel5) show a full-128 and a two-training-fold-PC internal advantage
over 37 emitted-geometry features in all three models. None of that connects to
the mechanistic object of the original study, which is one specific direction:
PC2 of the development basis. This script makes the connection, or fails to.

FROZEN SPECIFICATION (stated before results were read; matches the frozen spec
in prot_interp_files/current_evidence_review_and_next_runs.md):

  model        Boltz-2 only. Its PC2 lives in Boltz-2's channel coordinates and
               cannot be applied to OpenFold3 or Protenix.
  basis        fitted on the 12 development assays (gym3_* captures), final
               layer, per-assay channel z-scoring, one SVD, sign fixed on
               kl_glob with orient_k=2 — analyze_heldout.py's construction,
               through the same pi_basis call.
  DMS sign     frozen on the development assays (mean per-assay Spearman of the
               PC2 coordinate against DMS, sign only). NEVER re-chosen on any
               test assay: a per-assay sign flip converts |rho| into rho.
  test rows    the exact Boltz-2 captures the newer comparisons used —
               xm_boltz2_r1_* in runs/xmodel_layers (heldout16) and
               runs/xmodel_panel5 (panel5), restricted to the assays the
               geometry archives analysed (their paired exclusions kept).
  comparators  read from the geometry archives for the SAME captures:
               geometry (37), rich output (10), internal full 128, internal at
               2 training-fold PCs. Nothing recomputed, so the pairing is on
               identical rows by construction.
  normalisation both. transductive: the test assay's own channel mean/sd
               (matches the LOAO convention). inductive/training-statistics:
               pooled development-assay channel statistics; nothing from the
               test protein enters.
  statistic    within-assay Spearman; groups summarised over assays with a
               cluster bootstrap; paired gaps with a paired cluster bootstrap,
               assay as the unit; reported separately for held-out stability
               (12), held-out other phenotype (4), and panel5.

Interpretation, stated in advance: a positive PC2-minus-geometry gap connects
the newer generalization result to the named direction. PC2 weakening on panel5
while full dz stays strong means PC2 is stability-focused while the broader
subspace is phenotype-general — that is a finding, not a failure.

    uv run python experiments/analysis/frozen_pc2_bridge.py \
        --out $W/runs/frozen_pc2_bridge.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax_harness"))

import pi_archive          # noqa: E402
import pi_basis            # noqa: E402
import pi_protocol         # noqa: E402
import pi_stats            # noqa: E402

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
EPS = 1e-9

# Held-out assays that do not measure folding stability — kept in step with
# analyze_heldout.py, which owns the assignment.
NON_STABILITY = {"CCDB", "ENVZ", "PHOT", "TAT"}


def load_assay(f):
    cap = pi_archive.load_capture(f)
    return {"X": cap.pair_row(-1),
            "y": np.asarray(cap.field("score"), float),
            "kl": np.asarray(cap.field("kl_glob"), float)[:, -1]}


def short(name):
    return name.split("_")[0]


def summarise(vals):
    g = {n: [v] for n, v in vals.items() if np.isfinite(v)}
    if not g:
        return None
    pt, lo, hi, k = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                               hierarchical=False)
    return {"mean": pt, "ci_lo": lo, "ci_hi": hi, "n_assays": k}


def paired(a_vals, b_vals, sel):
    ga = {n: [a_vals[n]] for n in sel}
    gb = {n: [b_vals[n]] for n in sel}
    pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(ga, gb, n_boot=10000,
                                                      seed=0,
                                                      hierarchical=False)
    wins = sum(a_vals[n] > b_vals[n] for n in sel)
    return {"gap": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins, "n": len(sel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis-glob", default=str(W / "runs/gym3_*.npz"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # ------------------------------------------------------------------
    # the development basis, built once, and both signs frozen on it
    # ------------------------------------------------------------------
    B = {}
    for f in sorted(glob.glob(a.basis_glob)):
        B[Path(f).stem.split("_", 1)[1]] = load_assay(f)
    bn = sorted(B)
    if len(bn) != 12:
        raise SystemExit(f"expected the 12 development assays, got {len(bn)} "
                         f"from {a.basis_glob}")
    Bs = pi_basis.fit({n: B[n]["X"] for n in bn}, layer=-1,
                      orient_on="kl_glob",
                      orient_ref={n: B[n]["kl"] for n in bn},
                      orient_k=2, eps=EPS)
    V = Bs.components
    sgn = float(np.sign(np.mean(
        [pi_stats.spearman(Bs.project(B[n]["X"], layer=-1)[:, 1], B[n]["y"])
         for n in bn])))
    print(f"basis: {len(bn)} development assays; PC2 DMS sign frozen at "
          f"{sgn:+.0f}\n")

    # ------------------------------------------------------------------
    # apply to the exact rows of the newer comparisons
    # ------------------------------------------------------------------
    cohorts = {
        "heldout16": {"captures": W / "runs/xmodel_layers",
                      "geometry": W / "runs/geometry_heldout16.json"},
        "panel5":    {"captures": W / "runs/xmodel_panel5",
                      "geometry": W / "runs/geometry_panel5.json"},
    }

    out = {"cohorts": {}, "pc2_dms_sign_frozen": sgn}
    for cname, c in cohorts.items():
        G = json.loads(Path(c["geometry"]).read_text())
        boltz = G["models"]["boltz2"]
        assays = G["assays"]
        rows, missing = {}, []
        for assay in assays:
            path = c["captures"] / f"xm_boltz2_r1_{assay}.npz"
            if not path.exists():
                missing.append(assay)
                continue
            d = load_assay(path)
            Zt = Bs.features(d["X"], layer=-1)
            Zi = Bs.features(d["X"], layer=-1, standardise="train")
            k = short(assay)
            rows[k] = {
                "assay": assay, "n": int(len(d["y"])),
                "pc2_transductive": sgn * pi_stats.spearman(Zt @ V[1], d["y"]),
                "pc2_inductive":    sgn * pi_stats.spearman(Zi @ V[1], d["y"]),
                # comparators for the SAME capture, from the geometry archive
                "geometry37": boltz["per_assay"]["geometry"][k],
                "rich10":     boltz["per_assay"]["rich"][k],
                "internal128": boltz["per_assay"]["internal"][k],
                "internal_2pc": boltz["curve_per_assay"]["internal"]["2"][k],
            }
        if missing:
            raise SystemExit(f"{cname}: geometry archive analysed assays with "
                             f"no capture on disk: {missing}")

        groups = {"all": sorted(rows)}
        if cname == "heldout16":
            groups["stability"] = [k for k in sorted(rows)
                                   if k not in NON_STABILITY]
            groups["non_stability"] = [k for k in sorted(rows)
                                       if k in NON_STABILITY]

        KEYS = ["pc2_transductive", "pc2_inductive", "internal_2pc",
                "internal128", "geometry37", "rich10"]
        print(f"=== {cname}  ({len(rows)} assays, identical rows to "
              f"{Path(c['geometry']).name}) ===")
        print(f"{'assay':8s}" + "".join(f"{k[:14]:>16s}" for k in KEYS))
        for k in sorted(rows):
            print(f"{k:8s}" + "".join(f"{rows[k][q]:>+16.3f}" for q in KEYS))
        pos = sum(rows[k]["pc2_transductive"] > 0 for k in rows)
        print(f"  frozen-sign PC2 positive in {pos}/{len(rows)} assays "
              f"(transductive)")

        cohort_out = {"n_assays": len(rows), "per_assay": rows,
                      "pc2_sign_consistency": {"positive": pos,
                                               "n": len(rows)},
                      "summary": {}, "paired": {}}
        for gname, sel in groups.items():
            if not sel:
                continue
            cohort_out["summary"][gname] = {
                key: summarise({k: rows[k][key] for k in sel}) for key in KEYS}
            gp = {}
            for aa, bb in (("pc2_transductive", "geometry37"),
                           ("pc2_transductive", "rich10"),
                           ("pc2_inductive", "geometry37"),
                           ("pc2_inductive", "rich10"),
                           ("internal_2pc", "pc2_transductive"),
                           ("internal128", "pc2_transductive")):
                va = {k: rows[k][aa] for k in sel}
                vb = {k: rows[k][bb] for k in sel}
                gp[f"{aa} - {bb}"] = paired(va, vb, sel)
            cohort_out["paired"][gname] = gp
            print(f"\n  -- {gname} ({len(sel)})")
            for pk, pv in gp.items():
                flag = "" if (pv["ci_lo"] > 0 or pv["ci_hi"] < 0) \
                    else "   <- includes zero"
                print(f"     {pk:44s} {pv['gap']:+.3f} "
                      f"[{pv['ci_lo']:+.3f}, {pv['ci_hi']:+.3f}]  "
                      f"{pv['wins']}/{pv['n']}{flag}")
        print()
        out["cohorts"][cname] = cohort_out

    proto = pi_protocol.protocol(
        script="frozen_pc2_bridge.py",
        design="frozen development PC2 (basis and DMS sign fixed on the 12 "
               "development assays, never re-chosen) applied to the exact "
               "Boltz-2 rows of the newer heldout16/panel5 output "
               "comparisons; comparators read from the geometry archives for "
               "the same captures",
        layer=pi_protocol.layers("final"),
        features=pi_protocol.features("frozen PC2 coordinate of dz_site", 128,
                                      kept=1),
        source=f"basis {a.basis_glob}; rows from geometry_heldout16.json / "
               f"geometry_panel5.json captures",
        n_assays=sum(c["n_assays"] for c in out["cohorts"].values()),
        normalisation="both: transductive (test assay's own channel stats) "
                      "and inductive (development pooled stats)",
        **Bs.protocol)
    pi_archive.write_result(a.out, out, protocol=proto, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
