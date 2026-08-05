"""Read the conformational-axis run: is the trunk bimodal, and does it move?

Three questions, in the order that decides whether the later ones mean anything.

1. BIMODALITY. At the axis pairs, does wild type's distogram carry mass near
   both states' distances, or has it already committed to one? If it is
   unimodal on the dominant state there is no room for a mutation to move
   anything, and a null projection would say nothing about the model. This is
   the precondition, not a result.

2. DIRECTION. Each variant's shift is projected onto the A-B axis and the
   projection is signed: positive is toward Ltn10, negative toward Ltn40. The
   design is two-sided -- V21C/V59C must come out positive and A36C/A49C
   negative -- so a probe reacting to generic perturbation fails half the time.
   Confidence intervals come from a cluster bootstrap over RESIDUES, not over
   pairs: pairs sharing a residue are not independent and treating 1682 pairs as
   1682 observations would shrink every interval by roughly the square root of
   the pairs-per-residue.

3. CONFOUND. V21-V59 is the single most discriminating pair on the axis (4.6 A
   in Ltn10, 28.1 A in Ltn40), so a model that merely forms a disulfide between
   two cysteines lands on Ltn10 without knowing anything. Every projection is
   therefore also reported with the crosslinked residues excluded, and against
   the serine controls at the same positions. A36-A49 does not have this problem
   (6.8 vs 5.7 A), which is why it is the primary test.

The emitted structure is scored against the same two references, so the
internal-versus-output contrast is available on the same rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_conf  # noqa: E402
import pi_stats  # noqa: E402

# which residues carry each variant's crosslink, for the exclusion control
CROSSLINK = {"V21C_V59C": (21, 59), "V21S_V59S": (21, 59),
             "A36C_A49C": (36, 49), "A36S_A49S": (36, 49)}
EXPECT = {"V21C_V59C": +1, "V21S_V59S": +1, "A36C_A49C": -1, "A36S_A49S": -1,
          "W55D": +1}


def mean_ci(values, groups):
    """Cluster bootstrap over residues; `groups` is one residue id per value."""
    g = {}
    for v, k in zip(values, groups):
        g.setdefault(int(k), []).append(float(v))
    pt, lo, hi, n = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0)
    return pt, lo, hi, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--window", type=float, default=1.5,
                    help="angstrom half-width for the bimodality mass windows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = np.load(a.run, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    L = d["proj"].shape[1]
    layer = a.layer % L
    pi_, pj = d["pair_i"], d["pair_j"]
    da, db = d["d_a_pairs"], d["d_b_pairs"]
    c = pi_conf.bin_centers()
    res = {"family": str(d["family"]), "layer": int(layer), "n_layers": int(L),
           "n_pairs": int(len(pi_)), "capture_drift": float(d["capture_drift"])}

    # ---- 1. is wild type bimodal at the axis pairs? ----------------------
    p = d["p_wt_pairs"][layer].astype(np.float64)          # [P, 64]
    wA = (np.abs(c[None, :] - da[:, None]) <= a.window)
    wB = (np.abs(c[None, :] - db[:, None]) <= a.window)
    mass_a = (p * wA).sum(1)
    mass_b = (p * wB).sum(1)
    both = (mass_a > 0.10) & (mass_b > 0.10)
    print(f"{res['family']}  layer {layer} of {L}   {len(pi_)} axis pairs "
          f"(capture drift {res['capture_drift']:.2e})\n")
    print("1. Wild-type bimodality at the axis pairs "
          f"(mass within +-{a.window} A)\n")
    print(f"   mass near Ltn10 distance   {mass_a.mean():.3f}")
    print(f"   mass near Ltn40 distance   {mass_b.mean():.3f}")
    print(f"   pairs with >10% on BOTH    {both.sum()} / {len(pi_)} "
          f"({100*both.mean():.1f}%)")
    ratio = float(mass_a.mean() / max(mass_b.mean(), 1e-9))
    # The fraction of pairs carrying some mass on both is not the whole story --
    # a 10:1 mass ratio means the trunk has largely committed, and "there is a
    # minority of pairs with room to move" is a much weaker statement than
    # "the trunk represents an ensemble".
    if ratio < 2 and both.mean() > 0.25:
        verdict = "genuinely bimodal: the trunk carries both states"
    elif both.mean() > 0.05:
        verdict = (f"predominantly one state ({ratio:.0f}:1 mass ratio) with a "
                   f"{100*both.mean():.0f}% minority of pairs retaining mass on "
                   f"the other -- limited room to move")
    else:
        verdict = "committed to one state -- a null projection is uninformative"
    print(f"   mass ratio                 {ratio:.1f} : 1")
    print(f"   -> {verdict}\n")
    res["bimodality"] = {"mass_a": float(mass_a.mean()), "mass_b": float(mass_b.mean()),
                         "frac_both": float(both.mean()), "n_both": int(both.sum()),
                         "mass_ratio": ratio, "verdict": verdict}

    # ---- 2 & 3. signed projection, with and without the crosslink --------
    print("2. Signed projection onto the Ltn10 <- -> Ltn40 axis")
    print("   positive = toward Ltn10 (state A), negative = toward Ltn40 (state B)\n")
    hdr = (f"   {'variant':12s} {'expect':>7s} {'proj':>18s} {'excl. crosslink':>22s} "
           f"{'sign':>6s}")
    print(hdr + "\n   " + "-" * (len(hdr) - 3))
    out = {}
    for k, nm in enumerate(names):
        if nm == "WT":
            continue
        pp = d["proj_pairs"][k, layer].astype(np.float64)
        pt, lo, hi, _ = mean_ci(pp, pi_)
        cl = CROSSLINK.get(nm)
        if cl:
            keep = ~(np.isin(pi_, cl) | np.isin(pj, cl))
            pt2, lo2, hi2, _ = mean_ci(pp[keep], pi_[keep])
        else:
            pt2, lo2, hi2 = pt, lo, hi
        exp = EXPECT.get(nm, 0)
        ok = "OK" if (np.sign(pt2) == exp and (lo2 > 0 or hi2 < 0)) else \
             ("n.s." if lo2 <= 0 <= hi2 else "WRONG")
        print(f"   {nm:12s} {('+' if exp > 0 else '-'):>7s} "
              f"{pt:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{pt2:+.4f} [{lo2:+.4f},{hi2:+.4f}] {ok:>6s}")
        out[nm] = {"proj": pt, "ci": [lo, hi], "proj_excl": pt2, "ci_excl": [lo2, hi2],
                   "expected_sign": exp, "verdict": ok,
                   "n_pairs_excl": int(keep.sum()) if cl else int(len(pp))}
    res["projection"] = out

    # disulfide-specific effect: cysteine minus its serine control
    print("\n3. Disulfide-specific effect (cysteine minus serine at same positions)\n")
    contrasts = {}
    for cys, ser in (("V21C_V59C", "V21S_V59S"), ("A36C_A49C", "A36S_A49S")):
        if cys not in names or ser not in names:
            continue
        i, j = names.index(cys), names.index(ser)
        cl = CROSSLINK[cys]
        keep = ~(np.isin(pi_, cl) | np.isin(pj, cl))
        diff = (d["proj_pairs"][i, layer] - d["proj_pairs"][j, layer])[keep]
        pt, lo, hi, _ = mean_ci(diff, pi_[keep])
        contrasts[f"{cys} - {ser}"] = {"gap": pt, "ci": [lo, hi]}
        print(f"   {cys} - {ser}   {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
              f"{'' if (lo > 0 or hi < 0) else '   <- includes zero'}")
    res["disulfide_contrast"] = contrasts

    # ---- is the movement SPECIFIC to the axis, or just movement? ---------
    if "cos" in d.files:
        print("\n3b. Specificity: is the shift along the axis, or merely a shift?\n")
        print(f"   {'variant':12s} {'|dp|':>8s} {'cos':>8s} {'proj':>9s} "
              f"{'permuted axis':>18s} {'z':>7s}")
        spec = {}
        for k, nm in enumerate(names):
            if nm == "WT":
                continue
            pr_ = float(d["proj"][k, layer])
            mu, sd = float(d["null_mu"][k, layer]), float(d["null_sd"][k, layer])
            z = (pr_ - mu) / sd if sd > 0 else np.nan
            spec[nm] = {"dnorm": float(d["dnorm"][k, layer]),
                        "cos": float(d["cos"][k, layer]), "proj": pr_,
                        "null_mu": mu, "null_sd": sd, "z": z}
            print(f"   {nm:12s} {spec[nm]['dnorm']:8.4f} {spec[nm]['cos']:+8.4f} "
                  f"{pr_:+9.4f} {mu:+9.4f}+-{sd:.4f} {z:+7.1f}")
        res["specificity"] = spec
        rand = 1.0 / np.sqrt(p.shape[1])
        print(f"\n   The permuted-axis null is WEAK and its z-scores should not be")
        print(f"   read as specificity. Permuting destroys the pair correspondence,")
        print(f"   and a different pair's axis vector lives at entirely different")
        print(f"   distances, so its overlap with this pair's movement is ~0 by")
        print(f"   construction. It rejects 'no movement', not 'no direction'.")
        print(f"\n   The cosine is the informative number: |cos| ~ {rand:.3f} is what a")
        print(f"   random direction gives in {p.shape[1]} bins. Observed values sit "
              f"near that.")

        # The test that does discriminate: variants predicted to move in
        # OPPOSITE directions must separate in the predicted order. This needs
        # no null at all -- it is internal to the design.
        print("\n3c. Directional discrimination (needs no null)\n")
        disc = {}
        pos = [n for n in EXPECT if EXPECT.get(n, 0) > 0 and n in spec]
        neg = [n for n in EXPECT if EXPECT.get(n, 0) < 0 and n in spec]
        for a_ in pos:
            for b_ in neg:
                dv = spec[a_]["proj"] - spec[b_]["proj"]
                ok = dv > 0
                disc[f"{a_} vs {b_}"] = {"delta": dv, "ordered_correctly": bool(ok)}
                print(f"   {a_:12s} (expect +) vs {b_:12s} (expect -): "
                      f"delta {dv:+.4f}  {'ordered correctly' if ok else 'WRONG ORDER'}")
        res["discrimination"] = disc
        nok = sum(v["ordered_correctly"] for v in disc.values())
        print(f"\n   {nok}/{len(disc)} opposite-expectation pairs ordered correctly.")

    # ---- the emitted structure, same references --------------------------
    print("\n4. Does the emitted STRUCTURE move too?  (mean |d_pred - d_ref|, A)\n")
    print(f"   {'variant':12s} {'|d-Ltn10|':>10s} {'|d-Ltn40|':>10s} "
          f"{'leans':>8s} {'pLDDT':>7s}")
    st = {}
    for k, nm in enumerate(names):
        ea, eb = float(d["struct_err_a"][k]), float(d["struct_err_b"][k])
        st[nm] = {"err_a": ea, "err_b": eb, "plddt": float(d["plddt"][k])}
        print(f"   {nm:12s} {ea:10.2f} {eb:10.2f} "
              f"{('Ltn10' if ea < eb else 'Ltn40'):>8s} {d['plddt'][k]:7.3f}")
    res["structure"] = st

    # per-layer trace of the primary test, for the figure
    res["per_layer"] = {nm: d["proj"][k].tolist() for k, nm in enumerate(names)}
    res["wt_overlap_a"] = d["wt_overlap_a"].tolist()
    res["wt_overlap_b"] = d["wt_overlap_b"].tolist()

    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
