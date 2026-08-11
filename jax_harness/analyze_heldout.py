"""Does the frozen PC2 direction work on proteins that never touched the basis?

Every transfer number in this project so far is leave-one-assay-out INSIDE the
twelve proteins that built the basis. That is a real held-out test, but the
eleven training assays and the twelfth are all Tsuboyama cDNA-display
proteolysis, all 40-72 residues, all measuring the same thing. Two questions
survive it:

  1. does a basis frozen on those twelve transfer to proteins chosen after the
     fact, with no re-fitting of any kind;
  2. is the direction about STABILITY, or about mutation severity generally?

The panel here answers both because of how it was picked: sixteen ProteinGym
assays outside the basis panel, twelve of them more Tsuboyama stability, and
FOUR measuring something else entirely -- ccdB toxicity by growth, EnvZ
signalling by fluorescent reporter, phototropin by fluorescence, and HIV Tat by
viral replication. If PC2 predicts the four at roughly the stability level then
"mutation-severity direction" is a phenotype-general claim; if it predicts them
near zero the direction is specifically about folding, and the abstract has to
say so. Either way the answer is worth having, and the captures already exist.

WHAT IS FROZEN. Everything. The basis is built once from the twelve gym3_*
archives by the recipe the report states -- final-layer dz_site, z-scored per
channel within each protein, stacked, one SVD, sign fixed on kl_glob -- and then
applied unchanged. No held-out label is touched by any fit, and the sign is not
re-chosen per protein, which would quietly convert |rho| into rho.

BASIS CAPTURE. gym3_*, not gym2s_*, even though the published basis numbers come
from gym2s. The held-out panel was captured by exp_gym3, so using gym2s for the
basis would confound "unseen protein" with "different capture version". The two
agree at kl_glob rank rho 0.970-0.9998, so this costs nothing and removes a
confound. `--basis-glob` overrides it, and the report should carry both.

TWO NORMALISATIONS, because they answer different questions:

  transductive  the held-out protein's own channel mean and sd. Uses held-out
                INPUTS but no labels. This is what the leave-one-assay-out code
                does, so it is the comparable number.
  inductive     training-assay channel statistics only. Nothing whatsoever from
                the held-out protein enters. Stricter, and the honest number for
                "could you score a new protein today".

BASELINES, all frozen the same way -- fitted on the twelve, applied unchanged:

  chemistry     17 descriptors from the two amino-acid letters. This is the
                deciding control: a chemistry model transfers across proteins
                trivially, so "PC2 transfers" means nothing unless PC2 beats it.
  full dz       all 128 channels through ridge. The upper reference.
  PC1           the volume axis, as a within-basis contrast.
  random        20 random unit directions in the same 128-space, which bounds
                how much of PC2's performance is "any direction has some signal".

The output side is deliberately absent. Per-residue pLDDT is the best output
block in the report, and it cannot appear here at all: its width is the chain
length, so it is not a fixed-width block that can transfer between proteins of
different sizes. That is a real limitation of the output representation rather
than an omission, and it is stated rather than worked around.
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
import pi_metrics  # noqa: E402
import pi_stats  # noqa: E402
from compare_internal_output import grouped_split  # noqa: E402

EPS = 1e-9

# The four held-out assays that do not measure folding stability. Kept explicit
# rather than parsed out of the filename so that adding an assay to the panel
# forces a decision about which group it belongs in.
NON_STABILITY = {
    "CCDB_ECOLI_Tripathi_2016":  "ccdB toxicity (growth)",
    "ENVZ_ECOLI_Ghose_2023":     "EnvZ signalling (reporter)",
    "PHOT_CHLRE_Chen_2023":      "phototropin (fluorescence)",
    "TAT_HV1BR_Fernandes_2016":  "HIV Tat (viral replication)",
}


def zc(M):
    return (M - M.mean(0)) / (M.std(0) + EPS)


def short(name):
    return name.split("_")[0]


def load_assay(f):
    d = np.load(f, allow_pickle=True)
    return {"X": np.asarray(d["dz_site"], float)[:, -1, :],
            "y": np.asarray(d["score"], float),
            "kl": np.asarray(d["kl_glob"], float)[:, -1],
            "mutant": [str(m) for m in d["mutant"]],
            "pos": np.asarray(d["pos"]),
            "n_res": int(np.asarray(d["plddt_res"]).shape[1])}


def ridge_train(X, y, lam):
    """Closed-form ridge on centred, already-standardised features."""
    n, p = X.shape
    A = X.T @ X + lam * np.eye(p)
    return np.linalg.solve(A, X.T @ y)


def within_assay(X, y, pos, lam, seeds, frac, seed0):
    """The ordinary within-assay probe, on position-grouped splits.

    Needed so that "frozen beats within-assay" is a claim about the SAME
    proteins. Comparing a frozen probe measured here against the within-assay
    number reported for the BASIS panel would compare two protein sets and call
    the difference a method effect.
    """
    out = []
    for s in range(seeds):
        rng = np.random.default_rng(seed0 + s)
        tr, te = grouped_split(pos, frac, rng)
        if tr.sum() < 10 or te.sum() < 5:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + EPS
        ym, ys = y[tr].mean(), y[tr].std() + EPS
        w = ridge_train((X[tr] - mu) / sd, (y[tr] - ym) / ys, lam)
        out.append(pi_stats.spearman((X[te] - mu) / sd @ w, y[te]))
    return float(np.nanmean(out)) if out else np.nan


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/"
    ap.add_argument("--basis-glob", default=R + "runs/gym3_*.npz")
    ap.add_argument("--heldout-glob", default=R + "runs/gym3p3_*.npz")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref", default=R + "data/gym/ref.csv",
                    help="ProteinGym reference table, for DMS_binarization_cutoff")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # ------------------------------------------------------------------
    # the basis, built once and then never touched again
    # ------------------------------------------------------------------
    B = {}
    for f in sorted(glob.glob(a.basis_glob)):
        B[Path(f).stem.split("_", 1)[1]] = load_assay(f)
    bn = sorted(B)
    if not bn:
        raise SystemExit(f"no basis archives matched {a.basis_glob}")
    print(f"basis: {len(bn)} assays, {sum(len(B[n]['y']) for n in bn)} variants")

    Xg = np.concatenate([zc(B[n]["X"]) for n in bn], 0)
    gm = Xg.mean(0)
    V = np.linalg.svd(Xg - gm, full_matrices=False)[2]              # (128, 128)
    # Orient on kl_glob, exactly as analyze_svd/analyze_chem do. Singular-vector
    # signs are arbitrary; without a fixed convention the held-out correlations
    # would carry a meaningless sign and averaging them would cancel.
    for c in (0, 1):
        g = {n: [pi_stats.spearman((zc(B[n]["X"]) - gm) @ V[c], B[n]["kl"])]
             for n in bn}
        if pi_stats.cluster_bootstrap(g, n_boot=2000, seed=0,
                                      hierarchical=False)[0] < 0:
            V[c] = -V[c]

    # A second, separate sign: which way the component points relative to DMS.
    # Only the non-Spearman metrics need it (they are not sign-invariant), and
    # like everything else it is fixed on the BASIS and never re-chosen.
    sgn_dms = {c: float(np.sign(np.mean(
        [pi_stats.spearman((zc(B[n]["X"]) - gm) @ V[c], B[n]["y"]) for n in bn])))
        for c in (0, 1)}
    print(f"   DMS orientation from the basis: PC1 {sgn_dms[0]:+.0f}, "
          f"PC2 {sgn_dms[1]:+.0f}")

    # raw (un-z-scored) training channel statistics, for the inductive variant
    Xraw = np.concatenate([B[n]["X"] for n in bn], 0)
    mu_tr, sd_tr = Xraw.mean(0), Xraw.std(0) + EPS

    # frozen ridge baselines, fitted on the pooled basis with targets z-scored
    # per assay so no protein's dynamic range dominates
    ytr = np.concatenate([(B[n]["y"] - B[n]["y"].mean()) / (B[n]["y"].std() + EPS)
                          for n in bn], 0)
    w_dz = ridge_train(Xg - gm, ytr, a.lam)

    Ctr = np.concatenate([pi_chem.chem_matrix(B[n]["mutant"]) for n in bn], 0)
    cmu, csd = Ctr.mean(0), Ctr.std(0) + EPS
    w_chem = ridge_train((Ctr - cmu) / csd, ytr, a.lam)

    rng = np.random.default_rng(a.seed)
    Rdirs = rng.normal(size=(a.n_random, V.shape[1]))
    Rdirs /= np.linalg.norm(Rdirs, axis=1, keepdims=True)

    # sanity: within the basis itself, so the held-out numbers have a reference
    within = {n: pi_stats.spearman((zc(B[n]["X"]) - gm) @ V[1], B[n]["y"])
              for n in bn}
    wpt, wlo, whi, _ = pi_stats.cluster_bootstrap(
        {n: [v] for n, v in within.items()}, n_boot=10000, seed=0,
        hierarchical=False)
    print(f"   PC2 inside the basis (descriptive, not held out): "
          f"{wpt:+.3f} [{wlo:+.3f}, {whi:+.3f}]\n")

    # ------------------------------------------------------------------
    # apply, unchanged, to every held-out protein
    # ------------------------------------------------------------------
    H = {}
    for f in sorted(glob.glob(a.heldout_glob)):
        H[Path(f).stem.split("_", 1)[1]] = load_assay(f)
    hn = sorted(H)
    overlap = sorted(set(hn) & set(bn))
    if overlap:
        raise SystemExit(f"held-out panel overlaps the basis: {overlap}")
    print(f"held out: {len(hn)} assays, {sum(len(H[n]['y']) for n in hn)} "
          f"variants, none in the basis\n")

    # ProteinGym binarisation cutoffs, for the metric-robustness check
    cut = {}
    try:
        import csv
        for r_ in csv.DictReader(open(a.ref)):
            v = r_.get("DMS_binarization_cutoff", "")
            if v not in ("", None):
                cut[r_["DMS_id"]] = float(v)
    except Exception as e:                                   # noqa: BLE001
        print(f"   no binarisation cutoffs ({e}); AUC/MCC will be blank")

    rows = {}
    for n in hn:
        d = H[n]
        Zt = zc(d["X"]) - gm                       # transductive
        Zi = (d["X"] - mu_tr) / sd_tr - gm         # inductive
        C = (pi_chem.chem_matrix(d["mutant"]) - cmu) / csd
        y = d["y"]
        rr = [pi_stats.spearman(Zt @ r, y) for r in Rdirs]
        rows[n] = {
            "group": "non-stability" if n in NON_STABILITY else "stability",
            "phenotype": NON_STABILITY.get(n, "folding stability (proteolysis)"),
            "n": int(len(y)), "n_res": d["n_res"],
            "n_pos": int(len(np.unique(d["pos"]))),
            "dms_sd": float(np.std(y)),
            "pc2_transductive": pi_stats.spearman(Zt @ V[1], y),
            "pc2_inductive": pi_stats.spearman(Zi @ V[1], y),
            "pc1_transductive": pi_stats.spearman(Zt @ V[0], y),
            "chem_frozen": pi_stats.spearman(C @ w_chem, y),
            "dz_frozen": pi_stats.spearman(Zt @ w_dz, y),
            "dz_within": within_assay(d["X"], y, d["pos"], a.lam, a.seeds,
                                      a.frac, a.seed),
            "random_absmax": float(np.nanmax(np.abs(rr))),
            "random_absmean": float(np.nanmean(np.abs(rr))),
        }
        # Same predictions, four metrics. Orientation is frozen from the basis
        # (sgn_dms), never re-chosen here.
        rows[n]["metrics"] = {
            "PC2 frozen": pi_metrics.all_metrics(sgn_dms[1] * (Zt @ V[1]), y,
                                                 cut.get(n)),
            "chemistry frozen": pi_metrics.all_metrics(C @ w_chem, y, cut.get(n)),
            "full dz frozen": pi_metrics.all_metrics(Zt @ w_dz, y, cut.get(n)),
        }
        rows[n]["cutoff"] = cut.get(n)

    # ------------------------------------------------------------------
    # report. Every block is summarised over ASSAYS, which is the unit.
    # ------------------------------------------------------------------
    # PC2 is oriented on kl_glob, so against DMS it runs NEGATIVE -- that is the
    # report's convention and the published number is -0.652. The ridge blocks
    # are fitted to predict DMS and run positive. Comparing the two signed
    # quantities would subtract opposite-signed numbers and produce a margin of
    # about -1.0 that means nothing, so every paired test below is on |rho| and
    # the per-assay table keeps the signed value for comparability with the
    # report. Magnitudes are unaffected by which convention is used.
    for n in hn:
        for k in ("pc2_transductive", "pc2_inductive", "pc1_transductive",
                  "chem_frozen", "dz_frozen", "dz_within"):
            rows[n]["abs_" + k] = abs(rows[n][k])

    KEYS = [("pc2_transductive", "PC2 frozen (transductive)"),
            ("pc2_inductive",    "PC2 frozen (inductive)"),
            ("chem_frozen",      "chemistry frozen (17)"),
            ("dz_frozen",        "full dz frozen (128)"),
            ("dz_within",        "full dz within-assay (128)"),
            ("pc1_transductive", "PC1 frozen (volume axis)"),
            ("random_absmean",   "mean of 20 random dirs |rho|"),
            ("random_absmax",    "best of 20 random dirs |rho|")]

    def summarise(sel, key):
        g = {n: [rows[n][key]] for n in sel if np.isfinite(rows[n][key])}
        if not g:
            return None
        pt, lo, hi, k = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                   hierarchical=False)
        return {"mean": pt, "ci_lo": lo, "ci_hi": hi, "n_assays": k}

    groups = {"all 16": hn,
              "stability (12)": [n for n in hn if n not in NON_STABILITY],
              "non-stability (4)": [n for n in hn if n in NON_STABILITY]}

    print(f"{'assay':34s}{'n':>5s}{'len':>5s}{'sd':>7s}"
          + "".join(f"{lab.split(' (')[0][:13]:>15s}" for _, lab in KEYS[:5]))
    for gname, sel in groups.items():
        print(f"\n-- {gname}")
        for n in sel:
            r = rows[n]
            print(f"   {short(n) + ' ' + r['phenotype'][:22]:31s}"
                  f"{r['n']:>5d}{r['n_res']:>5d}{r['dms_sd']:>7.2f}"
                  + "".join(f"{r[k]:>+15.3f}" for k, _ in KEYS[:5]))

    summary, summary_abs = {}, {}
    print(f"\n{'block':34s}" + "".join(f"{g:>22s}" for g in groups))
    for key, lab in KEYS:
        summary[key] = {g: summarise(sel, key) for g, sel in groups.items()}
        # the same blocks on |rho|, which is what a figure can put on one axis
        akey = key if key.startswith("random") else "abs_" + key
        summary_abs[key] = {"label": lab,
                            **{g: summarise(sel, akey) for g, sel in groups.items()}}
        cells = []
        for g in groups:
            s = summary[key][g]
            cells.append("" if s is None else
                         f"{s['mean']:+.3f} [{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]")
        print(f"   {lab:31s}" + "".join(f"{c:>22s}" for c in cells))

    # paired margins, per assay, on |rho| (see the sign note above)
    print("\npaired per-assay margins on |rho| (the unit is the assay)\n")
    pairs = {}
    for aa, bb in (("abs_pc2_transductive", "abs_chem_frozen"),
                   ("abs_pc2_transductive", "random_absmax"),
                   ("abs_pc2_transductive", "random_absmean"),
                   ("abs_dz_frozen", "abs_chem_frozen"),
                   ("abs_dz_frozen", "abs_pc2_transductive"),
                   ("abs_dz_frozen", "abs_dz_within"),
                   ("abs_pc2_transductive", "abs_dz_within"),
                   ("abs_pc2_transductive", "abs_pc2_inductive")):
        for gname, sel in groups.items():
            ga = {n: [rows[n][aa]] for n in sel}
            gb = {n: [rows[n][bb]] for n in sel}
            pt, lo, hi, k = pi_stats.paired_cluster_bootstrap(
                ga, gb, n_boot=10000, seed=0, hierarchical=False)
            wins = sum(rows[n][aa] > rows[n][bb] for n in sel)
            pairs[f"{aa} - {bb} | {gname}"] = {
                "mean": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins, "n": len(sel)}
            print(f"   {aa[:20]:21s} - {bb[:18]:19s} {gname:19s} "
                  f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  {wins}/{len(sel)}")

    # ------------------------------------------------------------------
    # is the conclusion metric-bound?
    # ------------------------------------------------------------------
    print("\nthe same predictions under four metrics "
          "(mean over assays; AUC/MCC use the ProteinGym cutoff)\n")
    METRICS = [("spearman", "Spearman |rho|"), ("auc", "AUC"), ("mcc", "MCC"),
               ("ndcg_top10", "NDCG@10%"), ("recall_top10", "recall@10%")]
    BLOCKS = ["PC2 frozen", "chemistry frozen", "full dz frozen"]
    metric_tab = {}
    for gname, sel in groups.items():
        sel = [n for n in sel if rows[n].get("cutoff") is not None or True]
        print(f"-- {gname}")
        print(f"   {'block':20s}" + "".join(f"{lab:>16s}" for _, lab in METRICS))
        metric_tab[gname] = {}
        for blk in BLOCKS:
            cells, store = [], {}
            for mk, _ in METRICS:
                vals = [rows[n]["metrics"][blk][mk] for n in sel]
                vals = [abs(v) if mk == "spearman" else v
                        for v in vals if v is not None and np.isfinite(v)]
                if not vals:
                    cells.append("")
                    continue
                m = float(np.mean(vals))
                store[mk] = {"mean": m, "n": len(vals)}
                cells.append(f"{m:.3f}")
            metric_tab[gname][blk] = store
            print(f"   {blk:20s}" + "".join(f"{c:>16s}" for c in cells))
        # does the ordering of blocks change with the metric?
        orders = {}
        for mk, lab in METRICS:
            ok = [b for b in BLOCKS if mk in metric_tab[gname][b]]
            orders[lab] = tuple(sorted(ok, key=lambda b:
                                       -metric_tab[gname][b][mk]["mean"]))
        uniq = set(orders.values())
        metric_tab[gname]["_orderings"] = {k: list(v) for k, v in orders.items()}
        metric_tab[gname]["_stable"] = len(uniq) == 1
        print(f"   ordering of the three blocks is "
              f"{'IDENTICAL' if len(uniq) == 1 else 'NOT identical'} "
              f"under all {len(METRICS)} metrics\n")

    # Is the non-stability shortfall about phenotype, or about assay quality?
    # The four non-stability assays are also longer and two of them have a much
    # narrower DMS spread, so the contrast is confounded and the confound is
    # measurable rather than something to wave at.
    print("\nconfound check across the 16 held-out assays\n")
    conf = {}
    for lab, v in (("DMS sd", [rows[n]["dms_sd"] for n in hn]),
                   ("chain length", [rows[n]["n_res"] for n in hn]),
                   ("sites", [rows[n]["n_pos"] for n in hn])):
        r_all = pi_stats.spearman(v, [rows[n]["abs_pc2_transductive"] for n in hn])
        sel = [n for n in hn if n not in NON_STABILITY]
        r_stab = pi_stats.spearman([rows[n]["dms_sd" if lab == "DMS sd" else
                                            ("n_res" if lab == "chain length"
                                             else "n_pos")] for n in sel],
                                   [rows[n]["abs_pc2_transductive"] for n in sel])
        conf[lab] = {"all16": r_all, "stability_only": r_stab}
        print(f"   |PC2 rho| vs {lab:14s} all 16 {r_all:+.3f}   "
              f"within the 12 stability {r_stab:+.3f}")
    print("   A strong positive here means the phenotype contrast is partly an\n"
          "   assay-quality contrast and the four cannot be read as a clean test.")

    # The four non-stability assays are 60-118 residues against the stability
    # panel's 40-72, and |PC2 rho| falls with length, so the phenotype contrast
    # and a length contrast are partly the same contrast. Length-match what can
    # be matched: for each of the four, compare against stability assays within
    # +/-10 residues. Only ENVZ has neighbours, which is itself the finding --
    # the other three cannot be matched at all and their gap stays confounded.
    print("\nlength-matched contrast (+/-10 residues)\n")
    matched = {}
    stab = [n for n in hn if n not in NON_STABILITY]
    for n in sorted(NON_STABILITY):
        if n not in rows:
            continue
        near = [m for m in stab if abs(rows[m]["n_res"] - rows[n]["n_res"]) <= 10]
        if not near:
            matched[n] = {"n_res": rows[n]["n_res"], "neighbours": [],
                          "note": "no stability assay within 10 residues"}
            print(f"   {short(n):6s} {rows[n]['n_res']:3d} aa  "
                  f"{rows[n]['abs_pc2_transductive']:.3f}   "
                  f"no stability assay within 10 residues -- gap stays confounded")
            continue
        nb = float(np.mean([rows[m]["abs_pc2_transductive"] for m in near]))
        matched[n] = {"n_res": rows[n]["n_res"],
                      "neighbours": [short(m) for m in near],
                      "self": rows[n]["abs_pc2_transductive"],
                      "neighbour_mean": nb,
                      "gap": rows[n]["abs_pc2_transductive"] - nb}
        print(f"   {short(n):6s} {rows[n]['n_res']:3d} aa  "
              f"{rows[n]['abs_pc2_transductive']:.3f}   vs {nb:.3f} "
              f"from {len(near)} stability assays at "
              f"{min(rows[m]['n_res'] for m in near)}-"
              f"{max(rows[m]['n_res'] for m in near)} aa "
              f"({', '.join(short(m) for m in near)})  gap {matched[n]['gap']:+.3f}")

    out = {"protocol": {
               "basis_glob": a.basis_glob, "heldout_glob": a.heldout_glob,
               "basis_assays": bn, "lam": a.lam, "n_random": a.n_random,
               "note": "basis frozen on the basis assays; no held-out label "
                       "enters any fit and the sign is not re-chosen per assay"},
           "pc2_within_basis": {"mean": wpt, "ci_lo": wlo, "ci_hi": whi,
                                "per_assay": within},
           "per_assay": rows, "summary": summary, "summary_abs": summary_abs,
           "paired": pairs,
           "confounds": conf, "length_matched": matched,
           "metrics_table": metric_tab}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
