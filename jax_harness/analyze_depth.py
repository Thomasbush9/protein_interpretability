"""At what depth does the severity direction appear, and do the models agree there?

Two things this settles, one of them a correction.

THE CORRECTION. The cross-model comparison used each model's LAST layer, which
is matched relative depth by one definition but ignores that the three trunks
are 64, 48 and 16 blocks deep. Boltz-2 came out at CKA ~0.47 from both other
models while OpenFold3 and Protenix agreed at 0.80, and that was reported
alongside a lineage explanation that is simply wrong -- all three use the same
AF3-derived trunk. So the asymmetry is unexplained, and depth is the obvious
confound: "the last of 64" and "the last of 16" are not the same amount of
computation. If the gap closes at matched FRACTIONAL depth, it was never a fact
about the models.

THE RESULT. `dz_vec` is stored per layer for every model, so the same data
answers a question the project has not asked: where in the trunk does mutation
severity become decodable? Early would mean it rides on the alignment and the
substitution itself; late would mean it is built by the pair stack. And if the
three models build it at the same relative depth despite different absolute
depths, that is a much stronger generality claim than "they all have it at the
end".

Everything is plotted against fractional depth l/L, which is the only axis on
which 64, 48 and 16 layers can be compared at all.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402
import pi_stats  # noqa: E402
from analyze_xmodel import block_perm, cka, rdm  # noqa: E402
from compare_internal_output import fit_ridge_block, grouped_split  # noqa: E402

MODELS = ("boltz2", "of3", "protenix")
FRACS = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
EPS = 1e-9


def layer_at(frac, n_layers):
    """Index of the layer sitting at fractional depth `frac`."""
    return int(np.clip(round(frac * n_layers) - 1, 0, n_layers - 1))


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/runs/"
    ap.add_argument("--dir", default=R)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--frac-split", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assays = sorted({Path(f).stem.split("_", 3)[3]
                     for f in glob.glob(a.dir + "xm_boltz2_r1_*.npz")})
    D = {}
    for asy in assays:
        for m in MODELS:
            f = Path(a.dir) / f"xm_{m}_r1_{asy}.npz"
            d = np.load(f, allow_pickle=True)
            D[(asy.split("_")[0], m)] = {
                "z": np.asarray(d["dz_vec"], float),          # (n, L, C)
                "y": np.asarray(d["score"], float),
                "pos": np.asarray(d["pos"])}
    names = sorted({k[0] for k in D})
    nL = {m: D[(names[0], m)]["z"].shape[1] for m in MODELS}
    print(f"{len(names)} assays; layer counts " +
          ", ".join(f"{m}={nL[m]}" for m in MODELS) + "\n")

    res = {"fracs": FRACS.tolist(), "n_layers": nL, "assays": names}

    # ---- 1. decodability against fractional depth -------------------------
    print("Held-out Spearman of the pair representation, by fractional depth\n")
    print(f"   {'model':10s}" + "".join(f"{f:>8.3f}" for f in FRACS))
    prof = {}
    for m in MODELS:
        row = []
        for fr in FRACS:
            li = layer_at(fr, nL[m])
            g = {}
            for n in names:
                r = D[(n, m)]
                vals = []
                for s in range(a.seeds):
                    rng = np.random.default_rng(s)
                    tr, te = grouped_split(r["pos"], a.frac_split, rng)
                    vals.append(fit_ridge_block(r["z"][:, li, :], r["y"], r["pos"],
                                                tr, te, rng)[0])
                g[n] = [float(np.nanmean(vals))]
            pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                       hierarchical=False)
            row.append({"frac": float(fr), "layer": li, "mean": pt,
                        "ci_lo": lo, "ci_hi": hi})
        prof[m] = row
        print(f"   {m:10s}" + "".join(f"{r['mean']:>+8.3f}" for r in row))
    res["decodability_by_depth"] = prof

    # ---- 2. does the cross-model gap close at matched depth? --------------
    print("\nCross-model CKA at matched fractional depth "
          "(the confound in the last-layer comparison)\n")
    pairs = [(MODELS[i], MODELS[j]) for i in range(len(MODELS))
             for j in range(i + 1, len(MODELS))]
    print(f"   {'pair':22s}" + "".join(f"{f:>8.3f}" for f in FRACS))
    ck = {}
    for ma, mb in pairs:
        row = []
        for fr in FRACS:
            la, lb = layer_at(fr, nL[ma]), layer_at(fr, nL[mb])
            g = {n: [cka(D[(n, ma)]["z"][:, la, :], D[(n, mb)]["z"][:, lb, :])]
                 for n in names}
            pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                       hierarchical=False)
            row.append({"frac": float(fr), "mean": pt, "ci_lo": lo, "ci_hi": hi})
        ck[f"{ma}|{mb}"] = row
        print(f"   {ma}|{mb:12s}" + "".join(f"{r['mean']:>8.3f}" for r in row))
    res["cka_by_depth"] = ck

    # ---- 3. where is the severity direction strongest? --------------------
    # A single direction per (model, depth), fitted on training rows only, so
    # this is the depth profile of the PC2-analogue rather than of the whole
    # representation.
    print("\nSeverity direction alone (top DMS-associated component), by depth\n")
    print(f"   {'model':10s}" + "".join(f"{f:>8.3f}" for f in FRACS))
    sev = {}
    for m in MODELS:
        row = []
        for fr in FRACS:
            li = layer_at(fr, nL[m])
            g = {}
            for n in names:
                r = D[(n, m)]
                X = r["z"][:, li, :]
                vals = []
                for s in range(a.seeds):
                    rng = np.random.default_rng(s)
                    tr, te = grouped_split(r["pos"], a.frac_split, rng)
                    Xc = X - X[tr].mean(0)
                    V = np.linalg.svd(Xc[tr], full_matrices=False)[2][:8]
                    P = Xc @ V.T
                    j = int(np.argmax([abs(pi_stats.spearman(P[tr, c], r["y"][tr]))
                                       for c in range(P.shape[1])]))
                    sgn = np.sign(pi_stats.spearman(P[tr, j], r["y"][tr]))
                    vals.append(pi_stats.spearman(sgn * P[te, j], r["y"][te]))
                g[n] = [float(np.nanmean(vals))]
            pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                       hierarchical=False)
            row.append({"frac": float(fr), "mean": pt, "ci_lo": lo, "ci_hi": hi})
        sev[m] = row
        print(f"   {m:10s}" + "".join(f"{r['mean']:>+8.3f}" for r in row))
    res["severity_direction_by_depth"] = sev

    best = {m: FRACS[int(np.argmax([r["mean"] for r in sev[m]]))] for m in MODELS}
    print("\n   peak fractional depth: " +
          ", ".join(f"{m} {best[m]:.3f}" for m in MODELS))
    res["peak_fraction"] = {m: float(v) for m, v in best.items()}

    pi_archive.write_result(a.out, res, protocol=pi_protocol.protocol(
        script="analyze_depth.py",
        design="cross-model at MATCHED RELATIVE DEPTH, not at the last layer; "
               "the trunks are 64, 48 and 16 blocks deep and comparing their "
               "final layers is what produced the earlier unexplained asymmetry",
        layer=pi_protocol.layers("relative-depth grid", n_layers=max(nL.values())),
        features=pi_protocol.features("z at the matched layer", 128),
        source=a.dir, n_assays=len(names), n_layers_per_model=nL,
        seeds=a.seeds, frac=a.frac_split))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
