"""How far toward the sampler does the predictive signal stay accessible?

Part A of the conditioning experiment. Five representations of the same
mutation, on identical rows, scored by the identical protein-held-out protocol,
ordered by distance from the decoder:

    trunk_s     trunk single at the mutated token            (384)
    trunk_z     trunk pair row at the mutated token          (128)   <- the
                                                                      published
                                                                      internal
    cond_ttb    token_trans_bias row                         (384)
    cond_q      q, atoms aggregated to tokens                (128)
    cond_c      c, atoms aggregated to tokens                (128)
    geometry    what the sampler emitted                     (37)

MATCHED CAPACITY IS THE WHOLE COMPARISON. The blocks are 384, 128 and 37 wide,
so their native scores are not comparable -- a wider block can win on width
alone, which is the objection the project already had to answer once. Every
block is therefore also scored at k training-fold principal components for a
common ladder of k, with the PCA fitted on TRAINING assays only. Read the
curves at matched k; the native column is context, not the comparison.

WHAT A DROP CAN AND CANNOT MEAN. The conditioning tensors are a deterministic
function of the trunk state. They cannot hold more information than the trunk
did, so a fall from trunk to conditioning localizes a loss of ACCESSIBILITY
under a linear readout at matched capacity, and never shows that anything was
erased. The reverse finding is the strong one: signal still decodable at the
decoder entrance means the measured gap reaches the sampler rather than
stopping at an unused trunk feature.

    uv run python experiments/analysis/conditioning_probe.py --out ...
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax_harness"))

import geom                                                    # noqa: E402
import pi_archive                                              # noqa: E402
import pi_protocol                                             # noqa: E402
import pi_stats                                                # noqa: E402
from protein_interpretability.analysis.emitted_geometry import (  # noqa: E402
    GEOMETRY_FEATURES, geometry_matrix,
)

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
BLOCKS = ["trunk_s", "trunk_z", "cond_ttb", "cond_q", "cond_c"]
KS = [1, 2, 4, 8, 16, 32, 64]
EPS = 1e-9


def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def loao(blocks, target, lam, k=0):
    """Leave-one-assay-out; k>0 reduces by a TRAINING-fold PCA first."""
    names = sorted(target)
    rho = {}
    for held in names:
        tr = [n for n in names if n != held]
        if k:
            P = np.concatenate([blocks[n] for n in tr], 0)
            mu = P.mean(0)
            V = np.linalg.svd(P - mu, full_matrices=False)[2][:k]
            Xtr_raw = {n: (blocks[n] - mu) @ V.T for n in tr}
            Xte_raw = (blocks[held] - mu) @ V.T
        else:
            Xtr_raw = {n: blocks[n] for n in tr}
            Xte_raw = blocks[held]
        Xtr = np.concatenate(
            [(Xtr_raw[n] - Xtr_raw[n].mean(0)) / (Xtr_raw[n].std(0) + EPS)
             for n in tr], 0)
        ytr = np.concatenate([(target[n] - target[n].mean())
                              / (target[n].std() + EPS) for n in tr], 0)
        Xte = (Xte_raw - Xte_raw.mean(0)) / (Xte_raw.std(0) + EPS)
        rho[held] = pi_stats.spearman(Xte @ ridge(Xtr, ytr, lam), target[held])
    return rho


def summarise(rho, names):
    pt, lo, hi, _ = pi_stats.cluster_bootstrap(
        {k: [rho[k]] for k in names}, n_boot=10000, seed=0, hierarchical=False)
    return {"mean": pt, "ci_lo": lo, "ci_hi": hi, "per_assay": rho}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(W / "runs/cond/cond_*.npz"))
    ap.add_argument("--ensemble", default=str(W / "runs/ensemble/ens_full_*.npz"),
                    help="supplies the emitted-geometry block on the same rows")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"no conditioning archives matched {a.glob}")
    ens = {Path(f).name.replace("ens_full_", "").replace(".npz", ""): f
           for f in sorted(glob.glob(a.ensemble))}

    data, target, widths = {b: {} for b in BLOCKS}, {}, {}
    geo = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        assay = str(d["assay"])
        key = assay.split("_")[0]
        muts = [str(m) for m in d["mutant"]]
        for b in BLOCKS:
            data[b][key] = np.asarray(d[b], float)
            widths[b] = data[b][key].shape[1]
        target[key] = np.asarray(d["score"], float)

        # emitted geometry on the SAME rows, from the ensemble archive's draw 0
        if assay in ens:
            e = np.load(ens[assay], allow_pickle=True)
            emut = [str(m) for m in e["mutant"]]
            idx = [emut.index(m) for m in muts if m in emut]
            if len(idx) == len(muts):
                ca, ca_wt = e["ca"][:, 0], e["ca_wt"][0]
                tm = np.array([geom.tm_score(np.asarray(c, float),
                                             np.asarray(ca_wt, float))
                               for c in ca[idx]])
                geo[key] = geometry_matrix(
                    ca[idx], ca_wt, tm, e["plddt"][idx, 0],
                    e["plddt_site"][idx, 0], np.asarray(e["pos"])[idx])
        print(f"  {key:8s} n={len(muts):4d}"
              + ("  +geometry" if key in geo else "  (no geometry)"))

    names = sorted(target)
    have_geo = len(geo) == len(names)
    if have_geo:
        widths["geometry"] = len(GEOMETRY_FEATURES)

    order = BLOCKS + (["geometry"] if have_geo else [])
    blocks_all = {**data, **({"geometry": geo} if have_geo else {})}

    print(f"\n{len(names)} assays, leave-one-assay-out, lam={a.lam:g}\n")
    print(f"{'block':12s}{'width':>7s}{'native':>10s}"
          + "".join(f"{'k='+str(k):>9s}" for k in KS))
    out = {"assays": names, "widths": widths, "ks": KS, "native": {},
           "curve": {}, "gaps": {}}
    native = {}
    for b in order:
        rn = loao(blocks_all[b], target, a.lam)
        native[b] = rn
        out["native"][b] = summarise(rn, names)
        cells = []
        out["curve"][b] = {}
        for k in KS:
            if k > widths[b]:
                cells.append("")
                continue
            rk = loao(blocks_all[b], target, a.lam, k=k)
            out["curve"][b][str(k)] = summarise(rk, names)
            cells.append(f"{np.mean([rk[n] for n in names]):+.3f}")
        print(f"{b:12s}{widths[b]:>7d}"
              f"{np.mean([rn[n] for n in names]):>+10.3f}"
              + "".join(f"{c:>9s}" for c in cells))

    # The comparison, at matched capacity. trunk_z is the published internal.
    print("\npaired gaps at matched k (assay is the unit)\n")
    for k in (2, 8, 32):
        for other in [b for b in order if b != "trunk_z"]:
            if k > min(widths["trunk_z"], widths[other]):
                continue
            ra = out["curve"]["trunk_z"][str(k)]["per_assay"]
            rb = out["curve"][other][str(k)]["per_assay"]
            pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(
                {n: [ra[n]] for n in names}, {n: [rb[n]] for n in names},
                n_boot=10000, seed=0, hierarchical=False)
            wins = sum(ra[n] > rb[n] for n in names)
            out["gaps"][f"k={k}: trunk_z - {other}"] = {
                "gap": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins,
                "n": len(names)}
            flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
            print(f"   k={k:<3d} trunk_z - {other:10s} {pt:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]  {wins}/{len(names)}{flag}")

    if have_geo:
        print("\nconditioning against what was emitted, at matched k\n")
        for k in (2, 8, 32):
            for b in ("cond_ttb", "cond_q", "cond_c"):
                if k > min(widths[b], widths["geometry"]):
                    continue
                ra = out["curve"][b][str(k)]["per_assay"]
                rb = out["curve"]["geometry"][str(k)]["per_assay"]
                pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(
                    {n: [ra[n]] for n in names}, {n: [rb[n]] for n in names},
                    n_boot=10000, seed=0, hierarchical=False)
                wins = sum(ra[n] > rb[n] for n in names)
                out["gaps"][f"k={k}: {b} - geometry"] = {
                    "gap": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins,
                    "n": len(names)}
                flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
                print(f"   k={k:<3d} {b:10s} - geometry  {pt:+.3f} "
                      f"[{lo:+.3f}, {hi:+.3f}]  {wins}/{len(names)}{flag}")

    proto = pi_protocol.protocol(
        script="conditioning_probe.py",
        design="trunk versus diffusion-conditioning versus emitted geometry on "
               "identical rows, leave-one-assay-out, compared at matched "
               "capacity via training-fold PCA",
        layer=pi_protocol.layers("final"),
        features={b: pi_protocol.features(b, widths[b]) for b in order},
        source=a.glob, n_assays=len(names), lam=a.lam, ks=KS,
        capacity="every block also scored at k training-fold PCs; native "
                 "widths differ (384/128/37) and are not comparable",
        determinism="conditioning is a deterministic function of the trunk "
                    "state; a drop localizes accessibility under this readout "
                    "and cannot demonstrate erasure")
    pi_archive.write_result(a.out, out, protocol=proto, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
