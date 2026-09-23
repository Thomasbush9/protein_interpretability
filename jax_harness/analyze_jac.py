"""Pool the per-assay `exp_jac` runs: is any of this a property of the LAYER?

A single protein cannot separate "the pair transition works this way" from
"this protein's operating point happens to look like this". The transition
weights are shared across all twelve assays; only the base point differs. So
every claim below is stated across twelve unrelated folds, and the ones that
survive are the ones about the layer.

Four questions.

  rank        `exp_jac` found the Jacobian's effective rank around 24 of 128 for
              RCRO, against 73-94 for the bare weights. If that holds across
              twelve folds it is the substantive correction to `probe_wsvd`:
              the transition IS close to low-rank, but only once conditioned on
              where the model actually sits. The weights alone do not show it.

  gain        Does the transition amplify or attenuate the PC coordinates, and
              is it doing anything to them that it does not do to an arbitrary
              direction? The matched null carries the SAME per-channel spread
              `s` used to bridge the PC out of the standardised basis, so a
              difference cannot be an artefact of that bridge. Reported as a
              percentile against the null rather than a ratio, because the null
              is itself systematically negative at depth and a ratio would read
              a shared contraction as a PC-specific effect.

  agreement   Principal angles between the dominant subspaces of DIFFERENT
              assays, from the pair-averaged second moments. The per-pair
              singular vectors of J are not in correspondence between two
              proteins, but `E_pairs[J J^T]` is a 128x128 operator on the
              model's own channels and means the same thing everywhere. This is
              the test that turns "J is low rank" into "J is low rank in the
              same place every time", which is the claim the DEU framing needs.

  where PC2   Where the stability axis sits inside that shared subspace, if it
              is shared.

Sign conventions and the `e_c` / `w_c` distinction are inherited from
`exp_jac.py`; see its docstring.

  sbatch analysis.sbatch analyze_jac.py --glob '../runs/jac_*.npz' --out ...
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent))
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402

EPS = 1e-8


def eff_rank(sv):
    p = sv ** 2
    return (p.sum(-1) ** 2) / ((p ** 2).sum(-1) + EPS)


def top_subspace(M, k):
    """Leading k eigenvectors of a symmetric PSD second-moment matrix."""
    w, V = np.linalg.eigh(M)
    return V[:, ::-1][:, :k], w[::-1]


def principal_angles(A, B):
    """cos of the principal angles between two orthonormal column bases."""
    return np.clip(np.linalg.svd(A.T @ B, compute_uv=False), -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--k", type=int, default=16, help="subspace dim for agreement")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"no files matched {a.glob}")
    D, names = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "mom_out" not in d.files:
            print(f"  skipping {Path(f).name}: no second moments (stale run)")
            continue
        D.append(d)
        names.append(str(d["assay"]).split("_")[0])
    if not D:
        raise SystemExit("every archive was stale; rerun exp_jac")
    L, n_pc, P = D[0]["gain"].shape
    dim = D[0]["sv"].shape[-1]
    print(f"{len(D)} assays: {', '.join(names)}")
    print(f"L={L} layers, {n_pc} components, {P} pairs, dim={dim}\n")

    # ---- 1. rank ----------------------------------------------------------
    er = np.stack([eff_rank(d["sv"]) for d in D])          # [A, L, P]
    er_l = er.mean(-1)                                     # [A, L]
    print("Jacobian effective rank (ceiling 128), mean over pairs")
    print(f"  {'layer':>5s} " + " ".join(f"{n:>7s}" for n in names))
    for li in list(range(0, L, 8)) + [L - 1]:
        print(f"  {li:5d} " + " ".join(f"{er_l[ai, li]:7.1f}" for ai in range(len(D))))
    print(f"\n  across all layers and assays: min {er_l.min():.1f}  "
          f"median {np.median(er_l):.1f}  max {er_l.max():.1f}")
    print(f"  bare-weight comparison (probe_wsvd): fc1 73, fc2 94, fc3 87 median\n")

    # ---- 2. gain, against the matched null --------------------------------
    print("PC gain, median over pairs then over assays "
          "(percentile against matched null in brackets)\n")
    g = np.stack([np.median(d["gain"], -1) for d in D])       # [A, L, n_pc]
    gr = np.stack([np.median(d["gain_rand"], -1) for d in D])  # [A, L, n_rand]
    pct = np.zeros((len(D), L, n_pc))
    for ai in range(len(D)):
        for li in range(L):
            for c in range(n_pc):
                pct[ai, li, c] = (gr[ai, li] <= g[ai, li, c]).mean()

    print(f"  {'layer':>5s} " + " ".join(f"{'PC'+str(c+1):>16s}" for c in range(n_pc))
          + f" {'null':>9s}")
    for li in list(range(0, L, 8)) + [L - 1]:
        row = " ".join(f"{g[:, li, c].mean():9.4f} ({pct[:, li, c].mean():.2f})"
                       for c in range(n_pc))
        print(f"  {li:5d} {row} {np.median(gr[:, li]):9.4f}")
    print("\n  total multiplier on the coordinate is 1 + gain; percentile ~0.5")
    print("  means the component is treated like an arbitrary direction.\n")

    # Deepest-layer summary with the across-assay spread, since that is where
    # the effect is largest and where a claim would be made.
    peak = int(np.argmin(g.mean(0).mean(-1)))
    print(f"strongest attenuation at layer {peak}:")
    for c in range(n_pc):
        v, p = g[:, peak, c], pct[:, peak, c]
        print(f"   PC{c+1}: gain {v.mean():+.4f} +/- {v.std():.4f} across assays, "
              f"percentile {p.mean():.2f} (range {p.min():.2f}-{p.max():.2f})")
    print()

    # ---- 3. cross-assay subspace agreement --------------------------------
    k = a.k
    rng = np.random.default_rng(a.seed)
    agree = {}
    for side in ("out", "in"):
        B = [top_subspace(d[f"mom_{side}"][L - 1], k)[0] for d in D]
        M = np.ones((len(D), len(D)))
        for i in range(len(D)):
            for j in range(i + 1, len(D)):
                M[i, j] = M[j, i] = (principal_angles(B[i], B[j]) ** 2).mean()
        off = M[~np.eye(len(D), dtype=bool)]
        # Null: random k-dim subspaces of R^128 have E[mean cos^2] = k/dim.
        print(f"top-{k} {side}put subspace agreement at layer {L-1} "
              f"(mean cos^2 of principal angles)")
        print(f"   across assays: {off.mean():.3f}  (min {off.min():.3f}, "
              f"max {off.max():.3f});  random baseline {k/dim:.3f}")

        # Depth profile, since the trunk is not homogeneous.
        prof = []
        for li2 in range(0, L, 8):
            Bl = [top_subspace(d[f"mom_{side}"][li2], k)[0] for d in D]
            vals = [(principal_angles(Bl[i], Bl[j]) ** 2).mean()
                    for i in range(len(D)) for j in range(i + 1, len(D))]
            prof.append((li2, float(np.mean(vals))))
        print("   by depth: " + "  ".join(f"L{li2}:{v:.3f}" for li2, v in prof) + "\n")
        agree[side] = {"last_layer_mean": float(off.mean()),
                       "last_layer_min": float(off.min()),
                       "last_layer_max": float(off.max()),
                       "random_baseline": k / dim,
                       "by_depth": [[li2, v] for li2, v in prof]}

    # ---- 4. where the PC directions sit in the shared subspace ------------
    print(f"fraction of each PC captured by the shared top-{k} subspace "
          f"(layer {L-1}, mean over assays)\n")
    ks = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    ks = ks[ks <= dim]
    print(f"  {'side':>6s} {'dir':>5s}  " + "  ".join(f"k={kk:<5d}" for kk in ks))
    cap = {}
    for side, vecs in (("out", "W"), ("in", "E")):
        rows = np.zeros((n_pc, len(ks)))
        for ai, d in enumerate(D):
            V, sd = d["V"], d["sd"]
            X = (V / (sd + EPS)) if vecs == "W" else (V * sd)
            Q, _ = top_subspace(d[f"mom_{side}"][L - 1], dim)
            for c in range(n_pc):
                u = X[c] / (np.linalg.norm(X[c]) + EPS)
                cc = (Q.T @ u) ** 2
                rows[c] += (np.cumsum(cc) / (cc.sum() + EPS))[ks - 1]
        rows /= len(D)
        cap[side] = rows
        for c in range(n_pc):
            print(f"  {side:>6s} {'PC'+str(c+1):>5s}  "
                  + "  ".join(f"{x:6.3f} " for x in rows[c]))
    print(f"  {'':>6s} {'rand':>5s}  " + "  ".join(f"{kk/dim:6.3f} " for kk in ks))

    out = {
        "assays": names, "layers": L, "dim": dim, "k": k,
        "agreement": agree,
        "peak_layer": peak,
        "peak": {f"PC{c+1}": {"gain_mean": float(g[:, peak, c].mean()),
                              "gain_sd": float(g[:, peak, c].std()),
                              "pct_mean": float(pct[:, peak, c].mean()),
                              "pct_min": float(pct[:, peak, c].min()),
                              "pct_max": float(pct[:, peak, c].max())}
                 for c in range(n_pc)},
        "eff_rank_min": float(er_l.min()), "eff_rank_max": float(er_l.max()),
        "eff_rank_by_assay_layer": er_l.tolist(),
        "eff_rank_median": float(np.median(er_l)),
        "eff_rank_by_layer": er_l.mean(0).tolist(),
        "gain_by_layer": {f"PC{c+1}": g[:, :, c].mean(0).tolist() for c in range(n_pc)},
        "gain_pct_by_layer": {f"PC{c+1}": pct[:, :, c].mean(0).tolist()
                              for c in range(n_pc)},
        "null_gain_by_layer": np.median(gr, axis=(0, 2)).tolist(),
        "capture_last_layer": {s: cap[s].tolist() for s in cap},
        "ks": ks.tolist(),
    }
    pi_archive.write_result(a.out, out, protocol=pi_protocol.protocol(
        script="analyze_jac.py",
        design="pooled over per-assay Jacobians of the z-path at the operating "
               "point; subspace agreement measured across folds, not fitted",
        layer=pi_protocol.layers("final"),
        features=pi_protocol.features("Jacobian of transition_z, channel space",
                                      128),
        source=a.glob, n_assays=len(names), subspace_k=a.k, seed=a.seed))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
