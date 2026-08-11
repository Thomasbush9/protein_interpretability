"""Compare all five z-path operations: which one shapes the stability axis?

Reads the channel-space operators `exp_ops` archives, one 128x128 matrix per
(layer, operation, assay), and asks of each operation the four questions
`analyze_jac` asked of the transition alone.

  rank        Effective rank of M. The transition came out near 24 of 128 with
              the SwiGLU gate as the mechanism. There is no reason the triangle
              operations should agree, and if they are much richer that is where
              the pair representation's capacity actually lives.

  gain        `w_c . M e_c` -- the added PC-c coordinate per unit of PC-c, so
              the operation's total multiplier on that coordinate is 1 + gain.
              Compared against directions drawn orthonormally in the
              standardised basis and carried through the SAME per-channel spread
              `s`, reported as a percentile. A percentile near 0.5 means the
              operation does nothing to the component it does not do to an
              arbitrary direction, however large the raw gain looks; this is the
              distinction that overturned the first reading of the transition
              result.

  agreement   Principal angles between the top-k subspaces of DIFFERENT assays.
              M is expressed in the model's own 128 pair channels, which mean
              the same thing in every protein, so these are directly comparable.

  consistency `transition_z` is present in BOTH this run and `exp_jac`, by
              different routes -- here as a row-averaged channel operator built
              from `jax.linearize`, there as a per-pair `jacfwd`. Because the
              transition is pointwise in the pair index the two must agree up to
              the differing pair samples. That is checked with `--jac-glob`
              rather than assumed, and it is the only end-to-end validation
              that the channel-operator construction measures what it claims.

Sign and basis conventions are inherited from `exp_jac.py`; in particular
`e_c = s * v_c` is a vector and `w_c = v_c / s` a covector, with `w_c . e_c = 1`.

  sbatch analysis.sbatch analyze_ops.py --glob '../runs/ops_*.npz' \
      --jac-glob '../runs/jac_*.npz' --out ../runs/ops_pooled.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

EPS = 1e-8


def eff_rank(sv):
    p = sv ** 2
    return (p.sum(-1) ** 2) / ((p ** 2).sum(-1) + EPS)


def princ(A, B):
    return np.clip(np.linalg.svd(A.T @ B, compute_uv=False), -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--jac-glob", default="")
    ap.add_argument("--basis", default="", help="basis_depth.npz; use each "
                    "layer's OWN basis instead of the last layer's")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-rand", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    D = [np.load(f, allow_pickle=True) for f in files]
    if not D:
        raise SystemExit(f"no files matched {a.glob}")
    names = [str(d["assay"]).split("_")[0] for d in D]
    ops = [str(x) for x in D[0]["ops"]]
    L, n_op, dim, _ = D[0]["M"].shape
    n_pc = D[0]["V"].shape[0]
    print(f"{len(D)} assays: {', '.join(names)}")
    print(f"{L} layers, {n_op} operations, dim {dim}, {n_pc} components\n")

    M = np.stack([d["M"].astype(np.float64) for d in D])       # [A,L,n_op,128,128]
    SD = np.stack([d["sd"] for d in D])                        # [A,128]
    V = D[0]["V"]

    rng = np.random.default_rng(a.seed)
    Q = rng.standard_normal((a.n_rand, dim))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)

    # ---- 1. rank ----------------------------------------------------------
    sv = np.linalg.svd(M, compute_uv=False)                    # [A,L,n_op,128]
    er = eff_rank(sv)                                          # [A,L,n_op]
    print("effective rank of the channel operator (ceiling 128), mean over assays\n")
    print(f"  {'layer':>5s} " + " ".join(f"{o:>14s}" for o in ops))
    for li in list(range(0, L, 8)) + [L - 1]:
        print(f"  {li:5d} " + " ".join(f"{er[:, li, oi].mean():14.1f}"
                                       for oi in range(n_op)))
    print("\n  over all layers/assays: " + ",  ".join(
        f"{o} {np.median(er[:, :, oi]):.1f}" for oi, o in enumerate(ops)))

    # Operator scale: rank says nothing about how much the op actually moves z.
    fro = np.linalg.norm(M, axis=(-2, -1))
    print("\n  Frobenius norm (how much the operation moves the site row at all):")
    print("  " + ",  ".join(f"{o} {np.median(fro[:, :, oi]):.2f}"
                            for oi, o in enumerate(ops)) + "\n")

    # ---- 2. gain ----------------------------------------------------------
    # Per-layer bases when available. `analyze_basis` found the mutation
    # subspace rotating almost completely with depth -- the top-4 subspace at
    # mid-depth overlaps the last layer's at barely above chance -- so a gain
    # measured against the last layer's PC2 at layer 20 is a gain along a
    # direction layer 20 does not use. With --basis, layer l is scored in layer
    # l's own basis. The operator acts on a difference arriving in layer l-1's
    # basis, but adjacent layers agree at >=0.9 everywhere, so using layer l's
    # basis on both sides keeps w.e = 1 and the multiplier reading intact.
    BAS = np.load(a.basis, allow_pickle=True) if a.basis else None
    if BAS is not None:
        bnames = [str(x) for x in BAS["assays"]]
        if bnames != names:
            raise SystemExit(f"basis assays {bnames} do not match {names}; the "
                             f"per-assay spreads would be paired with the "
                             f"wrong protein")
        print(f"using per-layer bases from {a.basis}\n")
    else:
        print("using the LAST-LAYER basis at every depth -- depth profiles "
              "below inherit\nthe rotation analyze_basis measured; pass "
              "--basis to correct them\n")

    gain = np.zeros((len(D), L, n_op, n_pc))
    pct = np.zeros((len(D), L, n_op, n_pc))
    for ai in range(len(D)):
        for li in range(L):
            if BAS is not None:
                Vl, sd = BAS["V"][li], BAS["sd"][ai, li]
            else:
                Vl, sd = V, SD[ai]
            E, W = Vl * sd, Vl / (sd + EPS)
            Er, Wr = Q * sd, Q / (sd + EPS)
            for oi in range(n_op):
                m = M[ai, li, oi]
                gr = np.einsum("rd,rd->r", Wr, Er @ m.T)
                for c in range(n_pc):
                    gain[ai, li, oi, c] = W[c] @ (m @ E[c])
                    pct[ai, li, oi, c] = (gr <= gain[ai, li, oi, c]).mean()

    for c in range(min(2, n_pc)):
        print(f"PC{c+1} gain by operation, mean over assays "
              f"(percentile against matched null)\n")
        print(f"  {'layer':>5s} " + " ".join(f"{o:>18s}" for o in ops))
        for li in list(range(0, L, 8)) + [L - 1]:
            print(f"  {li:5d} " + " ".join(
                f"{gain[:, li, oi, c].mean():11.4f} ({pct[:, li, oi, c].mean():.2f})"
                for oi in range(n_op)))
        print()

    print("total multiplier on the coordinate is 1 + gain; percentile ~0.5 means")
    print("the component is treated like an arbitrary direction. Extreme")
    print("percentiles are the signal, not large |gain|.\n")

    # Which operation departs furthest from its own null, pooled over depth?
    print("departure from the null, |percentile - 0.5| averaged over layers "
          "and assays\n")
    print(f"  {'op':>16s} " + " ".join(f"{'PC'+str(c+1):>8s}" for c in range(n_pc)))
    dev = np.abs(pct - 0.5).mean((0, 1))                        # [n_op, n_pc]
    for oi, o in enumerate(ops):
        print(f"  {o:>16s} " + " ".join(f"{dev[oi, c]:8.3f}" for c in range(n_pc)))
    print()

    # ---- 3. cross-assay agreement -----------------------------------------
    k = a.k
    print(f"top-{k} subspace agreement across assays (mean cos^2 of principal")
    print(f"angles; random baseline {k/dim:.3f})\n")
    print(f"  {'op':>16s} {'output side':>13s} {'input side':>12s}")
    agree = {}
    for oi, o in enumerate(ops):
        res = []
        for side in ("out", "in"):
            vals = []
            for li in range(0, L, 8):
                B = []
                for ai in range(len(D)):
                    U, _, Vt = np.linalg.svd(M[ai, li, oi])
                    B.append(U[:, :k] if side == "out" else Vt[:k].T)
                vals += [(princ(B[i], B[j]) ** 2).mean()
                         for i in range(len(D)) for j in range(i + 1, len(D))]
            res.append(float(np.mean(vals)))
        agree[o] = res
        print(f"  {o:>16s} {res[0]:13.3f} {res[1]:12.3f}")
    print()

    # ---- 4. consistency with the per-pair Jacobian ------------------------
    # Only meaningful when both sides use the SAME basis. `exp_jac` projects
    # onto the last layer's basis at every depth, so under --basis the two are
    # not estimates of one quantity and a low correlation would be the expected
    # result of comparing different things, not evidence of a fault. Running it
    # anyway printed a WARNING that would have gone into a report as a defect.
    consistency = None
    if a.jac_glob and BAS is not None:
        print("consistency check skipped: exp_jac projects onto the LAST-LAYER\n"
              "basis while this run uses per-layer bases, so the two are not\n"
              "the same quantity. The check is valid in the last-layer run.\n")
    elif a.jac_glob:
        J = {}
        for f in sorted(glob.glob(a.jac_glob)):
            d = np.load(f, allow_pickle=True)
            if "gain" in d.files:
                J[str(d["assay"]).split("_")[0]] = np.median(d["gain"], -1)  # [L,n_pc]
        oi = ops.index("transition_z")
        shared = [n for n in names if n in J]
        if shared:
            x = np.concatenate([gain[names.index(n), :, oi, :].ravel() for n in shared])
            y = np.concatenate([J[n].ravel() for n in shared])
            r = float(np.corrcoef(x, y)[0, 1])
            rel = float(np.abs(x - y).mean() / (np.abs(y).mean() + EPS))
            consistency = {"assays": shared, "n_values": int(x.size),
                           "r": r, "mean_abs_diff_frac": rel}
            print(f"consistency check on transition_z, {len(shared)} assays:")
            print(f"  channel operator (this run) vs per-pair jacfwd (exp_jac)")
            print(f"  r = {r:.4f} over {len(x)} (layer, component) values, "
                  f"mean |difference| {rel:.1%} of mean |gain|")
            if r < 0.9:
                print("  WARNING: the two constructions disagree. One of them is not")
                print("  measuring the transition's Jacobian, and the triangle")
                print("  numbers above inherit whichever fault it is.")
            else:
                print("  -> the two routes to the same quantity agree; the")
                print("     channel-operator construction is doing what it claims.")
            print()

    out = {
        "assays": names, "ops": ops, "layers": L, "dim": dim, "k": k,
        "consistency": consistency,
        # Which basis produced the depth profiles. A reader who does not know
        # this cannot tell a corrected profile from the one it replaced.
        "basis": ("per-layer" if BAS is not None else "last-layer"),
        "basis_file": (a.basis or None),
        # |percentile - 0.5| for a percentile uniform on [0,1] has expectation
        # 0.25, so that -- not zero -- is the level `null_departure` must beat.
        "null_departure_chance": 0.25,
        "eff_rank_median": {o: float(np.median(er[:, :, oi]))
                            for oi, o in enumerate(ops)},
        "eff_rank_by_layer": {o: er[:, :, oi].mean(0).tolist()
                              for oi, o in enumerate(ops)},
        "frobenius_median": {o: float(np.median(fro[:, :, oi]))
                             for oi, o in enumerate(ops)},
        "gain_by_layer": {o: {f"PC{c+1}": gain[:, :, oi, c].mean(0).tolist()
                              for c in range(n_pc)} for oi, o in enumerate(ops)},
        "pct_by_layer": {o: {f"PC{c+1}": pct[:, :, oi, c].mean(0).tolist()
                             for c in range(n_pc)} for oi, o in enumerate(ops)},
        "null_departure": {o: dev[oi].tolist() for oi, o in enumerate(ops)},
        "agreement": agree,
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
