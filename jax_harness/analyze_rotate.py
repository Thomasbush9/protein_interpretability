"""Which operations rotate the mutation subspace, and where?

`analyze_basis` established that the mutation basis at the Pairformer entrance is
nearly unrelated to the one at its exit -- top-4 agreement 0.096 against a 0.031
chance floor -- so the stability axis is BUILT inside the stack rather than
carried into it. `analyze_ops` established that no single operation privileges
PC2 within the subspace. Together those say the direction forms by accumulated
rotation. This asks which operations do the rotating.

The measurement is a within-layer decomposition. Linearised, the z-path composes
as

    C_k = (I + M_k)(I + M_{k-1}) ... (I + M_1)

over the five operations in the order the layer applies them, where `M` is the
channel operator `exp_ops` archives. Pushing layer l-1's basis through the
cumulative maps and taking principal angles at each stage gives, per operation,
how much of that layer's rotation it accounts for.

Two things make this checkable rather than decorative.

  closure   The composed layer map must land on the NEXT layer's basis. If
            `C_5 E_{l-1}` does not span roughly `V_l`, the channel-operator
            composition is too lossy to attribute anything and the per-operation
            split below is arithmetic on a wrong object. Reported per layer, and
            against the rotation actually being explained -- a layer that barely
            rotates passes closure trivially, so closure is only informative
            where there is rotation to explain.

  baseline  A rotation attributed to an operation has to beat what its own size
            AND SHAPE would produce anyway. Two nulls are reported for that
            reason. A dense random matrix of matching Frobenius norm is the
            obvious one and it is not fair here: these operators have effective
            rank 7-26 of 128, while a dense random matrix has essentially full
            rank and spreads its energy isotropically, which walks a subspace out
            of itself far more efficiently per unit of norm. Every operation
            "loses" against that null for reasons that have nothing to do with
            direction. The informative null keeps each operator's exact singular
            SPECTRUM and randomises only its singular vectors, so norm, rank and
            conditioning are all held fixed and the only thing varying is where
            the operator points.

Bases and spreads are per-layer and per-assay, from `analyze_basis --npz`.
Directions enter as raw-space VECTORS `e = s * v` -- the same bridge the rest of
the study uses, and the reason it matters is in `build_jac_report.py`.

  sbatch analysis.sbatch analyze_rotate.py --glob '../runs/ops_*.npz' \
      --basis ../runs/basis_depth.npz --out ../runs/rotate_pooled.json
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


def onb(X):
    """Orthonormal basis for the columns of X (128 x k)."""
    Q, _ = np.linalg.qr(X)
    return Q


def agree(A, B):
    """Mean cos^2 of the principal angles between two orthonormal column bases."""
    s = np.clip(np.linalg.svd(A.T @ B, compute_uv=False), -1, 1)
    return float((s ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-rand", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    D = [np.load(f, allow_pickle=True) for f in files]
    names = [str(d["assay"]).split("_")[0] for d in D]
    ops = [str(x) for x in D[0]["ops"]]
    L, n_op, dim, _ = D[0]["M"].shape

    B = np.load(a.basis, allow_pickle=True)
    if [str(x) for x in B["assays"]] != names:
        raise SystemExit("basis assays do not match the operator assays")
    V, SD = B["V"], B["sd"]                      # [L,n_pc,128], [A,L,128]
    k = min(a.k, V.shape[1])
    print(f"{len(names)} assays: {', '.join(names)}")
    print(f"{L} layers, {n_op} operations, subspace dim {k}\n")

    rng = np.random.default_rng(a.seed)
    I = np.eye(dim)

    # step[a,l,o] : rotation contributed by operation o at layer l
    def rand_orth(n):
        Q, R = np.linalg.qr(rng.standard_normal((n, n)))
        return Q * np.sign(np.diag(R))

    step = np.zeros((len(D), L, n_op))
    step_rand = np.zeros((len(D), L, n_op))      # dense, norm-matched only
    step_spec = np.zeros((len(D), L, n_op))      # spectrum-matched
    layer_pred = np.zeros((len(D), L))           # rotation the layer map produces
    layer_act = np.zeros((len(D), L))            # rotation the bases actually show
    closure = np.zeros((len(D), L))              # composed map vs the next basis

    for ai, d in enumerate(D):
        M = d["M"].astype(np.float64)
        for li in range(1, L):
            Ein = (V[li - 1] * SD[ai, li - 1]).T          # [128, n_pc] raw vectors
            B0 = onb(Ein[:, :k])
            Bnext = onb((V[li] * SD[ai, li]).T[:, :k])
            layer_act[ai, li] = 1.0 - agree(B0, Bnext)

            cur = Ein[:, :k].copy()
            prev = B0
            for oi in range(n_op):
                cur = cur + M[li, oi] @ cur              # (I + M_oi) applied
                Bk = onb(cur)
                step[ai, li, oi] = 1.0 - agree(prev, Bk)
                prev = Bk
            closure[ai, li] = agree(prev, Bnext)
            layer_pred[ai, li] = 1.0 - agree(B0, prev)

            # Two nulls per operation: dense norm-matched, and spectrum-matched
            # (same singular values, random singular vectors).
            for oi in range(n_op):
                fro = np.linalg.norm(M[li, oi])
                sv = np.linalg.svd(M[li, oi], compute_uv=False)
                acc, acs = np.zeros(a.n_rand), np.zeros(a.n_rand)
                for r in range(a.n_rand):
                    Rm = rng.standard_normal((dim, dim))
                    Rm *= fro / (np.linalg.norm(Rm) + EPS)
                    acc[r] = 1.0 - agree(B0, onb(Ein[:, :k] + Rm @ Ein[:, :k]))
                    Sm = rand_orth(dim) @ np.diag(sv) @ rand_orth(dim).T
                    acs[r] = 1.0 - agree(B0, onb(Ein[:, :k] + Sm @ Ein[:, :k]))
                step_rand[ai, li, oi] = acc.mean()
                step_spec[ai, li, oi] = acs.mean()

    # ---- closure: is the decomposition operating on the right object? -----
    print("closure -- does the composed layer map land on the next basis?\n")
    print(f"  {'layer':>5s} {'closure':>8s} {'rotation to explain':>20s}")
    for li in list(range(1, L, 8)) + [L - 1]:
        print(f"  {li:5d} {closure[:, li].mean():8.3f} "
              f"{layer_act[:, li].mean():20.3f}")
    big = layer_act.mean(0) > 0.02
    big[0] = False
    cl_big = float(closure.mean(0)[big].mean()) if big.any() else float("nan")
    print(f"\n  mean closure over all layers: {closure[:, 1:].mean():.3f}")
    print(f"  mean closure where the layer actually rotates (>0.02): "
          f"{cl_big:.3f}  ({int(big.sum())} layers)")
    if cl_big < 0.8:
        print("\n  WARNING: the composed channel operators do not reproduce the\n"
              "  next layer's basis where it matters. The per-operation split\n"
              "  below is then arithmetic on an object that is not the layer's\n"
              "  real action, and should not be quoted.")
    print()

    # ---- per-operation attribution ---------------------------------------
    tot = step.sum(-1, keepdims=True)
    share = step / (tot + EPS)
    print("share of each layer's rotation, mean over assays and layers\n")
    print(f"  {'operation':>16s} {'share':>8s} {'rotation':>10s} "
          f"{'dense null':>11s} {'spectrum null':>14s} {'vs spectrum':>12s}")
    rows = []
    for oi, o in enumerate(ops):
        sh = float(share[:, 1:, oi].mean())
        rt = float(step[:, 1:, oi].mean())
        nl = float(step_rand[:, 1:, oi].mean())
        sp = float(step_spec[:, 1:, oi].mean())
        rows.append((o, sh, rt, nl, sp, rt - sp))
        print(f"  {o:>16s} {sh:8.3f} {rt:10.4f} {nl:11.4f} {sp:14.4f} "
              f"{rt-sp:+12.4f}")
    print("\n  'dense null' matches Frobenius norm only and is not a fair "
          "comparison:\n  these operators are rank 7-26, a dense random matrix "
          "is not, and rank\n  alone changes how fast a subspace is walked out "
          "of itself. 'spectrum\n  null' holds the singular values fixed and "
          "randomises only direction.\n")

    # ---- where in depth does the rotation happen? ------------------------
    print("rotation per layer by operation (mean over assays)\n")
    print(f"  {'layer':>5s} " + " ".join(f"{o:>14s}" for o in ops)
          + f" {'layer total':>12s}")
    for li in list(range(1, L, 6)) + [L - 1]:
        print(f"  {li:5d} " + " ".join(f"{step[:, li, oi].mean():14.4f}"
                                       for oi in range(n_op))
              + f" {layer_act[:, li].mean():12.4f}")
    print()

    early = layer_act[:, 1:32].mean()
    late = layer_act[:, 32:].mean()
    print(f"rotation per layer: layers 1-31 {early:.4f}, layers 32-63 {late:.4f} "
          f"({late/(early+EPS):.1f}x)")
    lead = max(rows, key=lambda r: r[5])
    print(f"largest excess over its own SPECTRUM-matched null: {lead[0]} "
          f"({lead[5]:+.4f} on a rotation of {lead[2]:.4f})")
    if all(r[5] <= 0 for r in rows):
        print("every operation rotates the subspace no more than a random "
              "operator\nof its own spectrum would -- the rotation is "
              "accounted for by size and\nshape, not by direction.")

    _res = {
        "assays": names, "ops": ops, "layers": L, "k": k,
        "closure_mean": float(closure[:, 1:].mean()),
        "closure_where_rotating": cl_big,
        "share": {o: float(share[:, 1:, oi].mean()) for oi, o in enumerate(ops)},
        "rotation": {o: float(step[:, 1:, oi].mean()) for oi, o in enumerate(ops)},
        "rotation_null_dense": {o: float(step_rand[:, 1:, oi].mean())
                                for oi, o in enumerate(ops)},
        "rotation_null_spectrum": {o: float(step_spec[:, 1:, oi].mean())
                                   for oi, o in enumerate(ops)},
        "excess_vs_spectrum": {
            o: float(step[:, 1:, oi].mean() - step_spec[:, 1:, oi].mean())
            for oi, o in enumerate(ops)},
        "rotation_by_layer": {o: step[:, :, oi].mean(0).tolist()
                              for oi, o in enumerate(ops)},
        "layer_actual": layer_act.mean(0).tolist(),
        "layer_predicted": layer_pred.mean(0).tolist(),
        "closure_by_layer": closure.mean(0).tolist(),
        "early_mean": float(early), "late_mean": float(late),
    }
    pi_archive.write_result(a.out, _res, protocol=pi_protocol.protocol(
        script="analyze_rotate.py",
        design="per-operation rotation attribution against a spectrum-matched "
               "null; directions enter as raw-space vectors e = s*v, the same "
               "bridge pi_basis.to_raw applies",
        layer=pi_protocol.layers("all", n_layers=L),
        features=pi_protocol.features("per-layer basis directions", 128,
                                      kept=a.k),
        source=a.glob, n_assays=len(names), basis=a.basis, n_random=a.n_rand))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
