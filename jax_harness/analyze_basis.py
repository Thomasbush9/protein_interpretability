"""Does "PC2" mean the same thing at every depth?

Every depth-resolved claim in the Jacobian study inherits an assumption nothing
has checked: the PC basis in `pc2_v2.npz` was fitted on `dz_site` at the FINAL
Pairformer layer, and then applied at all 64. If the mutation subspace rotates
with depth, a per-layer gain along the last layer's PC2 is a gain along a
direction that layer does not use, and the depth profiles describe an artefact.

The last-layer numbers are unaffected either way -- this is about whether the
profiles beneath them mean anything.

Three measurements, all on the archived `gym2_*` runs; no GPU capture and no
model load is involved.

  rotation    Fit the shared basis independently at each layer, by exactly the
              protocol `analyze_pc2.py` uses (pool the twelve assays, z-score
              per channel WITHIN each assay, subtract the pooled mean, SVD),
              then take principal angles against the layer-63 basis. Reported
              as mean cos^2 over the top-k subspace, against the k/128 that two
              unrelated bases would give.

  identity    A subspace can be stable while the individual components rotate
              inside it. So the per-component |cos| against the layer-63
              component of the same index is reported separately. If the
              subspace holds but the components swap, "PC2" is not a fixed
              direction even though the span is.

  meaning     Whether the layer-l component still tracks what PC2 was NAMED
              for. Each layer's own component is scored against `kl_glob` at
              that layer, pooled by assay with the project's cluster bootstrap,
              so a component that survives geometrically but stops predicting
              is visible as such.

Component signs are arbitrary in an SVD, so every comparison here is on |cos|
and every correlation is oriented by the same rule `analyze_pc2.py` uses --
positive score means the internal state moved more.

  sbatch analysis.sbatch analyze_basis.py --glob '../runs/gym2_*.npz' \
      --pc ../runs/pc2_v2.npz --out ../runs/basis_depth.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
import pi_basis  # noqa: E402
import pi_stats  # noqa: E402

EPS = 1e-8


def basis_at(layers_by_assay, li, n_pc, KL=None, n_boot=500):
    """The shared basis at layer li, through pi_basis.

    The docstring this replaces said "by analyze_pc2's protocol exactly" and
    the function did NOT orient at all -- orientation lived only in the --npz
    writer below, and there it used n_boot=500 against analyze_pc2's 2000. Two
    bootstrap sizes can disagree on the sign of a near-zero correlation, and
    this is the file that writes basis_depth.npz. Pass KL to orient; the
    bootstrap size is now an argument instead of a number in one branch.

    The full (n, L, dim) block goes in rather than a pre-sliced one, so `li` is
    a layer pi_basis MEASURES rather than one this function asserts.
    """
    blocks = {str(i): A for i, A in enumerate(layers_by_assay)}
    kw = dict(layer=li, n_pc=n_pc, eps=EPS)
    if KL is None:
        B = pi_basis.fit(blocks, orient_on=None, **kw)
    else:
        B = pi_basis.fit(blocks, orient_on="kl_glob", orient_k=n_pc,
                         n_boot=n_boot,
                         orient_ref={str(i): K[:, li] for i, K in enumerate(KL)},
                         **kw)
    return B


def princ(A, B):
    return np.clip(np.linalg.svd(A.T @ B, compute_uv=False), -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--pc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default="", help="per-layer bases, for analyze_ops")
    ap.add_argument("--n-pc", type=int, default=4)
    ap.add_argument("--k", type=int, default=4, help="subspace dim for rotation")
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"no files matched {a.glob}")
    names, DZ, KL = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        names.append(Path(f).stem.split("_", 1)[1].split("_")[0])
        DZ.append(np.asarray(d["dz_site"], np.float64))
        KL.append(np.asarray(d["kl_glob"], np.float64))
    L, dim = DZ[0].shape[1], DZ[0].shape[2]
    print(f"{len(names)} assays: {', '.join(names)}")
    print(f"{L} layers, dim {dim}\n")

    ref = np.load(a.pc, allow_pickle=True)
    V_ref_file = np.asarray(ref["V"]) * np.asarray(ref["orient"])[:, None]

    # The archived basis must be reproducible from the archives, or the
    # comparison below is against a basis of unknown provenance.
    V_last = basis_at(DZ, L - 1, a.n_pc).components
    agree = [float(abs(V_last[c] @ V_ref_file[c])) for c in range(a.n_pc)]
    print("refit of the last-layer basis vs the archived pc2_v2.npz (|cos|):")
    print("   " + "  ".join(f"PC{c+1} {agree[c]:.4f}" for c in range(a.n_pc)))
    if min(agree) < 0.99:
        print("   WARNING: the refit does not reproduce the archived basis; the\n"
              "   protocol here differs from analyze_pc2 in some way and the\n"
              "   rotation numbers below are not comparable to the study.")
    print()

    Vs = [basis_at(DZ, li, a.n_pc).components for li in range(L)]

    # ---- rotation of the subspace ----------------------------------------
    k = min(a.k, a.n_pc)
    rot = [float((princ(Vs[li][:k].T, Vs[L - 1][:k].T) ** 2).mean())
           for li in range(L)]
    # Per-component identity
    ident = np.array([[float(abs(Vs[li][c] @ Vs[L - 1][c])) for c in range(a.n_pc)]
                      for li in range(L)])
    # Adjacent-layer rotation says whether drift is gradual or has a hinge.
    adj = [float((princ(Vs[li][:k].T, Vs[li + 1][:k].T) ** 2).mean())
           for li in range(L - 1)]

    print(f"top-{k} subspace vs the layer-{L-1} basis (mean cos^2; "
          f"unrelated bases give {k/dim:.3f})\n")
    print(f"  {'layer':>5s} {'vs last':>8s} {'vs next':>8s}  "
          + "  ".join(f"{'|cos| PC'+str(c+1):>10s}" for c in range(a.n_pc)))
    for li in list(range(0, L, 4)) + [L - 1]:
        nxt = f"{adj[li]:8.3f}" if li < L - 1 else f"{'-':>8s}"
        print(f"  {li:5d} {rot[li]:8.3f} {nxt}  "
              + "  ".join(f"{ident[li, c]:10.3f}" for c in range(a.n_pc)))
    print()

    # ---- does the layer's own component still predict? -------------------
    print("each layer's OWN component vs kl_glob at that layer "
          "(cluster bootstrap over assays)\n")
    print(f"  {'layer':>5s} " + "  ".join(f"{'PC'+str(c+1):>16s}"
                                          for c in range(a.n_pc)))
    meaning = np.zeros((L, a.n_pc))
    for li in list(range(0, L, 8)) + [L - 1]:
        Bl = basis_at(DZ, li, a.n_pc)
        cells = []
        for c in range(a.n_pc):
            g = {}
            for ai, nm in enumerate(names):
                P = Bl.project(DZ[ai], layer=li)[:, c]
                g[nm] = [pi_stats.spearman(P, KL[ai][:, li])]
            m, lo, hi = pi_stats.cluster_bootstrap(g, n_boot=2000, seed=0,
                                                   hierarchical=False)[:3]
            s = 1.0 if m >= 0 else -1.0
            meaning[li, c] = s * m
            cells.append(f"{s*m:+.3f} [{min(s*lo, s*hi):+.2f},{max(s*lo, s*hi):+.2f}]")
        print(f"  {li:5d} " + "  ".join(f"{c:>16s}" for c in cells))
    print()

    hinge = int(np.argmin(adj)) if adj else -1
    print(f"lowest adjacent-layer agreement at the {hinge}->{hinge+1} boundary "
          f"({min(adj):.3f})" if adj else "")
    lo = min(rot[: L - 1]) if L > 1 else 1.0
    print(f"worst agreement with the last-layer basis over all depths: {lo:.3f}\n")

    # Per-layer bases for `analyze_ops --basis`. Every gain measured against
    # the LAST layer's basis at a shallow layer is a gain along a direction
    # that layer does not use -- the rotation above is what makes this file
    # necessary rather than a refinement. Each component is oriented so its
    # pooled correlation with kl_glob at its own layer is non-negative, the
    # same convention analyze_pc2 applies at the last layer, so the sign of a
    # gain means the same thing at every depth.
    if a.npz:
        Vout = np.zeros((L, a.n_pc, dim))
        SD = np.zeros((len(names), L, dim))
        GM = np.zeros((L, dim))
        for li in range(L):
            Bl = basis_at(DZ, li, a.n_pc, KL=KL, n_boot=500)
            GM[li] = Bl.gm
            Vout[li] = Bl.components
            for ai in range(len(names)):
                SD[ai, li] = DZ[ai][:, li, :].std(0)
        np.savez_compressed(a.npz, V=Vout, sd=SD, gm=GM,
                            assays=np.array(names))
        print(f"wrote {a.npz}  (per-layer bases, oriented)")

    Path(a.out).write_text(json.dumps({
        "assays": names, "layers": L, "dim": dim, "n_pc": a.n_pc, "k": k,
        "refit_vs_archived": agree,
        "rot_vs_last": rot, "rot_adjacent": adj,
        "identity_vs_last": ident.tolist(),
        "meaning_spearman": meaning.tolist(),
        "random_baseline": k / dim,
        "worst_vs_last": lo, "hinge_layer": hinge,
    }, indent=2, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
