"""Feasibility probe: SVD of the pair transition's OWN weights (NaNA-style).

Everything in the SVD study so far decomposes ACTIVATIONS -- `dz_site`, the
mutant-minus-wildtype pair-channel difference. That says which directions the
mutation response occupies. It says nothing about which directions the network
was BUILT to read and write. Xue & Andrzejak (ICML 2026, "SVD as a Fast
Interpretability Method for Transformers") decompose the weight matrices
instead, pairing each right singular vector (a "detector", what the unit reads)
with the matching left singular vector (an "effector", what it writes). This
probe asks whether that transfers to Boltz-2's `transition_z`.

Three things are measured, cheapest first.

  spectra     Singular values of fc1, fc2, fc3 in every Pairformer layer. If
              these are near-flat the rank-1 decomposition has no preferred
              units to find and the whole approach is decorative. Reported as
              a participation-ratio effective rank against the ceiling (128,
              since every one of these matrices is at most rank 128).

  effectors   fc3 has shape [128, 512]: its COLUMNS live in exactly the same
              128 pair channels that `dz_site` and the PC basis live in, so
              the comparison needs no bridge. For each layer the probe asks
              how much of the PC2 readout direction is spanned by the top-k
              left singular vectors of fc3, against the k/128 a random
              direction would give.

  detectors   fc1 and fc2 read z AFTER the transition's own LayerNorm, whose
              diagonal gain is folded in here (rows scaled by `norm.weight`,
              then projected off the constant vector, which LayerNorm's
              mean-subtraction removes). Skipping either step compares the
              wrong vectors and would flatter or flatten the result for a
              reason that has nothing to do with the model.

The PC basis in `pc2_v2.npz` was fitted on PER-ASSAY STANDARDISED `dz_site`
(`zc` in `analyze_pc2.py` divides by each channel's within-assay spread), so a
component `v` there is the raw-channel readout `v / s_assay`, NOT `v`. The
weights know nothing about that scaling, so the un-standardisation is applied
per assay and the spread across the twelve is reported. Comparing `v` directly
against a singular vector would be comparing two different bases.

WHAT THIS PROBE DOES NOT DO. The paper's detector-effector pairing assumes a
two-matrix MLP, `W_out @ W_in`. `transition_z` is SwiGLU --
`fc3(silu(fc1 v) * fc2 v)` -- so there is no single linear map to decompose and
the detector side is genuinely ambiguous: fc1 and fc2 feed a MULTIPLICATIVE
gate, and which of them "detects" depends on the operating point. The honest
composite is the Jacobian at a real z, which needs a capture; this probe is the
weights-only tier that says whether that is worth running.

  sbatch analysis.sbatch probe_wsvd.py --out ../runs/wsvd_probe.json \
      --npz ../runs/wsvd_probe.npz
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-8


def eff_rank(sv):
    """Participation ratio of the squared spectrum: (sum s^2)^2 / sum s^4.

    Preferred over a threshold count because it needs no cutoff, and over
    entropy-based rank because it is on the same scale as the raw dimension.
    """
    p = sv ** 2
    return float((p.sum() ** 2) / ((p ** 2).sum() + EPS))


def frac_in_top(sv, k):
    p = sv ** 2
    return float(p[:k].sum() / (p.sum() + EPS))


def subspace_capture(U, w):
    """Fraction of unit vector `w` captured by the span of columns U[:, :k].

    Returns the cumulative curve over k. U is orthonormal, so this is just the
    running sum of squared coefficients.
    """
    c = (U.T @ w) ** 2
    return np.cumsum(c) / (c.sum() + EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--pc", default="../runs/pc2_v2.npz")
    ap.add_argument("--glob", default="../runs/gym2_*.npz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"jax devices: {jax.devices()}\n")

    # ---- weights ---------------------------------------------------------
    from mosaic.models.boltz2 import Boltz2
    model = Boltz2().model
    tz = model.pairformer_module.stacked_parameters.transition_z

    W1 = np.asarray(tz.fc1.weight)          # [L, hidden, 128]
    W2 = np.asarray(tz.fc2.weight)          # [L, hidden, 128]
    W3 = np.asarray(tz.fc3.weight)          # [L, 128, hidden]
    gain = np.asarray(tz.norm.weight)       # [L, 128]
    L, hidden, dim = W1.shape
    print(f"transition_z: {L} layers, dim {dim}, hidden {hidden}")
    print(f"  fc1 {W1.shape}  fc2 {W2.shape}  fc3 {W3.shape}  norm {gain.shape}\n")
    assert W3.shape == (L, dim, hidden), W3.shape

    # LayerNorm folding for the detector side: gain is diagonal on the input,
    # and mean-subtraction kills the constant direction.
    def fold_ln(W, g):
        Wg = W * g[:, None, :]                       # [L, hidden, dim]
        return Wg - Wg.mean(-1, keepdims=True)

    W1f, W2f = fold_ln(W1, gain), fold_ln(W2, gain)

    # ---- spectra ---------------------------------------------------------
    svd = jax.jit(jax.vmap(lambda M: jnp.linalg.svd(M, full_matrices=False)))
    mats = {"fc1": W1, "fc2": W2, "fc3": W3,
            "fc1_ln": W1f, "fc2_ln": W2f}
    spec, bases = {}, {}
    for k, M in mats.items():
        U, s, Vt = svd(jnp.asarray(M, jnp.float32))
        spec[k] = np.asarray(s)                      # [L, min(dim,hidden)]
        bases[k] = (np.asarray(U), np.asarray(Vt))

    print(f"{'layer':>5s} " + " ".join(f"{k:>16s}" for k in mats))
    print(f"{'':>5s} " + " ".join(f"{'effrank  top8%':>16s}" for _ in mats))
    for li in list(range(min(4, L))) + ([L - 1] if L > 4 else []):
        row = []
        for k in mats:
            sv = spec[k][li]
            row.append(f"{eff_rank(sv):7.1f} {100*frac_in_top(sv, 8):6.1f}%")
        print(f"{li:5d} " + " ".join(f"{r:>16s}" for r in row))
    print(f"\n  (ceiling for effective rank is {min(dim, hidden)}; a matrix with "
          f"a flat\n   spectrum sits at the ceiling and has no preferred units)\n")

    # ---- PC directions, un-standardised per assay ------------------------
    P = np.load(a.pc, allow_pickle=True)
    V = np.asarray(P["V"])                            # [n_pc, 128]
    orient = np.asarray(P["orient"])
    V = V * orient[:, None]
    n_pc = V.shape[0]

    files = sorted(glob.glob(a.glob))
    stds, names = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        X = np.asarray(d["dz_site"])[:, -1, :]        # last layer, [n_var, 128]
        stds.append(X.std(0))
        names.append(Path(f).stem.split("_", 1)[1].split("_")[0])
    stds = np.stack(stds)                             # [n_assay, 128]
    print(f"assays for un-standardisation: {len(names)} -- {', '.join(names)}\n")

    # w_raw[a, c] = normalise(V[c] / s[a])
    W_raw = V[None, :, :] / (stds[:, None, :] + EPS)  # [n_assay, n_pc, 128]
    W_raw /= np.linalg.norm(W_raw, axis=-1, keepdims=True) + EPS

    # ---- how much of each PC does fc3's effector basis span? -------------
    ks = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    ks = ks[ks <= dim]
    cap = np.zeros((L, n_pc, len(stds), len(ks)))
    for li in range(L):
        U = bases["fc3"][0][li]                       # [128, min] left = effectors
        for c in range(n_pc):
            for ai in range(len(stds)):
                curve = subspace_capture(U, W_raw[ai, c])
                cap[li, c, ai] = curve[ks - 1]

    # Random baseline: a uniform unit vector in 128-d captures k/128 in
    # expectation under any fixed orthonormal basis.
    base = ks / dim

    rng = np.random.default_rng(a.seed)
    R = rng.standard_normal((256, dim))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    rand_cap = np.stack([subspace_capture(bases["fc3"][0][L - 1], r)[ks - 1]
                         for r in R])

    print("fc3 effector subspace vs the PC readout directions "
          f"(layer {L-1}, mean over assays)\n")
    hdr = "  ".join(f"k={k:<4d}" for k in ks)
    print(f"  {'direction':>12s}  {hdr}")
    for c in range(n_pc):
        v = cap[L - 1, c].mean(0)
        print(f"  {'PC'+str(c+1):>12s}  " + "  ".join(f"{x:6.3f}   " for x in v))
    print(f"  {'random':>12s}  " + "  ".join(f"{x:6.3f}   " for x in rand_cap.mean(0)))
    print(f"  {'analytic':>12s}  " + "  ".join(f"{x:6.3f}   " for x in base))
    print()

    # Depth profile at a fixed k, so "is this a late-layer property" is visible.
    kk = int(np.argmin(np.abs(ks - 8)))
    print(f"depth profile, capture at k={ks[kk]} (mean over assays)\n")
    print(f"  {'layer':>5s}  " + "  ".join(f"{'PC'+str(c+1):>7s}" for c in range(n_pc)))
    for li in range(L):
        print(f"  {li:5d}  " + "  ".join(f"{cap[li, c, :, kk].mean():7.3f}"
                                         for c in range(n_pc)))
    print()

    out = {
        "layers": L, "dim": dim, "hidden": hidden,
        "assays": names, "ks": ks.tolist(),
        "eff_rank": {k: [eff_rank(spec[k][li]) for li in range(L)] for k in mats},
        "top8_frac": {k: [frac_in_top(spec[k][li], 8) for li in range(L)] for k in mats},
        "capture_last_layer": {f"PC{c+1}": cap[L - 1, c].mean(0).tolist()
                               for c in range(n_pc)},
        "capture_random_last_layer": rand_cap.mean(0).tolist(),
        "capture_analytic_baseline": base.tolist(),
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {a.out}")

    if a.npz:
        np.savez_compressed(
            a.npz, ks=ks, capture=cap, rand_cap=rand_cap, assays=np.array(names),
            **{f"sv_{k}": spec[k] for k in mats},
            U_fc3=bases["fc3"][0], Vt_fc1_ln=bases["fc1_ln"][1],
            Vt_fc2_ln=bases["fc2_ln"][1], W_raw=W_raw)
        print(f"wrote {a.npz}")


if __name__ == "__main__":
    main()
