"""Why does the Jacobian's rank collapse from ~87 (weights) to ~24 (operating point)?

`analyze_jac` found the pair transition's Jacobian sitting at effective rank
~24 of 128 in every one of twelve folds, while `probe_wsvd` found fc3's bare
weights at 87. Something about the operating point is removing most of the
matrix. The obvious suspect is the SwiGLU gate: `silu(fc1 v) * fc2 v` is near
zero for any hidden unit whose fc1 pre-activation is well below zero, and a dead
unit contributes nothing to the Jacobian no matter what its fc3 column holds.

This checks that directly rather than asserting it. Each hidden unit u
contributes to J = W3 @ [diag(silu'(a) * b) W1 + diag(silu(a)) W2] @ dLN/dz a
term whose size is set by the row norm of

    r_u = silu'(a_u) * b_u * W1[u] + silu(a_u) * W2[u]

scaled by fc3's column u. The participation ratio of {||W3[:,u]|| * ||r_u||}
over the 512 units is how many units are effectively live. If that number is
close to the Jacobian's effective rank, the gate is the mechanism; if it is
close to 512, something else is.

Reuses the `z_pre` already archived by `exp_jac`, so no re-capture is needed.

  sbatch analysis.sbatch probe_gate.py --glob '../runs/jac_*.npz' --out ...
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import equinox as eqx
import jax

# jax defaults to float32 and downcasts a float64 array on the way in
# without saying so. That is not hypothetical here: pc2_v2.npz -- the PC
# basis six analyses inherit -- was computed in single precision for a
# week because analyze_pc2 passed float32 straight to numpy, and its
# orthonormality was 1.06e-08 where float64 gives 3.11e-15. Rank
# statistics hid it. x64 is enabled explicitly wherever this project
# reduces through jnp.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402
import pi_core as pi  # noqa: E402

EPS = 1e-8


def pr(x, axis=-1):
    """Participation ratio: how many entries carry the mass."""
    p = x ** 2
    return (p.sum(axis) ** 2) / ((p ** 2).sum(axis) + EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print(f"jax devices: {jax.devices()}\n", flush=True)
    model = pi.load_model(subsample_msa=False)
    pf = model.pairformer_module
    L = pf.stacked_parameters.transition_z.fc1.weight.shape[0]
    trans = [eqx.combine(pf.static,
                         jax.tree.map(lambda x, i=i: x[i], pf.stacked_parameters)
                         ).transition_z for i in range(L)]

    files = sorted(glob.glob(a.glob))
    live_all, rank_all, names = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "z_pre" not in d.files:
            continue
        names.append(str(d["assay"]).split("_")[0])
        z = d["z_pre"]                                    # [L, P, 128]
        sv = d["sv"]                                      # [L, P, 128]
        live = np.zeros((L, z.shape[1]))
        for li in range(L):
            t = trans[li]
            v = jax.vmap(t.norm)(jnp.asarray(z[li], jnp.float32))   # [P,128]
            a1 = jax.vmap(t.fc1)(v)                                  # [P,512]
            b = jax.vmap(t.fc2)(v)                                   # [P,512]
            sig = jax.nn.sigmoid(a1)
            dsilu = sig * (1 + a1 * (1 - sig))            # d/da silu(a)
            W1, W2 = t.fc1.weight, t.fc2.weight           # [512,128]
            # ||r_u|| per pair per unit, without materialising [P,512,128].
            n11 = jnp.einsum("ud,ud->u", W1, W1)
            n22 = jnp.einsum("ud,ud->u", W2, W2)
            n12 = jnp.einsum("ud,ud->u", W1, W2)
            c1, c2 = dsilu * b, jax.nn.silu(a1)
            rn = jnp.sqrt(jnp.clip(c1 ** 2 * n11 + c2 ** 2 * n22
                                   + 2 * c1 * c2 * n12, 0.0))        # [P,512]
            w3n = jnp.linalg.norm(t.fc3.weight, axis=0)               # [512]
            live[li] = np.asarray(pr(rn * w3n[None, :], axis=-1))
        live_all.append(live)
        rank_all.append((sv ** 2).sum(-1) ** 2 / (((sv ** 2) ** 2).sum(-1) + EPS))
        print(f"  {names[-1]}: live units median {np.median(live):.1f} / 512", flush=True)

    live = np.stack(live_all)                             # [A, L, P]
    rank = np.stack(rank_all)
    print(f"\n{len(names)} assays: {', '.join(names)}\n")
    print("live hidden units (participation ratio of per-unit contribution, of 512)")
    print("vs the Jacobian's own effective rank (of 128), mean over pairs & assays\n")
    print(f"  {'layer':>5s} {'live/512':>9s} {'effrank/128':>12s}")
    for li in list(range(0, L, 8)) + [L - 1]:
        print(f"  {li:5d} {live[:, li].mean():9.1f} {rank[:, li].mean():12.1f}")
    print(f"\n  overall: live {live.mean():.1f}/512 ({100*live.mean()/512:.1f}%), "
          f"effrank {rank.mean():.1f}/128")
    c = np.corrcoef(live.mean(-1).ravel(), rank.mean(-1).ravel())[0, 1]
    print(f"  correlation across (assay, layer): r = {c:.3f}")

    _res = {
        "assays": names,
        "live_by_layer": live.mean((0, 2)).tolist(),
        "rank_by_layer": rank.mean((0, 2)).tolist(),
        "live_mean": float(live.mean()), "rank_mean": float(rank.mean()),
        "corr": float(c)}
    pi_archive.write_result(a.out, _res, protocol=pi_protocol.protocol(
        script="probe_gate.py",
        design="descriptive: how many SwiGLU units are live at the operating "
               "point, and whether that count tracks the perturbation",
        layer=pi_protocol.layers("final"),
        features=pi_protocol.features("SwiGLU hidden units", 512),
        source=a.glob, n_assays=len(names)))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
