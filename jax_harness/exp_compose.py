"""Does the linearisation actually explain the observed mutation response?

Everything in the Jacobian study so far is descriptive: it measures what the
z-path operations do to directions in pair-channel space. None of it has been
asked to PREDICT anything. This does, and it can fail.

`gym2_*.npz` already archives, for 250 real variants per assay, the
mutant-minus-wildtype pair-row difference `dz_site` at every one of the 64
Pairformer layers. The linearised layer stack makes a specific claim about how
one layer's difference turns into the next one's. So take the archived
difference, push it through the layer's own Jacobian at the wild-type operating
point, and compare against the difference the model actually produced.

Two predictions, because they fail for different reasons.

  one-step     Predict `dz_site(l)` from the ARCHIVED `dz_site(l-1)`. Errors do
               not accumulate, so this isolates whether each individual layer is
               well described by its linearisation.

  free-running Build the tangent once from `dz_site(0)` and propagate it through
               every layer without ever re-reading the archive. This is the
               actual composition test: it asks whether 63 chained Jacobians
               reproduce the trajectory, and it is where accumulated error and
               any real nonlinearity will show.

The whole [N,N,128] tangent field is propagated -- NOT the 128-vector row
summary. Reducing to the row mean between layers would discard the off-row
structure that the triangle operations read, and would test a model of the
computation rather than the computation. `jax.linearize` is taken on the
layer's full z-path, so all five operations are differentiated together at the
wild-type base point.

THE ONE APPROXIMATION, stated plainly. The archive stores `z_site`, the row mean
over partners, not the full row -- so the initial tangent has to be
reconstructed by assuming the difference is uniform across row r and column r.
That is the same assumption `exp_ops.py` makes when it builds its channel
operators, so the two are consistent, but it is an assumption and it is the
most likely thing to break. A free-running prediction that decays is evidence
about the uniform-row reconstruction at least as much as about nonlinearity,
and the one-step numbers are the control that separates them: one-step stays
accurate if the layer is linear but the reconstruction is lossy, because it is
re-seeded from the archive every layer.

Reported per layer: cosine between predicted and actual `dz_site`, relative L2
error, and the same for the PC2 coordinate alone -- the quantity the rest of the
project is about.

  sbatch analysis.sbatch exp_compose.py --assay <name> ... \
      --gym $R/gym2_<stem>.npz --pc $R/pc2_v2.npz --out $R/comp_<stem>.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m  # noqa: E402
from joltz import TrunkState  # noqa: E402

EPS = 1e-8


def z_path_out(layer, z, pair_mask):
    """The layer's z output. Same five lines as PairformerLayer2, dropout-free.

    Verified against the real layer before use -- see `--tol`.
    """
    z = z + layer.tri_mul_out(z, pair_mask)
    z = z + layer.tri_mul_in(z, pair_mask)
    z = z + layer.tri_att_start(z, pair_mask)
    z = z + layer.tri_att_end(z, pair_mask)
    return z + layer.transition_z(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--gym", required=True)
    ap.add_argument("--pc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()

    print(f"jax devices: {jax.devices()}\n", flush=True)
    t0 = time.time()

    g = np.load(a.gym, allow_pickle=True)
    DZ = np.asarray(g["dz_site"], np.float32)          # [V, L, 128]
    POS = np.asarray(g["pos"]).astype(int)             # 0-indexed sequence pos
    V_n, L, dim = DZ.shape
    wt = str(g["wt_seq"])
    print(f"{a.assay}: {V_n} variants, {L} layers, dim {dim}, len {len(wt)}",
          flush=True)

    work = Path(a.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)

    model = pi.load_model(subsample_msa=False)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    a3m = work / "msa" / "wt_comp.a3m"
    graft_a3m(a3m, Path(a.a3m), wt, wt, cap=a.msa_cap)
    y = work / "yamls" / "wt_comp.yaml"
    y.write_text(YAML_TMPL.format(seq=wt, msa=a3m.resolve()))
    feats, _ = pi.load_features(y.read_text())

    # Rows must be resolved the way exp_gym2 resolved them, or the tangent is
    # placed at a different residue than the archived difference was read at.
    mask_np = np.asarray(feats["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask_np)[0]
    pos_of = {int(r): k for k, r in enumerate(valid)}
    keep = np.array([p in pos_of for p in POS])
    if not keep.all():
        print(f"   dropping {int((~keep).sum())} variants whose position is not "
              f"a valid token")
    rows = np.array([pos_of[int(p)] for p in POS[keep]])
    DZ = DZ[keep]
    V_n = DZ.shape[0]

    # ---- trunk to the Pairformer entry, exactly as exp_jac/exp_gym2 -------
    key = jax.random.key(0)
    emb = model.embed_inputs(feats)
    state = TrunkState(s=jnp.zeros_like(emb.s_init), z=jnp.zeros_like(emb.z_init))
    for i in range(a.recycles - 1):
        state = pi.iteration(model, state, emb, feats, key=jax.random.fold_in(key, i))
    mask = feats["token_pad_mask"]
    pair_mask = mask[:, :, None] * mask[:, None, :]
    k = jax.random.fold_in(key, a.recycles - 1)
    s = emb.s_init + model.s_recycle(model.s_norm(state.s))
    z = emb.z_init + model.z_recycle(model.z_norm(state.z))
    z = z + model.template_module(z, feats, pair_mask, deterministic=True, key=k)
    z = z + model.msa_module(z, emb.s_inputs, feats, deterministic=True,
                             key=jax.random.fold_in(k, 0))
    lay_key = jax.random.fold_in(k, 1)

    pf = model.pairformer_module

    def build(u, r):
        T = jnp.zeros_like(z)
        T = T.at[0, r, :, :].add(u)
        T = T.at[0, :, r, :].add(u)
        return T

    def read(T, r):
        return T[0, r, :, :].mean(axis=0)

    # Free-running tangent, seeded once from the archived layer-0 difference.
    free = None
    pred_one = np.zeros((V_n, L, dim), np.float32)
    pred_free = np.zeros((V_n, L, dim), np.float32)
    worst = 0.0
    kk = lay_key
    for li in range(L):
        layer = eqx.combine(pf.static,
                            jax.tree.map(lambda x, i=li: x[i], pf.stacked_parameters))
        z_rep = z_path_out(layer, z, pair_mask)
        s, z_real, kk_next = layer(s, z, mask, pair_mask, key=kk, deterministic=True)
        d = float(jnp.abs(z_rep - z_real).max() / (jnp.abs(z_real).max() + EPS))
        worst = max(worst, d)
        if d > a.tol:
            raise SystemExit(
                f"layer {li}: replicated z-path does not reproduce the layer "
                f"output (rel err {d:.2e}); every prediction below would be "
                f"taken from a computation the model never ran.")

        _, jvp = jax.linearize(lambda zz, lay=layer: z_path_out(lay, zz, pair_mask), z)
        push = jax.jit(jax.vmap(jvp))

        if li > 0:                       # one-step: re-seed from the archive
            for b in range(0, V_n, a.chunk):
                sl = slice(b, min(b + a.chunk, V_n))
                Ts = jax.vmap(build)(jnp.asarray(DZ[sl, li - 1]),
                                     jnp.asarray(rows[sl]))
                out = push(Ts)
                pred_one[sl, li] = np.asarray(
                    jax.vmap(read)(out, jnp.asarray(rows[sl])))

        if free is None:                 # free-running: seed once, never again
            free = [None] * ((V_n + a.chunk - 1) // a.chunk)
            for bi, b in enumerate(range(0, V_n, a.chunk)):
                sl = slice(b, min(b + a.chunk, V_n))
                free[bi] = jax.vmap(build)(jnp.asarray(DZ[sl, 0]),
                                           jnp.asarray(rows[sl]))
        for bi, b in enumerate(range(0, V_n, a.chunk)):
            sl = slice(b, min(b + a.chunk, V_n))
            if li > 0:
                free[bi] = push(free[bi])
            pred_free[sl, li] = np.asarray(
                jax.vmap(read)(free[bi], jnp.asarray(rows[sl])))

        z, kk = z_real, kk_next
        if li % 16 == 0 or li == L - 1:
            print(f"[{time.time()-t0:6.1f}s]   layer {li:2d} "
                  f"(z-path rel err {d:.2e})", flush=True)

    print(f"\nreplicated z-path vs the real layer: max rel err {worst:.2e}\n")

    # ---- scoring ----------------------------------------------------------
    Pz = np.load(a.pc, allow_pickle=True)
    Vb = np.asarray(Pz["V"]) * np.asarray(Pz["orient"])[:, None]
    sd = DZ[:, -1, :].std(0)
    Wc = Vb / (sd + EPS)                              # readout covectors

    def cos(A, B):
        na = np.linalg.norm(A, axis=-1) + EPS
        nb = np.linalg.norm(B, axis=-1) + EPS
        return (A * B).sum(-1) / (na * nb)

    def relerr(A, B):
        return np.linalg.norm(A - B, axis=-1) / (np.linalg.norm(B, axis=-1) + EPS)

    act = DZ
    stats = {}
    for tag, P in (("one_step", pred_one), ("free", pred_free)):
        c = np.array([cos(P[:, li], act[:, li]).mean() for li in range(L)])
        e = np.array([np.median(relerr(P[:, li], act[:, li])) for li in range(L)])
        pc = {}
        for ci in range(Vb.shape[0]):
            pa = act @ Wc[ci]                          # [V, L]
            pp = P @ Wc[ci]
            r = np.array([np.corrcoef(pp[:, li], pa[:, li])[0, 1] for li in range(L)])
            scale = np.array([
                float(np.polyfit(pa[:, li], pp[:, li], 1)[0]) for li in range(L)])
            pc[f"PC{ci+1}"] = {"r": r.tolist(), "slope": scale.tolist()}
        stats[tag] = {"cosine": c.tolist(), "rel_err": e.tolist(), "pc": pc}

    print("prediction vs the model's own dz_site\n")
    print(f"  {'layer':>5s} {'one-step cos':>13s} {'relerr':>8s} "
          f"{'free cos':>10s} {'relerr':>8s} {'PC2 r (free)':>13s}")
    for li in list(range(1, L, 8)) + [L - 1]:
        print(f"  {li:5d} {stats['one_step']['cosine'][li]:13.3f} "
              f"{stats['one_step']['rel_err'][li]:8.3f} "
              f"{stats['free']['cosine'][li]:10.3f} "
              f"{stats['free']['rel_err'][li]:8.3f} "
              f"{stats['free']['pc']['PC2']['r'][li]:13.3f}")
    print()

    np.savez_compressed(a.out, assay=a.assay, rows=rows,
                        pred_one=pred_one, pred_free=pred_free,
                        act=act.astype(np.float32),
                        stats=np.array(json.dumps(stats)))
    print(f"[{time.time()-t0:6.1f}s] wrote {a.out}")


if __name__ == "__main__":
    main()
