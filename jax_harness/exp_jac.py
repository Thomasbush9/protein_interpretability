"""The pair transition's Jacobian at a real operating point.

`probe_wsvd.py` decomposed `transition_z`'s raw weights and found the mutation
axis sitting at or below chance in fc3's effector basis. That result has a known
weakness: the singular ordering of fc3 reflects weight magnitude alone, while
what the layer actually WRITES depends on the hidden activations, which are
nowhere near isotropic. A direction with a small singular value can dominate the
output if its units fire hard.

It also has a structural problem. The paper's detector-effector pairing assumes
a two-matrix MLP, `W_out @ W_in`. This is SwiGLU --
`fc3(silu(fc1 v) * fc2 v)` -- so there is no single linear map to decompose, and
which of fc1/fc2 is "the detector" depends on where you are in activation space.

Both problems have the same fix: decompose the JACOBIAN at a real z instead of
the weights. That is an unambiguous 128x128 map, it needs no linearisation
choice, and it acts on exactly the `dz` the rest of the project measures.

THE OPERATING POINT. The Jacobian must be taken at the transition's INPUT, and
that is not a quantity anything in the harness currently stores. `z_layers` from
`pi_core.pairformer_capture` is the layer OUTPUT, i.e. z AFTER the transition's
residual has been added; the layer input is z BEFORE the four triangle ops. The
transition input sits between them.

It is recovered here without reimplementing the layer. Reading
`PairformerLayer2.__call__`, the transition is the last thing to touch z --

    z = z + dropout * self.tri_mul_out(z, pair_mask)
    z = z + dropout * self.tri_mul_in(z, pair_mask)
    z = z + dropout * self.tri_att_start(z, pair_mask)
    z = z + dropout * self.tri_att_end(z, pair_mask)
    z = z + self.transition_z(z)          <-- last z operation

-- so running the SAME layer object with `fc3.weight` set to zero returns
exactly the transition's input: fc3 has no bias, so the transition contributes
identically zero and everything upstream of it is untouched. Both runs start
from the same key and make the same four `get_dropout_mask` calls, so they agree
bit-for-bit up to that line. `--check` verifies the identity
`z_out == z_pre + transition_z(z_pre)` rather than trusting the argument.

The alternative -- reusing the layer output as the base point -- would be wrong
by the transition's own residual, which is the very quantity being measured.

WHAT IS COMPUTED, per layer and per sampled residue pair:

  gain      For PC component c, `w_c . J e_c`. This is the interpretable
            scalar: if the mutation moves z by one PC-c unit, how much
            additional PC-c does the transition write. Because the transition
            is a residual branch, the layer's total multiplier on that
            coordinate is `1 + gain`, so the sign says amplify or attenuate.

            `e_c` and `w_c` are NOT the same vector. `pc2_v2.npz` was fitted on
            per-assay standardised `dz_site`, so a unit step along component c
            in that space is the raw-space VECTOR `e_c = s * v_c`, while the
            raw-space READOUT of that coordinate is the COVECTOR `w_c = v_c / s`.
            They satisfy `w_c . e_c = 1`, which is what makes the gain a clean
            multiplier. Using `v_c` for both -- the obvious shortcut -- silently
            measures a different quantity in a basis the model never uses.

            Computed by `jax.jvp` through the real module, so LayerNorm's own
            Jacobian (mean removal and the 1/rms factor, both of which depend on
            z) is exact rather than hand-derived.

  spectrum  SVD of the full 128x128 J. Says how much of the pair space the
            transition can move at all, and how anisotropic it is.

  placement Where `e_c` sits in J's right singular basis and `w_c` in its left
            one -- the activation-aware version of the question `probe_wsvd`
            asked of the bare weights.

The null is matched to the construction: a random direction `r` drawn
orthonormally in the standardised space and mapped through the SAME `s`, so
`e_r = s * r`, `w_r = r / s`, `w_r . e_r = 1`. Comparing against an unstructured
random 128-vector would confound the PC's own scaling with the model's geometry.

  sbatch analysis.sbatch exp_jac.py --assay <name> --assay-dir ... --a3m ... \
      --work ... --out ../runs/jac_<name>.npz --gym ../runs/gym2_<stem>.npz
"""

from __future__ import annotations

import argparse
import csv
import re
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


def transition_capture(pf, s, z, mask, pair_mask, ii, jj, *, key):
    """Per-layer transition input and output at the sampled pairs.

    The zeroed-fc3 replica runs inside the same scan body as the real layer, so
    it sees the identical carry and key and costs one extra layer evaluation
    rather than a second pass over the stack.
    """
    def body(carry, params):
        s_, z_, k_ = carry
        layer = eqx.combine(pf.static, params)
        muted = eqx.tree_at(
            lambda l: l.transition_z.fc3.weight, layer,
            jnp.zeros_like(layer.transition_z.fc3.weight),
        )
        _, z_pre, _ = muted(s_, z_, mask, pair_mask, key=k_, deterministic=True)
        s_, z_, k_ = layer(s_, z_, mask, pair_mask, key=k_, deterministic=True)
        return (s_, z_, k_), {"z_pre": z_pre[0][ii, jj], "z_out": z_[0][ii, jj]}

    (s, z, key), per = jax.lax.scan(body, (s, z, key), pf.stacked_parameters)
    return s, z, per


def run_wt(model, feats, ii, jj, *, recycles, key):
    """Trunk to the final recycle, then the Pairformer with transition capture.

    Mirrors `exp_gym2.trunk_capture`'s key schedule exactly -- the operating
    point is only comparable to the archived runs if the model saw the same
    alignment and the same recycles.
    """
    emb = model.embed_inputs(feats)
    state = TrunkState(s=jnp.zeros_like(emb.s_init), z=jnp.zeros_like(emb.z_init))
    for i in range(recycles - 1):
        state = pi.iteration(model, state, emb, feats, key=jax.random.fold_in(key, i))
    mask = feats["token_pad_mask"]
    pair_mask = mask[:, :, None] * mask[:, None, :]
    k = jax.random.fold_in(key, recycles - 1)
    s = emb.s_init + model.s_recycle(model.s_norm(state.s))
    z = emb.z_init + model.z_recycle(model.z_norm(state.z))
    z = z + model.template_module(z, feats, pair_mask, deterministic=True, key=k)
    z = z + model.msa_module(z, emb.s_inputs, feats, deterministic=True,
                             key=jax.random.fold_in(k, 0))
    return transition_capture(model.pairformer_module, s, z, mask, pair_mask,
                              ii, jj, key=jax.random.fold_in(k, 1))


def layer_transitions(pf):
    """The 64 `transition_z` modules, unstacked."""
    L = pf.stacked_parameters.transition_z.fc1.weight.shape[0]
    return [eqx.combine(pf.static,
                        jax.tree.map(lambda x: x[i], pf.stacked_parameters)).transition_z
            for i in range(L)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--gym", required=True, help="gym2_<stem>.npz, for the per-channel spread")
    ap.add_argument("--pc", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--n-pairs", type=int, default=128)
    ap.add_argument("--n-rand", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", default=True)
    a = ap.parse_args()

    print(f"jax devices: {jax.devices()}\n", flush=True)
    t0 = time.time()

    rows = [r for r in csv.DictReader(open(Path(a.assay_dir) / f"{a.assay}.csv"))
            if ":" not in r["mutant"]]
    wt = list(rows[0]["mutated_sequence"])
    m0 = re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"])
    wt[int(m0.group(2)) - 1] = m0.group(1)
    wt = "".join(wt)

    work = Path(a.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)

    model = pi.load_model(subsample_msa=False)
    print(f"[{time.time()-t0:6.1f}s] model loaded; {a.assay} len={len(wt)}", flush=True)

    a3m = work / "msa" / "wt_jac.a3m"
    graft_a3m(a3m, Path(a.a3m), wt, wt, cap=a.msa_cap)
    y = work / "yamls" / "wt_jac.yaml"
    y.write_text(YAML_TMPL.format(seq=wt, msa=a3m.resolve()))
    feats, _ = pi.load_features(y.read_text())

    rng = np.random.default_rng(a.seed)
    mask_np = np.asarray(feats["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask_np)[0]
    p, q = rng.choice(valid, a.n_pairs * 2), rng.choice(valid, a.n_pairs * 2)
    keep = p != q
    ii_np, jj_np = p[keep][:a.n_pairs], q[keep][:a.n_pairs]
    ii, jj = jnp.asarray(ii_np), jnp.asarray(jj_np)
    print(f"[{time.time()-t0:6.1f}s] {len(ii_np)} pairs sampled from {len(valid)} "
          f"valid tokens", flush=True)

    _, _, per = run_wt(model, feats, ii, jj, recycles=a.recycles,
                       key=jax.random.key(0))
    z_pre = np.asarray(per["z_pre"], np.float64)      # [L, P, 128]
    z_out = np.asarray(per["z_out"], np.float64)
    L, P, dim = z_pre.shape
    print(f"[{time.time()-t0:6.1f}s] captured z_pre {z_pre.shape}\n", flush=True)

    trans = layer_transitions(model.pairformer_module)

    # ---- the identity that justifies the zeroed-fc3 trick ------------------
    if a.check:
        worst = 0.0
        for li in range(L):
            recon = np.asarray(jax.vmap(trans[li])(jnp.asarray(z_pre[li], jnp.float32)),
                               np.float64) + z_pre[li]
            d = np.abs(recon - z_out[li]).max() / (np.abs(z_out[li]).max() + EPS)
            worst = max(worst, d)
        print(f"z_out == z_pre + transition_z(z_pre): max rel err {worst:.2e}")
        if worst > 1e-4:
            raise SystemExit(
                f"the recovered transition input does not reproduce the layer "
                f"output (rel err {worst:.2e}). Every Jacobian below would be "
                f"taken at a point the model never visited.")
        print("  -> the recovered base point is the transition's real input\n")

    # ---- PC directions, bridged out of the standardised basis -------------
    g = np.load(a.gym, allow_pickle=True)
    sd = np.asarray(g["dz_site"])[:, -1, :].std(0)     # [128], per-channel spread
    if a.pc:
        Pz = np.load(a.pc, allow_pickle=True)
        V = np.asarray(Pz["V"]) * np.asarray(Pz["orient"])[:, None]
    else:
        raise SystemExit("--pc is required")
    n_pc = V.shape[0]

    E = V * sd[None, :]                                # vectors  e_c = s * v_c
    W = V / (sd[None, :] + EPS)                        # covectors w_c = v_c / s
    print(f"w_c . e_c = {[f'{float(W[c] @ E[c]):.4f}' for c in range(n_pc)]} "
          f"(should be 1)\n")

    # Matched null: orthonormal in the standardised basis, same s bridge.
    Q = rng.standard_normal((a.n_rand, dim))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    E_r, W_r = Q * sd[None, :], Q / (sd[None, :] + EPS)

    # ---- gains via JVP through the real module ----------------------------
    def gains_for(li, Evecs, Wvecs):
        zl = jnp.asarray(z_pre[li], jnp.float32)
        out = np.zeros((len(Evecs), P))
        for c, e in enumerate(Evecs):
            ev = jnp.broadcast_to(jnp.asarray(e, jnp.float32), (P, dim))
            _, jv = jax.jvp(jax.vmap(trans[li]), (zl,), (ev,))
            out[c] = np.asarray(jv, np.float64) @ Wvecs[c]
        return out

    gain = np.stack([gains_for(li, E, W) for li in range(L)])        # [L, n_pc, P]
    gain_r = np.stack([gains_for(li, E_r, W_r) for li in range(L)])  # [L, n_rand, P]
    print(f"[{time.time()-t0:6.1f}s] gains computed\n", flush=True)

    # ---- full Jacobian: spectrum and placement ----------------------------
    sv = np.zeros((L, P, dim))
    cap_in = np.zeros((L, n_pc, P, dim))
    cap_out = np.zeros((L, n_pc, P, dim))
    # Pair-averaged second moments. These are what makes the subspace
    # comparable ACROSS assays: the per-pair singular vectors of J are not in
    # correspondence between two proteins (pair 7 of RCRO is not pair 7 of
    # RS15), but E_pairs[J J^T] and E_pairs[J^T J] are 128x128 operators on the
    # model's own channels and mean the same thing in every protein. Their
    # leading eigenvectors are the transition's dominant write and read
    # subspaces at that layer, pooled over operating points.
    mom_out = np.zeros((L, dim, dim))
    mom_in = np.zeros((L, dim, dim))
    for li in range(L):
        zl = jnp.asarray(z_pre[li], jnp.float32)
        J = np.asarray(jax.vmap(jax.jacfwd(trans[li]))(zl), np.float64)  # [P,128,128]
        U, s_, Vt = np.linalg.svd(J)
        sv[li] = s_
        mom_out[li] = np.einsum("pij,pkj->ik", J, J) / P
        mom_in[li] = np.einsum("pji,pjk->ik", J, J) / P
        for c in range(n_pc):
            e = E[c] / (np.linalg.norm(E[c]) + EPS)
            w = W[c] / (np.linalg.norm(W[c]) + EPS)
            ci = np.einsum("pkd,d->pk", Vt, e) ** 2      # input side
            co = np.einsum("pdk,d->pk", U, w) ** 2       # output side
            cap_in[li, c] = np.cumsum(ci, -1) / (ci.sum(-1, keepdims=True) + EPS)
            cap_out[li, c] = np.cumsum(co, -1) / (co.sum(-1, keepdims=True) + EPS)
    print(f"[{time.time()-t0:6.1f}s] Jacobian SVD done\n", flush=True)

    # ---- report -----------------------------------------------------------
    pr = (sv ** 2).sum(-1) ** 2 / (((sv ** 2) ** 2).sum(-1) + EPS)   # [L, P]
    print("Jacobian of transition_z, per layer "
          "(median over pairs unless stated)\n")
    # The null is compared SIGNED, and as a percentile. Comparing |gain| against
    # median|null| would hide the fact that the null is itself systematically
    # negative at depth, and would read a shared contraction as a PC-specific
    # effect. `pct` is the fraction of the matched random directions whose gain
    # is at least as negative as the PC's -- small means the PC is attenuated
    # unusually strongly, ~0.5 means it is being treated like any direction.
    med_r = np.median(gain_r, axis=(1, 2))                       # [L], signed
    pct = np.zeros((L, n_pc))
    for li in range(L):
        gm = np.median(gain_r[li], axis=1)                       # [n_rand]
        for c in range(n_pc):
            pct[li, c] = float((gm <= np.median(gain[li, c])).mean())

    print(f"  {'layer':>5s} {'top sv':>7s} {'effrank':>8s} "
          + " ".join(f"{'PC'+str(c+1)+' (pct)':>15s}" for c in range(n_pc))
          + f" {'null med':>10s}")
    for li in range(L):
        row = " ".join(f"{np.median(gain[li, c]):9.4f} ({pct[li, c]:.2f})"
                       for c in range(n_pc))
        print(f"  {li:5d} {np.median(sv[li].max(-1)):7.3f} {np.median(pr[li]):8.1f} "
              f"{row} {med_r[li]:10.4f}")

    print("\n  gain is the ADDED PC coordinate per unit of that coordinate;")
    print("  the layer's total multiplier on it is 1 + gain. `pct` is the")
    print("  fraction of matched random directions attenuated at least as")
    print("  strongly -- ~0.5 means the PC is treated like any other direction.\n")

    ks = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    ks = ks[ks <= dim]
    print("placement of the PC directions in the Jacobian's own singular bases")
    print(f"  (layer {L-1}, median over pairs; random baseline is k/{dim})\n")
    print(f"  {'side':>6s} {'dir':>5s}  " + "  ".join(f"k={k:<5d}" for k in ks))
    for side, arr in (("input", cap_in), ("output", cap_out)):
        for c in range(n_pc):
            v = np.median(arr[L - 1, c], 0)[ks - 1]
            print(f"  {side:>6s} {'PC'+str(c+1):>5s}  "
                  + "  ".join(f"{x:6.3f} " for x in v))
    print(f"  {'':>6s} {'rand':>5s}  " + "  ".join(f"{k/dim:6.3f} " for k in ks))

    np.savez_compressed(
        a.out, assay=a.assay, sv=sv, gain=gain, gain_rand=gain_r,
        cap_in=cap_in, cap_out=cap_out, ks=ks, sd=sd, V=V,
        mom_in=mom_in, mom_out=mom_out,
        z_pre=z_pre.astype(np.float32), ii=ii_np, jj=jj_np)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {a.out}")


if __name__ == "__main__":
    main()
