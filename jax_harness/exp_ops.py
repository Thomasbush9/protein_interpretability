"""The same Jacobian treatment for all five z-path operations, not just the MLP.

`exp_jac` established that `transition_z`'s Jacobian is low rank (~24 of 128),
protein-general, and largely orthogonal to the stability axis -- so the pair
transition is not where PC2 is shaped. That leaves the obvious question of which
operation IS, and the four the transition sits beside are the candidates.
`PairformerLayer2` touches z five times and only five times:

    z = z + tri_mul_out(z, pair_mask)
    z = z + tri_mul_in(z, pair_mask)
    z = z + tri_att_start(z, pair_mask)
    z = z + tri_att_end(z, pair_mask)
    z = z + transition_z(z)

WHY THE JACOBIAN HAS TO BE REDEFINED. `transition_z` is pointwise in the pair
index, so "the Jacobian" is unambiguously a 128x128 matrix per pair and
`exp_jac` could take it with `jacfwd`. The four triangle operations are not:
`tri_mul_out` at pair (i,j) reads every (i,k) and (j,k), and the attentions
read whole rows or columns. Their true Jacobian is an operator on the entire
[N,N,128] tensor, which is far too large to decompose and is not comparable
across proteins of different length anyway.

What IS comparable, and is what the rest of the project actually asks about, is
the CHANNEL-SPACE operator: perturb z the way a point mutation does, and read
the response in the model's own 128 pair channels.

  perturb   A substitution at residue r changes every pair involving r, so the
            tangent puts the direction into row r AND column r. Perturbing only
            the row would measure a perturbation the model can never receive.
  read      Mean over partners of the response in row r -- which is exactly how
            `exp_gym2` defines the archived `z_site`
            (`z[0].mean(axis=1)[row]`), so the operator acts on the same
            quantity every downstream number in this project is built from.

That gives M[c', c] = response in channel c' to a unit perturbation in channel
c, a 128x128 matrix per (layer, operation), averaged over sampled residues r.
It is obtained by pushing the 128 basis tangents through `jax.linearize` of the
real module, so the primal is computed once and the operation's own nonlinearity
(softmax, sigmoid gates, LayerNorm) is differentiated exactly rather than
approximated.

For `transition_z` this construction REDUCES to the row-average of the pointwise
Jacobian `exp_jac` already computed, so the two runs must agree on that column.
That is checked in `analyze_ops.py` and is the reason the transition is
recomputed here rather than copied across.

BASE POINTS. Each operation must be linearised at its own input, and those are
the four intermediate z values inside the layer, which nothing stores. They are
recovered by replicating the five-line z-path above -- the only place in this
harness where model code is duplicated. Under `deterministic=True`
`get_dropout_mask` returns all ones (dropout=0 makes `bernoulli(key, 1.0)`
all-True and the scale 1/(1-0)), so the replica is exact, and `--check` verifies
it reproduces the real layer's z output before any Jacobian is taken. If that
assertion ever fires, every number below describes a computation the model never
ran.

  sbatch analysis.sbatch exp_ops.py --assay <name> ... --out ../runs/ops_<name>.npz
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
OPS = ["tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end", "transition_z"]


def z_path(layer, z, pair_mask):
    """The layer's z updates, in order, returning each operation's INPUT.

    Mirrors `PairformerLayer2.__call__` under deterministic=True, where every
    dropout mask is all ones. Returns (base points, final z); the final z is
    what `--check` compares against the real layer.
    """
    bases = []
    bases.append(z)
    z = z + layer.tri_mul_out(z, pair_mask)
    bases.append(z)
    z = z + layer.tri_mul_in(z, pair_mask)
    bases.append(z)
    z = z + layer.tri_att_start(z, pair_mask)
    bases.append(z)
    z = z + layer.tri_att_end(z, pair_mask)
    bases.append(z)
    z = z + layer.transition_z(z)
    return bases, z


def op_fn(layer, name, pair_mask):
    """The residual BRANCH alone, as a function of z -- not z + branch.

    Matches `exp_jac`'s convention so a gain is always the ADDED coordinate and
    the operation's total multiplier is 1 + gain.
    """
    mod = getattr(layer, name)
    if name == "transition_z":
        return lambda z: mod(z)
    return lambda z: mod(z, pair_mask)


def channel_operator(f, z0, rows, dim, chunk):
    """M[c', c]: response in channel c' at the site row to a unit perturbation
    in channel c across row r and column r, averaged over the sampled rows.

    `jax.linearize` computes the primal once and returns the exact tangent map,
    so the 128 basis directions cost 128 linear pushes rather than 128 forward
    passes.
    """
    _, jvp = jax.linearize(f, z0)
    eye = jnp.eye(dim, dtype=z0.dtype)

    def one_row(r):
        def push(basis):                      # basis: [chunk, dim]
            def tangent(e):
                t = jnp.zeros_like(z0)
                t = t.at[0, r, :, :].add(e)
                t = t.at[0, :, r, :].add(e)
                return t
            out = jax.vmap(lambda e: jvp(tangent(e)))(basis)   # [chunk,1,N,N,dim]
            return out[:, 0, r, :, :].mean(axis=1)             # [chunk, dim]
        parts = [push(eye[i:i + chunk]) for i in range(0, dim, chunk)]
        return jnp.concatenate(parts, 0)                       # [dim(in), dim(out)]

    return jnp.stack([one_row(int(r)) for r in rows]).mean(0).T  # [out, in]


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
    ap.add_argument("--n-rows", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()

    print(f"jax devices: {jax.devices()}\n", flush=True)
    t0 = time.time()

    rows_csv = [r for r in csv.DictReader(open(Path(a.assay_dir) / f"{a.assay}.csv"))
                if ":" not in r["mutant"]]
    wt = list(rows_csv[0]["mutated_sequence"])
    m0 = re.match(r"([A-Z])(\d+)([A-Z])", rows_csv[0]["mutant"])
    wt[int(m0.group(2)) - 1] = m0.group(1)
    wt = "".join(wt)

    work = Path(a.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)

    model = pi.load_model(subsample_msa=False)
    print(f"[{time.time()-t0:6.1f}s] model loaded; {a.assay} len={len(wt)}", flush=True)

    a3m = work / "msa" / "wt_ops.a3m"
    graft_a3m(a3m, Path(a.a3m), wt, wt, cap=a.msa_cap)
    y = work / "yamls" / "wt_ops.yaml"
    y.write_text(YAML_TMPL.format(seq=wt, msa=a3m.resolve()))
    feats, _ = pi.load_features(y.read_text())

    # ---- trunk to the final recycle, exactly as exp_jac / exp_gym2 do ------
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
    L = pf.stacked_parameters.transition_z.fc1.weight.shape[0]
    dim = pf.stacked_parameters.transition_z.fc1.weight.shape[-1]
    mask_np = np.asarray(feats["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask_np)[0]
    rng = np.random.default_rng(a.seed)
    rows = rng.choice(valid, min(a.n_rows, len(valid)), replace=False)
    print(f"[{time.time()-t0:6.1f}s] {L} layers, dim {dim}, "
          f"{len(valid)} valid tokens, rows {sorted(rows.tolist())}\n", flush=True)

    M = np.zeros((L, len(OPS), dim, dim), np.float32)
    worst = 0.0
    kk = lay_key
    for li in range(L):
        layer = eqx.combine(pf.static,
                            jax.tree.map(lambda x, i=li: x[i], pf.stacked_parameters))
        bases, z_rep = z_path(layer, z, pair_mask)
        s, z_real, kk_next = layer(s, z, mask, pair_mask, key=kk,
                                   deterministic=True)
        d = float(jnp.abs(z_rep - z_real).max() / (jnp.abs(z_real).max() + EPS))
        worst = max(worst, d)
        if d > a.tol:
            raise SystemExit(
                f"layer {li}: the replicated z-path does not reproduce the "
                f"layer output (rel err {d:.2e}). The base points below would "
                f"not be the operations' real inputs.")
        for oi, name in enumerate(OPS):
            M[li, oi] = np.asarray(
                channel_operator(op_fn(layer, name, pair_mask), bases[oi],
                                 rows, dim, a.chunk), np.float32)
        z, kk = z_real, kk_next
        if li % 16 == 0 or li == L - 1:
            print(f"[{time.time()-t0:6.1f}s]   layer {li:2d} done "
                  f"(z-path rel err {d:.2e})", flush=True)

    print(f"\nreplicated z-path vs the real layer: max rel err {worst:.2e}")
    print("  -> the base points are the operations' real inputs\n")

    g = np.load(a.gym, allow_pickle=True)
    sd = np.asarray(g["dz_site"])[:, -1, :].std(0)
    Pz = np.load(a.pc, allow_pickle=True)
    V = np.asarray(Pz["V"]) * np.asarray(Pz["orient"])[:, None]

    np.savez_compressed(a.out, assay=a.assay, ops=np.array(OPS), M=M,
                        sd=sd, V=V, rows=rows)
    print(f"[{time.time()-t0:6.1f}s] wrote {a.out}")


if __name__ == "__main__":
    main()
