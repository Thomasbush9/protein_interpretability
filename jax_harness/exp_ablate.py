"""Experiment 7 -- causal test: is `transition_z` responsible for the L37-45 dip?

exp_sublayers (in KL) attributes the transient reduction of the mutation's
distributional footprint at layers 37-45 to `transition_z`, the per-pair
feed-forward. That is an attribution, not a cause: it says where the divergence
went down, not that removing the op would change anything.

Here we remove it. `Pairformer2` keeps its 64 layers as a *stacked* parameter
pytree, so zeroing `transition_z`'s output weight on a chosen slice of layers
is one `eqx.tree_at` on `stacked_parameters` -- no hooks, no re-tracing of the
model, and the untouched layers are bit-identical.

Zeroing the final projection of the Transition block makes its residual
contribution exactly zero, i.e. `z += 0`, which is a clean deletion of that
write rather than a perturbation of it.

WHAT THE FIRST VERSION COULD NOT SHOW, and what this one adds.

The original run compared the L37-45 band against ONE control band (L10-18) on
ONE protein, using a band that had been chosen by looking at the same GFP
attribution curve the ablation then tested. Three things follow, and the August
2026 audit asked for all three:

  *sliding sweep* -- `--sliding` ablates EVERY width-matched band (56 of them at
   width 9), so the frozen band's effect is reported as a percentile against the
   full null distribution of same-width interventions rather than against a
   single hand-picked comparator. One control band cannot distinguish "this band
   is special" from "deleting nine MLPs anywhere does this".

  *graded scaling* -- `--scales` multiplies transition_z's output by alpha
   instead of deleting it. Full deletion is far out of distribution; a
   dose-response through alpha = 1 -> 0 is the evidence that the effect tracks
   the op's magnitude rather than the shock of removing it.

  *WT-quality diagnostics* -- every condition now reports what the ablation did
   to the wild-type prediction itself: distogram entropy, pLDDT, TM to the
   unablated structure, and the global scale of z. An intervention that raises
   mutant-WT divergence by degrading the model into noise is not evidence about
   mutation representation, and without these numbers the two are
   indistinguishable.

The band is FROZEN at whatever `--band` says. Discover it on GFP, then run this
unchanged on held-out proteins; do not re-tune the band per protein.
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
from geom import kabsch_rmsd  # noqa: E402  (numpy only; tmtools is login-node)
from joltz import TrunkState  # noqa: E402


def scale_transition_z(model, layers, alpha=0.0):
    """Multiply transition_z's residual contribution by `alpha` on `layers`.

    joltz's Transition is `fc3(silu(fc1(v)) * fc2(v))`, so **fc3** is the output
    projection -- fc1/fc2 are the gated inner branches. Scaling fc3's weight
    (and bias, if it has one) scales the block's output exactly, because fc3 is
    linear; alpha = 0 deletes the `z += transition_z(z)` write entirely. Layers
    not in `layers` are bit-identical because only the selected slice of the
    stacked parameter array is touched.
    """
    if not len(layers):
        return model
    idx = jnp.asarray(sorted(layers))

    def out_w(m):
        return m.pairformer_module.stacked_parameters.transition_z.fc3.weight

    model = eqx.tree_at(out_w, model, out_w(model).at[idx].multiply(alpha))

    def out_b(m):
        return m.pairformer_module.stacked_parameters.transition_z.fc3.bias

    if out_b(model) is not None:
        model = eqx.tree_at(out_b, model, out_b(model).at[idx].multiply(alpha))
    return model


def ablate_transition_z(model, layers):
    """Full deletion -- the alpha = 0 case, kept as a name the report cites."""
    return scale_transition_z(model, layers, alpha=0.0)


def run_trunk_capture(model, feats, ii, jj, *, recycles, key):
    """Per-layer distogram logits at sampled pairs, plus the final trunk state.

    The trunk state comes back so the caller can run the structure module on the
    SAME forward pass and report what the ablation did to the prediction, rather
    than inferring quality from the divergence it is trying to explain.
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
    z = z + model.msa_module(
        z, emb.s_inputs, feats, deterministic=True, key=jax.random.fold_in(k, 0)
    )
    s, z, per_layer = pi.pairformer_capture(
        model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=True,
        reduce_fn=lambda s_, z_: model.distogram_module(z_)[0, :, :, 0, :][ii, jj],
    )
    return per_layer, emb, TrunkState(s=s, z=z)


def skl(la, lb):
    def sm(x):
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)
    pa, pb = sm(la), sm(lb)
    return ((pa - pb) * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum(-1)


def trunk_quality(model, trunk, mask):
    """Cheap, trunk-only health of a prediction: entropy and representation scale."""
    logits = np.asarray(model.distogram_module(trunk.z)[0, :, :, 0, :])
    x = logits - logits.max(-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(-1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(-1)
    return {
        "disto_entropy": float(ent[np.ix_(mask, mask)].mean()),
        "z_norm": float(np.linalg.norm(np.asarray(trunk.z)[0]) /
                        np.sqrt(np.asarray(trunk.z)[0].size)),
        "s_norm": float(np.linalg.norm(np.asarray(trunk.s)[0]) /
                        np.sqrt(np.asarray(trunk.s)[0].size)),
    }


def structure_quality(model, feats, emb, trunk, mask, *, sampling_steps, key):
    """pLDDT and CA coordinates -- what the ablation did to the actual output."""
    from mosaic.losses.boltz2 import boltz2_forward_from_trunk
    out = boltz2_forward_from_trunk(
        model, feats, emb, trunk, num_sampling_steps=sampling_steps,
        deterministic=True, key=key)
    plddt = np.asarray(out.plddt)[mask]
    ca = np.asarray(out.backbone_coordinates)[mask][:, 1].astype(np.float64)
    return float(plddt.mean()), ca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wt", default="gfp_wt")
    ap.add_argument("--mutants", default="gfp_core_32",
                    help="comma-separated; each is compared against --wt")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-pairs", type=int, default=5000)
    ap.add_argument("--band", default="37,45",
                    help="FROZEN band under test, inclusive. Do not re-tune per protein.")
    ap.add_argument("--control-band", default="10,18")
    ap.add_argument("--sliding", action="store_true",
                    help="ablate every width-matched band, for a percentile null")
    ap.add_argument("--sliding-stride", type=int, default=1)
    ap.add_argument("--scales", default="0,0.25,0.5,0.75",
                    help="graded transition_z scale factors applied to the frozen band")
    ap.add_argument("--sampling-steps", type=int, default=50)
    ap.add_argument("--no-structure", action="store_true",
                    help="skip pLDDT/TM diagnostics (trunk-only, much faster)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.band.split(","))
    clo, chi = (int(x) for x in args.control_band.split(","))
    width = hi - lo + 1
    scales = [float(s) for s in args.scales.split(",") if s.strip() != ""]
    mutant_ids = [m for m in args.mutants.split(",") if m.strip()]
    data = Path(args.data)

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    f_wt, h = pi.load_features((data / "yamls" / f"{args.wt}.yaml").read_text())
    feats_m, handles = {}, [h]
    for mid in mutant_ids:
        fm, hm = pi.load_features((data / "yamls" / f"{mid}.yaml").read_text())
        feats_m[mid] = fm
        handles.append(hm)

    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    valid = np.where(mask)[0]
    rng = np.random.default_rng(0)
    a, b = rng.choice(valid, args.n_pairs), rng.choice(valid, args.n_pairs)
    keep = a != b
    ii, jj = jnp.asarray(a[keep]), jnp.asarray(b[keep])
    n_layers = int(model.pairformer_module.stacked_parameters
                   .transition_z.fc3.weight.shape[0])

    # ---- conditions ------------------------------------------------------
    conditions = {"intact": ([], 1.0)}
    conditions[f"ablate_band_{lo}_{hi}"] = (list(range(lo, hi + 1)), 0.0)
    conditions[f"ablate_control_{clo}_{chi}"] = (list(range(clo, chi + 1)), 0.0)
    conditions["ablate_all"] = (list(range(n_layers)), 0.0)
    for al in scales:
        if al == 0.0:
            continue                       # already covered by the deletion
        conditions[f"scale_band_{lo}_{hi}_a{al:g}"] = (list(range(lo, hi + 1)), al)
    sliding_names = []
    if args.sliding:
        for start in range(0, n_layers - width + 1, args.sliding_stride):
            nm = f"slide_{start}_{start + width - 1}"
            if nm in conditions:
                continue
            conditions[nm] = (list(range(start, start + width)), 0.0)
            sliding_names.append(nm)

    print(f"  {len(conditions)} conditions, width {width}, "
          f"{n_layers} layers, mutants {mutant_ids}", flush=True)

    out = {"band": [lo, hi], "control_band": [clo, chi], "width": width,
           "n_layers": n_layers, "wt": args.wt, "mutants": mutant_ids,
           "recycles": args.recycles, "n_pairs": int(len(ii)),
           "scales": scales, "sliding": bool(args.sliding),
           "sliding_names": sliding_names,
           "sampling_steps": None if args.no_structure else args.sampling_steps,
           "conditions": {}}

    ca_ref = None
    for n, (name, (layers, alpha)) in enumerate(conditions.items()):
        m = model if not layers else scale_transition_z(model, layers, alpha)
        lw, emb_w, tr_w = run_trunk_capture(m, f_wt, ii, jj,
                                            recycles=args.recycles, key=key)
        lw = np.asarray(lw)
        rec = {"layers": [int(x) for x in layers], "alpha": float(alpha),
               "n_layers_ablated": len(layers), "wt_quality": {}}
        rec["wt_quality"].update(trunk_quality(m, tr_w, mask))

        if not args.no_structure:
            pl, ca = structure_quality(m, f_wt, emb_w, tr_w, mask,
                                       sampling_steps=args.sampling_steps,
                                       key=jax.random.fold_in(key, 7))
            if ca_ref is None:
                ca_ref = ca                 # the intact condition runs first
            rec["wt_quality"]["plddt"] = pl
            # RMSD, not TM: tmtools is absent from the mosaic container, and the
            # two structures are the same chain in residue correspondence, so
            # Kabsch needs no alignment step. The coordinates are archived so a
            # TM-score can be computed on the login node without re-running.
            rec["wt_quality"]["rmsd_to_intact_wt"] = float(kabsch_rmsd(ca, ca_ref))
            rec["ca"] = np.asarray(ca, dtype=np.float32).round(3).tolist()

        for mid in mutant_ids:
            lm, _, _ = run_trunk_capture(m, feats_m[mid], ii, jj,
                                         recycles=args.recycles, key=key)
            kl = skl(np.asarray(lm), lw).mean(axis=1)
            rec[mid] = {
                "kl": [float(v) for v in kl],
                "kl_final": float(kl[-1]),
                # change across the FROZEN band, comparable across conditions
                "kl_change_frozen_band": float(kl[hi] - kl[lo - 1]),
                # change across this condition's OWN band, when it has one
                "kl_change_own_band": (
                    float(kl[layers[-1]] - kl[max(layers[0] - 1, 0)])
                    if layers else float(kl[hi] - kl[lo - 1])),
            }
        out["conditions"][name] = rec
        if name in ("intact", f"ablate_band_{lo}_{hi}",
                    f"ablate_control_{clo}_{chi}", "ablate_all") or \
                name.startswith("scale_"):
            q = rec["wt_quality"]
            print(f"  {name:26s} " + "  ".join(
                f"{mid}: dKL_band {rec[mid]['kl_change_frozen_band']:+.4f} "
                f"final {rec[mid]['kl_final']:.4f}" for mid in mutant_ids) +
                f"   [WT ent {q['disto_entropy']:.3f}"
                + (f" pLDDT {q['plddt']:.3f} RMSD {q['rmsd_to_intact_wt']:.2f}A"
                   if "plddt" in q else "") + "]", flush=True)
        elif (n % 10) == 0:
            print(f"  [{time.time()-t0:6.1f}s] {n+1}/{len(conditions)} conditions",
                  flush=True)

    # ---- the frozen band against the width-matched null -------------------
    if sliding_names:
        base = out["conditions"]["intact"]
        frozen = out["conditions"][f"ablate_band_{lo}_{hi}"]
        print(f"\n  Frozen band L{lo}-{hi} against {len(sliding_names)} "
              f"width-{width} bands:", flush=True)
        summary = {}
        for mid in mutant_ids:
            null_final = np.array([out["conditions"][s][mid]["kl_final"]
                                   for s in sliding_names])
            null_own = np.array([out["conditions"][s][mid]["kl_change_own_band"]
                                 for s in sliding_names])
            f_final = frozen[mid]["kl_final"]
            f_own = frozen[mid]["kl_change_own_band"]
            # how extreme is the frozen band among same-width interventions?
            pct_final = float((null_final <= f_final).mean())
            pct_own = float((null_own <= f_own).mean())
            summary[mid] = {
                "frozen_kl_final": f_final,
                "frozen_kl_change_own_band": f_own,
                "intact_kl_final": base[mid]["kl_final"],
                "null_kl_final_mean": float(null_final.mean()),
                "null_kl_final_sd": float(null_final.std()),
                "null_kl_change_own_mean": float(null_own.mean()),
                "null_kl_change_own_sd": float(null_own.std()),
                "percentile_kl_final": pct_final,
                "percentile_kl_change_own_band": pct_own,
                "n_null_bands": len(sliding_names),
            }
            print(f"    {mid}: final KL {f_final:.4f} vs null "
                  f"{null_final.mean():.4f}+/-{null_final.std():.4f} "
                  f"(percentile {pct_final:.2f});  own-band change {f_own:+.4f} "
                  f"vs null {null_own.mean():+.4f}+/-{null_own.std():.4f} "
                  f"(percentile {pct_own:.2f})", flush=True)
        out["frozen_vs_null"] = summary

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    for hh in handles:
        hh.cleanup()


if __name__ == "__main__":
    main()
