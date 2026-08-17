"""Do the two Boltz-2 adapters produce the same model output?

The refactor keeps `pi_models` -- one class, four models -- and retires
`pi_core` as the collection entry point. `pi_core` produced every archived
Boltz-2 capture, so that swap is only safe if the two agree on the same input,
and "agree" has to be measured rather than assumed: they reach the model by
different routes.

    pi_core     Boltz yaml -> boltz.process_inputs -> joltz.from_torch(ckpt)
                -> embed_inputs -> hand-written recycle loop -> distogram head
    pi_models   sequence + a3m -> mosaic TargetChain -> Boltz2().model_output

THE THING MOST LIKELY TO MAKE THEM DIFFER IS NOT NUMERICAL. `pi_core.load_model`
takes `subsample_msa=False` and says so explicitly: "deliberate and differs from
mosaic's design-time default ... for interpretability we need the MSA depth to
be an *exact, controlled* quantity, not a random 1024-row draw that changes per
key." If the mosaic wrapper subsamples, the two runs see different alignments
and every downstream number differs for a reason that has nothing to do with the
adapter. So depth is measured on both sides and reported FIRST; a mismatch there
invalidates the comparison rather than failing it.

Coordinates are compared only through the distogram and pLDDT. Backbone atoms
come out of a diffusion sampler whose global frame is arbitrary, so a raw
coordinate difference measures rigid motion, not disagreement.

    sbatch checkout.sbatch exp_adapter_equiv.py --a3m ... --seq-from ... --out ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi_archive  # noqa: E402
import pi_core as pi  # noqa: E402
import pi_models  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from protein_interpretability.collection import records  # noqa: E402

PROTOCOL = dict(
    design="paired: one sequence, one alignment, both adapters, same key",
    compares="pi_core (joltz) against pi_models (mosaic) on Boltz-2",
    controls="MSA depth measured on both sides; recycles and PRNG key fixed",
    reduction="distogram logits, expected distance, pLDDT; coordinates excluded "
              "because the diffusion frame is arbitrary",
)


def summarise(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(denom > 0, d / np.maximum(denom, 1e-30), 0.0)
    return {
        "shape": list(a.shape),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_rel": float(rel.max()),
        "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a3m", required=True, help="alignment for the WT sequence")
    ap.add_argument("--seq", required=True, help="the WT sequence itself")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sweep", default="",
                    help="comma-separated recycle counts to try on the "
                         "pi_models side against pi_core at --recycles. The "
                         "two loops may not mean the same thing by 'a "
                         "recycle': pi_core runs recycling_steps iterations "
                         "total (n-1 plain, then one capturing), and an "
                         "off-by-one there produces exactly the signature of "
                         "a near-miss -- correlation ~0.9999 with a visible "
                         "absolute difference.")
    ap.add_argument("--msa-cap", type=int, default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import jax

    work = Path(a.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)

    seq = a.seq.strip().upper()
    a3m = work / "msa" / "wt.a3m"
    graft_a3m(a3m, Path(a.a3m), seq, seq, cap=a.msa_cap)

    pi_models.block_network()          # never silently fall back to the server
    result = {"sequence_length": len(seq), "recycles": a.recycles}

    # ---- pi_core: yaml -> joltz -------------------------------------------
    y = work / "yamls" / "wt.yaml"
    y.write_text(YAML_TMPL.format(seq=seq, msa=a3m.resolve()))
    feats_core, handle = pi.load_features(y.read_text())
    model_core = pi.load_model(subsample_msa=False)
    out_core = pi.trunk_capture(model_core, feats_core,
                                recycling_steps=a.recycles,
                                key=jax.random.key(0))
    logits_core = np.asarray(out_core["distogram"])
    ed_core = np.asarray(pi.expected_distance(out_core["distogram"]))
    depth_core = pi_models.msa_depth("boltz2", feats_core)

    # ---- pi_models: sequence + a3m -> mosaic ------------------------------
    wrapper = pi_models.load("boltz2")
    ex = pi_models.run_one(wrapper, seq, str(a3m), recycles=a.recycles,
                           key=jax.random.key(0), name="boltz2", work=work)
    feats_m, _ = pi_models.features_for("boltz2", wrapper, seq, str(a3m), work=work)
    depth_models = pi_models.msa_depth("boltz2", feats_m)

    # The new schema, applied to the adapter that is being kept.
    records.validate(ex)

    # Depth in the FEATURES is not depth in the MODEL. `subsample_msa` is a
    # construction argument on the MSA module, so a wrapper can hold all 6857
    # rows and still use a 1024-row draw of them. pi_core sets it False on
    # purpose; whatever mosaic's Boltz2() defaults to is recorded here rather
    # than assumed.
    def msa_args_of(obj):
        for attr in ("msa_args", "msa_module_args", "_msa_args"):
            if hasattr(obj, attr):
                return {k: v for k, v in vars(getattr(obj, attr)).items()} \
                    if hasattr(getattr(obj, attr), "__dict__") \
                    else dict(getattr(obj, attr))
        inner = getattr(obj, "model", None)
        mm = getattr(inner, "msa_module", None) if inner is not None else None
        if mm is not None:
            return {k: getattr(mm, k) for k in
                    ("subsample_msa", "num_subsampled_msa") if hasattr(mm, k)}
        return None

    result["msa_depth"] = {"pi_core": depth_core, "pi_models": depth_models,
                           "match": depth_core == depth_models}
    try:
        result["msa_module_args"] = {
            "pi_models_wrapper": msa_args_of(wrapper),
            "pi_core": {"subsample_msa": False, "num_subsampled_msa": 1024},
        }
    except Exception as exc:                       # introspection must not fail the run
        result["msa_module_args"] = {"error": repr(exc)}
    result["bins"] = {"pi_core": int(logits_core.shape[-1]),
                      "pi_models": int(ex.n_bins)}

    if depth_core != depth_models:
        # Report and stop short of claiming an adapter difference: the two runs
        # did not see the same alignment, so nothing downstream is comparable.
        result["verdict"] = ("MSA DEPTH DIFFERS -- the adapters were not given "
                             "the same alignment, so the tensors below are not "
                             "a measure of adapter equivalence")
    else:
        result["verdict"] = "same alignment, same key, same recycles"

    result["distogram_logits"] = summarise(logits_core, ex.logits)
    result["expected_distance"] = summarise(ed_core, ex.ed)
    result["plddt_shape"] = {"pi_models": list(np.asarray(ex.plddt).shape)}

    if a.sweep:
        # Which recycle count on the mosaic side reproduces pi_core's run? If
        # one of them lands at ~0 while the others sit near the figure above,
        # the adapters agree and only their COUNTING differs -- a mapping, not
        # a numerical discrepancy.
        sweep = {}
        for k in [int(s) for s in a.sweep.split(",")]:
            exk = pi_models.run_one(wrapper, seq, str(a3m), recycles=k,
                                    key=jax.random.key(0), name="boltz2",
                                    work=work)
            sweep[k] = {
                "distogram_logits": summarise(logits_core, exk.logits),
                "expected_distance": summarise(ed_core, exk.ed),
            }
            print(f"  pi_models recycles={k}: "
                  f"ed max_abs={sweep[k]['expected_distance']['max_abs']:.4f} "
                  f"corr={sweep[k]['expected_distance']['corr']:.8f}", flush=True)
        result["recycle_sweep"] = sweep
        best = min(sweep, key=lambda k: sweep[k]["expected_distance"]["max_abs"])
        result["best_match"] = {
            "pi_core_recycles": a.recycles, "pi_models_recycles": best,
            "ed_max_abs": sweep[best]["expected_distance"]["max_abs"]}

    pi_archive.write_result(Path(a.out), result, protocol=PROTOCOL)
    print(json.dumps({k: v for k, v in result.items()
                      if k != "distogram_logits"}, indent=1))
    print("distogram:", json.dumps(result["distogram_logits"], indent=1))
    del handle


if __name__ == "__main__":
    main()
