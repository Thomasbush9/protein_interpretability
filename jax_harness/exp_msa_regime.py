"""Is the 1024-row MSA subsample good enough to adopt?

`pi_models` builds Boltz-2 with `subsample_msa=True, num_subsampled_msa=1024`;
`pi_core` sets it False and uses the whole alignment. Keeping the subsample
would let the mosaic adapter be adopted as-is. The question is whether it costs
anything that matters.

WHAT IS VARIED. Only the flag. Both arms run through `pi_core.load_model` and
`exp_gym2.trunk_capture`, so this is not the pi_core-vs-pi_models comparison
again -- the adapter, the featurisation, the recycle loop and the reductions are
identical, and the MSA regime is the only difference. The earlier comparison
confounded the two.

WHY WT STRUCTURE AGREEMENT IS NOT THE TEST. It is the obvious thing to measure
and it is not sufficient. Every result in this project is a DIFFERENCE between a
variant and its wild type, and subsampling is a random draw: if the draw moves
between the two runs of a pair, that movement lands in `dz_site` as though the
mutation had caused it. A regime can reproduce the wild-type structure closely
and still destroy the measurement built on top of it. So three things are
measured, and the third decides:

  1. structure       WT pLDDT and CA geometry, full vs subsampled
  2. agreement       does dz_site under subsampling track dz_site under the
                     full alignment, across variants?
  3. noise floor     two subsample DRAWS against each other. If re-drawing moves
                     dz_site as much as changing the regime does, the subsample
                     is adding variance of the same size as the signal, and no
                     amount of agreement in (1) rescues it.

The endpoint is the one the report depends on: does a dz-magnitude predictor
still rank the assay's stability scores?

    sbatch --time=180 checkout.sbatch exp_msa_regime.py --assay ... --out ...
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi_archive  # noqa: E402
import pi_core as pi  # noqa: E402
import pi_stats  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m  # noqa: E402
from exp_gym2 import trunk_capture  # noqa: E402

PROTOCOL = dict(
    design="one assay, one alignment; the ONLY variable is subsample_msa",
    arms="full (subsample_msa=False) and subsampled (True, 1024 rows) at "
         "several PRNG keys, so the draw-to-draw noise floor is measured "
         "rather than assumed",
    reduction="dz_site = z_site(variant) - z_site(WT), per layer, at the "
              "mutated position -- the quantity the shared basis is built on",
    endpoint="within-assay Spearman of a dz-magnitude predictor against the "
             "assay score, per arm",
    controls="identical features, recycles, reductions and adapter across arms",
)


def spear(a, b):
    return float(pi_stats.spearman(np.asarray(a, float), np.asarray(b, float)))


def flat_corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def ca_distance_matrix(ca):
    """Superposition-free structure descriptor: pairwise CA distances."""
    d = ca[:, None, :] - ca[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-variants", type=int, default=32)
    ap.add_argument("--n-pairs", type=int, default=1500)
    ap.add_argument("--msa-cap", type=int, default=2048,
                    help="matches exp_gym2's default, which is what the "
                         "archived captures were produced with")
    ap.add_argument("--sampling-steps", type=int, default=50)
    ap.add_argument("--sub-keys", default="0,1",
                    help="PRNG keys for the subsampled arm; two different "
                         "draws is what makes the noise floor measurable")
    ap.add_argument("--full-keys", default="0",
                    help="PRNG keys for the FULL arm. Give two to separate the "
                         "diffusion sampler's own noise from the MSA draw: "
                         "coordinates come from a stochastic sampler keyed by "
                         "the same seed, so comparing arms at different seeds "
                         "measures both at once. With subsample_msa=False the "
                         "key cannot change the alignment, so full-vs-full "
                         "across keys is the sampler's contribution alone.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import jax
    from mosaic.losses.boltz2 import boltz2_forward_from_trunk

    rows = [r for r in csv.DictReader(open(Path(a.assay_dir) / f"{a.assay}.csv"))
            if ":" not in r["mutant"]]
    wt = list(rows[0]["mutated_sequence"])
    m0 = re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"])
    wt[int(m0.group(2)) - 1] = m0.group(1)
    wt = "".join(wt)

    rng = np.random.default_rng(a.seed)
    if len(rows) > a.n_variants:
        rows = [rows[i] for i in
                sorted(rng.choice(len(rows), a.n_variants, replace=False))]

    work = Path(a.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)
    src = Path(a.a3m)

    def featurise(seq, tag):
        m = work / "msa" / f"{tag}.a3m"
        graft_a3m(m, src, seq, wt, cap=a.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=m.resolve()))
        return pi.load_features(y.read_text())

    # Features are identical across arms; build them once.
    f_wt, h_wt = featurise(wt, "wt")
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    n_tok = int(mask.sum())
    prng = np.random.default_rng(a.seed)
    ii = prng.integers(0, n_tok, a.n_pairs)
    jj = prng.integers(0, n_tok, a.n_pairs)

    variants, holds = [], [h_wt]
    for r in rows:
        m = re.match(r"([A-Z])(\d+)([A-Z])", r["mutant"])
        if not m:
            continue
        row = int(m.group(2)) - 1
        seq = r["mutated_sequence"]
        f, hh = featurise(seq, f"v{row}_{m.group(3)}")
        holds.append(hh)
        variants.append({"row": row, "score": float(r["DMS_score"]), "feats": f,
                         "mutant": r["mutant"]})
    print(f"{a.assay}: len={len(wt)} variants={len(variants)} "
          f"msa_cap={a.msa_cap}", flush=True)

    full_keys = [int(s) for s in a.full_keys.split(",")]
    arms = [("full" if i == 0 else f"full_key{k}", False, k)
            for i, k in enumerate(full_keys)]
    for k in [int(s) for s in a.sub_keys.split(",")]:
        arms.append((f"sub_key{k}", True, k))

    per_arm = {}
    for label, subsample, keyseed in arms:
        t0 = time.time()
        model = pi.load_model(subsample_msa=subsample)
        key = jax.random.key(keyseed)

        emb_w, tr_w, ref = trunk_capture(model, f_wt, ii, jj, None,
                                         recycles=a.recycles, key=key)
        ref_z = np.asarray(ref["z_full"])            # [L, N, 128]
        out_w = boltz2_forward_from_trunk(
            model, f_wt, emb_w, tr_w, num_sampling_steps=a.sampling_steps,
            deterministic=True, key=jax.random.fold_in(key, 7))
        plddt = np.asarray(out_w.plddt)[mask].astype(np.float32)
        ca = np.asarray(out_w.backbone_coordinates)[mask][:, 1].astype(np.float32)
        logits_w = np.asarray(ref["logits"]).astype(np.float32)

        dz, scores = [], []
        for v in variants:
            _, _, cur = trunk_capture(model, v["feats"], ii, jj, v["row"],
                                      recycles=a.recycles, key=key)
            dz.append((np.asarray(cur["z_site"]) - ref_z[:, v["row"]])
                      .astype(np.float32))
            scores.append(v["score"])
        per_arm[label] = {
            "dz": np.stack(dz),                      # [n_var, L, 128]
            "plddt": plddt, "ca": ca, "logits_wt": logits_w,
            "scores": np.asarray(scores, float),
        }
        print(f"  {label:10s} done in {time.time()-t0:6.1f}s  "
              f"plddt_mean={plddt.mean():.2f}", flush=True)
        del model

    # ---- report -----------------------------------------------------------
    full = per_arm["full"]
    sub_labels = [l for l in per_arm if l.startswith("sub_key")]
    full_extra = [l for l in per_arm if l.startswith("full_key")]
    s0 = per_arm[sub_labels[0]]

    dm_full = ca_distance_matrix(full["ca"])
    result = {
        "assay": a.assay, "length": len(wt), "n_variants": len(variants),
        "msa_cap": a.msa_cap, "recycles": a.recycles,
        "arms": [l for l, _, _ in arms],
        "wt_structure": {},
        "dz_agreement": {},
        "endpoint_spearman": {},
    }

    for label in sub_labels + full_extra:
        arm = per_arm[label]
        result["wt_structure"][f"full_vs_{label}"] = {
            "plddt_mean_full": float(full["plddt"].mean()),
            "plddt_mean_sub": float(arm["plddt"].mean()),
            "plddt_max_abs": float(np.abs(full["plddt"] - arm["plddt"]).max()),
            "plddt_corr": flat_corr(full["plddt"], arm["plddt"]),
            "ca_distmat_max_abs_A": float(
                np.abs(dm_full - ca_distance_matrix(arm["ca"])).max()),
            "ca_distmat_mean_abs_A": float(
                np.abs(dm_full - ca_distance_matrix(arm["ca"])).mean()),
        }
        result["dz_agreement"][f"full_vs_{label}"] = {
            "corr": flat_corr(full["dz"], arm["dz"]),
            "per_variant_norm_spearman": spear(
                np.linalg.norm(full["dz"], axis=(1, 2)),
                np.linalg.norm(arm["dz"], axis=(1, 2))),
        }

    if len(sub_labels) >= 2:
        s1 = per_arm[sub_labels[1]]
        result["noise_floor"] = {
            "pair": sub_labels[:2],
            "corr": flat_corr(s0["dz"], s1["dz"]),
            "per_variant_norm_spearman": spear(
                np.linalg.norm(s0["dz"], axis=(1, 2)),
                np.linalg.norm(s1["dz"], axis=(1, 2))),
            "reading": "if this is not clearly better than dz_agreement above, "
                       "the subsample is contributing as much variation as the "
                       "regime change itself",
        }

    for label, arm in per_arm.items():
        pred = np.linalg.norm(arm["dz"], axis=(1, 2))
        result["endpoint_spearman"][label] = spear(pred, arm["scores"])

    pi_archive.write_result(Path(a.out), result, protocol=PROTOCOL)
    import json
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=1))
    del holds


if __name__ == "__main__":
    main()
