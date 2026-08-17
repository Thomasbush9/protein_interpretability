"""Can the Boltz-2 loader be unified onto pi_models without moving the science?

`exp_adapter_equiv` answered this for the wild-type OUTPUT: at a matched MSA
regime the two loaders agree on expected distance to 0.030 A max, 0.0019 A mean.
That is reassuring and it is not the question. Every result in the project is a
DIFFERENCE between a variant and its wild type, taken from the trunk rather than
the output, and a loader difference that cancels in the absolute distogram need
not cancel in `dz_site` -- nor the reverse.

So this measures the thing the shared basis is built on, and it measures it
against a scale rather than against zero. The two loaders are not bit-identical
and will not produce corr 1.000000; the useful question is whether their
disagreement is small COMPARED WITH variation the pipeline already tolerates.
Two references for that, both measured here rather than quoted:

    key         same loader, same alignment, different PRNG key. Under the full
                MSA this is exactly 1.000000 -- the trunk is deterministic, so
                it sets the floor.
    subsample   same loader, the 1024-row draw. corr ~0.998, and this project
                has already decided that regime is acceptable for everyday runs.

If backend disagreement sits at or below the subsample band, swapping the loader
costs less than a choice already made. If it sits above, it does not.

    sbatch --time=240 checkout.sbatch exp_backend_equiv.py --assay ... --out ...
"""

from __future__ import annotations

import argparse
import csv
import json
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
    design="one assay, one alignment; arms vary the LOADER, the PRNG key and "
           "the MSA regime so the loader effect is read against both",
    reduction="dz_site = z_site(variant) - z_site(WT) per layer at the mutated "
              "position -- the quantity the shared basis is built on",
    references="key-only (deterministic floor) and subsample (an accepted band)",
    endpoint="within-assay Spearman of a dz-magnitude predictor vs assay score",
)


def spear(a, b):
    return float(pi_stats.spearman(np.asarray(a, float), np.asarray(b, float)))


def corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(np.corrcoef(a, b)[0, 1])


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
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import jax

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
        f, hh = featurise(r["mutated_sequence"], f"v{row}_{m.group(3)}")
        holds.append(hh)
        variants.append({"row": row, "score": float(r["DMS_score"]), "feats": f})
    print(f"{a.assay}: len={len(wt)} variants={len(variants)}", flush=True)

    #    label              backend   subsample  key
    arms = [
        ("joltz_full_k0",   "joltz",   False,    0),   # the reference: today's path
        ("joltz_full_k1",   "joltz",   False,    1),   # key-only floor
        ("joltz_sub_k0",    "joltz",   True,     0),   # an accepted band
        ("mosaic_full_k0",  "mosaic",  False,    0),   # the proposed path
    ]

    per_arm = {}
    for label, backend, sub, keyseed in arms:
        t0 = time.time()
        model = pi.load_model(subsample_msa=sub, backend=backend)
        key = jax.random.key(keyseed)
        _, _, ref = trunk_capture(model, f_wt, ii, jj, None,
                                  recycles=a.recycles, key=key)
        ref_z = np.asarray(ref["z_full"])
        dz = []
        for v in variants:
            _, _, cur = trunk_capture(model, v["feats"], ii, jj, v["row"],
                                      recycles=a.recycles, key=key)
            dz.append((np.asarray(cur["z_site"]) - ref_z[:, v["row"]])
                      .astype(np.float32))
        per_arm[label] = np.stack(dz)
        print(f"  {label:16s} {time.time()-t0:6.1f}s", flush=True)
        del model

    scores = np.array([v["score"] for v in variants], float)
    ref_arm = per_arm["joltz_full_k0"]

    def against_ref(label):
        return {"corr": corr(ref_arm, per_arm[label]),
                "per_variant_norm_spearman": spear(
                    np.linalg.norm(ref_arm, axis=(1, 2)),
                    np.linalg.norm(per_arm[label], axis=(1, 2)))}

    result = {
        "assay": a.assay, "n_variants": len(variants), "recycles": a.recycles,
        "msa_cap": a.msa_cap,
        "dz_vs_reference": {l: against_ref(l) for l in per_arm if l != "joltz_full_k0"},
        "endpoint_spearman": {l: spear(np.linalg.norm(v, axis=(1, 2)), scores)
                              for l, v in per_arm.items()},
    }

    backend_corr = result["dz_vs_reference"]["mosaic_full_k0"]["corr"]
    sub_corr = result["dz_vs_reference"]["joltz_sub_k0"]["corr"]
    key_corr = result["dz_vs_reference"]["joltz_full_k1"]["corr"]
    result["verdict"] = {
        "backend_corr": backend_corr,
        "subsample_band": sub_corr,
        "key_floor": key_corr,
        "backend_within_accepted_band": bool(backend_corr >= sub_corr),
        "reading": ("the loader swap costs no more than the subsample regime "
                    "this project already accepts" if backend_corr >= sub_corr
                    else "the loader swap costs MORE than the subsample regime; "
                         "it is not free and should not ride along with it"),
    }

    pi_archive.write_result(Path(a.out), result, protocol=PROTOCOL)
    print(json.dumps(result, indent=1))
    del holds


if __name__ == "__main__":
    main()
