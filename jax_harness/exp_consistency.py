"""Trunk-structure consistency: score the decoded structure against the trunk's OWN belief.

The thesis of this project is that the trunk's distogram moves under mutation
and the decoded structure does not. If that is true then for a destabilising
mutant the two should DISAGREE -- the sampler returns a wild-type-like fold while
the distogram has become a broad or shifted distribution that the fold does not
satisfy. That disagreement is measurable per variant, needs no training, and is a
direct quantification of the paper's central claim rather than a proxy for it.

Three readouts, in increasing cost:

  1. CONSISTENCY  log P_mut(structure) -- for each residue pair, the mutant's own
                  distogram evaluated at the distance the decoder actually
                  produced, averaged over pairs. One sample is enough.
                  If destabilising mutants score worse, this is a training-free
                  stability readout built from two things the model already
                  computes, and it uses the disagreement instead of ignoring it.

  2. RERANKING    sample K structures from the same trunk state, keep the one
                  with the highest consistency. Does best-of-K's TM-to-wild-type
                  track dG better than a random draw's? This asks whether the
                  sampler GENERATES an appropriate structure but fails to SELECT
                  it -- a decode-time fix -- or never generates one at all.

  3. SPREAD       the spread of consistency across the K samples, which separates
                  "the decoder cannot satisfy this distogram" from "it sometimes
                  can but usually does not".

Controls that make the numbers mean something:
  * consistency of the same structure under the WILD TYPE's distogram. If a
    variant's structure scores badly under everything, that is a bad structure,
    not a disagreement. The informative quantity is the DIFFERENCE.
  * the wild type's own consistency, as the reference point for "agreement".
  * distogram entropy per variant, because a broader distribution assigns lower
    log-likelihood to everything and would otherwise masquerade as disagreement.

Saves per-variant arrays; analysis in analyze_consistency.py on a login node.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m  # noqa: E402


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def bin_of(dist, edges):
    """Index of the distogram bin containing each distance; clipped at both ends."""
    return np.clip(np.digitize(dist, edges) - 1, 0, len(edges) - 2)


def consistency(logp, ca, edges, sep_min=3):
    """Mean log P(observed distance) over residue pairs, under a given distogram.

    Pairs closer than `sep_min` in sequence are excluded: they are geometrically
    constrained by the backbone, every model gets them right, and including them
    dilutes the signal with pairs that carry no information about the fold.
    """
    N = len(ca)
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    b = bin_of(d, edges)
    i, j = np.triu_indices(N, k=sep_min)
    return float(np.mean(np.log(logp[i, j, b[i, j]] + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--n-variants", type=int, default=60)
    ap.add_argument("--samples", type=int, default=8,
                    help="structures per variant, from one trunk state")
    ap.add_argument("--sampling-steps", type=int, default=100)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mosaic.losses.boltz2 import boltz2_forward_from_trunk

    work = Path(args.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader((Path(args.assay_dir) / f"{args.assay}.csv").open())
            if ":" not in r["mutant"]]
    wt = None
    for r in rows:
        m = re.match(r"([A-Z])(\d+)([A-Z])", r["mutant"])
        if m:
            s = list(r["mutated_sequence"])
            s[int(m.group(2)) - 1] = m.group(1)
            wt = "".join(s)
            break
    rows = sorted(rows, key=lambda r: float(r["DMS_score"]))
    idx = np.unique(np.linspace(0, len(rows) - 1, args.n_variants).round().astype(int))
    picked = [rows[i] for i in idx]
    src = Path(args.a3m)

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    centres = np.asarray(pi.BIN_CENTRES)
    w = float(centres[1] - centres[0])
    edges = np.concatenate([centres - w / 2, [centres[-1] + w / 2]])
    key = jax.random.key(args.seed)
    print(f"[{time.time()-t0:6.1f}s] {args.assay} n={len(picked)} K={args.samples} "
          f"steps={args.sampling_steps}", flush=True)

    def featurise(seq, tag):
        a3m = work / "msa" / f"{tag}.a3m"
        graft_a3m(a3m, src, seq, wt, cap=args.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=a3m.resolve()))
        return pi.load_features(y.read_text())

    def run(feats):
        """Trunk once, K structures from it, plus the trunk's distogram."""
        emb = model.embed_inputs(feats)
        tr = pi.run_trunk(model, emb, feats, recycling_steps=args.recycles,
                          key=key, deterministic=True, capture_last=False)
        st = tr["trunk_state"]
        mask = np.asarray(feats["token_pad_mask"][0]).astype(bool)
        logits = np.asarray(model.distogram_module(st.z)[0, :, :, 0, :])[np.ix_(mask, mask)]
        p = softmax(logits)
        cas, pls = [], []
        for i in range(args.samples):
            out = boltz2_forward_from_trunk(
                model, feats, emb, st, num_sampling_steps=args.sampling_steps,
                deterministic=True, key=jax.random.fold_in(jax.random.key(4000 + i), i))
            cas.append(np.asarray(out.backbone_coordinates)[mask][:, 1].astype(np.float32))
            pls.append(float(np.asarray(out.plddt)[mask].mean()))
        return p, np.stack(cas), np.array(pls), st

    f_wt, h = featurise(wt, "wt")
    p_wt, ca_wt, pl_wt, _ = run(f_wt)
    ent_wt = float(-(p_wt * np.log(p_wt + 1e-12)).sum(-1).mean())
    c_wt = np.array([consistency(p_wt, ca_wt[k].astype(float), edges)
                     for k in range(args.samples)])
    print(f"[{time.time()-t0:6.1f}s] WT: consistency {c_wt.mean():.4f} "
          f"+/- {c_wt.std():.4f}, entropy {ent_wt:.4f}, pLDDT {pl_wt.mean():.3f}",
          flush=True)
    h.cleanup()

    C_own, C_wtdist, TM_pairs, PL, ENT, meta, CAS = [], [], [], [], [], [], []
    for n, r in enumerate(picked):
        f_m, hm = featurise(r["mutated_sequence"], "mut")
        p_m, ca_m, pl_m, _ = run(f_m)
        # each sample scored under the MUTANT's distogram and under the WT's
        C_own.append([consistency(p_m, ca_m[k].astype(float), edges)
                      for k in range(args.samples)])
        C_wtdist.append([consistency(p_wt, ca_m[k].astype(float), edges)
                         for k in range(args.samples)])
        ENT.append(float(-(p_m * np.log(p_m + 1e-12)).sum(-1).mean()))
        PL.append(pl_m)
        CAS.append(ca_m)
        meta.append((r["mutant"], int(re.match(r"[A-Z](\d+)", r["mutant"]).group(1)) - 1,
                     float(r["DMS_score"])))
        hm.cleanup()
        if (n + 1) % 10 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(picked)}", flush=True)

    np.savez_compressed(
        args.out,
        c_own=np.array(C_own, np.float32),          # [n, K] under its own distogram
        c_wtdist=np.array(C_wtdist, np.float32),    # [n, K] under the WT's
        c_wt=c_wt.astype(np.float32),               # [K]    WT structure, WT distogram
        entropy=np.array(ENT, np.float32), entropy_wt=np.float32(ent_wt),
        plddt=np.array(PL, np.float32), plddt_wt=pl_wt.astype(np.float32),
        ca=np.stack(CAS).astype(np.float32), ca_wt=ca_wt.astype(np.float32),
        mutant=np.array([m[0] for m in meta]),
        pos=np.array([m[1] for m in meta]),
        score=np.array([m[2] for m in meta], np.float32),
        assay=np.array(args.assay), n_samples=np.array(args.samples),
    )
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)
    co = np.array(C_own).mean(1)
    print(f"    consistency under own distogram: {co.mean():.4f} "
          f"(WT {c_wt.mean():.4f})")


if __name__ == "__main__":
    main()
