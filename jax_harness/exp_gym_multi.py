"""The central claim, on any model: does the trunk beat the model's own output?

Boltz-2 result to replicate: a readout of internal state predicts measured dG at
rho = 0.548 on held-out residue positions while TM-to-wild-type reaches 0.214 and
pLDDT-at-the-mutated-site 0.037, on identical variants and splits.

Per variant this records, uniformly across models:

  INTERNAL (from the trunk's distogram, which every model exposes)
    kl_glob    mean symmetric KL vs wild type over all residue pairs
    kl_site    the same restricted to pairs touching the mutated residue
    ent_glob   distogram entropy, and
    ent_site   its value at the mutated residue's row
    ed_site    mean |E[d] - E[d]_wt| at the mutated residue's row

  OUTPUT (what the model actually returns to a user)
    tm_to_wt, rmsd_to_wt, plddt_mean, plddt_site

**Scope, stated honestly.** The Boltz-2 probe used 256 features -- four
divergence quantities at each of 64 Pairformer layers. Per-layer capture needs
model-specific plumbing, so this uses only the FINAL trunk state via the
distogram. It is therefore a weaker internal readout than the Boltz-2 one and
its rho should not be compared to 0.548. What it does support is the paired
comparison the claim actually rests on: internal vs output, same variants, same
splits, within each model.

Alignments are controlled (pi_models.block_network) so all models see the a3m we
supply, grafted per variant exactly as in the Boltz-2 experiments.
"""
from __future__ import annotations
import argparse, csv, re, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import pi_models  # noqa: E402
from exp_gym import graft_a3m  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=pi_models.available())
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-variants", type=int, default=100)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=100)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--single-sequence", action="store_true",
                    help="no MSA -- the only mode mosaic's AF2 wrapper supports")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax
    print(f"MSA server blocked at: {pi_models.block_network()}", flush=True)

    work = Path(args.work); (work / "msa").mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(
        (Path(args.assay_dir) / f"{args.assay}.csv").open()) if ":" not in r["mutant"]]
    wt = None
    for r in rows:
        m = re.match(r"([A-Z])(\d+)([A-Z])", r["mutant"])
        if m:
            s = list(r["mutated_sequence"]); s[int(m.group(2)) - 1] = m.group(1)
            wt = "".join(s); break
    rows = sorted(rows, key=lambda r: float(r["DMS_score"]))
    idx = np.unique(np.linspace(0, len(rows) - 1, args.n_variants).round().astype(int))
    picked = [rows[i] for i in idx]
    src = Path(args.a3m)

    t0 = time.time()
    model = pi_models.load(args.model)
    print(f"[{time.time()-t0:6.1f}s] {args.model} / {args.assay} "
          f"len={len(wt)} n={len(picked)}", flush=True)

    # NOTE: no TM-score in here. tmtools is not installed in the mosaic
    # container, so coordinates are saved and TM is computed on a login node by
    # analyze_gym_multi.py. geom.kabsch_rmsd is pure numpy and is safe to use.
    import geom

    def run(seq, tag):
        if args.single_sequence:
            a3m_path = None
        else:
            a3m = work / "msa" / f"{tag}.a3m"
            graft_a3m(a3m, src, seq, wt, cap=args.msa_cap)
            a3m_path = str(a3m)
        feats, depth = pi_models.features_for(
            args.model, model, seq, a3m_path, work=work / tag)
        o = model.model_output(features=feats, recycling_steps=args.recycles,
                               sampling_steps=args.sampling_steps,
                               key=jax.random.key(0))
        return pi_models.extraction_from(o, name=args.model), depth

    e_wt, depth_wt = run(wt, "wt")
    print(f"[{time.time()-t0:6.1f}s] WT done, msa={depth_wt}, "
          f"pLDDT={e_wt.plddt.mean():.3f}", flush=True)

    rec, cas = [], []
    for n, r in enumerate(picked):
        pos = int(re.match(r"[A-Z](\d+)", r["mutant"]).group(1)) - 1
        e, _ = run(r["mutated_sequence"], "mut")
        kl = pi_models.sym_kl(e.p, e_wt.p)
        d_ed = np.abs(e.ed - e_wt.ed)
        rmsd = geom.kabsch_rmsd(e.ca.astype(float), e_wt.ca.astype(float))
        cas.append(e.ca)
        rec.append(dict(
            mutant=r["mutant"], pos=pos, score=float(r["DMS_score"]),
            kl_glob=float(kl.mean()), kl_site=float(kl[pos].mean()),
            ent_glob=float(e.entropy.mean()), ent_site=float(e.entropy[pos].mean()),
            ed_site=float(d_ed[pos].mean()),
            rmsd_to_wt=rmsd,
            plddt_mean=float(e.plddt.mean()), plddt_site=float(e.plddt[pos]),
        ))
        if (n + 1) % 20 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(picked)}", flush=True)

    keys = list(rec[0])
    out = {k: np.array([r[k] for r in rec]) for k in list(rec[0])}
    out["ca"] = np.stack(cas).astype(np.float32)      # TM computed on login node
    out["ca_wt"] = e_wt.ca.astype(np.float32)
    out["model"] = np.array(args.model)
    out["assay"] = np.array(args.assay)
    out["msa_depth"] = np.array(depth_wt)
    out["single_sequence"] = np.array(bool(args.single_sequence))
    np.savez_compressed(args.out, **out)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
