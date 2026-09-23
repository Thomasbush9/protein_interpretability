"""What the diffusion-conditioning boundary actually holds, before designing on it.

`exp_trajectory.py` materialises the conditioning tensors before sampling, but
nothing records their shapes, widths, or which axis is tokens and which is
atoms. A probe that reads a mutation difference off the wrong axis would look
exactly like a result, so this prints the boundary first: what each tensor is,
how big it is per variant, and whether wild type and mutant are even
comparable elementwise.

Costs one trunk forward per sequence and NO diffusion sampling.

  sbatch checkout.sbatch probe_conditioning.py --assay ... --a3m ... --out ...
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exp_gym import YAML_TMPL, graft_a3m  # noqa: E402


def describe(name, x):
    if callable(x):
        return {"name": name, "kind": "callable", "note": "not an array; "
                "cannot be archived or differenced"}
    a = np.asarray(x)
    return {"name": name, "kind": "array", "shape": list(a.shape),
            "dtype": str(a.dtype), "mb": round(a.nbytes / 1e6, 3)}


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/"
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", default=R + "data/gym/assays/"
                                              "DMS_ProteinGym_substitutions")
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pi_core as pi

    rows = [r for r in csv.DictReader(
        open(Path(a.assay_dir) / f"{a.assay}.csv")) if ":" not in r["mutant"]]
    wt = list(rows[0]["mutated_sequence"])
    m0 = re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"])
    wt[int(m0.group(2)) - 1] = m0.group(1)
    wt = "".join(wt)

    work = Path(a.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    key = jax.random.key(0)

    def featurise(seq, tag):
        p = work / "msa" / f"{tag}.a3m"
        graft_a3m(p, Path(a.a3m), seq, wt, cap=a.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=p.resolve()))
        return pi.load_features(y.read_text())

    def conditioning(feats):
        emb = model.embed_inputs(feats)
        tr = pi.run_trunk(model, emb, feats, recycling_steps=a.recycles,
                          key=key, deterministic=True, capture_last=False)
        st = tr["trunk_state"]
        out = model.diffusion_conditioning(
            st.s, st.z, emb.relative_position_encoding, feats)
        return st, out, feats

    NAMES = ["q", "c", "to_keys", "atom_enc_bias", "atom_dec_bias",
             "token_trans_bias"]
    f_wt, h = featurise(wt, "wt")
    st_w, cond_w, _ = conditioning(f_wt)
    n_tok = int(np.asarray(f_wt["token_pad_mask"][0]).sum())
    n_atom = int(np.asarray(f_wt["atom_pad_mask"][0]).sum())
    print(f"[{time.time()-t0:6.1f}s] WT: {n_tok} tokens, {n_atom} atoms\n",
          flush=True)

    report = {"assay": a.assay, "wt": {"n_tokens": n_tok, "n_atoms": n_atom},
              "trunk": {"s": describe("s", st_w.s), "z": describe("z", st_w.z)},
              "conditioning": []}
    print(f"{'tensor':18s}{'kind':10s}{'shape':28s}{'MB':>8s}")
    for nm, x in zip(NAMES, cond_w):
        d = describe(nm, x)
        report["conditioning"].append(d)
        print(f"{nm:18s}{d['kind']:10s}"
              f"{str(d.get('shape','-')):28s}{d.get('mb','-'):>8}")
    for k in ("s", "z"):
        d = report["trunk"][k]
        print(f"trunk.{k:12s}{d['kind']:10s}"
              f"{str(d.get('shape','-')):28s}{d.get('mb','-'):>8}")

    # Is a mutant even comparable elementwise? Atom counts change; token counts
    # do not. This decides whether a difference can be taken before aggregation.
    mo = re.match(r"([A-Z])(\d+)([A-Z])", rows[1]["mutant"])
    p0 = int(mo.group(2)) - 1
    f_m, hm = featurise(rows[1]["mutated_sequence"], "mut")
    st_m, cond_m, _ = conditioning(f_m)
    n_tok_m = int(np.asarray(f_m["token_pad_mask"][0]).sum())
    n_atom_m = int(np.asarray(f_m["atom_pad_mask"][0]).sum())
    print(f"\nmutant {rows[1]['mutant']}: {n_tok_m} tokens, {n_atom_m} atoms")
    report["mutant"] = {"mutant": rows[1]["mutant"], "n_tokens": n_tok_m,
                        "n_atoms": n_atom_m,
                        "tokens_match": n_tok_m == n_tok,
                        "atoms_match": n_atom_m == n_atom}
    print(f"  tokens match WT: {n_tok_m == n_tok}   "
          f"atoms match WT: {n_atom_m == n_atom}")

    cmp = []
    for nm, xw, xm in zip(NAMES, cond_w, cond_m):
        if callable(xw):
            cmp.append({"name": nm, "differenceable": False,
                        "why": "callable"})
            continue
        aw, am = np.asarray(xw), np.asarray(xm)
        same = aw.shape == am.shape
        # MATCHING SHAPES ARE NOT COMPARABLE INDICES. Atom-level tensors are
        # padded to a fixed window count, so WT and mutant come out the same
        # SHAPE while holding different numbers of real atoms (820 vs 816
        # here) -- atom k is a different atom in the two runs from the
        # mutation site onward. Differencing them elementwise is well-formed
        # and wrong, which is this project's recurring failure mode. Only
        # token-indexed tensors may be differenced directly.
        axis = "token" if (aw.ndim >= 3 and aw.shape[1] == n_tok) else "atom"
        e = {"name": nm, "shapes_match": bool(same), "axis": axis,
             "safe_to_difference": bool(same and axis == "token"),
             "wt_shape": list(aw.shape), "mut_shape": list(am.shape)}
        if same:
            d = np.abs(aw - am)
            e["mean_abs_diff"] = float(d.mean())
            e["max_abs_diff"] = float(d.max())
        if axis == "atom":
            e["why"] = ("atom-indexed and padded: aggregate through "
                        "atom_to_token BEFORE differencing")
        cmp.append(e)
        print(f"  {nm:18s}axis={axis:6s}shapes_match={same}  "
              f"SAFE_TO_DIFFERENCE={e['safe_to_difference']}"
              + (f"  mean|d|={e['mean_abs_diff']:.3e}" if same else ""))
    report["wt_vs_mutant"] = cmp

    # atom -> token map, the aggregation any atom-level tensor needs
    a2t = np.asarray(f_wt["atom_to_token"][0])
    report["atom_to_token_shape"] = list(a2t.shape)
    report["mutated_position_0based"] = p0
    print(f"\natom_to_token {a2t.shape}  (atom-level tensors must be "
          f"aggregated through this before differencing)")

    h.cleanup(); hm.cleanup()
    Path(a.out).write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
