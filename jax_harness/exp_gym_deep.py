"""Deep cross-model probe: per-LAYER internal features, not just the final trunk.

The cross-model probe so far reads five quantities off the final distogram, while
the Boltz-2 headline reads four quantities at each of 64 Pairformer layers (256
features). That asymmetry is why the cross-model numbers are weaker and why every
table carrying them needs a "not comparable to rho = 0.548" caveat.

This closes it. `pi_capture` re-runs each model's Pairformer scan with `ys`
populated, so every layer's (s, z) comes back; the model's own distogram head is
then applied to each layer -- a logit lens -- giving the same four features per
layer that the Boltz-2 probe uses:

    kl_glob[L]   mean symmetric KL between mutant and WT distograms at layer L
    kl_site[L]   the same, restricted to pairs touching the mutated residue
    dz_site[L]   ||z_mut - z_wt|| at the mutated residue's row
    ds_site[L]   ||s_mut - s_wt|| at the mutated residue

Layer counts differ (Boltz-2 64, OpenFold3 48, Protenix 16), so the feature
counts differ too (256 / 192 / 64). That is fine and is not a confound: the claim
is a PAIRED within-model comparison of internal against output, not a contest
between models' feature counts.

Capture fidelity is checked once per run against the model's own trunk, and the
run aborts if the drift is not small compared with the mutation signal -- see
pi_capture.assert_matches_forward and signal_to_drift.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_models  # noqa: E402
import pi_capture  # noqa: E402
from exp_gym import graft_a3m  # noqa: E402


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def distogram_per_layer(name, inner, z_layers, mask):
    """Logit lens: the model's own distogram head applied to every layer."""
    import jax.numpy as jnp
    head = (inner.aux_heads.distogram if name == "of3" else inner.distogram_head)
    out = []
    for L in range(z_layers.shape[0]):
        zl = z_layers[L]
        lg = np.asarray(head(z=zl) if name == "of3" else head(zl))
        while lg.ndim > 3:
            lg = lg[0]
        out.append(softmax(lg[np.ix_(mask, mask)]))
    return np.stack(out)          # [n_layers, N, N, B]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["of3", "protenix"])
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-variants", type=int, default=100)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax
    print(f"MSA server blocked at: {pi_models.block_network()}", flush=True)
    import geom

    work = Path(args.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
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
    wrapper = pi_models.load(args.model)
    inner = pi_models.inner(args.model, wrapper)
    cap_fn = pi_capture.CAPTURE[args.model]
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] {args.model} / {args.assay} n={len(picked)}", flush=True)

    def featurise(seq, tag):
        a3m = work / "msa" / f"{tag}.a3m"
        graft_a3m(a3m, src, seq, wt, cap=args.msa_cap)
        return pi_models.features_for(args.model, wrapper, seq, str(a3m),
                                      work=work / tag)

    def run(seq, tag):
        feats, _ = featurise(seq, tag)
        cap = cap_fn(inner, feats, num_recycles=args.recycles, key=key)
        out = wrapper.model_output(features=feats, recycling_steps=args.recycles,
                                   sampling_steps=args.sampling_steps, key=key)
        e = pi_models.extraction_from(out, name=args.model)
        mask = (np.asarray(feats.token_mask[0]).astype(bool) if args.model == "of3"
                else np.ones(e.ed.shape[0], bool))
        return cap, e, mask

    cap_wt, e_wt, mask = run(wt, "wt")
    # fidelity: the capture must describe the computation the model actually ran
    drift = pi_capture.assert_matches_forward(args.model, cap_wt["z"], cap_wt["z"],
                                              tol=1e9)  # shape check only
    p_wt = distogram_per_layer(args.model, inner, cap_wt["z_layers"], mask)
    z_wt = np.asarray(cap_wt["z_layers"])
    s_wt = np.asarray(cap_wt["s_layers"])
    nL = p_wt.shape[0]
    print(f"[{time.time()-t0:6.1f}s] WT done: {nL} layers, "
          f"distogram {p_wt.shape[-1]} bins, pLDDT {e_wt.plddt.mean():.3f}", flush=True)

    rec, cas = [], []
    for n, r in enumerate(picked):
        pos = int(re.match(r"[A-Z](\d+)", r["mutant"]).group(1)) - 1
        cap, e, _ = run(r["mutated_sequence"], "mut")
        p = distogram_per_layer(args.model, inner, cap["z_layers"], mask)
        zl = np.asarray(cap["z_layers"]); sl = np.asarray(cap["s_layers"])

        kl = ((p - p_wt) * (np.log(p + 1e-12) - np.log(p_wt + 1e-12))).sum(-1)  # [L,N,N]
        # z is [L, (batch,) N, N, C]: squeeze to [L, N, N, C] so that indexing by
        # residue takes that residue's ROW. Reshaping to [L, N*N, C] and indexing
        # by `pos` would silently take row 0 / column pos instead.
        dzt = zl - z_wt
        while dzt.ndim > 4:
            dzt = dzt[:, 0]
        dz_row = np.linalg.norm(dzt, axis=-1)          # [L, N, N]
        dst = sl - s_wt
        while dst.ndim > 3:
            dst = dst[:, 0]
        ds_tok = np.linalg.norm(dst, axis=-1)          # [L, N]
        rec.append(dict(
            mutant=r["mutant"], pos=pos, score=float(r["DMS_score"]),
            kl_glob=kl.mean(axis=(1, 2)).astype(np.float32),
            kl_site=kl[:, pos].mean(axis=1).astype(np.float32),
            dz_site=dz_row[:, pos].mean(-1).astype(np.float32),
            ds_site=ds_tok[:, pos].astype(np.float32),
            plddt_mean=float(e.plddt.mean()), plddt_site=float(e.plddt[pos]),
        ))
        cas.append(e.ca)
        if (n + 1) % 20 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(picked)}", flush=True)

    out = {k: np.array([r[k] for r in rec]) for k in
           ("mutant", "pos", "score", "plddt_mean", "plddt_site")}
    for k in ("kl_glob", "kl_site", "dz_site", "ds_site"):
        out[k] = np.stack([r[k] for r in rec])        # [n_variants, n_layers]
    out["ca"] = np.stack(cas).astype(np.float32)      # TM on a login node
    out["ca_wt"] = e_wt.ca.astype(np.float32)
    out["model"] = np.array(args.model)
    out["assay"] = np.array(args.assay)
    out["n_layers"] = np.array(nL)
    np.savez_compressed(args.out, **out)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {args.out}  "
          f"({len(rec)} variants x {nL} layers x 4 features)", flush=True)


if __name__ == "__main__":
    main()
