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

All three models run through THIS script, so variant selection, feature
definitions, alignment handling, recycles and sampling steps are identical
across them by construction. Earlier, Boltz-2's per-layer features came from
exp_gym2.py, which samples 250 variants at random and reads a sampled subset of
residue pairs, while this script spreads 100 variants across the sorted score
range and uses all pairs; the two overlapped by only ~20 variants per assay, so
the models could be compared to their own outputs but not to each other.

Capture fidelity is checked twice per run and the run ABORTS on failure: the
captured final z is compared against the model's own trunk output
(pi_capture.verify_capture), and the first real mutation must move z far more
than that drift (pi_capture.check_signal_to_drift). Both numbers are saved into
the output archive next to the features they license.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_models  # noqa: E402
import pi_capture  # noqa: E402
from exp_gym import graft_a3m  # noqa: E402


@dataclass
class _Args:
    """The argparse namespace, as a type.

    `collect_assay` was lifted out of `main()` verbatim, and its body reads
    every setting off `args`. Rebuilding that object rather than rewriting a
    hundred lines of `args.model` into `model` is what makes the extraction a
    move rather than an edit -- the diff shows no expression changed, which is
    the only cheap way to be sure the numerics did not.
    """

    model: str
    assay: str
    assay_dir: str
    a3m: str
    work: object
    n_variants: int
    recycles: int
    sampling_steps: int
    msa_cap: int
    msa: str
    out: object


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def distogram_per_layer(name, inner, z_layers, mask):
    """Logit lens: the model's own distogram head applied to every layer."""
    out = []
    for L in range(z_layers.shape[0]):
        zl = z_layers[L]
        if name == "of3":
            lg = np.asarray(inner.aux_heads.distogram(z=zl))
        elif name == "protenix":
            lg = np.asarray(inner.distogram_head(zl))
        else:
            # Boltz-2's head returns [B, N, N, n_distograms, n_bins] and needs a
            # 4-D input (it does a `b n m d -> b m n d` rearrange internally).
            # The generic leading-axis squeeze below would strip the wrong axis
            # here -- [1,N,N,1,64] -> [N,1,64] -- so index it explicitly.
            if zl.ndim == 3:
                zl = zl[None]
            out.append(softmax(np.asarray(
                inner.distogram_module(zl)[0, :, :, 0, :])[np.ix_(mask, mask)]))
            continue
        while lg.ndim > 3:
            lg = lg[0]
        out.append(softmax(lg[np.ix_(mask, mask)]))
    return np.stack(out)          # [n_layers, N, N, B]


def collect_assay(model, assay, assay_dir, a3m, work, *, n_variants=100,
                  recycles=3, sampling_steps=200, msa_cap=2048,
                  msa="subsample", out_path=None):
    """One assay through one model. Returns the arrays; writes nothing.

    EXTRACTED FROM `main()` WITHOUT CHANGING AN EXPRESSION. Every line below was
    main's body; only the argument names changed (`args.model` -> `model`) and
    the `np.savez_compressed` moved out to the caller. It was pulled out so the
    package's adapter can run the same numerics rather than reimplementing them
    -- the archived cross-model captures came from this code, and a second
    implementation of it would be a second thing to validate.

    `out_path` is accepted and ignored except in the log line, so the message a
    reader sees still names the file the caller is about to write.
    """
    args = _Args(model=model, assay=assay, assay_dir=assay_dir, a3m=a3m,
                 work=work, n_variants=n_variants, recycles=recycles,
                 sampling_steps=sampling_steps, msa_cap=msa_cap, msa=msa,
                 out=out_path)

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
    wrapper = pi_models.load(args.model, msa=args.msa)
    inner = pi_models.inner(args.model, wrapper)
    cap_fn = pi_capture.CAPTURE[args.model]
    key = jax.random.key(0)
    print(f"[{time.time()-t0:6.1f}s] {args.model} / {args.assay} n={len(picked)}", flush=True)

    def featurise(seq, tag):
        a3m = work / "msa" / f"{tag}.a3m"
        graft_a3m(a3m, src, seq, wt, cap=args.msa_cap)
        feats, depth = pi_models.features_for(args.model, wrapper, seq, str(a3m),
                                              work=work / tag)
        # Boltz-2 replaces the alignment with a one-row dummy when the a3m query
        # does not match the input, which silently changes the computation being
        # captured. Catch it here rather than discovering it in the numbers.
        if depth < 2:
            raise AssertionError(
                f"{tag}: alignment collapsed to {depth} row(s); the model is not "
                "seeing the alignment this run is supposed to control.")
        return feats, depth

    depths = []

    def run(seq, tag):
        feats, depth = featurise(seq, tag)
        depths.append(depth)
        cap = cap_fn(inner, feats, num_recycles=args.recycles, key=key)
        out = wrapper.model_output(features=feats, recycling_steps=args.recycles,
                                   sampling_steps=args.sampling_steps, key=key)
        e = pi_models.extraction_from(out, name=args.model)
        if args.model == "of3":
            mask = np.asarray(feats.token_mask[0]).astype(bool)
        elif args.model == "boltz2":
            # boltz2 features are a plain dict, not an attribute-style Batch
            mask = np.asarray(feats["token_pad_mask"][0]).astype(bool)
        else:
            mask = np.ones(e.ed.shape[0], bool)
        return cap, feats, e, mask

    cap_wt, feats_wt, e_wt, mask = run(wt, "wt")
    # Fidelity: the capture must describe the computation the model actually ran.
    # This compares the captured final z against the model's OWN trunk output and
    # raises if they disagree beyond the per-model tolerance. An earlier version
    # compared cap_wt["z"] with itself at tol=1e9, which is a shape check and
    # cannot fail -- the per-layer features below were unverified because of it.
    drift = pi_capture.verify_capture(args.model, inner, feats_wt, cap_wt,
                                      num_recycles=args.recycles, key=key)
    print(f"[{time.time()-t0:6.1f}s] capture drift vs real trunk: rel {drift:.3e} "
          f"(tol {pi_capture.DRIFT_TOL[args.model]:g})", flush=True)
    p_wt = distogram_per_layer(args.model, inner, cap_wt["z_layers"], mask)
    z_wt = np.asarray(cap_wt["z_layers"])
    s_wt = np.asarray(cap_wt["s_layers"])
    nL = p_wt.shape[0]
    print(f"[{time.time()-t0:6.1f}s] WT done: {nL} layers, "
          f"distogram {p_wt.shape[-1]} bins, pLDDT {e_wt.plddt.mean():.3f}", flush=True)

    rec, cas = [], []
    ratio = None
    for n, r in enumerate(picked):
        pos = int(re.match(r"[A-Z](\d+)", r["mutant"]).group(1)) - 1
        cap, _, e, _ = run(r["mutated_sequence"], "mut")
        if ratio is None:
            # A small drift only means something next to the effect being
            # measured. Check once, on the first real mutation, before spending
            # an hour on variants whose features would be capture noise.
            sig, ratio = pi_capture.check_signal_to_drift(
                args.model, cap["z"], cap_wt["z"], drift)
            print(f"[{time.time()-t0:6.1f}s] one-mutation signal rel {sig:.3e}; "
                  f"signal/drift {ratio:.0f}x "
                  f"(need {pi_capture.MIN_SIGNAL_TO_DRIFT:g}x)", flush=True)
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
        # The VECTOR, not its norm. A norm cannot support a subspace
        # comparison: it discards the direction, which is the entire object of
        # interest. `deep2_*` stored only the norm, which is why the
        # cross-model analysis could not be done offline. Defined to match
        # Boltz-2's dz_site exactly -- mean over partner residues of the pair
        # row at the mutated position -- so the three models describe the same
        # quantity even though their channel spaces are unrelated.
        dz_vec = dzt[:, pos].mean(axis=1)              # [L, C]
        dst = sl - s_wt
        while dst.ndim > 3:
            dst = dst[:, 0]
        ds_tok = np.linalg.norm(dst, axis=-1)          # [L, N]
        ds_vec = dst[:, pos]                           # [L, C_s]
        rec.append(dict(
            mutant=r["mutant"], pos=pos, score=float(r["DMS_score"]),
            kl_glob=kl.mean(axis=(1, 2)).astype(np.float32),
            kl_site=kl[:, pos].mean(axis=1).astype(np.float32),
            dz_site=dz_row[:, pos].mean(-1).astype(np.float32),
            ds_site=ds_tok[:, pos].astype(np.float32),
            dz_vec=dz_vec.astype(np.float32),
            ds_vec=ds_vec.astype(np.float32),
            plddt_mean=float(e.plddt.mean()), plddt_site=float(e.plddt[pos]),
        ))
        cas.append(e.ca)
        if (n + 1) % 20 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(picked)}", flush=True)

    out = {k: np.array([r[k] for r in rec]) for k in
           ("mutant", "pos", "score", "plddt_mean", "plddt_site")}
    for k in ("kl_glob", "kl_site", "dz_site", "ds_site"):
        out[k] = np.stack([r[k] for r in rec])        # [n_variants, n_layers]
    for k in ("dz_vec", "ds_vec"):
        out[k] = np.stack([r[k] for r in rec])        # [n_variants, n_layers, C]
    out["ca"] = np.stack(cas).astype(np.float32)      # TM on a login node
    out["ca_wt"] = e_wt.ca.astype(np.float32)
    # the fidelity evidence travels with the features it licenses
    out["capture_drift"] = np.array(drift)
    out["signal_to_drift"] = np.array(ratio if ratio is not None else np.nan)
    out["drift_tol"] = np.array(pi_capture.DRIFT_TOL[args.model])
    out["msa_depth"] = np.array(depths[0] if depths else -1)
    out["recycles"] = np.array(args.recycles)
    out["msa_cap"] = np.array(args.msa_cap)
    out["sampling_steps"] = np.array(args.sampling_steps)
    out["model"] = np.array(args.model)
    out["assay"] = np.array(args.assay)
    out["n_layers"] = np.array(nL)
    # The MSA regime is not recoverable from the arrays, and the two are not
    # interchangeable: `subsample` redraws the alignment per key and is not
    # bit-reproducible. Read back off the built model rather than echoing the
    # argument, so the record describes what ran.
    for _k, _v in pi_models.regime_block(args.model, wrapper).items():
        out[_k] = np.array(str(_v))
    print(f"\n[{time.time()-t0:6.1f}s] collected {args.out}  "
          f"({len(rec)} variants x {nL} layers; dz_vec "
          f"{out['dz_vec'].shape})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["of3", "protenix", "boltz2"])
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-variants", type=int, default=100)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--msa", default="subsample",
                    choices=pi_models.MSA_REGIMES,
                    help="MSA regime. Default `subsample` reproduces the archives this script has already written; use `full` for numbers meant to be reproduced -- it is bit-reproducible across keys and the subsample is not.")
    args = ap.parse_args()

    out = collect_assay(
        args.model, args.assay, args.assay_dir, args.a3m, args.work,
        n_variants=args.n_variants, recycles=args.recycles,
        sampling_steps=args.sampling_steps, msa_cap=args.msa_cap,
        msa=args.msa, out_path=args.out)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
