"""Does the internal probe transfer to a protein it was never trained on?

Everything reported so far is WITHIN-assay decodability: train on some residue
positions of a protein, test on other positions of the same protein. That is the
right design for the comparison the paper makes -- internal against the same
model's own output, on identical rows -- but it leaves a question a reader will
ask immediately: is the Pairformer carrying a general stability signal, or does
each assay need its own decoder?

Leave-one-assay-out answers it. Train on 11 assays, test on the 12th, repeat.
All twelve Boltz-2 assays share the same 256-dimensional feature space (4
quantities x 64 layers), so the pooling is well defined.

Two normalisations are needed and neither is optional. Feature scales differ
between proteins -- ||dz|| depends on chain length and on the model's
representation scale for that fold -- so features are z-scored WITHIN each assay
before pooling. Targets differ in dynamic range for the same reason, so
DMS_score is z-scored within assay too. Without both, the ridge would spend its
capacity modelling which protein a row came from.

The comparison that matters here is NOT against a stability predictor; this
project does not compete on ProteinGym. It is whether internal still beats the
model's own emitted output when both have to generalise to an unseen protein.
So the transferred internal probe is scored against:

  TM to WT          unfitted, so transfer is free -- the same number either way
  output_rich       the 10 emitted quantities, transferred the same way
  chemistry         substitution chemistry, transferred the same way
  within-assay      the ordinary probe, as a ceiling reference

If internal transfers and the output side does not, the claim strengthens from
"decodable per assay" to "a general signal the coordinates do not carry". If
neither transfers, the honest description stays assay-specific decodability --
which does not damage the paired comparison, but does bound what can be said.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402
import pi_chem  # noqa: E402
import pi_basis  # noqa: E402
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402
import pi_stats  # noqa: E402
from compare_internal_output import (grouped_split, output_matrix,  # noqa: E402
                                     ridge_fit, ridge_pred, select_k)


def zscore(a):
    a = np.asarray(a, dtype=float)
    return (a - a.mean(0)) / (a.std(0) + 1e-9)


def zstats(a):
    a = np.asarray(a, dtype=float)
    return a.mean(0), a.std(0) + 1e-9


def zapply(a, mu, sd):
    return (np.asarray(a, dtype=float) - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--inductive", action="store_true",
                    help="scale held-out features with TRAINING-assay statistics "
                         "instead of the held-out assay's own")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--tm-cache",
                    default="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
                            "prot_interp_files/runs/tm_cache.npz",
                    help="tmtools is in the repo venv, not the container; "
                         "precompute_tm.py writes this")
    ap.add_argument("--k-sweep", default="",
                    help="comma-separated k values; records the leave-one-assay-out "
                         "curve for the 128-channel internal block, so a figure can "
                         "show how much of the representation is needed instead of "
                         "asserting one truncation")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    TM = np.load(a.tm_cache)
    A = {}
    for f in a.features:
        d = pi_archive.load_capture(f)
        # THE ASSAY, NOT THE FILENAME. `Path(f).stem.split("_")[1]` reads the
        # second underscore field, which is the assay for `gym2s_<assay>` and
        # the MODEL for `xm_<model>_<run>_<assay>` -- so pointing this script at
        # the cross-model family gave three files all keyed "boltz2". The xm
        # captures record their assay; a filename is not evidence when the
        # array is right there.
        assay = (str(d["assay"]) if "assay" in d
                 else Path(f).stem.split("_", 1)[1])
        name = assay.split("_")[0]
        y = d["score"]
        ca, ca_wt = d["ca"], d["ca_wt"]
        # TM comes from the cache: tmtools lives in the repo venv and JAX in
        # the container, and neither has the other. Recomputing it here would
        # simply fail; dropping it would weaken the output baseline, which is
        # the wrong direction to be sloppy in.
        if assay not in TM:
            raise SystemExit(f"{assay}: not in {a.tm_cache}; run precompute_tm.py")
        tm = np.asarray(TM[assay], float)
        # RAW blocks are kept alongside the z-scored ones. The published
        # protocol scales every assay by its own statistics, which is
        # unsupervised but transductive -- the held-out assay's rows decide its
        # own scale. `--inductive` instead scales it with the training assays'
        # statistics, so nothing about the held-out protein enters the fit.
        raw = {
            # `magnitudes` rather than an explicit norm on `ds_site`. Both
            # give (n, L) for gym2s, where ds_site is (n, L, 384) -- but in the
            # xm family ds_site is ALREADY the per-layer norm, and taking a norm
            # of it again collapses the layer axis to (n,). The accessor asks
            # the array what it is instead of assuming.
            "internal": np.concatenate(
                [d["kl_glob"], d["kl_site"],
                 d.magnitudes("dz"),
                 d.magnitudes("ds")], axis=1),
            # The 128 pair channels at the final layer -- the DIRECTION, not
            # just how far the row moved. `internal` above feeds dz_site in as
            # a per-layer norm, which is a defensible shared feature space but
            # discards exactly the quantity this project is about, and it is
            # why the transferred internal number came out well below the
            # within-assay figures reported elsewhere. The channels mean the
            # same thing in every protein (that is the shared-subspace result),
            # so pooling them across assays is well defined.
            # `pair_row` finds the DIRECTION wherever the family keeps it --
            # `dz_vec` in xm, `dz_site` in gym2s -- and raises on an archive
            # that holds only norms rather than quietly returning them.
            "internal_vec": d.pair_row(-1),
            "chemistry": pi_chem.chem_matrix([str(m) for m in d["mutant"]]),
            # Two families, two spellings of the chain mean.
            "output_rich": output_matrix(
                ca, ca_wt, tm,
                d["plddt"] if "plddt" in d else d["plddt_mean"],
                d["plddt_site"], d["pos"]),
        }
        A[name] = {"y": y, "pos": d["pos"], "yz": zscore(y), "tm": tm,
                   "raw": raw, **{k: zscore(v) for k, v in raw.items()}}
        print(f"  loaded {name:8s} n={len(y):4d}  internal dim "
              f"{A[name]['internal'].shape[1]}")

    names = sorted(A)
    dims = {A[n]["internal"].shape[1] for n in names}
    if len(dims) > 1:
        raise SystemExit(f"feature dimensions differ across assays: {dims}")

    BLOCKS = ["internal", "internal_vec", "chemistry", "output_rich"]
    res = {b: {} for b in BLOCKS}
    res["TM_to_WT"] = {}
    res["internal_within"] = {}

    for held in names:
        tr_names = [n for n in names if n != held]
        for b in BLOCKS:
            if a.inductive:
                # one scale, learned on the training assays, applied to all
                mu, sd = zstats(np.concatenate([A[n]["raw"][b] for n in tr_names], 0))
                Xtr = np.concatenate([zapply(A[n]["raw"][b], mu, sd)
                                      for n in tr_names], 0)
                Xte = zapply(A[held]["raw"][b], mu, sd)
            else:
                Xtr = np.concatenate([A[n][b] for n in tr_names], 0)
                Xte = A[held][b]
            ytr = np.concatenate([A[n]["yz"] for n in tr_names], 0)
            k = min(a.k, Xtr.shape[1])
            idx = select_k(Xtr, ytr, k)
            w = ridge_fit(Xtr[:, idx], ytr, a.lam)
            pred = ridge_pred(w, Xte[:, idx])
            res[b][held] = pi_stats.spearman(pred, A[held]["y"])
        # unfitted comparator: transfer is free, it is the same number
        res["TM_to_WT"][held] = pi_stats.spearman(A[held]["tm"], A[held]["y"])
        # ceiling: the ordinary within-assay probe, position-grouped
        vals = []
        for s in range(5):
            tr, te = grouped_split(A[held]["pos"], 0.25, np.random.default_rng(s))
            X = A[held]["internal"]
            idx = select_k(X[tr], A[held]["y"][tr], a.k)
            w = ridge_fit(X[tr][:, idx], A[held]["yz"][tr], 1.0)
            vals.append(pi_stats.spearman(ridge_pred(w, X[te][:, idx]),
                                          A[held]["y"][te]))
        res["internal_within"][held] = float(np.nanmean(vals))

    # ---- truncation curve for the 128-channel block -----------------------
    # Same leave-one-assay-out loop, same selection rule, k varied. The ridge
    # solve runs on the accelerator: this account has no CPU partition, so the
    # job holds a GPU either way and should use it.
    sweep = {}
    if a.k_sweep:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        print(f"\n   jax devices: {jax.devices()}", flush=True)

        @jax.jit
        def solve(Xtr, ytr, Xte, lam):
            Xb = jnp.column_stack([Xtr, jnp.ones(len(Xtr))])
            A = Xb.T @ Xb + lam * jnp.eye(Xb.shape[1])
            A = A.at[-1, -1].add(-lam)          # never penalise the intercept
            w = jnp.linalg.solve(A, Xb.T @ ytr)
            return jnp.column_stack([Xte, jnp.ones(len(Xte))]) @ w

        ks_sweep = [int(x) for x in a.k_sweep.split(",")]
        print(f"   k-sweep on internal_vec: {ks_sweep}")
        # Two bases, one protocol. CHANNELS keeps the model's own coordinates
        # and selects among them; ROTATED fits an SVD on the training assays
        # only and selects among components. At k = full rank a rotation is
        # invertible, so a linear probe cannot distinguish them and the two
        # curves MUST meet -- that endpoint is the built-in check, not a result.
        for kk in ks_sweep:
            per, per_rot = {}, {}
            for held in names:
                tr_names = [n for n in names if n != held]
                Xtr = np.concatenate([A[n]["internal_vec"] for n in tr_names], 0)
                Xte = A[held]["internal_vec"]
                ytr = np.concatenate([A[n]["yz"] for n in tr_names], 0)
                kc = min(kk, Xtr.shape[1])

                idx = select_k(Xtr, ytr, kc)
                pred = np.asarray(solve(jnp.asarray(Xtr[:, idx]), jnp.asarray(ytr),
                                        jnp.asarray(Xte[:, idx]), float(a.lam)))
                per[held] = pi_stats.spearman(pred, A[held]["y"])

                # zscore=False: this basis is NOT PC2. Every other basis in
                # the project standardises per assay first; this one
                # decomposes the model's own channel units, so a protein with
                # a larger channel scale pulls the components toward itself.
                # The two directions overlap at |cos| = 0.87, measured in
                # pi_basis_test.py -- close enough to look like the same
                # object and not be one. Nothing said so before; the flag says
                # it at the call site now.
                Bk = pi_basis.fit(
                    {n: A[n]["internal_vec"] for n in tr_names}, layer=-1,
                    orient_on=None, zscore=False)
                Ptr = Bk.features(Xtr, layer=-1) @ Bk.components.T
                Pte = Bk.project(Xte, layer=-1)
                jdx = select_k(Ptr, ytr, kc)
                predr = np.asarray(solve(jnp.asarray(Ptr[:, jdx]), jnp.asarray(ytr),
                                         jnp.asarray(Pte[:, jdx]), float(a.lam)))
                per_rot[held] = pi_stats.spearman(predr, A[held]["y"])

            out_k = {}
            for tag, d_ in (("channels", per), ("rotated", per_rot)):
                pt, lo, hi, _ = pi_stats.cluster_bootstrap(
                    {n: [d_[n]] for n in names}, n_boot=10000, seed=0,
                    hierarchical=False)
                out_k[tag] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                              "per_assay": d_}
            # backward-compatible: bare keys are the channel basis
            sweep[str(kk)] = {**out_k["channels"], **out_k,
                              "rotated_basis": dict(Bk.protocol)["basis"]}
            d_gap = out_k["rotated"]["mean"] - out_k["channels"]["mean"]
            print(f"     k={kk:4d}  channels {out_k['channels']['mean']:+.3f}  "
                  f"rotated {out_k['rotated']['mean']:+.3f}  "
                  f"diff {d_gap:+.4f}")

    ORDER = ["internal_within", "internal", "internal_vec", "chemistry",
             "output_rich", "TM_to_WT"]
    LAB = {"internal_within": "internal (within-assay)",
           "internal_vec": "internal 128-dim TRANSFERRED",
           "internal": "internal TRANSFERRED", "chemistry": "chemistry TRANSFERRED",
           "output_rich": "output-rich TRANSFERRED", "TM_to_WT": "TM to WT"}
    print(f"\nLeave-one-assay-out: trained on 11 assays, tested on the 12th\n")
    print(f"{'held-out assay':16s}" + "".join(f"{LAB[k][:22]:>24s}" for k in ORDER))
    print("-" * (16 + 24 * len(ORDER)))
    for n in names:
        print(f"{n:16s}" + "".join(f"{res[k][n]:>+24.3f}" for k in ORDER))
    print("-" * (16 + 24 * len(ORDER)))
    print(f"{'mean':16s}" + "".join(
        f"{np.nanmean([res[k][n] for n in names]):>+24.3f}" for k in ORDER))

    print("\nAssay-level bootstrap over the 12 held-out assays\n")
    summary = {}
    for k in ORDER:
        pt, lo, hi, nk = pi_stats.cluster_bootstrap(
            {n: [res[k][n]] for n in names}, n_boot=10000, seed=0,
            hierarchical=False)
        summary[k] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                      "per_assay": {n: res[k][n] for n in names}}
        print(f"  {LAB[k]:26s} {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    print("\nPaired differences across the 12 held-out assays\n")
    gaps = {}
    for lab, ka, kb in (("transferred internal - TM", "internal", "TM_to_WT"),
                        ("transferred internal - output-rich", "internal", "output_rich"),
                        ("internal 128-dim - output-rich", "internal_vec", "output_rich"),
                        ("internal 128-dim - chemistry", "internal_vec", "chemistry"),
                        ("internal 128-dim - internal norms", "internal_vec", "internal"),
                        ("transferred internal - chemistry", "internal", "chemistry"),
                        ("within-assay - transferred", "internal_within", "internal")):
        pt, lo, hi, nk = pi_stats.paired_cluster_bootstrap(
            {n: [res[ka][n]] for n in names}, {n: [res[kb][n]] for n in names},
            n_boot=10000, seed=0, hierarchical=False)
        wins = sum(1 for n in names if res[ka][n] > res[kb][n])
        gaps[lab] = {"gap": pt, "ci_lo": lo, "ci_hi": hi, "wins": wins,
                     "n_assays": len(names)}
        # An interval entirely BELOW zero is a significant difference in the
        # other direction, not an inconclusive one. The old test flagged
        # "within-assay - transferred" at [-0.098, -0.036] as including zero.
        excl = np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
        flag = "" if excl else "   <- includes zero"
        print(f"  {lab:36s} {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"{wins}/{len(names)} assays{flag}")

    _res = (
        {"k_sweep": sweep, "protocol": {**pi_protocol.protocol(
             script="analyze_transfer.py",
             design="leave-one-assay-out (train on 11 assays, test on the 12th)",
             layer=pi_protocol.layers("final", n_layers=A[names[0]]["raw"]["internal"].shape[1] // 4),
             features={b: pi_protocol.features(
                 b, A[names[0]]["raw"][b].shape[1],
                 kept=min(a.k, A[names[0]]["raw"][b].shape[1])) for b in BLOCKS},
             source=" ".join(a.features) if isinstance(a.features, list) else a.features,
             n_assays=len(names),
             n_train_rows=int(sum(len(A[n]["y"]) for n in names)
                              - len(A[names[0]]["y"])),
             selection="top-k by |Spearman| on TRAINING rows only",
             normalisation_variant=("inductive (training-assay statistics)"
                                    if a.inductive else
                                    "transductive (each assay scaled by its own)"),
             note="internal_vec is the 128 pair channels at the FINAL layer; "
                  "internal is 4 scalar quantities x every layer, so dz enters "
                  "it only as a per-layer norm."),
             "design_short": "leave-one-assay-out", "k": a.k, "lam": a.lam,
                      "normalisation": "features and target z-scored within assay",
                      "n_assays": len(names)},
         "normalisation_mode": "inductive (training-assay statistics)"
             if a.inductive else "transductive (each assay's own statistics)",
         "predictors": summary, "gaps": gaps})
    pi_archive.write_result(a.out, _res, protocol=_res.pop("protocol"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
