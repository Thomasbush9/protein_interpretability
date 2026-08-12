"""Where in the protein does PC2 act?

PC2 of the shared mutation basis is simultaneously the stability axis and the
predictive-certainty axis. That says WHAT it is but not WHERE it is: `z_site` is
averaged over partner residues before archiving, so the component score is a
whole-protein quantity.

The full pair row would answer it directly and needs a GPU re-capture. This does
not, because the per-pair DISTOGRAM is already archived. `exp_gym2` stores
`disto` as the final-layer logits at ~1479 sampled residue pairs, and those
pairs are recoverable exactly: they were drawn from `np.random.default_rng(0)`
after a single fixed `rng.choice` for the variant subsample, over `valid =
arange(len(wt))`. The reconstruction is checked against `disto.shape[1]` for
every assay before use -- the kept counts differ between assays (1476 to 1482),
so twelve independent matches is not a coincidence that a wrong reconstruction
could produce.

So for every variant we have, per residue pair, how much the model's predicted
distance distribution moved and how much it widened. The question becomes
answerable as stated:

  localisation   Weight each pair's |d sigma| by its distance from the mutated
                 residue and take the centroid. Divide by the same centroid
                 computed with uniform weights, which is what a perturbation
                 with no spatial preference would give. Below 1 means the
                 change concentrates near the mutation. The ratio is used
                 rather than the raw radius because a mutation near a terminus
                 has a different distance distribution available to it, and
                 that has nothing to do with the model.

  does PC2 differ from PC1   PC1 is substitution volume. If PC1 is local
                 packing and PC2 is delocalised, that is a real functional
                 distinction between the two directions rather than a
                 restatement of their annotation.

  is it above noise   The `gym2`/`gym2s` replicate gives a per-pair noise
                 estimate at no cost, so the spatial profile is plotted against
                 the level at which it would be indistinguishable from
                 inference drift. Per-pair values are far noisier than the
                 per-variant means used elsewhere, so this is not optional.

Component signs are arbitrary in an SVD. Every component is oriented here so
that its pooled correlation with `kl_glob` is non-negative, i.e. a positive
score always means "the internal state moved more". That convention is applied
before anything is interpreted and is reported in the output.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import sys

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import pi_basis  # noqa: E402
import pi_conf  # noqa: E402
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402
import pi_stats  # noqa: E402

EPS = 1e-9
CENTERS = jnp.asarray(pi_conf.bin_centers())
BINS = np.array([0, 6, 9, 12, 16, 20, 25, 32, 100.0])       # angstrom edges
N_PC = 4


def sigma_mu(logits):
    p = jax.nn.softmax(jnp.asarray(logits, jnp.float64), axis=-1)
    mu = (p * CENTERS).sum(-1)
    var = (p * (CENTERS - mu[..., None]) ** 2).sum(-1)
    return mu, jnp.sqrt(jnp.maximum(var, 1e-12))


def rebuild_pairs(assay_csv, n_variants, n_pairs, length, seed=0):
    """Reproduce the (ii, jj) pair sample `exp_gym2` drew for this assay."""
    rows = [r for r in csv.DictReader(open(assay_csv)) if ":" not in r["mutant"]]
    rng = np.random.default_rng(seed)
    if len(rows) > n_variants:
        rng.choice(len(rows), n_variants, replace=False)
    valid = np.arange(length)
    a, b = rng.choice(valid, n_pairs), rng.choice(valid, n_pairs)
    keep = a != b
    return a[keep], b[keep]


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/"
    ap.add_argument("--glob", default=R + "runs/gym2s_*.npz")
    ap.add_argument("--replicate-glob", default=R + "runs/gym2_*.npz")
    ap.add_argument("--assay-dir",
                    default=R + "data/gym/assays/DMS_ProteinGym_substitutions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default="")
    a = ap.parse_args()

    print(f"jax devices: {jax.devices()}\n")
    files = sorted(glob.glob(a.glob))
    rep = {Path(f).stem.split("_", 1)[1]: f for f in glob.glob(a.replicate_glob)}

    A = {}
    print("Reconstructing the archived pair sample\n")
    print(f"   {'assay':8s} {'len':>4s} {'pairs':>6s} {'archived':>9s} {'check':>6s}")
    for f in files:
        stem = Path(f).stem.split("_", 1)[1]
        name = stem.split("_")[0]
        d = np.load(f, allow_pickle=True)
        L = len(str(d["wt_seq"]))
        ii, jj = rebuild_pairs(Path(a.assay_dir) / f"{stem}.csv", 250, 1500, L)
        ok = len(ii) == d["disto"].shape[1]
        print(f"   {name:8s} {L:4d} {len(ii):6d} {d['disto'].shape[1]:9d} "
              f"{'OK' if ok else 'FAIL':>6s}")
        if not ok:
            raise SystemExit(f"{name}: pair reconstruction does not match the "
                             f"archive; the analysis would silently pair the "
                             f"wrong residues with the wrong distograms")
        A[name] = {"d": d, "ii": ii, "jj": jj, "stem": stem, "L": L,
                   "rep": rep.get(stem)}
    print()

    # ---- shared basis at the last layer, oriented -------------------------
    # pi_basis, not a local rebuild. This script writes pc2_v2.npz, so what it
    # constructs here IS the basis every other analysis inherits; it was one of
    # eight independent reconstructions of it. Verified against the previous
    # code bit-for-bit by pi_basis_test.py.
    blocks = {n: np.asarray(A[n]["d"]["dz_site"])[:, -1, :] for n in A}
    kl = {n: np.asarray(A[n]["d"]["kl_glob"])[:, -1] for n in A}
    B = pi_basis.fit(blocks, layer=-1, orient_on="kl_glob", orient_ref=kl,
                     orient_k=N_PC, n_pc=N_PC, eps=EPS)
    for n in A:
        A[n]["P"] = B.project(blocks[n], layer=-1)
    V, orient = B.V, [float(s) for s in B.orient]
    print(f"Component orientation (so that rho with kl_glob >= 0): {orient}\n")

    # ---- per-pair change, and its distance from the mutated residue -------
    prof, per_assay = {}, {}
    nb = len(BINS) - 1
    # Three sets of profiles, because the obvious one is confounded. Splitting
    # variants by raw PC score mixes in perturbation MAGNITUDE, which PC2
    # tracks at +0.54, and a larger perturbation need not have the same spatial
    # shape as a smaller one. So each component is also split on its score
    # AFTER residualising on magnitude, and magnitude itself is profiled on its
    # own. If a gradient survives residualisation it belongs to the component;
    # if it moves to the magnitude panel it never did.
    acc = {c: np.zeros((4, nb)) for c in range(N_PC)}       # raw quartiles
    acc_n = {c: np.zeros((4, nb)) for c in range(N_PC)}
    accr = {c: np.zeros((4, nb)) for c in range(N_PC)}      # magnitude-adjusted
    accr_n = {c: np.zeros((4, nb)) for c in range(N_PC)}
    accm, accm_n = np.zeros((4, nb)), np.zeros((4, nb))     # magnitude itself
    noise_acc, noise_n = np.zeros(nb), np.zeros(nb)

    def resid_on(x, cov):
        """Rank-residualise x on cov, the construction used in pi_stats."""
        from scipy.stats import rankdata
        rx, rc = rankdata(x), rankdata(cov)
        Z = np.column_stack([rc, np.ones(len(rc))])
        return rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]

    def quartile(v):
        return np.clip(np.searchsorted(np.nanpercentile(v, [25, 50, 75]), v), 0, 3)

    print("Localisation of the per-pair width change\n")
    print(f"   {'assay':8s} {'radius ratio':>13s} {'noise/signal':>13s} "
          + "".join(f"{('PC' + str(c + 1)):>8s}" for c in range(N_PC)))
    for n, R_ in A.items():
        d, ii, jj = R_["d"], R_["ii"], R_["jj"]
        _, sw = sigma_mu(d["disto_wt"])                     # (P,)
        _, sm = sigma_mu(d["disto"])                        # (Nv, P)
        dsig = np.asarray(sm - sw)                          # (Nv, P)

        ca = np.asarray(d["ca_wt"], float)
        pos = np.asarray(d["pos"])
        # distance of each pair from the mutated residue: the nearer endpoint
        dmat = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
        dist = np.minimum(dmat[pos][:, ii], dmat[pos][:, jj])    # (Nv, P)

        w = np.abs(dsig)
        rad = (w * dist).sum(1) / np.maximum(w.sum(1), EPS)
        rad0 = dist.mean(1)                                  # uniform-weight
        ratio = rad / np.maximum(rad0, EPS)

        # replicate noise on the same quantity
        ns = np.nan
        if R_["rep"]:
            d2 = np.load(R_["rep"], allow_pickle=True)
            if [str(m) for m in d2["mutant"]] == [str(m) for m in d["mutant"]]:
                _, sw2 = sigma_mu(d2["disto_wt"])
                _, sm2 = sigma_mu(d2["disto"])
                dsig2 = np.asarray(sm2 - sw2)
                nse = np.std(dsig - dsig2) / np.sqrt(2)
                ns = float(nse / (np.abs(dsig).mean() + EPS))
                bi = np.clip(np.digitize(dist.ravel(), BINS) - 1, 0, nb - 1)
                np.add.at(noise_acc, bi, np.abs(dsig - dsig2).ravel() / np.sqrt(2))
                np.add.at(noise_n, bi, 1.0)

        rows = {"radius_ratio_mean": float(np.nanmean(ratio)),
                "noise_to_signal": ns, "n_var": int(len(ratio))}
        # correlate each component with the localisation ratio, holding the
        # overall magnitude fixed: a bigger perturbation is not the same claim
        # as a more delocalised one
        mag = w.mean(1)
        rows["mag_vs_ratio"] = pi_stats.spearman(mag, ratio)
        bi = np.clip(np.digitize(dist, BINS) - 1, 0, nb - 1)

        def fill(target, target_n, qq):
            for qi in range(4):
                m = qq == qi
                if not m.any():
                    continue
                nrm = w[m] / np.maximum(w[m].mean(1, keepdims=True), EPS)
                np.add.at(target[qi], bi[m].ravel(), nrm.ravel())
                np.add.at(target_n[qi], bi[m].ravel(), 1.0)

        fill(accm, accm_n, quartile(mag))
        for c in range(N_PC):
            sc = A[n]["P"][:, c]
            rows[f"PC{c+1}_vs_ratio"] = pi_stats.partial_spearman(sc, ratio, [mag])
            rows[f"PC{c+1}_vs_mag"] = pi_stats.spearman(sc, mag)
            fill(acc[c], acc_n[c], quartile(sc))
            fill(accr[c], accr_n[c], quartile(resid_on(sc, mag)))
        per_assay[n] = rows
        print(f"   {n:8s} {rows['radius_ratio_mean']:13.4f} "
              f"{ns:13.3f}"
              + "".join(f"{rows[f'PC{c+1}_vs_ratio']:+8.3f}" for c in range(N_PC)))

    print("\n   radius ratio < 1 means the width change concentrates near the")
    print("   mutated residue; = 1 means no spatial preference at all.\n")

    out = {"protocol": {"orientation": orient, "bins": BINS.tolist(),
                        "n_pc": N_PC, "assays": sorted(A), **B.protocol},
           "per_assay": per_assay}

    for lab, key in ([("localisation radius ratio", "radius_ratio_mean"),
                      ("magnitude vs radius ratio", "mag_vs_ratio")]
                     + [(f"PC{c+1} vs radius ratio | magnitude",
                         f"PC{c+1}_vs_ratio") for c in range(N_PC)]
                     + [(f"PC{c+1} vs perturbation magnitude", f"PC{c+1}_vs_mag")
                        for c in range(N_PC)]):
        g = {n: [v[key]] for n, v in per_assay.items() if np.isfinite(v[key])}
        if not g:
            continue
        pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                   hierarchical=False)
        out[lab] = {"mean": pt, "ci_lo": lo, "ci_hi": hi}
        flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
        print(f"   {lab:38s} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]{flag}")

    prof = {f"PC{c+1}": (acc[c] / np.maximum(acc_n[c], 1)).tolist()
            for c in range(N_PC)}
    profr = {f"PC{c+1}": (accr[c] / np.maximum(accr_n[c], 1)).tolist()
             for c in range(N_PC)}
    profm = (accm / np.maximum(accm_n, 1)).tolist()
    out["profiles"] = {"bins": BINS.tolist(), "by_quartile": prof,
                       "by_quartile_mag_adjusted": profr,
                       "by_magnitude_quartile": profm,
                       "noise": (noise_acc / np.maximum(noise_n, 1)).tolist()}

    lab = [f"{BINS[i]:.0f}-{BINS[i+1]:.0f}" for i in range(nb)]
    lab[-1] = f">{BINS[-2]:.0f}"
    print("\nFar-minus-near gradient of the top-quartile / bottom-quartile "
          "difference\n")
    print(f"   {'split':34s}" + "".join(f"{l:>9s}" for l in lab))
    for tag, P in ([("magnitude", np.array(profm))]
                   + [(f"PC{c+1}, raw", np.array(prof[f'PC{c+1}']))
                      for c in range(2)]
                   + [(f"PC{c+1}, magnitude-adjusted", np.array(profr[f'PC{c+1}']))
                      for c in range(2)]):
        print(f"   {tag:34s}" + "".join(f"{v:9.3f}" for v in P[3] - P[0]))
    print("\n   A gradient that survives the magnitude adjustment belongs to the")
    print("   component. One that does not was the component standing in for")
    print("   how large the perturbation is.")

    pi_archive.write_result(a.out, out, protocol=pi_protocol.protocol(
        script="analyze_pc2.py",
        design="within-assay: variants split by component score into quartiles, "
               "profiled against distance from the mutated residue; the "
               "magnitude-adjusted panels residualise on perturbation size",
        layer=pi_protocol.layers("final"),
        features=pi_protocol.features("dz_site, final-layer pair row", 128,
                                      kept=N_PC,
                                      note="components, not raw channels"),
        source=a.glob, n_assays=len(A),
        pair_sample="reconstructed from exp_gym2's rng and checked against "
                    "disto.shape[1] for every assay before use"))
    print(f"\nwrote {a.out}")
    if a.npz:
        np.savez_compressed(a.npz, V=V, orient=np.array(orient), bins=BINS,
                            **{f"prof_PC{c+1}": acc[c] / np.maximum(acc_n[c], 1)
                               for c in range(N_PC)},
                            noise=noise_acc / np.maximum(noise_n, 1))
        print(f"wrote {a.npz}")


if __name__ == "__main__":
    main()
