"""Give the OUTPUT side the same treatment the internal side just got.

`compare_internal_output.output_matrix` says in its own docstring that the real
danger in this project's headline comparison is dimensional asymmetry, and that
it closes it by giving the structure module "the richest description of its own
product that the saved coordinates allow". That was true when internal was 256
scalars and output was ten hand-built summaries. It stopped being true the
moment the SVD study showed that internal reaches +0.731 when it is given its
raw 128 pair channels instead of four summaries per layer -- because the output
side is still ten summaries, and the saved coordinates allow far more than that.

So the comparison that carries the paper is currently asymmetric in exactly the
way the project already identified as its own weak point. This script removes
the asymmetry by running one protocol over both sides.

Output blocks, in increasing generosity:

  rich (10)          the published baseline, unchanged, as the reference point
  displacement (L)   per-residue displacement magnitude after Kabsch
                     superposition onto wild type
  coordinates (3L)   the signed per-residue residual after the same
                     superposition -- direction as well as magnitude
  pair distances (P) the change in CA-CA distance at the SAME ~1479 residue
                     pairs the distogram was sampled at

That last block is the one that makes the comparison fair rather than merely
larger. It is superposition-free, so it cannot be penalised by a bad global
alignment, and it is structurally matched to the internal representation: both
sides are then described by what happened between residue pairs, over the same
pairs, with the same number of rows.

Internal is deliberately handicapped in three ways, so that a win is not an
artifact of the setup:

  * a single layer, the last, fixed a priori -- no layer search, no averaging
    over the depth window where the probe is known to be strongest
  * 128 dimensions against the output's 3L (183-216) and P (~1479)
  * identical splits, identical standardisation, identical ridge with the same
    inner-fold lambda grid, identical k-truncation grid, identical bootstrap

If internal still wins under those conditions the comparison is hard to attack.
If output closes the gap, that is something to find out here rather than in
review.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402
import pi_stats  # noqa: E402
from analyze_svd import EPS, basis_of, fit_select, spearman_batch  # noqa: E402
from analyze_pc2 import rebuild_pairs  # noqa: E402
from compare_internal_output import grouped_split, output_matrix  # noqa: E402

KS = (1, 2, 4, 8, 16, 32, 64, 128)
REPORT_K = (8, 32, "full")


def superposed(ca, ca_wt):
    """Per-residue residual after Kabsch superposition. Returns (disp, xyz)."""
    B = np.asarray(ca_wt, float) - np.asarray(ca_wt, float).mean(0)
    d, x = [], []
    for c in np.asarray(ca, float):
        A = c - c.mean(0)
        res = A @ geom.kabsch(A, B).T - B
        d.append(np.linalg.norm(res, axis=1))
        x.append(res.ravel())
    return np.asarray(d), np.asarray(x)


def evaluate(X, y, pos, seeds, frac, ks):
    """Held-out Spearman at each k, for both the rotated and the selected view.

    Same construction as `analyze_svd`: basis and scaler fitted on training
    positions only, components left at natural scale so that k = D is exactly
    ridge on the centred matrix, lambda chosen on an inner grouped split.
    """
    X = jnp.asarray(np.asarray(X, float))[None]              # (1, n, D)
    y = jnp.asarray(np.asarray(y, float))
    out = np.full((seeds, len(ks), 2), np.nan)
    for s in range(seeds):
        rng = np.random.default_rng(s)
        tr, te = grouped_split(pos, frac, rng)
        itr, iva = grouped_split(pos[tr], frac, rng)
        ti, ei = np.where(tr)[0], np.where(te)[0]
        if len(ei) < 5 or len(ti) < 20:
            continue
        Xtr = X[:, ti]
        Z = (X - Xtr.mean(1, keepdims=True)) / (Xtr.std(1, keepdims=True) + EPS)
        Zc = Z - Z[:, ti].mean(1, keepdims=True)
        vt, _ = basis_of(Zc[:, ti], center=False)
        P = jnp.einsum("lnd,lkd->lnk", Zc, vt)
        rho_z = spearman_batch(Zc[:, ti].transpose(0, 2, 1), y[ti])
        order_z = jnp.argsort(-jnp.abs(jnp.nan_to_num(rho_z)), axis=-1)
        yc = y - y[ti].mean()
        for ki, k in enumerate(ks):
            kk = min(k, P.shape[-1])
            for vi, Q in enumerate((P[..., :kk],
                                    jnp.take_along_axis(Zc, order_z[:, None, :kk],
                                                        axis=2))):
                pred, _ = fit_select(Q[:, ti], yc[ti], Q[:, ti][:, itr],
                                     yc[ti][itr], Q[:, ti][:, iva], yc[ti][iva],
                                     Q[:, ei])
                out[s, ki, vi] = float(spearman_batch(pred, y[ei])[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/"
    ap.add_argument("--glob", default=R + "runs/gym2s_*.npz")
    ap.add_argument("--assay-dir",
                    default=R + "data/gym/assays/DMS_ProteinGym_substitutions")
    ap.add_argument("--tm-cache", default=R + "runs/tm_cache.npz",
                    help="written by precompute_tm.py in the repo venv; tmtools "
                         "is not installed in the mosaic container")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print(f"jax devices: {jax.devices()}\n")
    # Loaded rather than computed, and required rather than optional: silently
    # dropping the TM column would weaken the output baseline, and it would do
    # so in the direction that favours the conclusion this test exists to be
    # able to refute.
    TM = np.load(a.tm_cache)
    res, dims = {}, {}
    for f in sorted(glob.glob(a.glob)):
        stem = Path(f).stem[len("gym2s_"):]
        name = stem.split("_")[0]
        d = np.load(f, allow_pickle=True)
        y, pos = np.asarray(d["score"], float), np.asarray(d["pos"])
        ca, ca_wt = d["ca"], np.asarray(d["ca_wt"], float)
        L = len(ca_wt)
        disp, xyz = superposed(ca, ca_wt)

        ii, jj = rebuild_pairs(Path(a.assay_dir) / f"{stem}.csv", 250, 1500, L)
        assert len(ii) == d["disto"].shape[1], f"{name}: pair rebuild mismatch"
        dwt = np.linalg.norm(ca_wt[ii] - ca_wt[jj], axis=-1)
        cam = np.asarray(ca, float)
        ddist = np.linalg.norm(cam[:, ii] - cam[:, jj], axis=-1) - dwt

        if stem not in TM:
            raise SystemExit(f"{name}: no TM in {a.tm_cache}; run "
                             f"precompute_tm.py with the repo venv first")
        tm = np.asarray(TM[stem], float)
        assert len(tm) == len(y), f"{name}: TM cache length {len(tm)} != {len(y)}"
        rich = output_matrix(ca, ca_wt, tm, d["plddt"], d["plddt_site"], pos)
        # The raw geometry blocks carry no confidence information at all, while
        # `rich` includes pLDDT. Without this last block, "extra dimensions did
        # not help the output" could just mean "raw coordinates lack pLDDT".
        # `output all` is every emitted description at once -- geometry at full
        # dimensionality PLUS the hand-built summaries PLUS confidence -- so it
        # is the most generous output side the archives can express, and it
        # removes that escape route.
        blocks = {
            "internal dz (last layer)": np.asarray(d["dz_site"])[:, -1, :],
            "output all (max generosity)": np.column_stack([ddist, xyz, disp, rich]),
            "output pair distances": ddist,
            "output coordinates": xyz,
            "output displacement": disp,
            "output rich (published)": rich,
        }
        dims[name] = {k: int(v.shape[1]) for k, v in blocks.items()}
        for bn, X in blocks.items():
            res.setdefault(bn, {})[name] = evaluate(
                X, y, pos, a.seeds, a.frac, KS)
        print(f"   {name:8s} L={L:3d}  " +
              "  ".join(f"{k.split('(')[0].strip()}={v}" for k, v in dims[name].items()),
              flush=True)

    names = sorted(res["internal dz (last layer)"])
    BN = list(res)
    print("\nHeld-out Spearman, identical protocol on both sides "
          "(components view)\n")
    print(f"   {'block':28s}" + "".join(f"{('k=' + str(k)):>10s}" for k in KS))
    summary = {}
    for bn in BN:
        row = {}
        for ki, k in enumerate(KS):
            g = {n: [float(np.nanmean(res[bn][n][:, ki, 0]))] for n in names}
            pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                       hierarchical=False)
            row[k] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                      "per_assay": {n: g[n][0] for n in names}}
        summary[bn] = row
        print(f"   {bn:28s}" + "".join(f"{row[k]['mean']:>+10.3f}" for k in KS))

    print("\nSelected-features view (each block's own best channels)\n")
    print(f"   {'block':28s}" + "".join(f"{('k=' + str(k)):>10s}" for k in KS))
    summary_sel = {}
    for bn in BN:
        row = {}
        for ki, k in enumerate(KS):
            g = {n: [float(np.nanmean(res[bn][n][:, ki, 1]))] for n in names}
            pt, lo, hi, _ = pi_stats.cluster_bootstrap(g, n_boot=10000, seed=0,
                                                       hierarchical=False)
            row[k] = {"mean": pt, "ci_lo": lo, "ci_hi": hi}
        summary_sel[bn] = row
        print(f"   {bn:28s}" + "".join(f"{row[k]['mean']:>+10.3f}" for k in KS))

    print("\nPaired internal - output, per assay, at pre-specified k\n")
    gaps = {}
    ref = "internal dz (last layer)"
    for bn in BN:
        if bn == ref:
            continue
        for k in (8, 32, 128):  # pre-specified, not each block's own best
            ki = KS.index(k)
            A = {n: [float(np.nanmean(res[ref][n][:, ki, 0]))] for n in names}
            B = {n: [float(np.nanmean(res[bn][n][:, ki, 0]))] for n in names}
            pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(
                A, B, n_boot=10000, seed=0, hierarchical=False)
            wins = sum(1 for n in names if A[n][0] > B[n][0])
            gaps[f"{bn} @ k={k}"] = {"gap": pt, "ci_lo": lo, "ci_hi": hi,
                                     "wins": wins, "n": len(names)}
            flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
            print(f"   vs {bn:26s} k={k:<4d} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"{wins}/{len(names)}{flag}")

    out = {"protocol": {"seeds": a.seeds, "frac": a.frac, "ks": list(KS),
                        "assays": names, "dims": dims,
                        "internal_handicap": "single fixed layer (last), no "
                                             "layer search or averaging"},
           "components_view": summary, "selected_view": summary_sel,
           "paired_gaps": gaps}
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
