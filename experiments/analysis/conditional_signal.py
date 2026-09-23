"""Does the internal representation add ANYTHING beyond chemistry and context?

Gate 1B of the 2026-09-06 audit, and the question the whole paper turns on. The
project has already shown the trunk beats a 17-descriptor chemistry baseline and
beats emitted geometry. Neither answers the sharper question: a mutation
representation can encode substitution identity, evolutionary surprise, and the
wild-type environment without carrying anything a folding model uniquely knows.
So the test is not another race against a weak baseline -- it is INCREMENTAL:

    B          a prespecified chemistry/context baseline
    B + trunk  the same, plus a reduced internal representation

and the claim survives only if the second beats the first on held-out proteins.

B, PRESPECIFIED, four blocks (the audit names all four):

  chem17     the existing substitution descriptors
  ident40    separate wild-type and mutant one-hot. NOTE these are ADDITIVE in
             a linear model: they represent a(wt) + b(mut), NOT all 380 ordered
             substitutions. The attribution code once claimed otherwise.
  ordered    the ordered substitution itself, wt->mut, as a 380-column
             indicator. Regularized, never selected. This is what "substitution
             identity" actually means, and it subsumes ident40.
  context    evolutionary and structural environment of the SITE, from inputs
             the folding model also sees but which are not its representation:
               - alignment column entropy and gap fraction at the site
               - frequency of the WT and of the MUTANT residue in that column
                 (MSA compatibility -- "is this substitution surprising here")
               - log N_eff of the column
               - wild-type burial: CA neighbours within 8 A and 12 A, and the
                 site's distance from the centroid
               - relative position along the chain
             plus two prespecified interactions: (hydropathy change x burial)
             and (volume change x burial), because a buried polar substitution
             is the textbook case where chemistry alone is not enough.

EVERYTHING IS FITTED ON TRAINING PROTEINS ONLY. Column statistics come from the
assay's own alignment (an input, never a label). The internal block is reduced
by a PCA fitted on the training assays only, so the held-out protein contributes
nothing to the basis -- the audit's objection to a globally-fitted PC2 in the
chemistry increments.

Reported both ways: transductive (each assay's own feature scale, the protocol
every other number here uses) and inductive (training statistics only).

    uv run python experiments/analysis/conditional_signal.py \
        --out $W/runs/conditional_signal.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax_harness"))

import pi_archive                                              # noqa: E402
import pi_chem                                                 # noqa: E402
import pi_protocol                                             # noqa: E402
import pi_stats                                                # noqa: E402
from protein_interpretability import artifacts                 # noqa: E402
from protein_interpretability.collection import Cohort         # noqa: E402

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
AA = pi_chem.AA
EPS = 1e-9


# ---- the context block ----------------------------------------------------
def read_a3m(path, cap=None):
    """Aligned rows of an a3m: query-length strings, insertions dropped."""
    rows, cur = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if cur:
                rows.append("".join(cur))
                cur = []
            if cap and len(rows) >= cap:
                break
        else:
            cur.append(line.strip())
    if cur and not (cap and len(rows) >= cap):
        rows.append("".join(cur))
    if not rows:
        return []
    q = rows[0]
    keep = [i for i, c in enumerate(q) if c != "-" and not c.islower()]
    out = []
    for r in rows:
        if len(r) < len(q):
            continue
        out.append("".join(r[i].upper() if i < len(r) else "-" for i in keep))
    return out


def column_stats(msa, n_res):
    """Per-position entropy, gap fraction, residue frequencies, log N_eff."""
    ent = np.zeros(n_res); gap = np.zeros(n_res)
    freq = np.zeros((n_res, 20)); neff = np.zeros(n_res)
    if not msa:
        return ent, gap, freq, neff
    A = np.array([list(r) for r in msa])
    for p in range(min(n_res, A.shape[1])):
        col = A[:, p]
        g = float((col == "-").mean())
        aas = col[col != "-"]
        counts = np.array([(aas == c).sum() for c in AA], float)
        tot = counts.sum()
        f = counts / tot if tot else counts
        nz = f[f > 0]
        ent[p] = float(-(nz * np.log(nz)).sum())
        gap[p] = g
        freq[p] = f
        neff[p] = np.log1p(tot)
    return ent, gap, freq, neff


def context_matrix(mutants, pos, ca_wt, msa):
    """[n, 13] evolutionary and structural environment of the mutated site."""
    n_res = len(ca_wt)
    ent, gap, freq, neff = column_stats(msa, n_res)
    d = np.linalg.norm(np.asarray(ca_wt)[:, None] - np.asarray(ca_wt)[None], axis=-1)
    nb8 = (d < 8.0).sum(1) - 1.0
    nb12 = (d < 12.0).sum(1) - 1.0
    cen = np.linalg.norm(np.asarray(ca_wt) - np.asarray(ca_wt).mean(0), axis=1)
    rows = []
    for i, mt in enumerate(mutants):
        w, _, m = pi_chem.parse(mt)
        p = int(pos[i]) if int(pos[i]) < n_res else 0
        fw = freq[p][AA.index(w)] if w in AA else 0.0
        fm = freq[p][AA.index(m)] if m in AA else 0.0
        dh = (pi_chem.HYDROPATHY.get(m, 0.0) - pi_chem.HYDROPATHY.get(w, 0.0))
        dv = (pi_chem.VOLUME.get(m, 0.0) - pi_chem.VOLUME.get(w, 0.0))
        bur = nb8[p]
        rows.append([ent[p], gap[p], fw, fm, np.log(fm + 1e-3), neff[p],
                     nb8[p], nb12[p], cen[p], p / max(n_res - 1, 1),
                     dh * bur, dv * bur, float(w == m)])
    return np.asarray(rows, float)


CONTEXT_FEATURES = ["col_entropy", "col_gap_frac", "freq_wt", "freq_mut",
                    "log_freq_mut", "log_neff", "nb8", "nb12",
                    "dist_centroid", "rel_position", "dhyd_x_burial",
                    "dvol_x_burial", "is_synonymous"]


def ordered_matrix(mutants):
    """[n, 400] the ordered substitution wt->mut. Regularized, never selected."""
    out = np.zeros((len(mutants), 400))
    for i, mt in enumerate(mutants):
        w, _, m = pi_chem.parse(mt)
        if w in AA and m in AA:
            out[i, AA.index(w) * 20 + AA.index(m)] = 1.0
    return out


# ---- probes ---------------------------------------------------------------
def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def loao(blocks, target, lam, inductive=False, pcs=0):
    """Leave-one-assay-out. PCA, when used, is fitted on TRAINING assays only."""
    names = sorted(target)
    rho = {}
    for held in names:
        tr = [n for n in names if n != held]
        Xtr_raw = {n: blocks[n] for n in tr}
        if pcs:
            P = np.concatenate([Xtr_raw[n] for n in tr], 0)
            mu = P.mean(0)
            V = np.linalg.svd(P - mu, full_matrices=False)[2][:pcs]
            Xtr_raw = {n: (blocks[n] - mu) @ V.T for n in tr}
            Xte_raw = (blocks[held] - mu) @ V.T
        else:
            Xte_raw = blocks[held]
        if inductive:
            P = np.concatenate([Xtr_raw[n] for n in tr], 0)
            mu, sd = P.mean(0), P.std(0) + EPS
            Xtr = np.concatenate([(Xtr_raw[n] - mu) / sd for n in tr], 0)
            Xte = (Xte_raw - mu) / sd
        else:
            Xtr = np.concatenate(
                [(Xtr_raw[n] - Xtr_raw[n].mean(0)) / (Xtr_raw[n].std(0) + EPS)
                 for n in tr], 0)
            Xte = (Xte_raw - Xte_raw.mean(0)) / (Xte_raw.std(0) + EPS)
        ytr = np.concatenate([(target[n] - target[n].mean())
                              / (target[n].std() + EPS) for n in tr], 0)
        w = ridge(Xtr, ytr, lam)
        rho[held] = pi_stats.spearman(Xte @ w, target[held])
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="heldout_assays")
    ap.add_argument("--captures", default=str(W / "runs/xmodel_layers"))
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--pcs", type=int, default=2,
                    help="internal PCs, fitted on training assays only")
    ap.add_argument("--msa-cap", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cohort = Cohort.load(a.cohort)
    cohort.verify()

    chem, ident, order, ctx, internal, target = {}, {}, {}, {}, {}, {}
    for assay in cohort:
        p = Path(a.captures) / f"xm_{a.model}_r1_{assay.id}.npz"
        if not p.exists():
            continue
        cap = artifacts.load_capture(p, require_vectors=True)
        key = assay.id.split("_")[0]
        muts = [str(m) for m in cap.field("mutant")]
        pos = np.asarray(cap.field("pos"))
        msa = read_a3m(assay.msa_path, cap=a.msa_cap) if assay.msa_path else []
        chem[key] = pi_chem.chem_matrix(muts)
        ident[key] = pi_chem.identity_matrix(muts)
        order[key] = ordered_matrix(muts)
        ctx[key] = context_matrix(muts, pos, cap.field("ca_wt"), msa)
        internal[key] = cap.pair_row(-1)
        target[key] = np.asarray(cap.field("score"), float)
        print(f"  {key:8s} n={len(muts):4d}  msa rows={len(msa):5d}")

    names = sorted(target)
    if not names:
        raise SystemExit("no captures found")

    def cat(*parts):
        return {k: np.concatenate([p[k] for p in parts], 1) for k in names}

    B = cat(chem, ident, order, ctx)
    BLOCKS = {
        "chem17": chem,
        "context13": ctx,
        "ordered400": order,
        "B (chem+ident+ordered+context)": B,
        "internal128": internal,
    }
    print(f"\n{len(names)} assays, lam={a.lam:g}, "
          f"internal reduced to {a.pcs} training-fold PCs\n")

    out = {"assays": names, "regimes": {}}
    for regime, ind in (("transductive", False), ("inductive", True)):
        rho = {lab: loao(blk, target, a.lam, inductive=ind)
               for lab, blk in BLOCKS.items()}
        rho[f"internal {a.pcs}PC"] = loao(internal, target, a.lam,
                                          inductive=ind, pcs=a.pcs)
        # the incremental tests: B against B plus the trunk
        rho["B + internal128"] = loao(cat(B, internal), target, a.lam,
                                      inductive=ind)
        rho[f"B + internal {a.pcs}PC"] = loao_bplus(
            B, internal, target, a.lam, ind, a.pcs)

        ORDER = list(BLOCKS) + [f"internal {a.pcs}PC", "B + internal128",
                                f"B + internal {a.pcs}PC"]
        print(f"-- {regime}")
        print(f"{'assay':9s}" + "".join(f"{k[:15]:>17s}" for k in ORDER))
        for k in names:
            print(f"{k:9s}" + "".join(f"{rho[b][k]:>+17.3f}" for b in ORDER))
        print(f"{'mean':9s}" + "".join(
            f"{np.mean([rho[b][k] for k in names]):>+17.3f}" for b in ORDER))

        summ, gaps = {}, {}
        for b in ORDER:
            pt, lo, hi, _ = pi_stats.cluster_bootstrap(
                {k: [rho[b][k]] for k in names}, n_boot=10000, seed=0,
                hierarchical=False)
            summ[b] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                       "per_assay": rho[b]}
        print(f"\n   incremental gains over B ({regime}):")
        for aa, bb in ((f"B + internal {a.pcs}PC", "B (chem+ident+ordered+context)"),
                       ("B + internal128", "B (chem+ident+ordered+context)"),
                       ("internal128", "B (chem+ident+ordered+context)"),
                       (f"internal {a.pcs}PC", "chem17")):
            pt, lo, hi, _ = pi_stats.paired_cluster_bootstrap(
                {k: [rho[aa][k]] for k in names},
                {k: [rho[bb][k]] for k in names},
                n_boot=10000, seed=0, hierarchical=False)
            wins = sum(rho[aa][k] > rho[bb][k] for k in names)
            gaps[f"{aa} - {bb}"] = {"gap": pt, "ci_lo": lo, "ci_hi": hi,
                                    "wins": wins, "n": len(names)}
            flag = "" if (lo > 0 or hi < 0) else "   <- includes zero"
            print(f"     {aa[:26]:27s} - {bb[:26]:27s} {pt:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]  {wins}/{len(names)}{flag}")
        out["regimes"][regime] = {"summary": summ, "gaps": gaps}
        print()

    proto = pi_protocol.protocol(
        script="conditional_signal.py",
        design="incremental held-out-protein prediction: a prespecified "
               "chemistry/context baseline B against B plus a reduced internal "
               "representation; PCA and all nuisance fitting on TRAINING "
               "assays only; leave-one-assay-out",
        layer=pi_protocol.layers("final"),
        features={"chem": pi_protocol.features("substitution chemistry", 17),
                  "identity": pi_protocol.features("wt/mut one-hot (ADDITIVE, "
                                                   "not 380 ordered)", 40),
                  "ordered": pi_protocol.features("ordered substitution "
                                                  "wt->mut", 400),
                  "context": pi_protocol.features(
                      "MSA column and WT burial context, 2 prespecified "
                      "interactions", len(CONTEXT_FEATURES)),
                  "internal": pi_protocol.features("dz_vec final pair row",
                                                   128, kept=a.pcs)},
        source=f"{a.captures}/xm_{a.model}_r1_*.npz",
        n_assays=len(names), lam=a.lam, model=a.model,
        normalisation="both transductive and inductive reported",
        note="context statistics come from the assay's own alignment, an "
             "INPUT; no label enters any fit")
    pi_archive.write_result(a.out, out, protocol=proto, indent=1)
    print(f"wrote {a.out}")


def loao_bplus(B, internal, target, lam, inductive, pcs):
    """B concatenated with training-fold PCs of the internal block.

    Separate from `loao` because the PCA must see only the training assays'
    internal features while B passes through whole -- reducing the two blocks
    together would let the held-out protein into the basis.
    """
    names = sorted(target)
    rho = {}
    for held in names:
        tr = [n for n in names if n != held]
        P = np.concatenate([internal[n] for n in tr], 0)
        mu = P.mean(0)
        V = np.linalg.svd(P - mu, full_matrices=False)[2][:pcs]
        blocks = {n: np.concatenate([B[n], (internal[n] - mu) @ V.T], 1)
                  for n in names}
        if inductive:
            Q = np.concatenate([blocks[n] for n in tr], 0)
            m2, s2 = Q.mean(0), Q.std(0) + EPS
            Xtr = np.concatenate([(blocks[n] - m2) / s2 for n in tr], 0)
            Xte = (blocks[held] - m2) / s2
        else:
            Xtr = np.concatenate(
                [(blocks[n] - blocks[n].mean(0)) / (blocks[n].std(0) + EPS)
                 for n in tr], 0)
            Xte = ((blocks[held] - blocks[held].mean(0))
                   / (blocks[held].std(0) + EPS))
        ytr = np.concatenate([(target[n] - target[n].mean())
                              / (target[n].std() + EPS) for n in tr], 0)
        w = ridge(Xtr, ytr, lam)
        rho[held] = pi_stats.spearman(Xte @ w, target[held])
    return rho


if __name__ == "__main__":
    main()
