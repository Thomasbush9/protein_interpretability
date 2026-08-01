"""RSA: how the Pairformer separates WT from mutant, and how much survives the structure module.

The question is not "can we score ProteinGym". It is: the Pairformer builds some
picture of what a mutation does; how much of that picture is still there in the
predicted structure, and how much gets flattened away?

Four representational spaces, each giving a variant x variant dissimilarity
matrix over the *same* set of single mutants:

  PF[L]    pair-representation footprint of the mutation at its own residue,
           Delta-z = z_mut - z_wt, one RDM per Pairformer layer (64 of them)
  DISTO    the trunk's distogram at its output -- the last thing before the
           structure module, expressed as a distribution
  STRUCT   the predicted CA coordinates, pairwise TM-score between variants
  EXP      the measured stability, |dG_i - dG_j|

Reading them together:

  RSA(PF[L], EXP)      does the Pairformer's notion of "these two mutations are
                       alike" match the experiment's, and at which depth?
  RSA(STRUCT, EXP)     does the *output* match the experiment?
  RSA(PF[L], STRUCT)   how much of the Pairformer's variant structure is still
                       visible in the coordinates?
  spread ratios        the blunt version: how much variant-to-variant variation
                       exists in each space at all, relative to its own scale.

If RSA(PF, EXP) is substantially higher than RSA(STRUCT, EXP), the trunk holds a
picture of the mutation that the structure module discards -- which is the
"correction" this analysis is meant to show, stated as a number.

Nuisance controls, partialled out of every RSA against EXP, because each would
otherwise masquerade as signal: two substitutions at the same residue share a
structural context and correlate in measured effect; chemically similar
substitutions are alike for reasons unrelated to this model; and both RDMs
inherit the marginal spread of the scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from geom import tm_score  # noqa: E402

AA = "ARNDCQEGHILKMFPSTWYV"
B62 = np.array([
[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
[-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
[-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
[-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
[ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
[-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
[-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
[ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
[-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
[-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
[-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
[-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
[-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
[-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
[ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
[-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4]], float)
IDX = {a: i for i, a in enumerate(AA)}


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def partial_spearman(x, y, Zs):
    rx, ry = rank(x), rank(y)
    Z = np.column_stack([rank(z) for z in Zs] + [np.ones(len(x))])

    def resid(v):
        c, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ c

    return spearman(resid(rx), resid(ry))


def cosine_rdm(V, iu):
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return (1.0 - Vn @ Vn.T)[iu]


def js_rdm(logits, iu):
    """Jensen-Shannon distance between variants' distogram distributions."""
    x = logits - logits.max(-1, keepdims=True)
    p = np.exp(x); p /= p.sum(-1, keepdims=True)          # [n, P, bins]
    n = len(p)
    out = np.zeros((n, n))
    for i in range(n):
        m = 0.5 * (p[i][None] + p)                        # [n, P, bins]
        kl1 = (p[i][None] * (np.log(p[i][None] + 1e-12) - np.log(m + 1e-12))).sum(-1)
        kl2 = (p * (np.log(p + 1e-12) - np.log(m + 1e-12))).sum(-1)
        out[i] = (0.5 * (kl1 + kl2)).mean(-1)
    return ((out + out.T) / 2)[iu]


def struct_rdm(ca, seq, iu):
    n = len(ca)
    out = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = 1.0 - tm_score(ca[i], ca[j], seq, seq)
    np.fill_diagonal(out, 0.0)
    return out[iu]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--max-variants", type=int, default=120,
                    help="structure RDM is O(n^2) TM-aligns; 120 -> ~7k aligns")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    results = {}
    for f in a.features:
        d = np.load(f, allow_pickle=True)
        name = Path(f).stem.replace("gym2_", "")
        n_all = len(d["score"])
        n = min(n_all, a.max_variants)
        sel = np.sort(np.random.default_rng(0).choice(n_all, n, replace=False))
        y, pos, mut = d["score"][sel], d["pos"][sel], d["mutant"][sel]
        dz, ca, dis = d["dz_site"][sel], d["ca"][sel], d["disto"][sel]
        pl, pls = d["plddt"][sel], d["plddt_site"][sel]
        L = int(d["n_layers"]); wt_seq = str(d["wt_seq"])
        iu = np.triu_indices(n, 1)

        exp = np.abs(y[:, None] - y[None, :])[iu]
        same_pos = (pos[:, None] == pos[None, :]).astype(float)[iu]
        w_aa = np.array([m[0] for m in mut]); m_aa = np.array([m[-1] for m in mut])
        bl = np.array([B62[IDX.get(w, 0), IDX.get(m, 0)] for w, m in zip(w_aa, m_aa)])
        blo = np.abs(bl[:, None] - bl[None, :])[iu]
        mag = (np.abs(y)[:, None] + np.abs(y)[None, :])[iu]
        ctrl = [same_pos, blo, mag]

        pf_exp, pf_struct = [], []
        rdm_struct = struct_rdm(ca, wt_seq, iu)
        rdm_disto = js_rdm(dis, iu)
        for l in range(L):
            r = cosine_rdm(dz[:, l, :], iu)
            pf_exp.append(partial_spearman(r, exp, ctrl))
            pf_struct.append(spearman(r, rdm_struct))
        pf_exp, pf_struct = np.array(pf_exp), np.array(pf_struct)

        rsa_struct_exp = partial_spearman(rdm_struct, exp, ctrl)
        rsa_disto_exp = partial_spearman(rdm_disto, exp, ctrl)
        rsa_plddt_exp = partial_spearman(
            np.abs(pls[:, None] - pls[None, :])[iu], exp, ctrl)

        # blunt spread: how much variant-to-variant variation each space holds,
        # normalised so spaces with different units are comparable
        def cv(v):
            return float(np.std(v) / (np.mean(v) + 1e-12))

        results[name] = {
            "n": int(n), "n_layers": L,
            "rsa_pf_vs_exp": [float(v) for v in pf_exp],
            "rsa_pf_vs_struct": [float(v) for v in pf_struct],
            "rsa_struct_vs_exp": rsa_struct_exp,
            "rsa_disto_vs_exp": rsa_disto_exp,
            "rsa_plddt_vs_exp": rsa_plddt_exp,
            "best_pf_layer": int(np.argmax(pf_exp)),
            "best_pf_vs_exp": float(pf_exp.max()),
            "final_pf_vs_exp": float(pf_exp[-1]),
            "cv_struct": cv(rdm_struct), "cv_disto": cv(rdm_disto),
            "cv_pf_best": cv(cosine_rdm(dz[:, int(np.argmax(pf_exp)), :], iu)),
            "tm_variant_to_variant_mean": float(1.0 - np.mean(rdm_struct)),
        }
        r = results[name]
        print(f"\n=== {name} ===  n={n} variants, {len(iu[0])} pairs")
        print("  matching the EXPERIMENT (partial: same-position, BLOSUM, |dG|)")
        print(f"    Pairformer, best layer L{r['best_pf_layer']:2d} : {r['best_pf_vs_exp']:+.3f}")
        print(f"    Pairformer, final layer      : {r['final_pf_vs_exp']:+.3f}")
        print(f"    distogram (trunk output)     : {rsa_disto_exp:+.3f}")
        print(f"    predicted STRUCTURE          : {rsa_struct_exp:+.3f}")
        print(f"    pLDDT at mutated residue     : {rsa_plddt_exp:+.3f}")
        print("  how much Pairformer structure survives into coordinates")
        print(f"    RSA(PF[best], STRUCT) = {pf_struct[int(np.argmax(pf_exp))]:+.3f}   "
              f"RSA(PF[final], STRUCT) = {pf_struct[-1]:+.3f}")
        print(f"  mean variant-to-variant TM = {r['tm_variant_to_variant_mean']:.4f} "
              f"(1.0 = all variants predicted identical)")
        print(f"  spread (CV): pairformer {r['cv_pf_best']:.3f}  "
              f"distogram {r['cv_disto']:.3f}  structure {r['cv_struct']:.3f}")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
