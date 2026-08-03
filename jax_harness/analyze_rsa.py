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
  EXP      the measured stability, |DMS_i - DMS_j|

Reading them together:

  RSA(PF[L], EXP)      does the Pairformer's notion of "these two mutations are
                       alike" match the experiment's, and at which depth?
  RSA(STRUCT, EXP)     does the *output* match the experiment?
  RSA(PF[L], STRUCT)   how much of the Pairformer's variant structure is still
                       visible in the coordinates?

Nuisance controls, partialled out of every RSA against EXP, because each would
otherwise masquerade as signal: two substitutions at the same residue share a
structural context and correlate in measured effect; chemically similar
substitutions are alike for reasons unrelated to this model; and both RDMs
inherit the marginal spread of the scores.

FIVE CORRECTIONS from the August 2026 audit; the previous version of this file
had all five, and together they are why its ordering could not carry a headline.

1. *Ranks were tie-breaking permutations.* `np.argsort(np.argsort(x))` assigned
   distinct ranks to tied values in array order. The same-position control is
   binary by construction, so it was the worst affected -- and it is a control
   that gets partialled out of every number here.

2. *Partial Spearman re-ranked its own residuals.* The old code computed
   `spearman(resid(rx), resid(ry))`, which discards the adjustment it just made.
   pi_stats.partial_spearman takes the Pearson correlation of the residuals,
   which is the standard procedure.

3. *Pair entries were treated as independent.* With n variants there are
   n(n-1)/2 entries but only n things that can be permuted, and every variant
   appears in n-1 pairs. Inference is now a Mantel test that permutes VARIANT
   LABELS.

4. *The BLOSUM control did not measure what it claimed.* It used
   |BLOSUM(mut_i) - BLOSUM(mut_j)|, the gap between two tolerance scores, so
   L->A and W->F counted as the same substitution because both score -1. The
   control now compares the residues actually involved (pi_chem.blosum_similarity).

5. *The best layer was chosen and reported on the same data.* The peak of a
   64-layer curve is an optimistic estimate of that layer's value. The layer is
   now chosen on one half of the variants and scored on the other, and the raw
   curve is labelled descriptive.

Also added: a magnitude-sensitive RDM. Cosine distance normalises Delta-z away,
yet the ProteinGym probe finds ||Delta-z|| to be the strongest single feature --
so discarding magnitude was discarding the part that works. Both are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_chem  # noqa: E402
import pi_stats  # noqa: E402
from geom import tm_score  # noqa: E402


def cosine_rdm(V, iu):
    """Directional: which way did the representation move, ignoring how far."""
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return (1.0 - Vn @ Vn.T)[iu]


def euclidean_rdm(V, iu):
    """Magnitude-sensitive: distance between the Delta-z vectors themselves."""
    d = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=-1)
    return d[iu]


def js_rdm(logits, iu):
    """Jensen-Shannon distance between variants' distogram distributions."""
    x = logits - logits.max(-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(-1, keepdims=True)                         # [n, P, bins]
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


def held_out_layer(rdms, exp, ctrl, n, rng):
    """Choose the best layer on half the VARIANTS, score it on the other half.

    Splitting pairs would leak: a pair in the training half and a pair in the
    test half can share a variant. Splitting variants and keeping only the pairs
    whose BOTH members fall on one side keeps the two halves disjoint.
    """
    order = rng.permutation(n)
    a = np.zeros(n, bool)
    a[order[: n // 2]] = True
    iu_i, iu_j = np.triu_indices(n, 1)
    in_a = a[iu_i] & a[iu_j]
    in_b = (~a[iu_i]) & (~a[iu_j])
    if in_a.sum() < 20 or in_b.sum() < 20:
        return None

    def score(mask, L):
        return pi_stats.partial_spearman(rdms[L][mask], exp[mask],
                                         [c[mask] for c in ctrl])

    sel = [score(in_a, L) for L in range(len(rdms))]
    sel = [s if np.isfinite(s) else 0.0 for s in sel]
    best = int(np.argmax(np.abs(sel)))
    return {"selected_layer": best,
            "rho_selection_half": float(sel[best]),
            "rho_held_out_half": float(score(in_b, best)),
            "n_pairs_selection": int(in_a.sum()),
            "n_pairs_held_out": int(in_b.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--max-variants", type=int, default=120,
                    help="structure RDM is O(n^2) TM-aligns; 120 -> ~7k aligns")
    ap.add_argument("--n-perm", type=int, default=2000,
                    help="Mantel permutations of variant labels")
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
        pls = d["plddt_site"][sel]
        L = int(d["n_layers"])
        wt_seq = str(d["wt_seq"])
        iu = np.triu_indices(n, 1)
        mut = [str(m) for m in mut]

        exp = np.abs(y[:, None] - y[None, :])[iu]
        same_pos = (pos[:, None] == pos[None, :]).astype(float)[iu]
        # a real substitution-similarity control, converted to a dissimilarity
        chem = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                chem[i, j] = chem[j, i] = -pi_chem.blosum_similarity(mut[i], mut[j])
        chem = chem[iu]
        mag = (np.abs(y)[:, None] + np.abs(y)[None, :])[iu]
        ctrl = [same_pos, chem, mag]

        rdm_struct = struct_rdm(ca, wt_seq, iu)
        rdm_disto = js_rdm(dis, iu)
        cos_rdms = [cosine_rdm(dz[:, l, :], iu) for l in range(L)]
        euc_rdms = [euclidean_rdm(dz[:, l, :], iu) for l in range(L)]

        pf_exp = np.array([pi_stats.partial_spearman(r, exp, ctrl) for r in cos_rdms])
        pf_exp_e = np.array([pi_stats.partial_spearman(r, exp, ctrl) for r in euc_rdms])
        pf_struct = np.array([pi_stats.spearman(r, rdm_struct) for r in cos_rdms])

        rng = np.random.default_rng(0)
        ho_cos = held_out_layer(cos_rdms, exp, ctrl, n, rng)
        ho_euc = held_out_layer(euc_rdms, exp, ctrl, n, np.random.default_rng(0))

        # Mantel inference on the endpoints that carry the comparison
        st_obs, st_p, st_sd = pi_stats.mantel_permutation(
            rdm_struct, exp, n, iu, covars=ctrl, n_perm=a.n_perm, seed=0)
        di_obs, di_p, di_sd = pi_stats.mantel_permutation(
            rdm_disto, exp, n, iu, covars=ctrl, n_perm=a.n_perm, seed=0)
        pl_obs, pl_p, _ = pi_stats.mantel_permutation(
            np.abs(pls[:, None] - pls[None, :])[iu], exp, n, iu, covars=ctrl,
            n_perm=a.n_perm, seed=0)
        pf_obs, pf_p, pf_sd = (float("nan"),) * 3
        if ho_cos:
            pf_obs, pf_p, pf_sd = pi_stats.mantel_permutation(
                cos_rdms[ho_cos["selected_layer"]], exp, n, iu, covars=ctrl,
                n_perm=a.n_perm, seed=0)

        def cv(v):
            return float(np.std(v) / (np.mean(v) + 1e-12))

        results[name] = {
            "n": int(n), "n_layers": L, "n_pairs": int(len(iu[0])),
            "target": "ProteinGym DMS_score",
            "controls": ["same position", "BLOSUM substitution similarity",
                         "marginal |DMS| spread"],
            "descriptive_curve_cosine": [float(v) for v in pf_exp],
            "descriptive_curve_euclidean": [float(v) for v in pf_exp_e],
            "rsa_pf_vs_struct": [float(v) for v in pf_struct],
            "held_out_layer_cosine": ho_cos,
            "held_out_layer_euclidean": ho_euc,
            "rsa_struct_vs_exp": {"rho": st_obs, "mantel_p": st_p, "null_sd": st_sd},
            "rsa_disto_vs_exp": {"rho": di_obs, "mantel_p": di_p, "null_sd": di_sd},
            "rsa_plddt_vs_exp": {"rho": pl_obs, "mantel_p": pl_p},
            "rsa_pf_selected_vs_exp": {"rho": pf_obs, "mantel_p": pf_p,
                                       "null_sd": pf_sd},
            "cv_struct": cv(rdm_struct), "cv_disto": cv(rdm_disto),
            "tm_variant_to_variant_mean": float(1.0 - np.mean(rdm_struct)),
        }

        r = results[name]
        print(f"\n=== {name} ===  n={n} variants, {len(iu[0])} pairs "
              f"(only {n} independent units)")
        print("  matching the EXPERIMENT (partial: same-position, BLOSUM "
              "similarity, |DMS| spread)")
        if ho_cos:
            print(f"    Pairformer cosine, layer L{ho_cos['selected_layer']:2d} "
                  f"chosen on half the variants:")
            print(f"        selection half {ho_cos['rho_selection_half']:+.3f}   "
                  f"HELD-OUT half {ho_cos['rho_held_out_half']:+.3f}")
        if ho_euc:
            print(f"    Pairformer euclidean, layer L{ho_euc['selected_layer']:2d}: "
                  f"held-out {ho_euc['rho_held_out_half']:+.3f}")
        print(f"    distogram (trunk output)     : {di_obs:+.3f}  "
              f"Mantel p={di_p:.4f}")
        print(f"    predicted STRUCTURE          : {st_obs:+.3f}  "
              f"Mantel p={st_p:.4f}")
        print(f"    pLDDT at mutated residue     : {pl_obs:+.3f}  "
              f"Mantel p={pl_p:.4f}")
        print(f"    descriptive curve peak (cos) : {np.nanmax(np.abs(pf_exp)):+.3f} "
              f"at L{int(np.nanargmax(np.abs(pf_exp)))} "
              f"-- selected AND scored on all pairs, so optimistic")
        print(f"  mean variant-to-variant TM = {r['tm_variant_to_variant_mean']:.4f} "
              f"(1.0 = all variants predicted identical)")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")
    print("\nNOTE: the descriptive curves are selected and scored on the same "
          "pairs and are\n  not effect-size estimates. Use the held-out-layer "
          "values for any comparison.")


if __name__ == "__main__":
    main()
