"""Amino-acid substitution chemistry, as baseline features.

Position-grouped splits stop a probe from memorising which SITES are
intolerant, but they do nothing about the other trivial route to a good
stability score: most of the variance in a deep mutational scan is explained by
what the substitution *is*. Proline anywhere in a helix, charge buried in a
core, a large residue into a small pocket -- none of that requires a folding
model to predict.

So "the Pairformer's internal state predicts stability" is only a claim about
the model once it is shown to beat these. The audit made this the deciding
baseline for the ProteinGym result, and it is the honest one: if a ridge on
BLOSUM score and hydropathy change does as well as 256 internal features, the
internal features are re-encoding substitution chemistry, not contributing
structural knowledge.

BLOSUM62 as published; Kyte-Doolittle hydropathy; Zamyatnin residue volumes in
cubic angstroms; charge at pH 7 with histidine at +0.1 to reflect partial
protonation rather than forcing it to 0 or 1.
"""

from __future__ import annotations

import numpy as np

AA = "ARNDCQEGHILKMFPSTWYV"
IDX = {a: i for i, a in enumerate(AA)}

BLOSUM62 = np.array([
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]],
    dtype=float)

HYDROPATHY = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
              "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
              "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
              "Y": -1.3, "V": 4.2}

VOLUME = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
          "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
          "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
          "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}

CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}

# The properties that make a substitution disruptive for reasons having nothing
# to do with where it sits: introducing a helix-breaker, losing a crosslink.
SPECIAL = {"P": "proline", "G": "glycine", "C": "cysteine"}

CHEM_FEATURES = ["blosum", "d_hydropathy", "abs_d_hydropathy", "d_volume",
                 "abs_d_volume", "d_charge", "abs_d_charge",
                 "wt_hydropathy", "mut_hydropathy", "wt_volume", "mut_volume",
                 "to_proline", "from_glycine", "from_proline", "to_glycine",
                 "from_cysteine", "to_cysteine"]


def parse(mutant: str):
    """'L40P' -> ('L', 40, 'P'). Returns (None, None, None) if unparseable."""
    m = str(mutant).strip()
    if len(m) < 3 or m[0] not in IDX or m[-1] not in IDX:
        return None, None, None
    try:
        return m[0], int(m[1:-1]), m[-1]
    except ValueError:
        return None, None, None


def blosum(wt: str, mut: str) -> float:
    if wt not in IDX or mut not in IDX:
        return 0.0
    return float(BLOSUM62[IDX[wt], IDX[mut]])


def chem_matrix(mutants) -> np.ndarray:
    """[n, len(CHEM_FEATURES)] substitution-chemistry design matrix.

    Signed and absolute versions of each contrast are both included: the sign
    matters for direction of effect, the magnitude for how disruptive the swap
    is, and a linear model cannot recover one from the other.
    """
    rows = []
    for mt in mutants:
        w, _, m = parse(mt)
        if w is None:
            rows.append(np.zeros(len(CHEM_FEATURES)))
            continue
        dh = HYDROPATHY[m] - HYDROPATHY[w]
        dv = VOLUME[m] - VOLUME[w]
        dq = CHARGE.get(m, 0.0) - CHARGE.get(w, 0.0)
        rows.append(np.array([
            blosum(w, m), dh, abs(dh), dv, abs(dv), dq, abs(dq),
            HYDROPATHY[w], HYDROPATHY[m], VOLUME[w], VOLUME[m],
            float(m == "P"), float(w == "G"), float(w == "P"), float(m == "G"),
            float(w == "C"), float(m == "C"),
        ]))
    return np.asarray(rows, dtype=float)


def identity_matrix(mutants) -> np.ndarray:
    """[n, 40] one-hot of wild-type and mutant amino acid identity.

    The most permissive chemistry baseline: no hand-chosen property scale, just
    'which residue became which'. If the internal features cannot beat this,
    they are not carrying structural information about the substitution.
    """
    out = np.zeros((len(mutants), 40))
    for i, mt in enumerate(mutants):
        w, _, m = parse(mt)
        if w is None:
            continue
        out[i, IDX[w]] = 1.0
        out[i, 20 + IDX[m]] = 1.0
    return out


def blosum_similarity(mut_a, mut_b) -> float:
    """How alike are two SUBSTITUTIONS -- not the gap between their scores.

    The RSA control previously used |BLOSUM(a) - BLOSUM(b)|, which says two
    substitutions are identical whenever they happen to be equally tolerated.
    L->A and W->F both score -1 under that measure and are called the same
    mutation. Comparing the residues actually involved is the intended control.
    """
    wa, _, ma = parse(mut_a)
    wb, _, mb = parse(mut_b)
    if wa is None or wb is None:
        return 0.0
    return 0.5 * (blosum(wa, wb) + blosum(ma, mb))
