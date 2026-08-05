"""Two-state conformational references, in the distogram's own coordinates.

The ProteinGym probe asks how far a mutant's internal state moved from wild type.
KL answers that with a magnitude and no direction. This module supplies the
direction: two experimentally determined conformations of the SAME sequence,
binned into the exact scheme Boltz-2's distogram head is trained on, so that a
mutant's shift can be projected onto the axis between them.

Why the reference comes from experiment and not from a second forward pass.
The trunk runs once per sequence; every diffusion sample is conditioned on the
same `z`, so two samples give identical per-layer features and cannot supply two
internal states. The only ways to make the trunk itself produce a second
conformation -- MSA subsampling, templates -- change the input, which
reintroduces exactly the confound the rest of this project spends its effort
eliminating (see `exp_gym.graft_a3m`: the alignment is held identical across
every variant in an assay). Experimental structures cost no forward pass and
carry no MSA.

Binning matches `boltz.data.feature.featurizerv2` exactly:

    boundaries = linspace(2.0, 22.0, 63)          # 63 edges -> 64 bins
    bin        = (d > boundaries).sum(-1)         # bin 0: d<=2 A, bin 63: d>22 A
    atom       = CB, and CA for glycine           # const.res_to_disto_atom

Two consequences of that scheme are easy to miss and both bound the experiment.
Distances above 22 A land in the last bin in BOTH states, so a pair that is
far apart in each conformation carries no signal however much it moved -- for a
187-residue protein that silently removes 40 % of pairs. And the bin width is
0.32 A, which is below the coordinate uncertainty of most NMR ensembles, so a
one-hot reference built from a single model would claim a precision the data
does not have.

The second point is why a reference here is a DISTRIBUTION, not a one-hot. For a
20-model NMR ensemble every model is binned and the results averaged, giving a
per-pair distribution whose width is the ensemble's real spread. That is the same
kind of object the distogram head emits, which is what makes the comparison
meaningful rather than merely dimensionally valid.

Selecting the pairs that carry the axis needs two criteria, not one, and the
reason is the bin width again. Total-variation distance between the two
reference distributions says whether the states are DISTINGUISHABLE at a pair;
it is the natural measure because it already lives in distribution space. But at
0.32 A per bin it saturates almost immediately -- a pair that moves 1 A has
disjoint support and scores TV = 1.0, the same as one that moves 15 A. Measured
here the median TV is 1.00 for all four systems, so TV alone selects roughly
everything and buys no signal-to-noise. It is kept as the distinguishability
gate and paired with a floor on |E[d_A] - E[d_B]|, which is what actually
separates a conformational change from sub-angstrom jitter.

Single-model entries get smoothed. Four of the eight structures deposit one
model (X-ray, or an NMR minimised average), and binning one model produces a
one-hot reference asserting the distance is known to 0.32 A. It is not: crystal
coordinate error at 2.1-2.7 A resolution puts distance uncertainty near 0.5 A.
Those references are convolved with a Gaussian of `sigma_ang` before use, so a
one-model state does not enter the axis looking more certain than a 20-model
ensemble.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

BOLTZ_MIN, BOLTZ_MAX, BOLTZ_BINS = 2.0, 22.0, 64
BOUNDARIES = np.linspace(BOLTZ_MIN, BOLTZ_MAX, BOLTZ_BINS - 1)

# Same fold, two states. Chain choices: the most complete protomer, and for
# 1KLQ chain A only -- chain B is the bound Mad1 peptide, not Mad2.
PAIRS = {
    "XCL1": {
        "a": ("1J8I", "A", "Ltn10 chemokine fold"),
        "b": ("2JP1", "A", "Ltn40 beta-sandwich dimer"),
        "note": "same construct, genuine equilibrium; W55D locks Ltn10, CC3 locks Ltn40",
    },
    "RfaH": {
        "a": ("2OUG", "A", "CTD alpha-helical (autoinhibited)"),
        "b": ("2LCL", "A", "CTD beta-barrel (released)"),
        "note": "axis defined on the CTD only; 2LCL carries a GAM expression tag",
    },
    "Mad2": {
        "a": ("1DUJ", "A", "O-Mad2 open"),
        "b": ("1KLQ", "A", "C-Mad2 closed"),
        "note": "1KLQ chain B is the Mad1 peptide and is excluded",
    },
    "KaiB": {
        "a": ("2QKE", "B", "ground state (thioredoxin-like)"),
        "b": ("5JYT", "A", "fold-switched"),
        # 5JYT is truncated after D99 and carries a C-terminal FLAG tag, so
        # residues above 99 are tag rather than KaiB. Dropping sequence
        # mismatches is NOT sufficient: tag residues 103 and 104 happen to match
        # KaiB's own residues at those numbers and survived into the axis until
        # this range was imposed. The locking substitutions themselves are
        # Y8A, N29A, G89A, D91R, Y94A.
        "res_range": (1, 99),
        "note": "5JYT = fold-switch-locked variant (Y8A/N29A/G89A/D91R/Y94A), "
                "truncated after D99 with a FLAG tag; axis capped at 99",
    },
}


def bin_distances(d):
    """Boltz's exact binning: index = number of boundaries the distance exceeds."""
    return (d[..., None] > BOUNDARIES).sum(-1)


def bin_centers():
    """Distance in angstroms represented by each of the 64 bins.

    Bins 0 and 63 are open-ended (d <= 2 A, d > 22 A); they are given the centre
    they would have if the grid continued, which is the least-bad finite choice
    and only matters for pairs whose mass sits in a tail.
    """
    w = BOUNDARIES[1] - BOUNDARIES[0]
    return np.concatenate([[BOUNDARIES[0] - w / 2],
                           (BOUNDARIES[:-1] + BOUNDARIES[1:]) / 2,
                           [BOUNDARIES[-1] + w / 2]])


def moments(p, centers=None):
    """Mean distance and spread, in angstroms, of a distribution over bins."""
    c = bin_centers() if centers is None else centers
    mu = (p * c).sum(-1)
    var = (p * (c - mu[..., None]) ** 2).sum(-1)
    return mu, np.sqrt(np.maximum(var, 1e-12))


def jeffreys_split(mu_w, sd_w, mu_m, sd_m):
    """Split a Jeffreys divergence into relocation and broadening.

    For two one-dimensional normals the symmetric KL separates additively with
    no cross term:

        J = [ sm^2/(2 sw^2) + sw^2/(2 sm^2) - 1 ]    SPREAD, zero iff sw == sm
          + [ d^2 (1/(2 sw^2) + 1/(2 sm^2)) ]        SHIFT,  zero iff mu equal

    Both terms are non-negative and each vanishes exactly when its own effect is
    absent, so "share of the divergence due to relocation" is a real quantity
    rather than the output of a regression. Distograms are not Gaussian, so
    callers must check the split against the true divergence and report the
    agreement -- see `analyze_klshape`.
    """
    vw, vm = sd_w ** 2, sd_m ** 2
    spread = vm / (2 * vw) + vw / (2 * vm) - 1.0
    shift = (mu_m - mu_w) ** 2 * (1.0 / (2 * vw) + 1.0 / (2 * vm))
    return shift, spread


def load_state(cif, chain, atom="CB"):
    """{resnum: {'aa', 'xyz' [n_models,3]}} at CB (CA for glycine), plus metadata.

    `atom="CA"` gives backbone carbons instead, which is what the structure
    module emits; the distogram references need CB, but comparing a PREDICTED
    structure against these conformations needs CA on both sides.

    Residues absent from any model are dropped rather than imputed -- a partially
    observed residue would otherwise contribute a distance averaged over a
    changing set of conformers.
    """
    warnings.filterwarnings("ignore")
    from Bio.PDB import MMCIFParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.PDB.Polypeptide import protein_letters_3to1 as T31

    st = MMCIFParser(QUIET=True).get_structure("s", cif)
    models = list(st)
    per = {}
    for m in models:
        for r in m[chain]:
            if r.id[0] != " ":
                continue
            aa = T31.get(r.get_resname().capitalize()) or T31.get(r.get_resname())
            if aa is None:
                continue
            at = atom if atom in r else ("CA" if "CA" in r else None)
            if at is None:
                continue
            per.setdefault(r.id[1], {"aa": aa, "xyz": []})["xyz"].append(r[at].coord)

    n = len(models)
    out = {k: {"aa": v["aa"], "xyz": np.array(v["xyz"], float)}
           for k, v in per.items() if len(v["xyz"]) == n}
    d = MMCIF2Dict(cif)
    seq = "".join(d.get("_entity_poly.pdbx_seq_one_letter_code_can", [""])[0].split())
    # Author residue numbers are not sequence positions -- 1DUJ's chain starts at
    # author 11 -- so carry the crystallographers' own mapping rather than an
    # offset guessed from the first residue.
    amap = {}
    if "_pdbx_poly_seq_scheme.seq_id" in d:
        for sid, num, strand in zip(d["_pdbx_poly_seq_scheme.seq_id"],
                                    d["_pdbx_poly_seq_scheme.pdb_seq_num"],
                                    d["_pdbx_poly_seq_scheme.pdb_strand_id"]):
            if strand == chain and num not in (".", "?"):
                amap[int(num)] = int(sid) - 1          # 0-based into entity_seq
    meta = {"n_models": n, "dropped": len(per) - len(out),
            "method": ";".join(d.get("_exptl.method", ["?"])),
            "title": d.get("_struct.title", ["?"])[0],
            "entity_seq": seq, "auth_to_index": amap}
    return out, meta


def reference_distogram(state, resnums, sigma_ang=0.5):
    """[n, n, 64] -- each model binned, averaged, then blurred by coordinate error.

    `sigma_ang` is the assumed per-distance uncertainty in angstroms; it is
    converted to bins and applied along the bin axis. Passing 0 disables it.
    """
    X = np.stack([np.array([state[r]["xyz"][m] for r in resnums])
                  for m in range(state[resnums[0]]["xyz"].shape[0])])   # [M,n,3]
    D = np.linalg.norm(X[:, :, None, :] - X[:, None, :, :], axis=-1)    # [M,n,n]
    b = bin_distances(D)                                                # [M,n,n]
    P = np.zeros(b.shape[1:] + (BOLTZ_BINS,))
    for m in range(b.shape[0]):
        np.add.at(P, (*np.indices(b.shape[1:]), b[m]), 1.0)
    P /= b.shape[0]
    if sigma_ang > 0:
        from scipy.ndimage import gaussian_filter1d
        width = (BOLTZ_MAX - BOLTZ_MIN) / (BOLTZ_BINS - 2)   # 0.323 A per bin
        P = gaussian_filter1d(P, sigma_ang / width, axis=-1, mode="nearest")
        P /= P.sum(-1, keepdims=True)
    return P, D


def build_pair(fam, root, spec=None, sigma_ang=0.5):
    """Align two states on shared numbering and return binned references."""
    spec = spec or PAIRS[fam]
    root = Path(root)
    (pa, ca, la), (pb, cb, lb) = spec["a"], spec["b"]
    A, mA = load_state(root / f"{pa}.cif", ca)
    B, mB = load_state(root / f"{pb}.cif", cb)

    shared = sorted(set(A) & set(B))
    # An explicit range where one entry contains non-native residues (tags,
    # truncations). Matching amino acids alone does not exclude them.
    lo, hi = spec.get("res_range", (-10**9, 10**9))
    shared = [r for r in shared if lo <= r <= hi]
    mismatch = [(r, A[r]["aa"], B[r]["aa"]) for r in shared if A[r]["aa"] != B[r]["aa"]]
    keep = [r for r in shared if A[r]["aa"] == B[r]["aa"]]
    if len(keep) < 10:
        raise ValueError(f"{fam}: only {len(keep)} matched residues")

    PA, DA = reference_distogram(A, keep, sigma_ang)
    PB, DB = reference_distogram(B, keep, sigma_ang)

    # CA distances for the same residues, so a predicted structure can be scored
    # against both states on the same footing as the distogram is.
    Aca, _ = load_state(root / f"{pa}.cif", ca, atom="CA")
    Bca, _ = load_state(root / f"{pb}.cif", cb, atom="CA")
    ca_ok = [r for r in keep if r in Aca and r in Bca]
    def _ca(S):
        X = np.stack([np.array([S[r]["xyz"][m] for r in ca_ok])
                      for m in range(S[ca_ok[0]]["xyz"].shape[0])])
        return np.linalg.norm(X[:, :, None] - X[:, None, :], axis=-1).mean(0)
    d_a_ca, d_b_ca = _ca(Aca), _ca(Bca)
    tv = 0.5 * np.abs(PA - PB).sum(-1)                  # [n,n] in [0,1]
    dA, dB = DA.mean(0), DB.mean(0)
    representable = (dA < BOLTZ_MAX) | (dB < BOLTZ_MAX)
    n = len(keep)
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

    # Where each axis residue sits in the sequence we will actually fold. State A
    # is the native construct in all four systems, so its entity sequence is the
    # fold target and its numbering defines the token indices.
    seq = "".join(A[r]["aa"] for r in keep)
    fold_seq = mA["entity_seq"]
    seq_index = np.array([mA["auth_to_index"][r] for r in keep])
    bad = [(r, i, seq[j], fold_seq[i]) for j, (r, i) in enumerate(zip(keep, seq_index))
           if fold_seq[i] != seq[j]]
    if bad:
        raise ValueError(f"{fam}: numbering map disagrees with residues: {bad[:5]}")

    return {
        "family": fam, "resnums": np.array(keep), "n": n,
        "seq": seq, "fold_seq": fold_seq, "seq_index": seq_index,
        "state_a": pa, "state_b": pb, "chain_a": ca, "chain_b": cb,
        "label_a": la, "label_b": lb, "note": spec["note"],
        "meta_a": mA, "meta_b": mB, "mismatch": mismatch,
        "P_a": PA, "P_b": PB, "d_a": dA, "d_b": dB,
        "d_a_ca": d_a_ca, "d_b_ca": d_b_ca, "ca_resnums": np.array(ca_ok),
        "tv": tv, "representable": representable, "sep": sep,
        # the axis a mutant's shift gets projected onto
        "axis": PA - PB,
    }


def load_reference(path):
    """Reload a saved conf_*.npz with the mask already applied."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def discriminating_mask(pair, tv_min=0.8, dmin_ang=2.0, min_sep=3):
    """Pairs that separate the two states, by a real distance, that the head can express.

    Both thresholds are load-bearing: TV alone saturates at this bin width and
    `dmin_ang` alone would admit pairs whose two states are far apart in the mean
    but overlapping across the ensemble.
    """
    return ((pair["tv"] >= tv_min)
            & (np.abs(pair["d_a"] - pair["d_b"]) >= dmin_ang)
            & pair["representable"]
            & (pair["sep"] >= min_sep))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/"
                                      "tbush/prot_interp_files/data/conformations")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tv-min", type=float, default=0.8)
    ap.add_argument("--dmin", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=0.5)
    a = ap.parse_args()
    out = Path(a.out or Path(a.root) / "refs")
    out.mkdir(parents=True, exist_ok=True)

    hdr = (f"{'family':7s} {'res':>4s} {'drop':>5s} {'models':>12s} {'pairs':>7s} "
           f"{'repr':>6s} {'axis':>7s} {'%':>6s} {'|dd| med':>9s} {'max':>6s} "
           f"{'fold':>6s}")
    print(hdr + "\n" + "-" * len(hdr))
    for fam in PAIRS:
        p = build_pair(fam, a.root, sigma_ang=a.sigma)
        m = discriminating_mask(p, a.tv_min, a.dmin)
        iu = np.triu_indices(p["n"], k=3)
        rep, sel = p["representable"][iu], m[iu]
        dd = np.abs(p["d_a"] - p["d_b"])[iu]
        np.savez_compressed(out / f"conf_{fam}.npz", mask=m, **{
            k: v for k, v in p.items()
            if isinstance(v, (np.ndarray, str, int)) or k in ("n",)})
        (out / f"{fam}.fasta").write_text(
            f">{fam}|{p['state_a']}_{p['label_a']}|fold target, axis on "
            f"{p['n']} residues {p['resnums'].min()}-{p['resnums'].max()}\n"
            f"{p['fold_seq']}\n")
        print(f"{fam:7s} {p['n']:4d} {len(p['mismatch']):5d} "
              f"{p['meta_a']['n_models']:5d}/{p['meta_b']['n_models']:<6d} "
              f"{len(dd):7d} {100*rep.mean():5.1f}% {sel.sum():7d} "
              f"{100*sel.mean():5.1f}% {np.median(dd[sel]):9.1f} {dd.max():6.1f} "
              f"{len(p['fold_seq']):6d}")
    print(f"\naxis = pairs with TV>={a.tv_min}, |d_a-d_b|>={a.dmin} A, "
          f"representable, |i-j|>=3;  sigma={a.sigma} A")
    print(f"wrote {out}/conf_*.npz and {out}/*.fasta")
