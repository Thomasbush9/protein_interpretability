"""Does a mutation move the trunk's internal state ALONG a known conformational axis?

Everything else in this project measures how FAR a mutant's internal state moved
from wild type. A symmetric KL has no direction, so "it moved" is all it can say.
Two experimentally determined conformations of the same sequence supply the
missing direction: the mutant's shift can be projected onto the axis between
them, and the projection is signed.

XCL1 is the system because it gives a TWO-SIDED prediction. Both variants are
engineered disulfides with solved structures, and they lock opposite states:

    V21C/V59C   (2HDM)  ->  Ltn10, the chemokine fold      projection > 0
    A36C/A49C   (2N54)  ->  Ltn40, the beta-sandwich       projection < 0

Two mutations of the same protein that must move the same quantity in opposite
directions is much harder to satisfy by accident than a one-sided "the mutant
moved" -- a probe reacting to generic perturbation gets the sign wrong half the
time.

The two are not equally informative, and the difference decides what a positive
result means. The V21-V59 pair sits at 4.6 A in Ltn10 and 28.1 A in Ltn40: it is
the single most discriminating pair on the axis, so a model that simply forms a
disulfide between two cysteines is forced onto Ltn10 geometry with no
understanding of the fold switch at all. That variant is therefore a POSITIVE
CONTROL -- it shows the readout works, not that the model knows anything. The
A36-A49 pair sits at 6.8 A and 5.7 A in the two states, a 1.1 A difference that
does not even clear the axis threshold, so the crosslink itself says nothing
about which conformation to adopt. A36C/A49C is the real test.

Serine controls run at the same positions with no disulfide possible. If the
projection only appears with cysteine, it is crosslink geometry rather than
anything about the fold.

The alignment is grafted from wild type onto every variant, exactly as
`exp_gym.graft_a3m` does for ProteinGym, so alignment composition is identical
across the comparison and cannot be what moved the internal state.

Both the internal state and the emitted structure are scored against the same
two references, so the internal-versus-output contrast the rest of the project
makes is available here too.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_capture  # noqa: E402
import pi_conf  # noqa: E402
import pi_models  # noqa: E402
from exp_gym import graft_a3m  # noqa: E402
from exp_gym_deep import distogram_per_layer  # noqa: E402

# name -> substitutions, in the reference's author numbering
NULL_PERM = 20          # permuted-axis draws per variant

VARIANTS = {
    "XCL1": [
        ("WT", "", "wild type"),
        ("V21C_V59C", "V21C+V59C", "2HDM: locks Ltn10 -- POSITIVE CONTROL, "
                                   "crosslink is itself the top axis pair"),
        ("A36C_A49C", "A36C+A49C", "2N54: locks Ltn40 -- PRIMARY TEST, "
                                   "crosslink carries no axis information"),
        ("V21S_V59S", "V21S+V59S", "serine control for V21C/V59C"),
        ("A36S_A49S", "A36S+A49S", "serine control for A36C/A49C"),
        ("W55D", "W55D", "literature-reported Ltn10-stabilising point mutation; "
                         "endpoint NOT independently verified here"),
    ],
}


def apply_muts(seq, spec, index_of):
    """Apply 'V21C+V59C' to the fold sequence, checking every wild-type residue."""
    s = list(seq)
    applied = []
    for tok in filter(None, spec.split("+")):
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", tok)
        if not m:
            raise ValueError(f"bad substitution {tok!r}")
        wt, num, mut = m.group(1), int(m.group(2)), m.group(3)
        if num not in index_of:
            raise ValueError(f"residue {num} is not in the axis numbering map")
        i = index_of[num]
        if s[i] != wt:
            raise ValueError(f"{tok}: sequence has {s[i]} at {num}, not {wt}")
        s[i] = mut
        applied.append((num, i, wt, mut))
    return "".join(s), applied


def overlap(p, q):
    """Overlap coefficient sum(min(p,q)) in [0,1]: shared probability mass."""
    return np.minimum(p, q).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="XCL1")
    ap.add_argument("--ref", required=True, help="conf_<family>.npz")
    ap.add_argument("--a3m", required=True, help="wild-type alignment")
    ap.add_argument("--work", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import jax
    print(f"MSA server blocked at: {pi_models.block_network()}", flush=True)

    R = np.load(a.ref, allow_pickle=True)
    fold_seq = str(R["fold_seq"])
    resnums, seq_index = R["resnums"], R["seq_index"]
    index_of = {int(r): int(i) for r, i in zip(resnums, seq_index)}
    P_a, P_b, mask = R["P_a"], R["P_b"], R["mask"]
    axis = (P_a - P_b).astype(np.float64)                 # [n, n, 64]
    axis_sq = (axis ** 2).sum(-1)                         # [n, n]
    iu = np.where(mask)
    print(f"{a.family}: fold {len(fold_seq)} aa, axis {len(resnums)} residues, "
          f"{int(mask.sum())} masked pairs ({int(mask.sum())//2} unordered)", flush=True)

    work = Path(a.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wrapper = pi_models.load("boltz2")
    inner = pi_models.inner("boltz2", wrapper)
    cap_fn = pi_capture.CAPTURE["boltz2"]
    key = jax.random.key(0)

    def run(seq, tag):
        p = work / "msa" / f"{tag}.a3m"
        graft_a3m(p, Path(a.a3m), seq, fold_seq, cap=a.msa_cap)
        feats, depth = pi_models.features_for("boltz2", wrapper, seq, str(p),
                                              work=work / tag)
        if depth < 2:
            raise AssertionError(
                f"{tag}: alignment collapsed to {depth} row(s); the model is not "
                "seeing the alignment this run is supposed to control.")
        cap = cap_fn(inner, feats, num_recycles=a.recycles, key=key)
        out = wrapper.model_output(features=feats, recycling_steps=a.recycles,
                                   sampling_steps=a.sampling_steps, key=key)
        e = pi_models.extraction_from(out, name="boltz2")
        tok = np.asarray(feats["token_pad_mask"][0]).astype(bool)
        return cap, feats, e, tok, depth

    rows, dep = [], []
    p_wt = None
    for name, spec, note in VARIANTS[a.family]:
        seq, applied = apply_muts(fold_seq, spec, index_of)
        cap, feats, e, tok, depth = run(seq, name)
        dep.append(depth)
        if p_wt is None:
            drift = pi_capture.verify_capture("boltz2", inner, feats, cap,
                                              num_recycles=a.recycles, key=key)
            print(f"[{time.time()-t0:6.1f}s] capture drift {drift:.3e} "
                  f"(tol {pi_capture.DRIFT_TOL['boltz2']:g})", flush=True)

        # per-layer distograms restricted to the axis residues
        P = distogram_per_layer("boltz2", inner, cap["z_layers"], tok)  # [L,N,N,B]
        valid = np.where(tok)[0]
        sel = np.array([int(np.where(valid == i)[0][0]) if i in valid else -1
                        for i in seq_index])
        if (sel < 0).any():
            raise AssertionError("axis residue missing from the token mask")
        P = P[:, sel][:, :, sel].astype(np.float64)                     # [L,n,n,B]
        L = P.shape[0]

        av = axis[iu[0], iu[1]]                            # [P, B]
        den = np.maximum(axis_sq[iu[0], iu[1]], 1e-12)     # [P]
        anorm = np.linalg.norm(av, axis=-1)
        if p_wt is None:
            p_wt = P
            pp = np.zeros((L, len(iu[0])))
            cos = np.zeros((L, len(iu[0])))
            dnorm = np.zeros((L, len(iu[0])))
            null_mu = null_sd = np.zeros(L)
            # Keep wild type's own distributions at the axis pairs. The
            # bimodality question -- does the trunk carry mass on BOTH states,
            # or has it already committed to one -- decides whether a null
            # projection means anything, and it cannot be answered from a
            # summary scalar.
            p_wt_pairs = P[:, iu[0], iu[1]].astype(np.float32)
        else:
            # fraction of the full A-B displacement, PER PAIR so that the mask
            # can be narrowed later -- the V21C/V59C crosslink has to be
            # excludable after the fact, or its own pair dominates the mean.
            dv = (P - p_wt)[:, iu[0], iu[1]]               # [L, P, B]
            num = (dv * av).sum(-1)                        # [L, P]
            pp = num / den
            # How much did the distogram move AT ALL, and how much of that
            # movement lies along the axis? Wild type sits confidently on one
            # state, so any perturbation pushes mass off it and registers with a
            # fixed sign on this axis whether or not the model knows anything
            # about the other conformation. The cosine and the permuted-axis
            # null are what separate "moved along the axis" from "moved".
            dnorm = np.linalg.norm(dv, axis=-1)            # [L, P]
            cos = num / np.maximum(dnorm * anorm, 1e-12)
            rng = np.random.default_rng(0)
            ns = []
            for _ in range(NULL_PERM):
                pm = rng.permutation(len(av))
                ns.append(((dv * av[pm]).sum(-1) / den[pm]).mean(1))
            ns = np.stack(ns)                              # [T, L]
            null_mu, null_sd = ns.mean(0), ns.std(0)
        proj = pp.mean(1)

        ovA = np.array([overlap(P[l], P_a)[iu].mean() for l in range(L)])
        ovB = np.array([overlap(P[l], P_b)[iu].mean() for l in range(L)])

        # the emitted structure, scored against the same two references
        ca = np.asarray(e.ca)[tok]
        ca_ax = ca[sel]
        dpred = np.linalg.norm(ca_ax[:, None] - ca_ax[None, :], axis=-1)
        eA = float(np.abs(dpred - R["d_a_ca"])[iu].mean())
        eB = float(np.abs(dpred - R["d_b_ca"])[iu].mean())

        plddt = float(np.asarray(e.plddt)[tok].mean())
        rows.append(dict(name=name, spec=spec, note=note, seq=seq,
                         proj=proj, proj_pairs=pp.astype(np.float32),
                         cos=cos.mean(1), dnorm=dnorm.mean(1),
                         null_mu=null_mu, null_sd=null_sd,
                         ov_a=ovA, ov_b=ovB,
                         plddt=plddt, ca=ca_ax.astype(np.float32),
                         struct_err_a=eA, struct_err_b=eB, msa_depth=depth))
        print(f"[{time.time()-t0:6.1f}s] {name:12s} proj(final)={proj[-1]:+.4f}  "
              f"overlapA={ovA[-1]:.3f} overlapB={ovB[-1]:.3f}  "
              f"pLDDT={plddt:.3f}  struct |d-A|={eA:.2f} |d-B|={eB:.2f}",
              flush=True)

    out = {"names": np.array([r["name"] for r in rows]),
           "specs": np.array([r["spec"] for r in rows]),
           "notes": np.array([r["note"] for r in rows]),
           "proj": np.stack([r["proj"] for r in rows]),
           "proj_pairs": np.stack([r["proj_pairs"] for r in rows]),
           "cos": np.stack([r["cos"] for r in rows]),
           "dnorm": np.stack([r["dnorm"] for r in rows]),
           "null_mu": np.stack([r["null_mu"] for r in rows]),
           "null_sd": np.stack([r["null_sd"] for r in rows]),
           "pair_i": resnums[iu[0]], "pair_j": resnums[iu[1]],
           "p_wt_pairs": p_wt_pairs,
           "d_a_pairs": R["d_a"][iu], "d_b_pairs": R["d_b"][iu],
           "ov_a": np.stack([r["ov_a"] for r in rows]),
           "ov_b": np.stack([r["ov_b"] for r in rows]),
           "plddt": np.array([r["plddt"] for r in rows]),
           "struct_err_a": np.array([r["struct_err_a"] for r in rows]),
           "struct_err_b": np.array([r["struct_err_b"] for r in rows]),
           "ca": np.stack([r["ca"] for r in rows]),
           "msa_depth": np.array(dep), "family": np.array(a.family),
           "resnums": resnums, "capture_drift": np.array(drift),
           "wt_overlap_a": rows[0]["ov_a"], "wt_overlap_b": rows[0]["ov_b"]}
    np.savez_compressed(a.out, **out)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
