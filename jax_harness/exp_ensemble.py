"""Is the emitted-output baseline weak because it is ONE diffusion draw?

Gate 2 of the 2026-09-06 publication audit, and the sharpest falsification the
representation-versus-output claim still faces. Every internal-minus-output gap
in this project compares a deterministic trunk representation against geometry
measured from a single stochastic realization of the structure module. If the
output baseline is weak mostly because one draw is noisy, then the result is
about sampling variance, not about what the trunk knows.

THE DESIGN, FROZEN BEFORE ANY OUTPUT WAS READ:

  what varies      the diffusion key, and nothing else. The MSA is the archived
                   alignment, the regime is full (never subsample -- a changed
                   key must not be allowed to redraw the alignment), and the
                   trunk is run ONCE per sequence and reused across draws, so
                   the trunk state is bit-identical across the ensemble.
  draws            4, keys 0..3, prespecified.
  rows             the same variants as the xm_boltz2_r1_* captures, so the
                   ensemble baseline is paired row-for-row with the internal
                   probe it will be compared against.
  cohort           a balanced subset of heldout16 chosen by a stated rule
                   (below), not by looking at outcomes.
  aggregation      FEATURES, never coordinates. Coordinates from different
                   draws are not in a common frame, and averaging them would
                   shrink every structure toward its own mean and manufacture
                   the stability the test is asking about. Each draw's geometry
                   is computed against THAT draw's wild type, then the 37
                   features are averaged (mean; mean+SD secondary).

WHAT THIS DOES NOT CLAIM. Not common random numbers. A substitution changes the
side chain's atom count, so the atom array is re-indexed from the mutation site
onward and the same key does NOT give a WT and a mutant the same per-atom noise
past that point. The audit is explicit about this and so is the script: it
records `atoms_wt` and `atoms_mut` per variant so the claim can be checked
rather than assumed, and the analysis treats the draws as independent samples
summarised distributionally.

  sbatch checkout_array.sbatch <task file>       # one assay per array task
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m  # noqa: E402


def _rmsd(x, y):
    """CA RMSD after optimal superposition. Never compare unaligned frames."""
    A = np.asarray(x, float) - np.asarray(x, float).mean(0)
    B = np.asarray(y, float) - np.asarray(y, float).mean(0)
    return float(np.sqrt(
        (np.linalg.norm(A @ geom.kabsch(A, B).T - B, axis=1) ** 2).mean()))

# The balanced subset rule, stated here so it is part of the code rather than
# part of a submission command: all four non-stability assays, plus the two
# highest- and two lowest-confidence stability assays by the archived Boltz-2
# chain-mean pLDDT. "Not only the easiest proteins", as the audit requires.
SUBSET_RULE = ("all 4 non-stability heldout16 assays + the 2 highest and 2 "
               "lowest confidence stability assays by archived Boltz-2 "
               "plddt_mean")


def pick_subset(captures: Path):
    non_stab, stab = [], []
    for p in sorted(captures.glob("xm_boltz2_r1_*.npz")):
        assay = p.name[len("xm_boltz2_r1_"):-len(".npz")]
        d = np.load(p, allow_pickle=True)
        conf = float(np.asarray(d["plddt_mean"], float).mean())
        (stab if "Tsuboyama_2023" in assay else non_stab).append((conf, assay))
    stab.sort()
    return sorted([a for _, a in non_stab]
                  + [a for _, a in stab[:2] + stab[-2:]])


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/"
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", default=R + "data/gym/assays/"
                                              "DMS_ProteinGym_substitutions")
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--capture", required=True,
                    help="xm_boltz2_r1_<assay>.npz -- supplies the exact "
                         "variant rows, so the ensemble is paired with the "
                         "internal probe")
    ap.add_argument("--work", required=True)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N archived variants only -- for a smoke test; "
                         "a limited run is not comparable to a full one and "
                         "says so in its protocol block")
    ap.add_argument("--convergence-steps", type=int, default=0,
                    help="if set, re-run draw 0 at this many steps as an "
                         "inference-convergence check on the same rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pi_core as pi
    from exp_gym3 import trunk_capture
    from mosaic.losses.boltz2 import boltz2_forward_from_trunk

    cap = np.load(a.capture, allow_pickle=True)
    want = [str(m) for m in cap["mutant"]]
    want_pos = np.asarray(cap["pos"])
    rows_by_mut = {}
    for r in csv.DictReader(open(Path(a.assay_dir) / f"{a.assay}.csv")):
        if ":" not in r["mutant"]:
            rows_by_mut[r["mutant"]] = r
    missing = [m for m in want if m not in rows_by_mut]
    if missing:
        raise SystemExit(f"{len(missing)} archived variants absent from the "
                         f"assay table, e.g. {missing[:3]}")
    rows = [rows_by_mut[m] for m in want]
    if a.limit:
        rows, want = rows[:a.limit], want[:a.limit]
        want_pos = want_pos[:a.limit]

    wt = list(rows[0]["mutated_sequence"])
    m0 = re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"])
    wt[int(m0.group(2)) - 1] = m0.group(1)
    wt = "".join(wt)

    work = Path(a.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # full MSA, never subsample: a changed key must not redraw the alignment.
    model = pi.load_model(subsample_msa=False)

    def featurise(seq, tag):
        a3m = work / "msa" / f"{tag}.a3m"
        graft_a3m(a3m, Path(a.a3m), seq, wt, cap=a.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=a3m.resolve()))
        return pi.load_features(y.read_text())

    def sample(emb, tr, feats, draw, steps):
        """One diffusion draw from a FIXED trunk state."""
        out = boltz2_forward_from_trunk(
            model, feats, emb, tr, num_sampling_steps=steps,
            deterministic=True, key=jax.random.fold_in(jax.random.key(draw), 11))
        p = np.asarray(out.plddt).reshape(-1)
        ca = np.asarray(out.backbone_coordinates)[:, 1].astype(np.float32)
        return ca, p

    key0 = jax.random.key(0)
    f_wt, h = featurise(wt, "wt")
    mask = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    n_atom_wt = int(np.asarray(f_wt["atom_pad_mask"][0]).sum())
    ii = jnp.asarray([0]); jj = jnp.asarray([1])
    emb_w, tr_w, _ = trunk_capture(model, f_wt, ii, jj, 0,
                                   recycles=a.recycles, key=key0)
    ca_wt = np.stack([sample(emb_w, tr_w, f_wt, d, a.sampling_steps)[0][mask]
                      for d in range(a.draws)])
    print(f"[{time.time()-t0:6.1f}s] WT trunk + {a.draws} draws done; "
          f"{len(rows)} variants, {int(mask.sum())} residues", flush=True)

    # How far apart are two draws of the SAME sequence? This is the scale any
    # mutation effect has to clear, and it is measured here rather than assumed.
    #
    # SUPERPOSED, which is not a detail. Two draws come out in different frames,
    # so the raw difference is ~19 A and is almost entirely a rigid-body offset;
    # superposed it is ~0.3 A. Reporting the unaligned number as a structural
    # difference is how a sampler that agrees with itself gets described as
    # producing degenerate coordinates.
    wt_spread = float(np.mean([
        _rmsd(ca_wt[i], ca_wt[j])
        for i in range(a.draws) for j in range(i + 1, a.draws)]))
    print(f"   mean pairwise WT draw-to-draw CA RMS (unaligned): "
          f"{wt_spread:.3f} A", flush=True)

    CA, PL, PLS, ATM, CONV = [], [], [], [], []
    for n, r in enumerate(rows):
        mo = re.match(r"([A-Z])(\d+)([A-Z])", r["mutant"])
        p0 = int(mo.group(2)) - 1
        f_m, hm = featurise(r["mutated_sequence"], f"mut{n % 2}")
        ATM.append(int(np.asarray(f_m["atom_pad_mask"][0]).sum()))
        emb_m, tr_m, _ = trunk_capture(model, f_m, ii, jj, 0,
                                       recycles=a.recycles, key=key0)
        cas, pls, plss = [], [], []
        for d in range(a.draws):
            ca, pl = sample(emb_m, tr_m, f_m, d, a.sampling_steps)
            cas.append(ca[mask])
            pls.append(float(pl[mask].mean()))
            plss.append(float(pl[mask][p0]) if p0 < int(mask.sum()) else np.nan)
        CA.append(np.stack(cas)); PL.append(pls); PLS.append(plss)
        if a.convergence_steps and n < 8:
            ca_c, pl_c = sample(emb_m, tr_m, f_m, 0, a.convergence_steps)
            CONV.append(ca_c[mask])
        hm.cleanup()
        if (n + 1) % 10 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(rows)}", flush=True)
    h.cleanup()

    conv_wt = None
    if a.convergence_steps:
        conv_wt = sample(emb_w, tr_w, f_wt, 0, a.convergence_steps)[0][mask]

    extra = {}
    if CONV:
        extra["ca_converge"] = np.stack(CONV).astype(np.float32)
        extra["ca_wt_converge"] = conv_wt.astype(np.float32)
        extra["convergence_steps"] = a.convergence_steps

    proto = pi_protocol.protocol(
        script="exp_ensemble.py",
        design=(f"diffusion ensemble: trunk run ONCE per sequence and reused, "
                f"{a.draws} diffusion draws (keys 0..{a.draws-1}) varying only "
                f"the sampler; MSA regime full so a changed key cannot redraw "
                f"the alignment; rows taken from the archived capture so the "
                f"ensemble is paired with the internal probe"),
        layer=pi_protocol.layers("final"),
        features=pi_protocol.features(
            "emitted CA coordinates and pLDDT, per draw", 0),
        source=a.capture, n_assays=1,
        draws=a.draws, sampling_steps=a.sampling_steps,
        msa_regime="full",
        n_variants=len(rows),
        limited=(f"SMOKE: first {a.limit} variants only, not comparable to a "
                 f"full run" if a.limit else False),
        common_random_numbers=(
            "NOT claimed: a substitution changes the side-chain atom count, "
            "so the atom array is re-indexed from the site onward and the "
            "same key does not give WT and mutant matched per-atom noise. "
            "atoms_wt/atoms_mut are recorded so this is checkable."),
        aggregation="features per draw against THAT draw's WT, then averaged; "
                    "coordinates are never averaged across draws")
    pi_archive.write_npz(a.out, {
        "assay": np.array(a.assay), "mutant": np.array(want), "pos": want_pos,
        "score": np.asarray(cap["score"], float)[:len(want)],
        "ca": np.stack(CA).astype(np.float32),       # (n_var, draws, res, 3)
        "ca_wt": ca_wt.astype(np.float32),           # (draws, res, 3)
        "plddt": np.asarray(PL, np.float32),         # (n_var, draws)
        "plddt_site": np.asarray(PLS, np.float32),
        "atoms_wt": np.int32(n_atom_wt),
        "atoms_mut": np.asarray(ATM, np.int32),
        "wt_draw_spread": np.float32(wt_spread),
        "draws": np.int32(a.draws), **extra}, protocol=proto)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {a.out}")


if __name__ == "__main__":
    main()
