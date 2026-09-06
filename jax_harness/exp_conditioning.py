"""Trace the predictive signal from the trunk to the decoder entrance.

Part A of the conditioning experiment. Every internal-versus-output number in
this project compares the trunk against what the sampler emitted, leaving the
question of WHERE between them the predictive signal stops being linearly
accessible. The diffusion-conditioning tensors are the last thing the structure
module sees before it samples, so they are the natural intermediate point.

FROZEN SPECIFICATION, stated before any result was read:

  cohort      the SAME prespecified balanced 8-assay subset of heldout16 used
              for the ensemble campaign. No new cohort search.
  rows        the variants of the archived xm_boltz2_r1_* captures, so the
              conditioning features are paired row-for-row with the internal
              probe and the emitted baseline.
  regime      msa='full', stated, never left implicit.
  features    all taken at the MUTATED TOKEN, as mutant minus wild type:
                trunk_z    pair row, averaged over partners            (128)
                trunk_s    single representation                       (384)
                cond_ttb   token_trans_bias row, averaged over partners (384)
                cond_q     q, aggregated atoms -> tokens               (128)
                cond_c     c, aggregated atoms -> tokens               (128)

WHY q AND c ARE AGGREGATED AND NOT DIFFERENCED DIRECTLY. They are atom-indexed
and padded to a fixed window count, so a wild type with 820 atoms and a mutant
with 816 both arrive as [832, 128] -- the same SHAPE holding different atoms,
re-indexed from the mutation site onward. An elementwise difference is
well-formed and wrong. Each is aggregated through `atom_to_token` (mean over a
token's real atoms) BEFORE the difference, which is the only operation that puts
the two runs on the same index. `token_trans_bias` and the trunk tensors are
already token-indexed and need no such step.

`to_keys` is a closure, not an array, and is not captured. The windowed atom
attention biases are excluded from this pass: aggregating a [26, 32, 128, 12]
window structure to tokens is a separate decision that should not ride along
inside a first measurement.

WHAT A DROP WOULD AND WOULD NOT MEAN. Conditioning is a deterministic function
of the trunk state, so it cannot hold more information than the trunk did; a
fall in score localizes a loss of ACCESSIBILITY under this readout and never
proves erasure. The informative direction is the other one: signal still
decodable at the decoder entrance extends the measured gap right up to the
sampler.

Costs one trunk forward per sequence and NO diffusion sampling.

  sbatch checkout_array.sbatch <task file>
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402
from exp_gym import YAML_TMPL, graft_a3m  # noqa: E402

BLOCKS = ("trunk_z", "trunk_s", "cond_ttb", "cond_q", "cond_c")


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/"
    ap.add_argument("--assay", required=True)
    ap.add_argument("--assay-dir", default=R + "data/gym/assays/"
                                              "DMS_ProteinGym_substitutions")
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--capture", required=True,
                    help="xm_boltz2_r1_<assay>.npz -- fixes the variant rows")
    ap.add_argument("--work", required=True)
    ap.add_argument("--msa", choices=("full", "subsample"), required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--msa-cap", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pi_core as pi

    cap = np.load(a.capture, allow_pickle=True)
    want = [str(m) for m in cap["mutant"]]
    want_pos = np.asarray(cap["pos"])
    score = np.asarray(cap["score"], float)
    by_mut = {r["mutant"]: r for r in csv.DictReader(
        open(Path(a.assay_dir) / f"{a.assay}.csv")) if ":" not in r["mutant"]}
    missing = [m for m in want if m not in by_mut]
    if missing:
        raise SystemExit(f"{len(missing)} archived variants absent from the "
                         f"assay table, e.g. {missing[:3]}")
    rows = [by_mut[m] for m in want]
    if a.limit:
        rows, want = rows[:a.limit], want[:a.limit]
        want_pos, score = want_pos[:a.limit], score[:a.limit]

    wt = list(rows[0]["mutated_sequence"])
    m0 = re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"])
    wt[int(m0.group(2)) - 1] = m0.group(1)
    wt = "".join(wt)

    work = Path(a.work)
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = pi.load_model(subsample_msa=(a.msa == "subsample"))
    key = jax.random.key(0)

    def featurise(seq, tag):
        p = work / "msa" / f"{tag}.a3m"
        graft_a3m(p, Path(a.a3m), seq, wt, cap=a.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=p.resolve()))
        return pi.load_features(y.read_text())

    def site_features(feats, pos0):
        """The five blocks at the mutated token, for one sequence."""
        emb = model.embed_inputs(feats)
        tr = pi.run_trunk(model, emb, feats, recycling_steps=a.recycles,
                          key=key, deterministic=True, capture_last=False)
        st = tr["trunk_state"]
        q, c, _to_keys, _aeb, _adb, ttb = model.diffusion_conditioning(
            st.s, st.z, emb.relative_position_encoding, feats)

        tok = np.asarray(feats["token_pad_mask"][0]).astype(bool)
        z = np.asarray(st.z)[0]                       # (N, N, 128)
        s = np.asarray(st.s)[0]                       # (N, 384)
        T = np.asarray(ttb)[0]                        # (N, N, 384)

        # atoms -> tokens. a2t is (n_atom_padded, n_token) one-hot; padded
        # atoms belong to no token, so the column sums count real atoms only.
        a2t = np.asarray(feats["atom_to_token"][0]).astype(np.float64)
        apad = np.asarray(feats["atom_pad_mask"][0]).astype(bool)
        a2t = a2t * apad[:, None]
        per_tok = a2t.sum(0)                          # (N,)
        denom = np.where(per_tok > 0, per_tok, 1.0)[:, None]
        Q = (a2t.T @ np.asarray(q)[0]) / denom        # (N, 128)
        C = (a2t.T @ np.asarray(c)[0]) / denom        # (N, 128)

        return {
            "trunk_z": z[pos0][tok].mean(0),
            "trunk_s": s[pos0],
            "cond_ttb": T[pos0][tok].mean(0),
            "cond_q": Q[pos0],
            "cond_c": C[pos0],
        }, int(tok.sum()), int(apad.sum())

    p_wt = int(re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"]).group(2)) - 1
    f_wt, h = featurise(wt, "wt")
    # WT features are read at each variant's own site below, so the wild-type
    # forward is done once and its per-token arrays kept.
    emb_w = model.embed_inputs(f_wt)
    tr_w = pi.run_trunk(model, emb_w, f_wt, recycling_steps=a.recycles,
                        key=key, deterministic=True, capture_last=False)
    st_w = tr_w["trunk_state"]
    qw, cw, _tk, _ae, _ad, ttbw = model.diffusion_conditioning(
        st_w.s, st_w.z, emb_w.relative_position_encoding, f_wt)
    tok_w = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    a2t_w = np.asarray(f_wt["atom_to_token"][0]).astype(np.float64)
    apad_w = np.asarray(f_wt["atom_pad_mask"][0]).astype(bool)
    a2t_w = a2t_w * apad_w[:, None]
    den_w = np.where(a2t_w.sum(0) > 0, a2t_w.sum(0), 1.0)[:, None]
    Zw, Sw = np.asarray(st_w.z)[0], np.asarray(st_w.s)[0]
    Tw = np.asarray(ttbw)[0]
    Qw = (a2t_w.T @ np.asarray(qw)[0]) / den_w
    Cw = (a2t_w.T @ np.asarray(cw)[0]) / den_w
    n_tok_w, n_atom_w = int(tok_w.sum()), int(apad_w.sum())
    print(f"[{time.time()-t0:6.1f}s] WT {n_tok_w} tokens {n_atom_w} atoms; "
          f"{len(rows)} variants, msa={a.msa}", flush=True)

    def wt_site(p0):
        return {"trunk_z": Zw[p0][tok_w].mean(0), "trunk_s": Sw[p0],
                "cond_ttb": Tw[p0][tok_w].mean(0),
                "cond_q": Qw[p0], "cond_c": Cw[p0]}

    acc = {b: [] for b in BLOCKS}
    atoms_mut, keep = [], []
    for n, r in enumerate(rows):
        mo = re.match(r"([A-Z])(\d+)([A-Z])", r["mutant"])
        p0 = int(mo.group(2)) - 1
        if p0 >= n_tok_w:
            continue
        f_m, hm = featurise(r["mutated_sequence"], f"mut{n % 2}")
        fm, n_tok_m, n_atom_m = site_features(f_m, p0)
        if n_tok_m != n_tok_w:
            hm.cleanup()
            continue
        w = wt_site(p0)
        for b in BLOCKS:
            acc[b].append(np.asarray(fm[b], np.float64)
                          - np.asarray(w[b], np.float64))
        atoms_mut.append(n_atom_m)
        keep.append(n)
        hm.cleanup()
        if (n + 1) % 20 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(rows)}", flush=True)
    h.cleanup()

    keep = np.asarray(keep)
    proto = pi_protocol.protocol(
        script="exp_conditioning.py",
        design="mutation difference at the mutated token, captured at the "
               "trunk and at the diffusion-conditioning boundary, on the rows "
               "of the archived capture; no diffusion sampling",
        layer=pi_protocol.layers("final"),
        features={b: pi_protocol.features(b, int(np.asarray(acc[b]).shape[1]))
                  for b in BLOCKS},
        source=a.capture, n_assays=1, msa_regime=a.msa,
        n_variants=int(len(keep)),
        aggregation="q and c are atom-indexed and padded; aggregated through "
                    "atom_to_token (mean over a token's real atoms) BEFORE "
                    "differencing. token_trans_bias and the trunk tensors are "
                    "token-indexed and differenced directly.",
        excluded="to_keys (a closure); atom_enc_bias / atom_dec_bias (windowed "
                 "atom attention, aggregation is a separate decision)",
        determinism="conditioning is a deterministic function of the trunk "
                    "state, so a score drop localizes accessibility under this "
                    "readout and cannot show erasure")
    pi_archive.write_npz(a.out, {
        "assay": np.array(a.assay),
        "mutant": np.array([want[i] for i in keep]),
        "pos": want_pos[keep], "score": score[keep],
        "atoms_wt": np.int32(n_atom_w),
        "atoms_mut": np.asarray(atoms_mut, np.int32),
        "n_tokens": np.int32(n_tok_w),
        **{b: np.asarray(acc[b], np.float32) for b in BLOCKS}},
        protocol=proto)
    print(f"\n[{time.time()-t0:6.1f}s] wrote {a.out}  "
          f"({len(keep)} variants)")


if __name__ == "__main__":
    main()
