"""Capture the Pairformer pair row at every layer, for one assay's variants.

The reference vertical slice. What it is meant to demonstrate is that the
scientific choices can be read off the top of the file while the numerical
capture stays exactly the code that produced the archives:

    cohort      which assay, checksummed
    spec        which fields, which layers, vector or norm, precision
    capture     exp_gym2.trunk_capture -- UNCHANGED, and deliberately so
    artifact    written through the seam, carrying its own protocol
    check       the written archive validated against the spec that asked for it

THE TWO KL FIELDS ARE TWO MEASUREMENTS. `kl_glob` averages the per-layer
divergence over every sampled pair; `kl_site` averages it over only the pairs
that touch the mutated token. This file once computed the first one twice and
stored it under both names, which no shape check could catch -- see
`collection.reductions`, which now owns both reductions and returns them
together.

`trunk_capture` is imported rather than reimplemented. It reproduces
`joltz.Joltz2.trunk_iteration` and is verified bit-identical to it by
`test_equivalence.py`; rewriting it prettily would throw that away, and the plan
says as much -- do not rewrite validated numerical kernels to make the tree look
uniform.

WHY --MUTANTS-FROM AND --PAIRS-FROM EXIST. To show this path reproduces the
harness, it has to collect the SAME variants an archive already holds. Selecting
"the first eight" independently would give eight different variants and nothing
to compare. So the mutants can be taken from an existing capture, and then
dz_site can be diffed row for row.

The KL fields need the same treatment for a different reason: they are averages
over a sampled set of token pairs, so two runs agree pair-for-pair only if they
share the sample. `--pairs-from` reads `pair_i`/`pair_j` from an earlier
capture; this script writes them, and the `gym2s_*` archives predate them.

    # login node: resolve and price the job without loading anything
    uv run python experiments/collection/collect_pairformer_layers.py --inspect

    sbatch jax_harness/checkout.sbatch \\
        ../experiments/collection/collect_pairformer_layers.py \\
        --mutants-from $W/runs/gym2s_ARGR_ECOLI_Tsuboyama_2023_1AOY.npz \\
        --pairs-from $W/runs/pairs_gym2s_ARGR_ECOLI_Tsuboyama_2023_1AOY.npz \\
        --n-variants 8 --out $W/runs/slice_pairformer_v2.npz
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

# The container's interpreter does not have this package installed -- jobs exec
# a bare `python` inside the mosaic image -- so the checkout's src/ is located
# the same way the pi_* shims locate it. Must precede the package imports.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from protein_interpretability.collection import CaptureSpec, Cohort
from protein_interpretability.collection import capabilities as caps
from protein_interpretability.collection import reductions as red
from protein_interpretability.experiments import protocol as P

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")

# The scientific declaration. Everything below this is plumbing.
SPEC = CaptureSpec(
    model="boltz2",
    fields=("dz_site", "kl_site", "kl_glob"),
    layers="all",                 # all 64 Pairformer blocks
    reduction="vector",           # the DIRECTION the row moved, not how far
    recycles=3,
    dtype="float32",
    n_pairs=1500,                 # exp_gym2's default; the KL sample, not disto
    notes="pair row at the mutated position, every layer, final recycle",
)

# The pair sample is drawn once and every variant is measured against it, so its
# seed is part of what makes two captures comparable -- kl_glob over a different
# 1500 pairs is a different measurement of the same run. exp_gym2 drew it from
# its --seed, which defaulted to 0.
PAIR_SEED = 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="smoke_pairformer")
    ap.add_argument("--assay", help="defaults to the cohort's only assay")
    ap.add_argument("--n-variants", type=int, default=8)
    ap.add_argument("--mutants-from", help="take the variants from this capture")
    ap.add_argument("--mutants", help="explicit comma-separated list; also sets "
                                      "the ORDER they are collected in, which "
                                      "is what isolates first-call effects")
    ap.add_argument("--pairs-from", help="reuse the sampled pair indices "
                                         "recorded by an earlier capture, so "
                                         "two runs' KL fields are averages "
                                         "over the SAME pairs")
    ap.add_argument("--msa", default="full", choices=("full", "subsample"))
    ap.add_argument("--msa-cap", type=int, default=2048,
                    help="matches exp_gym2's default, which is what the "
                         "archived captures were produced with")
    ap.add_argument("--inspect", action="store_true",
                    help="resolve, price and exit -- loads no model")
    ap.add_argument("--out")
    return ap.parse_args(argv)


def chosen_mutants(a, rows) -> list[str]:
    if a.mutants:
        return [m.strip() for m in a.mutants.split(",") if m.strip()]
    if a.mutants_from:
        with np.load(a.mutants_from, allow_pickle=True) as z:
            names = [str(m) for m in z["mutant"]]
        return names[: a.n_variants]
    return [r["mutant"] for r in rows][: a.n_variants]


def main(argv=None) -> int:
    a = parse_args(argv)

    # ---- 1. the declaration, checked before anything is loaded ------------
    SPEC.validate()
    caps.check_msa(SPEC.model, use_msa=a.msa != "none")

    cohort = Cohort.load(a.cohort)
    cohort.verify()
    assay = next((x for x in cohort if x.id == a.assay), None) if a.assay \
        else cohort.assays[0]
    if assay is None:
        raise SystemExit(f"{a.assay!r} is not in cohort {a.cohort!r}: "
                         f"{cohort.ids}")

    n_tokens = assay.wt_length
    est = SPEC.estimate_bytes(n_variants=a.n_variants, n_tokens=n_tokens)
    print(f"cohort   {cohort.name} -> {assay.id}")
    print(f"model    {SPEC.model}, {SPEC.n_layers} layers, {SPEC.reduction}, "
          f"recycles={SPEC.recycles}, msa={a.msa}")
    print(f"kl       {SPEC.n_pairs} sampled pairs, "
          f"{'reused from ' + a.pairs_from if a.pairs_from else f'seed {PAIR_SEED}'}"
          f"; kl_glob over all of them, kl_site over those touching the site")
    print(f"size     {a.n_variants} variants x {n_tokens} tokens "
          f"-> {est / 1e6:.1f} MB")
    print(f"         (the unreduced pair tensor would be "
          f"{SPEC.full_pair_tensor_bytes(n_variants=a.n_variants, n_tokens=n_tokens) / 1e9:.0f} GB)")
    if a.inspect:
        print("\ninspect only; no model loaded")
        return 0
    if not a.out:
        raise SystemExit("--out is required unless --inspect")

    # ---- 2. the model, and the capture code that produced the archives ----
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax_harness"))
    import jax                                                     # noqa: E402
    import pi_archive                                              # noqa: E402
    import pi_core as pi                                           # noqa: E402
    import pi_models                                               # noqa: E402
    from exp_gym import YAML_TMPL, graft_a3m                       # noqa: E402
    from exp_gym2 import trunk_capture                             # noqa: E402

    rows = [r for r in csv.DictReader(open(assay.assay_csv))
            if ":" not in r["mutant"]]
    by_mutant = {r["mutant"]: r for r in rows}
    wt = list(rows[0]["mutated_sequence"])
    m0 = re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"])
    wt[int(m0.group(2)) - 1] = m0.group(1)
    wt = "".join(wt)

    wanted = chosen_mutants(a, rows)
    missing = [m for m in wanted if m not in by_mutant]
    if missing:
        raise SystemExit(f"not in this assay's table: {missing}")

    work = Path(a.out).with_suffix("")
    (work / "msa").mkdir(parents=True, exist_ok=True)
    (work / "yamls").mkdir(parents=True, exist_ok=True)

    def featurise(seq, tag):
        m = work / "msa" / f"{tag}.a3m"
        graft_a3m(m, Path(assay.msa_path), seq, wt, cap=a.msa_cap)
        y = work / "yamls" / f"{tag}.yaml"
        y.write_text(YAML_TMPL.format(seq=seq, msa=m.resolve()))
        return pi.load_features(y.read_text())

    pi_models.block_network()
    model = pi.load_model(subsample_msa=a.msa == "subsample")
    key = jax.random.key(0)

    f_wt, h_wt = featurise(wt, "wt")

    # ---- 2a. the token grid, resolved once --------------------------------
    # The mutant name gives a RESIDUE number; `ii`, `jj` and the z rows are
    # indexed by TOKEN. Those coincide only while the token grid is this
    # sequence followed by padding, and the failure when they do not is silent:
    # the site mask selects a neighbourhood that is not the mutation's and
    # `kl_site` still comes out a plausible size. So the assumption is checked
    # rather than carried.
    pad = np.asarray(f_wt["token_pad_mask"][0]).astype(bool)
    valid = np.where(pad)[0]
    n_tok = int(valid.size)
    if not np.array_equal(valid, np.arange(n_tok)):
        raise SystemExit(
            f"the valid tokens of {assay.id} are not a prefix of the token "
            f"grid ({valid[:8]}...), so a residue number is not its token "
            f"index. Resolve the residue-to-token mapping from the features "
            f"before capturing against this input.")

    # ---- 2b. the pair sample, which is half of what a KL field means ------
    if a.pairs_from:
        # The archives do not record their pair sample, so `kl_glob` from two
        # runs is only comparable pair-for-pair if one of them can be handed the
        # other's indices. New captures carry `pair_i`/`pair_j` for exactly this.
        with np.load(a.pairs_from, allow_pickle=True) as z:
            if "pair_i" not in z.files:
                raise SystemExit(
                    f"{a.pairs_from} does not record its sampled pairs, so they "
                    f"cannot be reused. Captures written before `pair_i`/"
                    f"`pair_j` existed can only be matched by replaying the "
                    f"producer's RNG stream, which is not what this flag does.")
            ii, jj = np.asarray(z["pair_i"]), np.asarray(z["pair_j"])
        out_of_grid = ii[(ii >= n_tok)].tolist() + jj[(jj >= n_tok)].tolist()
        if out_of_grid:
            raise SystemExit(
                f"{a.pairs_from} samples tokens {sorted(set(out_of_grid))[:8]} "
                f"but this input has {n_tok}; the two captures are not of the "
                f"same molecule and their KL fields are not comparable.")
        pair_source = str(a.pairs_from)
    else:
        # Drawn from the valid tokens with self-pairs dropped, exactly as
        # exp_gym2 draws them. The diagonal is a delta at distance zero for
        # every variant, so keeping it would dilute both reductions with
        # guaranteed zeros.
        prng = np.random.default_rng(PAIR_SEED)
        first = prng.choice(valid, SPEC.n_pairs)
        second = prng.choice(valid, SPEC.n_pairs)
        keep = first != second
        ii, jj = first[keep], second[keep]
        pair_source = f"default_rng({PAIR_SEED}), self-pairs dropped"

    def token_of(mutant: str) -> int:
        p0 = int(re.match(r"([A-Z])(\d+)([A-Z])", mutant).group(2)) - 1
        if not (0 <= p0 < n_tok):
            raise SystemExit(
                f"{mutant} names residue {p0 + 1}, outside the {n_tok} tokens "
                f"this input has")
        return p0                      # identity, checked above, not assumed

    tokens = {name: token_of(name) for name in wanted}

    # Knowable now, and the answer does not change once the GPU starts: the
    # sample is fixed and so are the sites. A variant whose site no pair touches
    # cannot have a `kl_site`, and the archived producer wrote 0.0 for it --
    # which reads as "this mutation moved nothing", the strongest claim in the
    # file, for the one variant that was never measured.
    blind = red.uncovered_sites(ii, jj, sorted(set(tokens.values())))
    if blind:
        raise SystemExit(
            f"tokens {blind} are touched by none of the {len(ii)} sampled "
            f"pairs, so `kl_site` for the variant(s) there would be an average "
            f"over nothing. Raise SPEC.n_pairs or change PAIR_SEED; do not let "
            f"it write a zero.")

    # ---- 2c. the captures --------------------------------------------------
    _, _, ref = trunk_capture(model, f_wt, ii, jj, None,
                              recycles=SPEC.recycles, key=key)
    ref_z = np.asarray(ref["z_full"])
    # Not cast: exp_gym2 handed `skl` the float32 it got back from the model,
    # and the archived kl_glob is that arithmetic.
    lw = np.asarray(ref["logits"])

    dz, kl_site, kl_glob, positions, scores, holds = [], [], [], [], [], [h_wt]
    for name in wanted:
        r = by_mutant[name]
        tok = tokens[name]
        feats, hh = featurise(r["mutated_sequence"], f"v_{name}")
        holds.append(hh)
        _, _, cur = trunk_capture(model, feats, ii, jj, tok,
                                  recycles=SPEC.recycles, key=key)
        dz.append((np.asarray(cur["z_site"]) - ref_z[:, tok]).astype(np.float32))
        # One divergence tensor, two reductions, returned together -- see
        # collection.reductions for why they are not computed side by side.
        kl = red.kl_reductions(np.asarray(cur["logits"]), lw, ii, jj, tok)
        kl_site.append(kl["kl_site"].astype(np.float32))
        kl_glob.append(kl["kl_glob"].astype(np.float32))
        positions.append(tok)
        scores.append(float(r["DMS_score"]))
        print(f"  {name:8s} token {tok:3d}  score {scores[-1]:+.3f}", flush=True)

    arrays = {
        "dz_site": np.stack(dz),
        "kl_site": np.stack(kl_site),
        "kl_glob": np.stack(kl_glob),
        "mutant": np.array(wanted),
        "pos": np.array(positions),
        # The pair sample IS part of what kl_glob and kl_site mean. Recorded so
        # a later run can be an average over the same pairs rather than over a
        # different draw that happens to have the same size.
        "pair_i": np.asarray(ii, np.int32),
        "pair_j": np.asarray(jj, np.int32),
        "score": np.array(scores),
        "wt_seq": np.array(wt),
    }

    # ---- 3. write through the seam, carrying the spec ---------------------
    proto = P.protocol(
        script=Path(__file__).name,
        design="pair row at the mutated position, every Pairformer layer, "
               "final recycle; WT reference captured once. kl_glob is the "
               "symmetric KL over all sampled pairs, kl_site the same "
               "divergence over only the pairs touching the mutated token",
        layer=P.layers("all", n_layers=SPEC.n_layers),
        features=P.features("dz_site, pair row", SPEC.n_layers * 128),
        source=str(assay.msa_path),
        n_assays=1,
        cohort=cohort.name,
        assay=assay.id,
        msa_regime=a.msa,
        msa_cap=a.msa_cap,
        # n_pairs from the spec is what was ASKED for; this is what survived
        # dropping the diagonal, and it is the divisor of every kl_glob here.
        pair_sample=pair_source,
        n_pairs_kept=int(len(ii)),
        divergence="symmetric KL (Jeffreys), exp_gym.skl",
        **SPEC.protocol(),
    )
    pi_archive.write_npz(Path(a.out), arrays, protocol=proto)

    # ---- 4. check the archive against the spec that asked for it ----------
    cap = pi_archive.load_capture(a.out, require_meta=True)
    SPEC.validate_capture(cap, n_variants=len(wanted), n_tokens=n_tokens)
    print(f"\nwrote {a.out} -- matches the spec it was collected from")
    del holds
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
