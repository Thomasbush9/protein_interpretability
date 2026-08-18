"""Capture the Pairformer pair row at every layer, for one assay's variants.

The reference vertical slice. What it is meant to demonstrate is that the
scientific choices can be read off the top of the file while the numerical
capture stays exactly the code that produced the archives:

    cohort      which assay, checksummed
    spec        which fields, which layers, vector or norm, precision
    capture     exp_gym2.trunk_capture -- UNCHANGED, and deliberately so
    artifact    written through the seam, carrying its own protocol
    check       the written archive validated against the spec that asked for it

`trunk_capture` is imported rather than reimplemented. It reproduces
`joltz.Joltz2.trunk_iteration` and is verified bit-identical to it by
`test_equivalence.py`; rewriting it prettily would throw that away, and the plan
says as much -- do not rewrite validated numerical kernels to make the tree look
uniform.

WHY --MUTANTS-FROM EXISTS. To show this path reproduces the harness, it has to
collect the SAME variants an archive already holds. Selecting "the first eight"
independently would give eight different variants and nothing to compare. So the
mutants can be taken from an existing capture, and then dz_site can be diffed
row for row.

    # login node: resolve and price the job without loading anything
    uv run python experiments/collection/collect_pairformer_layers.py --inspect

    sbatch jax_harness/checkout.sbatch \\
        ../experiments/collection/collect_pairformer_layers.py \\
        --mutants-from $W/runs/gym2s_ARGR_ECOLI_Tsuboyama_2023_1AOY.npz \\
        --n-variants 8 --out $W/runs/slice_pairformer.npz
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

# The container's interpreter does not have this package installed -- jobs exec
# a bare `python` inside the mosaic image -- so the checkout's src/ is located
# the same way the pi_* shims locate it. Must precede the package imports.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from protein_interpretability.collection import CaptureSpec, Cohort
from protein_interpretability.collection import capabilities as caps
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
    notes="pair row at the mutated position, every layer, final recycle",
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="smoke_pairformer")
    ap.add_argument("--assay", help="defaults to the cohort's only assay")
    ap.add_argument("--n-variants", type=int, default=8)
    ap.add_argument("--mutants-from", help="take the variants from this capture")
    ap.add_argument("--msa", default="full", choices=("full", "subsample"))
    ap.add_argument("--msa-cap", type=int, default=2048,
                    help="matches exp_gym2's default, which is what the "
                         "archived captures were produced with")
    ap.add_argument("--inspect", action="store_true",
                    help="resolve, price and exit -- loads no model")
    ap.add_argument("--out")
    return ap.parse_args(argv)


def chosen_mutants(a, rows) -> list[str]:
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
    n_tok = int(np.asarray(f_wt["token_pad_mask"][0]).astype(bool).sum())
    prng = np.random.default_rng(0)
    ii = prng.integers(0, n_tok, 1500)
    jj = prng.integers(0, n_tok, 1500)

    _, _, ref = trunk_capture(model, f_wt, ii, jj, None,
                              recycles=SPEC.recycles, key=key)
    ref_z = np.asarray(ref["z_full"])

    dz, kl_site, kl_glob, positions, scores, holds = [], [], [], [], [], [h_wt]
    for name in wanted:
        r = by_mutant[name]
        m = re.match(r"([A-Z])(\d+)([A-Z])", name)
        row = int(m.group(2)) - 1
        feats, hh = featurise(r["mutated_sequence"], f"v_{name}")
        holds.append(hh)
        _, _, cur = trunk_capture(model, feats, ii, jj, row,
                                  recycles=SPEC.recycles, key=key)
        dz.append((np.asarray(cur["z_site"]) - ref_z[:, row]).astype(np.float32))
        lw = np.asarray(ref["logits"], np.float64)
        lm = np.asarray(cur["logits"], np.float64)
        kl_site.append(_kl(lw, lm).astype(np.float32))
        kl_glob.append(_kl(lw, lm).astype(np.float32))
        positions.append(row)
        scores.append(float(r["DMS_score"]))
        print(f"  {name:8s} row {row:3d}  score {scores[-1]:+.3f}", flush=True)

    arrays = {
        "dz_site": np.stack(dz),
        "kl_site": np.stack(kl_site),
        "kl_glob": np.stack(kl_glob),
        "mutant": np.array(wanted),
        "pos": np.array(positions),
        "score": np.array(scores),
        "wt_seq": np.array(wt),
    }

    # ---- 3. write through the seam, carrying the spec ---------------------
    proto = P.protocol(
        script=Path(__file__).name,
        design="pair row at the mutated position, every Pairformer layer, "
               "final recycle; WT reference captured once",
        layer=P.layers("all", n_layers=SPEC.n_layers),
        features=P.features("dz_site, pair row", SPEC.n_layers * 128),
        source=str(assay.msa_path),
        n_assays=1,
        cohort=cohort.name,
        assay=assay.id,
        msa_regime=a.msa,
        msa_cap=a.msa_cap,
        **SPEC.protocol(),
    )
    pi_archive.write_npz(Path(a.out), arrays, protocol=proto)

    # ---- 4. check the archive against the spec that asked for it ----------
    cap = pi_archive.load_capture(a.out, require_meta=True)
    SPEC.validate_capture(cap, n_variants=len(wanted), n_tokens=n_tokens)
    print(f"\nwrote {a.out} -- matches the spec it was collected from")
    del holds
    return 0


def _kl(logits_wt, logits_mut):
    """Per-layer KL(mut || wt) over the sampled pairs. float64 on the way in."""
    def soft(x):
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)
    p, q = soft(logits_mut), soft(logits_wt)
    return (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1).mean(-1)


if __name__ == "__main__":
    raise SystemExit(main())
