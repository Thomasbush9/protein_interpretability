"""The corrected reductions, checked against an archive `exp_gym2` produced.

`gym2s_*.npz` stores the final-layer distogram logits it sampled (`disto`, one
row per variant, and `disto_wt`) alongside the `kl_glob` and `kl_site` curves it
reduced from them. So the last layer of both fields can be RECOMPUTED from the
archive's own numbers and compared with what the archive recorded -- an
old-versus-new equivalence check that needs no GPU, no model environment and no
rerun.

The pair indices are the one thing the archive does not store. They are
recovered by replaying the producer's RNG stream (`default_rng(0)`, the
250-of-1287 variant draw first, then two `choice(valid, 1500)` draws with
self-pairs dropped), and that reconstruction is not taken on faith: it is
accepted only because the pair count it yields matches the archive's and the
`kl_glob` it produces is bit-identical to the recorded one over all 250
variants.

This is the check the buggy code could not have passed. `kl_site` in this
archive runs about 8x `kl_glob`, so wiring both fields to the global reduction
-- which is what `collect_pairformer_layers.py` did -- moves `kl_site` by
roughly an order of magnitude while leaving every shape and every dtype intact.

    uv run pytest tests/test_kl_matches_the_archive.py -q
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from protein_interpretability.collection.reductions import (
    site_mask,
    symmetric_kl,
)

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
ASSAY = "ARGR_ECOLI_Tsuboyama_2023_1AOY"
ARCHIVE = W / "runs" / f"gym2s_{ASSAY}.npz"

pytestmark = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="the gym2s captures are not mounted here")

N_VARIANTS = 250          # exp_gym2's --n-variants default
N_PAIRS = 1500            # exp_gym2's --n-pairs default
SEED = 0                  # exp_gym2's --seed default


@pytest.fixture(scope="module")
def archive():
    return np.load(ARCHIVE, allow_pickle=True)


@pytest.fixture(scope="module")
def pairs(archive):
    """The producer's pair sample, replayed and then checked against the file."""
    from protein_interpretability.collection import Cohort

    assay = next(a for a in Cohort.load("smoke_pairformer") if a.id == ASSAY)
    rows = [r for r in csv.DictReader(open(assay.assay_csv))
            if ":" not in r["mutant"]]

    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(len(rows), N_VARIANTS, replace=False))
    assert [rows[i]["mutant"] for i in chosen] == \
        [str(m) for m in archive["mutant"]], \
        "the variant draw does not replay, so neither does the pair draw"

    valid = np.arange(assay.wt_length)
    first, second = rng.choice(valid, N_PAIRS), rng.choice(valid, N_PAIRS)
    keep = first != second
    ii, jj = first[keep], second[keep]
    assert len(ii) == archive["disto"].shape[1], (
        "the replayed sample has a different number of pairs than the archive "
        "stored; the reconstruction is wrong and nothing below means anything")
    return ii, jj


def test_the_global_reduction_reproduces_the_archive(archive, pairs):
    """Bit-identical: same expression, same dtype, same pairs, same order."""
    kl = symmetric_kl(archive["disto"], archive["disto_wt"][None])
    assert np.array_equal(kl.mean(-1), archive["kl_glob"][:, -1])


def test_the_site_reduction_reproduces_the_archive(archive, pairs):
    """Float32 summation over a subset, so exact to rounding rather than bitwise."""
    ii, jj = pairs
    kl = symmetric_kl(archive["disto"], archive["disto_wt"][None])
    pos = np.asarray(archive["pos"])
    mine = np.array([kl[k][site_mask(ii, jj, int(pos[k]))].mean()
                     for k in range(len(pos))])
    assert np.abs(mine - archive["kl_site"][:, -1]).max() < 1e-5


def test_the_two_fields_are_far_apart_in_this_archive(archive):
    """What the bug destroyed, stated as a number: `kl_site` is not `kl_glob`
    to within any tolerance a shape or dtype check could ever notice."""
    ratio = archive["kl_site"][:, -1] / archive["kl_glob"][:, -1]
    assert np.median(ratio) > 4.0
