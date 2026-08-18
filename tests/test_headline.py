"""The headline number, as a test.

`transfer_full.json` reports the internal 128-dim probe at Spearman +0.758, and
that is the figure the report opens with. This runs the recipe from scratch --
final-layer dz_site, per-assay z-score, leave-one-assay-out ridge -- and checks
it against the archive per assay, not only on the pooled mean, because twelve
per-assay values that each match is a much stronger statement than one average
that happens to land.

It takes about 4 s and reads the captures under prot_interp_files, so it skips
cleanly anywhere those are not mounted.

    uv run pytest tests/test_headline.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
ARCHIVE = W / "report_master" / "data" / "transfer_full.json"

pytestmark = pytest.mark.skipif(
    not (W / "runs").is_dir() or not ARCHIVE.exists(),
    reason="captures or the archived report input are not available here")


@pytest.fixture(scope="module")
def computed():
    from experiments.analysis.reproduce_headline_transfer import (  # noqa: E402
        ridge, predict, zscore)
    import numpy as np
    from protein_interpretability import artifacts
    from protein_interpretability.analysis import statistics as st
    from protein_interpretability.collection import Cohort

    data = {}
    for assay in Cohort.load("basis_assays"):
        cap = artifacts.load_capture(W / "runs" / f"gym2s_{assay.id}.npz")
        dz = np.asarray(cap.field("dz_site"), float)
        y = np.asarray(cap.field("score"), float)
        data[assay.id.split("_")[0]] = {
            "X": zscore(dz[:, -1, :]), "y": y, "yz": zscore(y)}

    out = {}
    names = sorted(data)
    for held in names:
        train = [n for n in names if n != held]
        Xtr = np.concatenate([data[n]["X"] for n in train], 0)
        ytr = np.concatenate([data[n]["yz"] for n in train], 0)
        w = ridge(Xtr, ytr, 10.0)
        out[held] = float(st.spearman(predict(w, data[held]["X"]), data[held]["y"]))
    return out


@pytest.fixture(scope="module")
def archived():
    return json.loads(ARCHIVE.read_text())["predictors"]["internal_vec"]


def test_pooled_mean_matches_the_report(computed, archived):
    import numpy as np
    assert np.mean(list(computed.values())) == pytest.approx(
        archived["mean"], abs=1e-9)


def test_every_assay_matches_the_archive(computed, archived):
    """Twelve values that each match, not one mean that happens to land."""
    for assay, expected in archived["per_assay"].items():
        assert computed[assay] == pytest.approx(expected, abs=1e-9), assay


def test_the_cohort_is_the_twelve_the_report_used(computed, archived):
    assert set(computed) == set(archived["per_assay"])
    assert len(computed) == 12


def test_the_headline_is_still_what_the_report_claims(archived):
    """Guards the sentence, not just the pipeline: if this figure is ever
    regenerated below ~0.75, the report's opening claim needs rewriting rather
    than the number quietly updating."""
    assert archived["mean"] == pytest.approx(0.758, abs=5e-4)
