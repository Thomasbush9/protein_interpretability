"""Protocol text must be derived from the cohort, never typed as a constant.

`analyze_transfer.py` hardcoded "train on 11 assays, test on the 12th" and the
string was archived, verbatim, into the 16- and 25-assay transfer results
(found by the 2026-09-06 publication audit). The design sentence is now built
by `loao_design`, and this test is the refusal the audit asked for: the text
must track the number it describes, and a cohort the protocol cannot describe
must raise rather than produce a plausible sentence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jax_harness"))

from analyze_transfer import loao_design  # noqa: E402


@pytest.mark.parametrize("n", [12, 16, 25])
def test_design_text_tracks_cohort_size(n):
    text = loao_design(n)
    assert f"train on {n - 1} assays" in text
    assert f"{n} in the cohort" in text


def test_the_archived_constant_is_gone():
    """The exact string the audit found in the panel5 archives must never be
    producible for a cohort that is not the 12-assay one."""
    for n in (16, 25):
        assert "train on 11 assays" not in loao_design(n)
    src = (Path(__file__).resolve().parents[1]
           / "jax_harness/analyze_transfer.py").read_text()
    assert "train on 11 assays" not in src


def test_refuses_a_cohort_it_cannot_describe():
    with pytest.raises(ValueError):
        loao_design(1)
    with pytest.raises(ValueError):
        loao_design(0)
