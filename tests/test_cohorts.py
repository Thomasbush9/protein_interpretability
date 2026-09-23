"""Does a cohort notice when the thing underneath it changed?

That is the only reason the manifests exist. A cohort that loads is not
interesting; a cohort that refuses to load because an alignment was regenerated
in place is the whole point, because that failure is otherwise silent -- the run
succeeds and returns a plausible number computed from an input nobody chose.

    uv run pytest tests/test_cohorts.py
"""

from __future__ import annotations

import hashlib

import pytest

from protein_interpretability.collection import Cohort, CohortError

MANIFEST = """\
cohort: tiny
description: two assays for testing
derived_from: nothing
n_assays: 2
assays:
  - id: AAA_TEST
    assay_csv:
      path: {csv_a}
      sha256: {sha_a}
    n_single_variants: 11
    wt_length: 7
    msa:
      path: {msa_a}
      sha256: {sha_msa_a}
      rows: 3
      panel: panelX
  - id: BBB_TEST
    assay_csv:
      path: {csv_b}
      sha256: {sha_b}
    n_single_variants: 22
    wt_length: 9
    msa: null
"""


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def cohort(tmp_path):
    csv_a = tmp_path / "a.csv"; csv_a.write_text("mutant,x\nA1B,1\n")
    csv_b = tmp_path / "b.csv"; csv_b.write_text("mutant,x\nC2D,2\n")
    msa_a = tmp_path / "a.a3m"; msa_a.write_text(">q\nAAAA\n>h\nAAAB\n")
    man = tmp_path / "tiny.yaml"
    man.write_text(MANIFEST.format(
        csv_a=csv_a, sha_a=sha(csv_a), csv_b=csv_b, sha_b=sha(csv_b),
        msa_a=msa_a, sha_msa_a=sha(msa_a)))
    return Cohort.from_manifest(man), tmp_path


def test_parses_the_manifest(cohort):
    c, _ = cohort
    assert c.name == "tiny" and len(c) == 2
    assert c.ids == ["AAA_TEST", "BBB_TEST"]
    a = c.assays[0]
    assert a.wt_length == 7 and a.n_single_variants == 11 and a.msa_rows == 3
    assert c.assays[1].msa_path is None, "a null msa must parse as absent"


def test_verify_passes_when_nothing_moved(cohort):
    c, _ = cohort
    c.verify()


def test_verify_catches_an_input_rewritten_in_place(cohort):
    """The failure this exists for: same path, same size, different content."""
    c, tmp = cohort
    (tmp / "a.a3m").write_text(">q\nAAAA\n>h\nAAAC\n")
    with pytest.raises(CohortError, match="has changed since the manifest"):
        c.verify()


def test_verify_catches_a_deleted_input(cohort):
    c, tmp = cohort
    (tmp / "a.csv").unlink()
    with pytest.raises(CohortError, match="missing"):
        c.verify()


def test_existence_only_check_skips_hashing(cohort):
    c, tmp = cohort
    (tmp / "a.a3m").write_text(">q\nAAAA\n>h\nAAAC\n")
    c.verify(checksums=False)
    with pytest.raises(CohortError):
        c.verify()


def test_verify_reports_every_problem_not_just_the_first(cohort):
    c, tmp = cohort
    (tmp / "a.csv").unlink()
    (tmp / "b.csv").unlink()
    with pytest.raises(CohortError) as exc:
        c.verify()
    assert "AAA_TEST" in str(exc.value) and "BBB_TEST" in str(exc.value)


def test_disjointness_is_checkable(cohort):
    c, _ = cohort
    with pytest.raises(CohortError, match="not held out"):
        c.assert_disjoint(c)


def test_unknown_cohort_lists_what_exists():
    with pytest.raises(KeyError, match="have"):
        Cohort.load("no_such_cohort")


# ---- the manifest document itself ------------------------------------------

HEADER = "cohort: tiny\ndescription: d\nderived_from: x\nn_assays: {n}\nassays:\n"


def entry(name, csv_path, sha):
    return (f"  - id: {name}\n"
            f"    assay_csv:\n"
            f"      path: {csv_path}\n"
            f"      sha256: {sha}\n")


def write(tmp_path, text) -> "object":
    p = tmp_path / "m.yaml"
    p.write_text(text)
    return p


def test_refuses_a_manifest_with_no_assays(tmp_path):
    """The dangerous one: an empty cohort loads, iterates zero times, and
    verify() passes for having checked nothing."""
    with pytest.raises(CohortError, match="no assays"):
        Cohort.from_manifest(write(tmp_path, HEADER.format(n=0)))


def test_refuses_duplicate_ids(tmp_path):
    csv_p = tmp_path / "a.csv"; csv_p.write_text("x\n")
    s = sha(csv_p)
    doc = HEADER.format(n=2) + entry("SAME", csv_p, s) + entry("SAME", csv_p, s)
    with pytest.raises(CohortError, match="duplicate ids"):
        Cohort.from_manifest(write(tmp_path, doc))


def test_refuses_a_header_that_disagrees_with_the_list(tmp_path):
    """A manifest that contradicts itself was hand-edited, not regenerated."""
    csv_p = tmp_path / "a.csv"; csv_p.write_text("x\n")
    doc = HEADER.format(n=12) + entry("ONE", csv_p, sha(csv_p))
    with pytest.raises(CohortError, match="disagrees with itself"):
        Cohort.from_manifest(write(tmp_path, doc))


def test_refuses_an_assay_naming_no_inputs(tmp_path):
    doc = HEADER.format(n=1) + "  - id: NOTHING\n    wt_length: 7\n"
    with pytest.raises(CohortError, match="name no inputs"):
        Cohort.from_manifest(write(tmp_path, doc))


def test_refuses_an_assay_with_no_id(tmp_path):
    csv_p = tmp_path / "a.csv"; csv_p.write_text("x\n")
    doc = (HEADER.format(n=1)
           + f"  - assay_csv:\n      path: {csv_p}\n      sha256: {sha(csv_p)}\n")
    with pytest.raises(CohortError, match="no id"):
        Cohort.from_manifest(write(tmp_path, doc))


def test_an_empty_cohort_cannot_be_verified_even_if_constructed(tmp_path):
    """Belt and braces: the loader refuses one, and verify() refuses it too, so
    a cohort built another way cannot report a vacuous pass."""
    from protein_interpretability.collection import Cohort as C
    with pytest.raises(CohortError, match="holds no assays"):
        C("empty", "", []).verify()


# ---- the real manifests ----------------------------------------------------

def test_the_shipped_cohorts_load_and_are_complete():
    names = Cohort.available()
    assert {"basis_assays", "heldout_assays", "cross_model_assays"} <= set(names)
    sizes = {n: len(Cohort.load(n)) for n in names}
    assert sizes["basis_assays"] == 12
    assert sizes["cross_model_assays"] == 4
    for name in names:
        for assay in Cohort.load(name):
            assert assay.id and assay.inputs(), f"{name}/{assay.id} has no inputs"


def test_heldout_is_disjoint_from_the_basis():
    """heldout_v1's claim, as a test rather than a sentence in a docstring."""
    Cohort.load("basis_assays").assert_disjoint(Cohort.load("heldout_assays"))
