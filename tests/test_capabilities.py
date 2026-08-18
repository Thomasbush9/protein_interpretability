"""What the model registry refuses, and how it notices it has gone stale.

A static description of a model is exactly the kind of thing that stops being
true when a wrapper is upgraded, so the tests that matter are: it refuses to
invent what has not been measured, and it can detect its own drift against a
real model.

    uv run pytest tests/test_capabilities.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from protein_interpretability.collection import CaptureSpec, CaptureSpecError
from protein_interpretability.collection import capabilities as caps

SRC = Path(__file__).resolve().parent.parent / "src" / "protein_interpretability"
BACKENDS = {"joltz", "mosaic", "boltz", "torch", "transformers", "jopenfold3",
            "equinox", "esm", "jax", "numpy", "scipy"}


# ---- the login-node property ----------------------------------------------

def test_the_registry_imports_nothing_heavy():
    """It has to resolve while a job is being planned, on a login node, with no
    backend and no CUDA. That is the entire point of a static table."""
    tree = ast.parse((SRC / "collection" / "capabilities.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & BACKENDS), f"imports {sorted(imported & BACKENDS)}"


# ---- refusing to invent ----------------------------------------------------

def test_unknown_model_is_refused():
    with pytest.raises(caps.CapabilityError, match="unknown model"):
        caps.capabilities("esmfold")


def test_an_unmeasured_attribute_raises_rather_than_defaulting():
    """AlphaFold2's trunk depth is not recorded anywhere in this project."""
    af2 = caps.capabilities("af2")
    assert af2.n_trunk_blocks is None
    with pytest.raises(caps.CapabilityError, match="not recorded in this project"):
        af2.require("n_trunk_blocks")


def test_every_unmeasured_attribute_is_declared_unknown():
    """`unknown` must list what is None, so a reader is told rather than
    discovering it from a raise."""
    for name in caps.available():
        c = caps.capabilities(name)
        for attr in ("n_trunk_blocks", "pair_width", "single_width",
                     "distogram_bins", "plddt_granularity"):
            if getattr(c, attr) is None:
                assert attr in c.unknown, f"{name}.{attr} is None but not listed"


def test_every_entry_records_where_its_numbers_came_from():
    for name in caps.available():
        assert len(caps.capabilities(name).evidence) > 40, name


# ---- model-specific semantics that must NOT be normalised away -------------

def test_plddt_granularity_differs_between_models():
    """Per-ATOM in OpenFold3, per-TOKEN in Protenix and Boltz-2. Smoothing this
    over is how a length mismatch becomes a silent reindex."""
    assert caps.capabilities("of3").plddt_granularity == "atom"
    assert caps.capabilities("protenix").plddt_granularity == "token"
    assert caps.capabilities("boltz2").plddt_granularity == "token"


def test_trunk_depths_differ_and_are_recorded():
    depths = {n: caps.capabilities(n).n_trunk_blocks
              for n in ("boltz2", "of3", "protenix")}
    assert depths == {"boltz2": 64, "of3": 48, "protenix": 16}


def test_only_boltz2_has_a_recorded_distogram_grid():
    """The others' grids must be read from the model, not assumed -- a KL across
    two different grids is well-formed and meaningless."""
    assert caps.capabilities("boltz2").distogram_centres is not None
    for n in ("of3", "protenix"):
        assert caps.capabilities(n).distogram_centres is None
        assert "distogram_centres" in caps.capabilities(n).unknown


def test_the_boltz2_grid_is_the_one_every_archived_number_used():
    c = caps.capabilities("boltz2").distogram_centres
    assert len(c) == 64
    assert c[0] == pytest.approx(2.15625)
    assert c[-1] == pytest.approx(21.84375)


# ---- asking a model for something it cannot do -----------------------------

def test_an_alignment_is_refused_for_a_single_sequence_model():
    caps.check_msa("boltz2", use_msa=True)
    with pytest.raises(caps.CapabilityError, match="single-sequence only"):
        caps.check_msa("af2", use_msa=True)
    caps.check_msa("af2", use_msa=False)


def test_a_field_the_wrapper_does_not_emit_is_refused():
    spec = CaptureSpec(model="of3", fields=("kl_glob",), layers="final")
    with pytest.raises(CaptureSpecError, match="does not produce"):
        spec.validate()


def test_a_capture_against_a_model_with_no_recorded_fields_is_refused():
    spec = CaptureSpec(model="af2", fields=("ca",), layers="final")
    with pytest.raises(CaptureSpecError):
        spec.validate()


def test_a_supported_field_passes():
    CaptureSpec(model="of3", fields=("dz_vec",), layers="final").validate()


# ---- drift detection against a real model ---------------------------------

class FakeStacked:
    def __init__(self, depth):
        self.transition_z = type("T", (), {
            "fc1": type("F", (), {"weight": type("W", (), {"shape": (depth, 4)})()})()
        })()


class FakeModel:
    """Shaped like the joltz network the wrappers hold in `.model`."""

    def __init__(self, depth, subsample=True):
        self.pairformer_module = type("PF", (), {})()
        self.pairformer_module.stacked_parameters = FakeStacked(depth)
        self.msa_module = type("MM", (), {"subsample_msa": subsample})()


def test_verify_accepts_a_model_matching_the_table():
    out = caps.verify_against_model("boltz2", FakeModel(64))
    assert out["checked"]["n_trunk_blocks"] == 64
    assert out["unverified"] == []


def test_verify_reports_what_it_could_not_check_rather_than_implying_agreement():
    """The first real run reported of3 and protenix as agreeing on the strength
    of having read nothing: their wrappers do not expose the trunk the way
    Boltz-2's does. "Nothing contradicted me" is not agreement."""
    opaque = object()
    out = caps.verify_against_model("of3", opaque)
    assert out["checked"] == {}
    assert out["unverified"] == ["n_trunk_blocks"], out


def test_verify_raises_when_the_table_no_longer_describes_the_model():
    """The failure this exists for: a wrapper upgrade changes the trunk and the
    table keeps confidently reporting the old number."""
    with pytest.raises(caps.CapabilityError, match="no longer describe"):
        caps.verify_against_model("boltz2", FakeModel(48))


def test_a_deliberately_changed_msa_flag_is_reported_not_refused():
    """pi_models.load(msa='full') sets exactly this; it is a choice, not drift."""
    out = caps.verify_against_model("boltz2", FakeModel(64, subsample=False))
    assert out["checked"]["subsamples_msa"] is False
    assert out["checked"]["subsample_differs_from_default"] is True


def test_describe_mentions_what_is_not_recorded():
    text = caps.describe("af2")
    assert "single-sequence only" in text
    assert "not recorded" in text
