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

class _Node:
    """A plain attribute holder, so the walker sees instance attributes the way
    it does on a real equinox module."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def fake_model(depth, subsample=True, *, stack_attr="stacked_parameters",
               container="pairformer_module", inner="model"):
    """Shaped like the network a wrapper holds in `.model`.

    Leaves are real arrays: every leaf under a stacked block shares the leading
    scan axis, which is what the depth is read from.
    """
    import numpy as np

    stacked = _Node(transition_z=_Node(fc1=_Node(weight=np.zeros((depth, 4)))))
    net = _Node(msa_module=_Node(subsample_msa=subsample))
    setattr(net, container, _Node(**{stack_attr: stacked}))
    # Wrappers hold the network one level down, and not always under the same
    # name: Protenix uses `.protenix` where the others use `.model`.
    return _Node(**{inner: net})


def FakeModel(depth, subsample=True):
    return fake_model(depth, subsample)


def test_verify_accepts_a_model_matching_the_table():
    out = caps.verify_against_model("boltz2", FakeModel(64))
    assert out["checked"]["n_trunk_blocks"] == 64
    assert out["unverified"] == []


def test_verify_reports_what_it_could_not_check_rather_than_implying_agreement():
    """"Nothing contradicted me" is not agreement. The first real run reported
    of3 and protenix as agreeing having read nothing from either."""
    out = caps.verify_against_model("of3", _Node())
    assert out["checked"] == {}
    assert out["unverified"] == ["n_trunk_blocks"], out


def test_each_model_is_read_from_where_it_actually_keeps_its_stack():
    """The three wrappers name it differently -- pairformer_module /
    stacked_parameters for Boltz-2, pairformer_stack / stacked_params for
    OpenFold3, pairformer_stack / stacked_parameters for Protenix. One generic
    accessor found only the first and called the others agreement."""
    of3 = fake_model(48, container="pairformer_stack", stack_attr="stacked_params")
    assert caps.observed_trunk_depth("of3", of3) == 48

    ptx = fake_model(16, container="pairformer_stack",
                     stack_attr="stacked_parameters", inner="protenix")
    assert caps.observed_trunk_depth("protenix", ptx) == 16

    # Read through the wrong accessor, OpenFold3's stack is invisible.
    assert caps.observed_trunk_depth("boltz2", of3) is None


def test_a_wrong_depth_is_caught_for_of3_too():
    with pytest.raises(caps.CapabilityError, match="no longer describe"):
        caps.verify_against_model(
            "of3", fake_model(64, container="pairformer_stack",
                              stack_attr="stacked_params"))


def test_protenix_is_read_through_its_own_inner_field():
    """`.protenix`, not `.model`. Reading it as `.model` is why the real run
    reported it unverified after of3 was already working."""
    ptx = fake_model(16, container="pairformer_stack",
                     stack_attr="stacked_parameters", inner="protenix")
    assert caps.observed_trunk_depth("protenix", ptx) == 16
    assert caps.verify_against_model("protenix", ptx)["checked"][
        "n_trunk_blocks"] == 16


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
