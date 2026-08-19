"""A task must refuse before a GPU, not after one.

Everything here runs on a login node in milliseconds, which is the property
under test as much as any individual refusal: `inspect()` resolves a whole
collection run -- model, regime, fields, layers, cohort, size -- and imports no
backend to do it.

    uv run pytest tests/test_task.py -q
"""

from __future__ import annotations

import pytest

from protein_interpretability.collection import CaptureSpec, Cohort
from protein_interpretability.collection.task import (
    CollectionTask,
    ModelSpec,
    ResolvedTask,
    TaskError,
    resolve_layers,
)


# ---- layer resolution ------------------------------------------------------

def test_all_and_final_resolve_against_the_models_own_depth():
    """'final' is block 63 in Boltz-2 and 15 in Protenix. Until it is resolved
    against a depth it is not an answer."""
    assert resolve_layers("all", 16) == tuple(range(16))
    assert resolve_layers("final", 64) == (63,)
    assert resolve_layers("final", 16) == (15,)


def test_negative_indices_become_absolute():
    assert resolve_layers((0, -1), 48) == (0, 47)
    assert resolve_layers((-2, -1), 64) == (62, 63)


def test_an_index_valid_for_one_model_is_refused_for_a_shallower_one():
    assert resolve_layers((15, 31, 63), 64) == (15, 31, 63)
    with pytest.raises(TaskError, match="outside a 16-block trunk"):
        resolve_layers((15, 31, 63), 16)


def test_duplicates_are_refused():
    """A repeated layer is weighted twice by anything pooling over the axis."""
    with pytest.raises(TaskError, match="more than once"):
        resolve_layers((5, 5), 64)
    with pytest.raises(TaskError, match="more than once"):
        resolve_layers((63, -1), 64)          # the same block, spelled twice


def test_out_of_order_is_refused():
    with pytest.raises(TaskError, match="increasing depth order"):
        resolve_layers((63, 15), 64)


def test_an_empty_selection_is_refused():
    with pytest.raises(TaskError, match="captures nothing"):
        resolve_layers((), 64)


# ---- the model spec --------------------------------------------------------

def test_an_unknown_model_is_refused():
    with pytest.raises(Exception, match="unknown model"):
        ModelSpec(name="alphafold9").validate()


def test_an_msa_asked_of_a_single_sequence_model_is_refused():
    with pytest.raises(Exception, match="single-sequence only"):
        ModelSpec(name="af2", msa="full").validate()


def test_a_backend_the_model_is_not_wrapped_by_is_refused():
    with pytest.raises(TaskError, match="not one of them"):
        ModelSpec(name="of3", backend="joltz").validate()
    ModelSpec(name="of3", backend="mosaic").validate()


def test_zero_recycles_is_refused():
    with pytest.raises(TaskError, match="recycles must be >= 1"):
        ModelSpec(name="boltz2", recycles=0).validate()


def test_the_declaration_records_the_regime_not_just_the_name():
    d = ModelSpec(name="boltz2", recycles=3, seed=7, msa="subsample").declaration()
    assert d["seed"] == 7 and d["msa"] == "subsample"
    assert d["network"] == "blocked", "network blocking is the default"


# ---- the task --------------------------------------------------------------

def _task(**over):
    kw = dict(
        name="t",
        cohort=Cohort.load("cross_model_assays"),
        model=ModelSpec(name="of3", recycles=3),
        capture=CaptureSpec(model="of3", fields=("dz_vec",), layers="all",
                            reduction="vector", recycles=3),
        output="runs/t",
        n_variants=10,
    )
    kw.update(over)
    return CollectionTask(**kw)


def test_a_task_resolves_without_importing_a_backend():
    r = _task().inspect(verify_inputs=False)
    assert isinstance(r, ResolvedTask)
    assert r.trunk_depth == 48 and len(r.layers) == 48
    assert r.estimated_bytes > 0


def test_a_model_named_twice_and_differently_is_refused():
    """The capture spec and the model spec both carry a model name."""
    with pytest.raises(TaskError, match="left over from another experiment"):
        _task(model=ModelSpec(name="boltz2", recycles=3)).inspect(
            verify_inputs=False)


def test_recycles_declared_twice_and_disagreeing_is_refused():
    with pytest.raises(TaskError, match="declared twice and disagrees"):
        _task(model=ModelSpec(name="of3", recycles=4)).inspect(
            verify_inputs=False)


def test_an_unknown_resume_policy_is_refused():
    with pytest.raises(TaskError, match="resume must be"):
        _task(resume="clobber").inspect(verify_inputs=False)


def test_the_task_id_is_stable_and_ignores_where_the_output_goes():
    """Writing the same measurement to a second path does not make it a
    different measurement; changing the seed does."""
    a = _task().inspect(verify_inputs=False)
    b = _task(output="runs/somewhere_else").inspect(verify_inputs=False)
    c = _task(model=ModelSpec(name="of3", recycles=3, seed=1)).inspect(
        verify_inputs=False)
    assert a.task_id == b.task_id
    assert a.task_id != c.task_id


def test_the_task_id_changes_with_the_layer_selection():
    a = _task().inspect(verify_inputs=False)
    b = _task(capture=CaptureSpec(model="of3", fields=("dz_vec",),
                                  layers=(0, 47), reduction="vector",
                                  recycles=3)).inspect(verify_inputs=False)
    assert a.task_id != b.task_id
    assert b.layers == (0, 47)


def test_a_resolved_task_round_trips_through_json():
    import json

    r = _task().inspect(verify_inputs=False)
    doc = json.loads(json.dumps(r.to_dict()))
    assert doc["task_id"] == r.task_id
    assert doc["layers"] == list(r.layers)
    assert doc["model"]["model"] == "of3"


def test_selected_layers_are_priced_not_the_whole_trunk():
    """Requesting three layers must not be estimated as if it were 48."""
    everything = _task().inspect(verify_inputs=False)
    three = _task(capture=CaptureSpec(model="of3", fields=("dz_vec",),
                                      layers=(0, 24, 47), reduction="vector",
                                      recycles=3)).inspect(verify_inputs=False)
    assert three.estimated_bytes * 10 < everything.estimated_bytes
