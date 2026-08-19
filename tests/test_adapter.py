"""The adapter's decisions that do not need a GPU to be wrong.

Layer selection, resume, and the registry are all places where a mistake
produces a plausible artifact rather than an error, so they are tested here
against synthetic arrays instead of waiting for a job to come back.

    uv run pytest tests/test_adapter.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from protein_interpretability.collection import CaptureSpec, Cohort
from protein_interpretability.collection.models import (
    AdapterError,
    adapter_for,
    available,
)
from protein_interpretability.collection.models.trunk import (
    _resume,
    _select_layers,
    _work_dir,
)
from protein_interpretability.collection.task import CollectionTask, ModelSpec


def _resolved(layers="all", model="protenix", **over):
    kw = dict(
        name="t",
        cohort=Cohort.load("cross_model_assays"),
        model=ModelSpec(name=model, recycles=3),
        capture=CaptureSpec(model=model, fields=("dz_vec",), layers=layers,
                            reduction="vector", recycles=3),
        output="runs/t",
        n_variants=5,
    )
    kw.update(over)
    return CollectionTask(**kw).inspect(verify_inputs=False)


def _arrays(depth=16, n=5):
    return {
        "dz_vec": np.arange(n * depth * 4, dtype=np.float32).reshape(n, depth, 4),
        "ds_vec": np.zeros((n, depth, 8), np.float32),
        "kl_glob": np.zeros((n, depth), np.float32),
        "kl_site": np.zeros((n, depth), np.float32),
        "dz_site": np.zeros((n, depth), np.float32),
        "ds_site": np.zeros((n, depth), np.float32),
        "score": np.zeros(n),
        "ca": np.zeros((n, 40, 3), np.float32),
        "n_layers": np.array(depth),
    }


# ---- the registry ----------------------------------------------------------

def test_the_registry_is_narrower_than_the_capability_table():
    """af2 is describable and not runnable, and the error has to say so."""
    assert set(available()) == {"boltz2", "of3", "protenix"}
    with pytest.raises(AdapterError, match="being describable is not being"):
        adapter_for("af2", ModelSpec(name="af2", msa="none"))


def test_listing_adapters_imports_no_backend():
    import sys

    available()
    for backend in ("joltz", "mosaic", "torch", "jopenfold3"):
        assert backend not in sys.modules, (
            f"listing the adapters imported {backend}; inspect and render have "
            f"to stay runnable on a login node")


# ---- layer selection -------------------------------------------------------

def test_asking_for_every_layer_changes_nothing():
    arrays = _arrays()
    out = _select_layers(arrays, _resolved("all"), np)
    assert out["dz_vec"].shape == (5, 16, 4)
    assert out is arrays, "an all-layers request should not copy the arrays"


def test_a_selection_keeps_only_those_layers_and_labels_them():
    resolved = _resolved((0, 7, 15))
    out = _select_layers(_arrays(), resolved, np)
    assert resolved.layers == (0, 7, 15)
    assert out["dz_vec"].shape == (5, 3, 4)
    assert out["kl_site"].shape == (5, 3)
    assert int(out["n_layers"]) == 3


def test_the_selection_takes_the_right_rows():
    """Not merely the right SHAPE: an off-by-one here is undetectable later."""
    arrays = _arrays()
    full = arrays["dz_vec"].copy()
    out = _select_layers(arrays, _resolved((0, 7, 15)), np)
    assert np.array_equal(out["dz_vec"][:, 0], full[:, 0])
    assert np.array_equal(out["dz_vec"][:, 1], full[:, 7])
    assert np.array_equal(out["dz_vec"][:, 2], full[:, 15])


def test_fields_without_a_layer_axis_are_untouched():
    out = _select_layers(_arrays(), _resolved((0, 7, 15)), np)
    assert out["ca"].shape == (5, 40, 3)
    assert out["score"].shape == (5,)


def test_a_backend_that_returns_the_wrong_depth_is_refused():
    """The whole point of resolving layers: 48 blocks against a 16-block task
    means every index in the artifact denotes a different layer."""
    with pytest.raises(AdapterError, match="returned 48 layers"):
        _select_layers(_arrays(depth=48), _resolved((0, 7, 15)), np)


def test_a_field_whose_layer_axis_is_the_wrong_length_is_refused():
    arrays = _arrays()
    arrays["kl_site"] = np.zeros((5, 9), np.float32)
    with pytest.raises(AdapterError, match="layer axis is not"):
        _select_layers(arrays, _resolved((0, 7, 15)), np)


# ---- resume ----------------------------------------------------------------

def _write(path, task_id):
    from protein_interpretability import artifacts

    artifacts.write_npz(path, {"x": np.zeros(3)},
                        protocol={"script": "t", "task_id": task_id})


def test_an_artifact_from_this_task_is_resumable(tmp_path):
    resolved = _resolved()
    path = tmp_path / "a.npz"
    _write(path, resolved.task_id)
    assert _resume(path, resolved) == "skip"


def test_an_artifact_from_a_different_task_is_never_overwritten(tmp_path):
    resolved = _resolved()
    path = tmp_path / "a.npz"
    _write(path, "some_other_task")
    with pytest.raises(AdapterError, match="not "):
        _resume(path, resolved)


def test_an_artifact_with_no_recorded_task_is_refused(tmp_path):
    """A capture that cannot say which task made it is not resumable work."""
    resolved = _resolved()
    path = tmp_path / "a.npz"
    from protein_interpretability import artifacts
    artifacts.write_npz(path, {"x": np.zeros(3)}, protocol={"script": "t"})
    with pytest.raises(AdapterError, match="<unrecorded>"):
        _resume(path, resolved)


def test_overwrite_is_only_ever_deliberate(tmp_path):
    resolved = _resolved(resume="overwrite")
    path = tmp_path / "a.npz"
    _write(path, "some_other_task")
    assert _resume(path, resolved) == "write"


# ---- the login-node fast path ----------------------------------------------

def test_a_finished_sweep_is_a_no_op_that_loads_no_model(tmp_path, monkeypatch):
    """Relaunching after a partial sweep is the NORMAL case -- that is what the
    resume policy is for -- so discovering there is nothing to do must not cost
    a model load. It also makes "what is left to collect?" answerable on a login
    node, which is where you actually want to ask it.
    """
    import sys

    task = CollectionTask(
        name="t",
        cohort=Cohort.load("cross_model_assays"),
        model=ModelSpec(name="protenix", recycles=3),
        capture=CaptureSpec(model="protenix", fields=("dz_vec",), layers="all",
                            reduction="vector", recycles=3),
        output=str(tmp_path),
        resume="resume",
        n_variants=5,
    )
    resolved = task.inspect(verify_inputs=False)
    for assay_id in resolved.assays:
        _write(resolved.output_for(assay_id), resolved.task_id)

    adapter = adapter_for("protenix", task.model)
    written = adapter.collect_cohort(task, resolved)

    assert len(written) == len(resolved.assays)
    for backend in ("jax", "mosaic", "joltz", "exp_gym_deep"):
        assert backend not in sys.modules, (
            f"an all-skip run imported {backend}; the whole point of deciding "
            f"resume before loading is that this path costs no GPU time")


def test_one_missing_assay_still_reaches_the_backend(tmp_path, monkeypatch):
    """The converse: the fast path must not swallow real work.

    The backend stage is stubbed to raise a known error rather than left to
    fail on a missing import. Otherwise this test passes for the wrong reason
    where jax is absent, and LOADS A MODEL where jax is present -- which is the
    one thing a unit test must never do.
    """
    from protein_interpretability.collection.models import trunk

    def _no_harness():
        raise AdapterError("reached the backend stage")

    monkeypatch.setattr(trunk, "_harness", _no_harness)

    task = CollectionTask(
        name="t",
        cohort=Cohort.load("cross_model_assays"),
        model=ModelSpec(name="protenix", recycles=3),
        capture=CaptureSpec(model="protenix", fields=("dz_vec",), layers="all",
                            reduction="vector", recycles=3),
        output=str(tmp_path),
        resume="resume",
        n_variants=5,
    )
    resolved = task.inspect(verify_inputs=False)
    for assay_id in resolved.assays[:-1]:
        _write(resolved.output_for(assay_id), resolved.task_id)

    adapter = adapter_for("protenix", task.model)
    with pytest.raises(AdapterError, match="reached the backend stage"):
        adapter.collect_cohort(task, resolved)


# ---- work directories ------------------------------------------------------

def test_work_directories_are_unique_per_job(monkeypatch):
    """Two jobs sharing one work directory race on the per-variant alignment,
    which is the most likely explanation on record for the A10E archive row."""
    resolved = _resolved()
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    a = _work_dir(resolved, "ASSAY")
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    b = _work_dir(resolved, "ASSAY")
    assert a != b and "111" in str(a) and "222" in str(b)


def test_two_assays_in_one_job_do_not_share_a_work_directory(monkeypatch):
    resolved = _resolved()
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    assert _work_dir(resolved, "A") != _work_dir(resolved, "B")
