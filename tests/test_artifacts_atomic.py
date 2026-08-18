"""Can a killed job leave something that looks like an archive?

These jobs run under a SLURM wall clock, so they are killed mid-write for real,
and a truncated JSON with a plausible name and size is worse than no file: it is
read, it parses far enough to be interesting, and nothing says it is partial.
`pi_report` already carries an error message about exactly that truncation.

Every test below simulates the failure rather than describing it.

    uv run pytest tests/test_artifacts_atomic.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from protein_interpretability import artifacts

PROTO = {"script": "t.py", "design": "d", "layer": {"which": "final"},
         "features": {"name": "f", "width": 1}, "source": "s", "n_assays": 1}


# ---- JSON results ----------------------------------------------------------

def test_write_result_round_trips(tmp_path):
    p = artifacts.write_result(tmp_path / "r.json", {"x": 1.5}, protocol=PROTO)
    doc = json.loads(p.read_text())
    assert doc["x"] == 1.5
    assert doc["protocol"]["design"] == "d"
    assert doc["provenance"]["argv"]


def test_a_failed_write_leaves_the_previous_result_intact(tmp_path, monkeypatch):
    """The property that matters: a reader sees the old file or the new one."""
    target = tmp_path / "r.json"
    artifacts.write_result(target, {"x": 1.0}, protocol=PROTO)

    boom = json.dumps

    def explode(*a, **k):
        raise RuntimeError("killed mid-serialise")

    monkeypatch.setattr(artifacts.json, "dumps", explode)
    with pytest.raises(RuntimeError):
        artifacts.write_result(target, {"x": 2.0}, protocol=PROTO)
    monkeypatch.setattr(artifacts.json, "dumps", boom)

    assert json.loads(target.read_text())["x"] == 1.0, (
        "the previous archive must survive a failed rewrite")


def test_a_failed_write_leaves_no_scratch_file_behind(tmp_path, monkeypatch):
    """A leftover temp file is a future archive someone mistakes for real."""
    target = tmp_path / "r.json"

    def explode(fh):
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(artifacts, "_atomic_write",
                        artifacts._atomic_write)  # keep the real one
    with pytest.raises(RuntimeError):
        artifacts._atomic_write(target, explode)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == [], f"left {list(tmp_path.iterdir())}"


def test_no_partial_file_is_ever_visible_at_the_target_path(tmp_path):
    """While the body is being written, the target must not exist yet."""
    target = tmp_path / "r.json"
    seen = {}

    def slow(fh):
        fh.write(b'{"partial":')
        seen["existed_midway"] = target.exists()
        fh.write(b" 1}")

    artifacts._atomic_write(target, slow)
    assert seen["existed_midway"] is False
    assert json.loads(target.read_text()) == {"partial": 1}


def test_write_result_still_refuses_a_missing_protocol(tmp_path):
    with pytest.raises(TypeError):
        artifacts.write_result(tmp_path / "r.json", {"x": 1})
    with pytest.raises(ValueError, match="protocol"):
        artifacts.write_result(tmp_path / "r.json", {"x": 1}, protocol={})


# ---- npz captures ----------------------------------------------------------

def test_write_npz_round_trips_and_embeds_its_block(tmp_path):
    p = artifacts.write_npz(tmp_path / "c.npz", {"dz_site": np.zeros((3, 4, 5))},
                            protocol=PROTO)
    with np.load(p, allow_pickle=True) as z:
        assert "dz_site" in z.files
        assert artifacts.NPZ_META_KEY in z.files
        meta = json.loads(str(z[artifacts.NPZ_META_KEY]))
    assert meta["protocol"]["design"] == "d"
    assert meta["arrays"]["dz_site"]["shape"] == [3, 4, 5]


def test_write_npz_does_not_append_npz_to_the_final_name(tmp_path):
    """np.savez appends `.npz` to a PATH but not to a file object. Writing
    through a handle is what keeps the target name exact."""
    p = artifacts.write_npz(tmp_path / "c.npz", {"a": np.arange(3)},
                            protocol=PROTO)
    assert p.name == "c.npz"
    assert not (tmp_path / "c.npz.npz").exists()
    assert sorted(x.name for x in tmp_path.iterdir()) == ["c.npz"]


def test_a_failed_npz_write_leaves_the_previous_capture_intact(tmp_path):
    target = tmp_path / "c.npz"
    artifacts.write_npz(target, {"a": np.arange(3)}, protocol=PROTO)

    class Unwritable:
        def __array__(self, *a, **k):
            raise RuntimeError("killed mid-save")

    with pytest.raises(Exception):
        artifacts.write_npz(target, {"a": Unwritable()}, protocol=PROTO)

    with np.load(target, allow_pickle=True) as z:
        assert list(z["a"]) == [0, 1, 2], "the previous capture must survive"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["c.npz"]


# ---- metadata on read ------------------------------------------------------

def test_require_meta_accepts_an_archive_written_through_the_seam(tmp_path):
    p = artifacts.write_npz(tmp_path / "c.npz", {"a": np.arange(3)},
                            protocol=PROTO)
    cap = artifacts.load_capture(p, require_meta=True)
    assert "a" in cap.files


def test_require_meta_refuses_an_archive_that_cannot_describe_itself(tmp_path):
    raw = tmp_path / "legacy.npz"
    np.savez_compressed(raw, a=np.arange(3))
    artifacts.load_capture(raw)                       # still loads by default
    with pytest.raises(ValueError, match="carries no embedded"):
        artifacts.load_capture(raw, require_meta=True)


def test_require_meta_is_satisfied_by_a_sidecar(tmp_path):
    """The 12 GB of pre-convention captures cost GPU-hours; a sidecar records
    their shapes without rewriting them."""
    raw = tmp_path / "legacy.npz"
    np.savez_compressed(raw, a=np.arange(3))
    artifacts.write_capture_sidecar(raw)
    artifacts.load_capture(raw, require_meta=True)
