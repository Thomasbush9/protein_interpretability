"""A named twelve-assay cohort must not quietly become an eleven-assay result.

The example analysis printed `skip` for a capture it could not find and carried
on. What it wrote was a real number over whatever happened to be on disk, with
`n_assays` recording the survivors -- true, and impossible to read as an
omission unless you already knew what the cohort held. The pooled interval is
over assays, so a missing one moves the reported figure.

    uv run pytest tests/test_example_transfer_probe.py -q
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
SCRIPT = REPO / "experiments" / "analysis" / "example_transfer_probe.py"


def _load():
    """Import the experiment script by path.

    `uv run pytest` does not put the working directory on sys.path -- which is
    what broke an earlier attempt to reach into experiments/ -- so the file is
    loaded by location rather than by package name.
    """
    spec = importlib.util.spec_from_file_location("_example_transfer", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load()

needs_captures = pytest.mark.skipif(
    not (W / "runs").is_dir(),
    reason="the captures are not mounted here")


# ---- the check itself, with no data at all ---------------------------------

def _stub_cohort(ids):
    return [SimpleNamespace(id=i) for i in ids]


def test_missing_captures_names_every_absent_assay(tmp_path):
    (tmp_path / "gym2s_B.npz").write_bytes(b"")
    assert probe.missing_captures(_stub_cohort(["A", "B", "C"]), tmp_path) \
        == ["A", "C"]


def test_nothing_is_missing_when_everything_is_there(tmp_path):
    for i in ("A", "B"):
        (tmp_path / f"gym2s_{i}.npz").write_bytes(b"")
    assert probe.missing_captures(_stub_cohort(["A", "B"]), tmp_path) == []


# ---- the refusal, on the real cohort ---------------------------------------

@needs_captures
def test_a_missing_capture_refuses_by_default(tmp_path):
    """The default path produces no result at all -- not a result over eleven."""
    out = tmp_path / "result.json"
    with pytest.raises(SystemExit) as exc:
        probe.main(["--captures", str(tmp_path), "--out", str(out)])
    assert "no capture" in str(exc.value)
    assert not out.exists(), "a refusal must not leave a file behind"


@needs_captures
def test_a_partial_run_says_so_in_the_result_and_the_protocol(tmp_path):
    """`--allow-partial` is allowed to run. It is not allowed to be quiet."""
    from protein_interpretability.collection import Cohort

    cohort = Cohort.load("basis_assays")
    present = cohort.ids[:2]
    captures = tmp_path / "captures"
    captures.mkdir()
    for assay_id in present:
        src = W / "runs" / f"gym2s_{assay_id}.npz"
        if not src.exists():
            pytest.skip(f"{src.name} is not on disk here")
        (captures / src.name).symlink_to(src)

    out = tmp_path / "partial.json"
    probe.main(["--captures", str(captures), "--allow-partial",
                "--out", str(out)])

    doc = json.loads(out.read_text())
    assert doc["partial"] is True
    assert doc["cohort_size"] == len(cohort)
    assert set(doc["missing_assays"]) == set(cohort.ids) - set(present)
    assert doc["pooled"]["n_assays"] == len(present)
    # The protocol block is what travels with a quoted number, so the omission
    # has to be legible there too and not only in the payload.
    assert doc["protocol"]["partial"] is True
    assert doc["protocol"]["n_assays"] == len(present)
    assert set(doc["protocol"]["missing_assays"]) == set(doc["missing_assays"])
