"""The registry says what each model produces. The archives say what it did.

`capabilities.REGISTRY[...].capture_fields` is the list `check_spec` refuses
against, so a field missing from it is a refusal for a capture that demonstrably
works. That is not hypothetical: `of3` and `protenix` were both declared as not
producing `kl_site`/`kl_glob` while every `xm_of3_r1_*.npz` and
`xm_protenix_r1_*.npz` on disk carried both, and `boltz2` was declared without
`dz_vec`/`ds_vec` while its own cross-model captures hold them.

Nothing caught it because nothing compared the declaration to the evidence it
cites. This does, in the direction that matters: every array an archive actually
holds must be declared. The converse is deliberately not asserted -- a model may
be able to produce a field no archive here happens to contain.

    uv run pytest tests/test_capture_fields_match_archives.py -q
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from protein_interpretability.collection import capabilities as caps
from protein_interpretability.collection.capture_spec import FIELDS

RUNS = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/runs")

pytestmark = pytest.mark.skipif(
    not RUNS.is_dir(), reason="the captures are not mounted here")

# Bookkeeping an archive carries that is not a captured QUANTITY: labels, the
# protocol scalars, the wild-type references and the fidelity evidence. These
# describe the capture rather than being one of its fields.
NOT_FIELDS = {
    "_pi_meta", "mutant", "assay", "model", "wt_seq", "bin",
    "n_layers", "recycles", "msa_cap", "msa_depth", "sampling_steps",
    "msa_regime", "msa_subsample_rows", "reproducibility",
    "capture_drift", "signal_to_drift", "drift_tol",
    "ca_wt", "disto_wt", "plddt_wt",
    "pair_i", "pair_j", "n_tokens",
    "dmu_site", "dmu_glob", "dsd_site", "dsd_glob",   # jeffreys split halves
}

FAMILY_MODEL = {"xm_boltz2": "boltz2", "xm_of3": "of3", "xm_protenix": "protenix",
                "gym2s": "boltz2"}


def archives():
    out = []
    for prefix, model in FAMILY_MODEL.items():
        hits = sorted(glob.glob(str(RUNS / f"{prefix}_*.npz")))
        if hits:
            out.append((prefix, model, hits[0]))
    return out


def test_there_are_archives_to_check():
    """Guard the guard: an empty list would make every case below vacuous."""
    assert len(archives()) >= 3


@pytest.mark.parametrize("prefix,model,path", archives(),
                         ids=lambda v: v if isinstance(v, str) and "/" not in str(v) else "")
def test_every_field_an_archive_holds_is_declared(prefix, model, path):
    import numpy as np

    with np.load(path, allow_pickle=True) as z:
        present = {k for k in z.files} - NOT_FIELDS
    declared = set(caps.capabilities(model).capture_fields)
    undeclared = sorted(present - declared)
    assert not undeclared, (
        f"{Path(path).name} holds {undeclared}, which "
        f"capabilities.REGISTRY[{model!r}].capture_fields does not declare. A "
        f"spec asking for one of them is refused by check_spec before the run "
        f"-- for a capture this archive proves works.")


@pytest.mark.parametrize("prefix,model,path", archives(),
                         ids=lambda v: v if isinstance(v, str) and "/" not in str(v) else "")
def test_every_declared_field_has_a_shape_contract(prefix, model, path):
    """A field the registry offers but `CaptureSpec` cannot shape-check is a
    field no archive can be validated against."""
    import numpy as np

    with np.load(path, allow_pickle=True) as z:
        present = {k for k in z.files} - NOT_FIELDS
    unknown = sorted(present - set(FIELDS))
    assert not unknown, (
        f"{Path(path).name} holds {unknown}, which capture_spec.FIELDS does "
        f"not describe, so expected_shapes() cannot check them.")


def test_the_three_trunk_depths_are_the_ones_on_disk():
    """Depth is what `final` resolves to, so a wrong one silently moves which
    layer every cross-model comparison reads."""
    import numpy as np

    for prefix, model, path in archives():
        if not prefix.startswith("xm_"):
            continue
        with np.load(path, allow_pickle=True) as z:
            on_disk = int(z["n_layers"])
        assert caps.capabilities(model).n_trunk_blocks == on_disk, model
