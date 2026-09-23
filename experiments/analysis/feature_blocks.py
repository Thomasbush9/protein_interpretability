"""Loading the four feature blocks the two new analyses share.

`reproduce_xmodel_transfer.py` is frozen: it reproduces the archived numbers to
0.00e+00 and states the recipe in one file, so it keeps its own loader even
though this one would serve. What is shared here is only what the new analyses
both need -- the same accessors, the same TM policy, the same paired-cohort
exclusion rule -- so the two of them cannot drift apart from each other.

    internal   dz_vec at the FINAL trunk layer: 128 pair channels at the
               mutated position, mutant minus wild type.
    rich       output_rich, the ten emitted features the archived result is
               reported against.
    geometry   the 37-feature emitted block, prespecified in
               `analysis.emitted_geometry`, from the same coordinates.
    chem       substitution chemistry, 17 features. Model-independent, so it
               transfers between cohorts for free -- which is why a
               cross-cohort internal number means nothing without it alongside.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax_harness"))

from compare_internal_output import OUTPUT_FEATURES, output_matrix  # noqa: E402

from protein_interpretability import artifacts                      # noqa: E402
from protein_interpretability.analysis.chemistry import (            # noqa: E402
    CHEM_FEATURES, chem_matrix,
)
from protein_interpretability.analysis.emitted_geometry import (     # noqa: E402
    GEOMETRY_FEATURES, geometry_matrix,
)

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
MODELS = ("boltz2", "of3", "protenix")
BLOCKS = ("internal", "rich", "geometry", "chem")

# Which side of the comparison each block is on. `internal` is the trunk;
# `rich` and `geometry` are both descriptions of what the structure module
# emitted; `chem` is neither -- it is a model-independent control.
EMITTED = ("rich", "geometry")

FEATURE_NAMES = {"rich": OUTPUT_FEATURES, "geometry": GEOMETRY_FEATURES,
                 "chem": CHEM_FEATURES}


def key_of(assay_id: str) -> str:
    """The short display name, as the archived per-assay tables use."""
    return assay_id.split("_")[0]


def complete_assays(cohort, captures, models, run="r1"):
    """Assays captured in EVERY model. The paired comparison needs all three."""
    ok, missing = [], {}
    for assay in cohort:
        absent = [m for m in models
                  if not (Path(captures) / f"xm_{m}_{run}_{assay.id}.npz").exists()]
        (ok.append(assay) if not absent
         else missing.setdefault(assay.id, absent))
    return ok, missing


def load_blocks(model, assays, captures, tm_cache, run="r1", blocks=BLOCKS):
    """{block: {assay key: X}}, {assay key: y}, for one model.

    Short keys are asserted unique. Two assays on the same protein would
    otherwise silently overwrite one another, and in a CROSS-COHORT design that
    is not a cosmetic collision: it would put a test assay into the training
    pool under another assay's name.
    """
    TM = np.load(tm_cache)
    out = {b: {} for b in blocks}
    target, seen = {}, {}
    for assay in assays:
        k = key_of(assay.id)
        if k in seen:
            raise SystemExit(
                f"short key {k!r} is shared by {seen[k]} and {assay.id}; "
                f"these analyses key assays by it, so the collision must be "
                f"resolved rather than tolerated")
        seen[k] = assay.id

        path = Path(captures) / f"xm_{model}_{run}_{assay.id}.npz"
        if not path.exists():
            raise SystemExit(f"missing {path.name}; collect it before analysing")
        cap = artifacts.load_capture(path, require_meta=True, require_vectors=True)

        if "internal" in out:
            out["internal"][k] = cap.pair_row(-1)
        if "chem" in out:
            out["chem"][k] = chem_matrix(cap.field("mutant"))
        if {"rich", "geometry"} & set(blocks):
            if assay.id not in TM:
                raise SystemExit(
                    f"{assay.id}: no TM in {Path(tm_cache).name}. Run "
                    f"jax_harness/precompute_tm.py over these captures first -- "
                    f"TM comes from each model's OWN coordinates, so a cache "
                    f"cannot be borrowed from another model.")
            args = (cap.field("ca"), cap.field("ca_wt"),
                    np.asarray(TM[assay.id], float),
                    cap.field("plddt_mean"), cap.field("plddt_site"),
                    cap.field("pos"))
            if "rich" in out:
                out["rich"][k] = output_matrix(*args)
            if "geometry" in out:
                out["geometry"][k] = geometry_matrix(*args)
        target[k] = np.asarray(cap.field("score"), float)
    return out, target


def as_probe_blocks(X_by_assay, target):
    return {k: {"X": X_by_assay[k], "y": target[k]} for k in X_by_assay}
