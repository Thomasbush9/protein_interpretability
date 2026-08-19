"""Live-model side: adapters, capture, and artifact writing.

Importing this subpackage does not import a model backend. The backends are
imported inside the adapter factories, so `inspect` and `render` can resolve a
capture spec, estimate memory and render a job without initialising CUDA.
"""

from protein_interpretability.collection.capture_spec import (
    CaptureSpec,
    CaptureSpecError,
)
from protein_interpretability.collection.cohorts import (
    Assay,
    Cohort,
    CohortError,
)

# `reductions` is NOT re-exported here on purpose: it imports numpy, and
# `cohorts` is deliberately dependency-free so a cohort can be inspected in an
# environment with no scientific stack. Import it as a submodule --
# `from protein_interpretability.collection import reductions` -- which is the
# only place that cost is paid.

__all__ = ["Assay", "CaptureSpec", "CaptureSpecError", "Cohort", "CohortError"]
