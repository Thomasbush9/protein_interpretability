"""Live alterations to a model, described separately from what is measured after.

The plan asks for the intervention and its measurements to be separate objects,
and there is a concrete reason here: `exp_steer` interleaves them in a
quadruple loop, so the injection algebra -- which is pure tensor arithmetic and
testable anywhere -- cannot be exercised without a GPU and a checkpoint.

`pair` holds the algebra. It imports no backend, so every decision inside it
(the three modes, the doubled diagonal, alpha=0 being exactly the identity) is
pinned by a test that runs on a login node in milliseconds.
"""

from protein_interpretability.intervention.pair import (
    MODES,
    InterventionError,
    PairDirectionIntervention,
    random_directions,
    unit,
)

__all__ = ["MODES", "InterventionError", "PairDirectionIntervention",
           "random_directions", "unit"]
