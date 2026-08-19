"""Which adapter runs which model. Importing this imports no backend.

The registry maps a model name to a FACTORY, not to an instance, and the factory
imports its backend when it is called. That is what keeps `pi models`, `pi
collect inspect` and `CollectionTask.inspect()` runnable on a login node while
`CollectionTask.run()` reaches the same object by the same name.

All three supported models share one adapter on purpose. `exp_gym_deep.py` runs
Boltz-2, OpenFold3 and Protenix through a single script so that "variant
selection, feature definitions, alignment handling, recycles and sampling steps
are identical across them by construction", and that construction is the reason
their numbers are comparable at all. Three adapter classes that had to be kept
identical would put that guarantee back into a maintainer's hands, which is
where it was before the cross-model captures were rebuilt.

Per-model behaviour still exists -- each model has its own capture kernel, its
own feature builder and its own distogram head -- but it lives where it already
lived, in `pi_capture.CAPTURE[name]` and `pi_models.features_for`, dispatched by
name. The adapter does not re-describe those differences; it carries them.
"""

from __future__ import annotations

from protein_interpretability.collection.models.base import (
    AdapterError,
    ModelAdapter,
    ModelIdentity,
    ResolvedInput,
)

__all__ = ["AdapterError", "ModelAdapter", "ModelIdentity", "ResolvedInput",
           "adapter_for", "available"]

# name -> (module, attribute). Resolved lazily; nothing here is imported until
# `adapter_for` is called, and the module it names is what pulls in a backend.
_ADAPTERS: dict[str, tuple[str, str]] = {
    "boltz2": ("protein_interpretability.collection.models.trunk", "TrunkAdapter"),
    "of3": ("protein_interpretability.collection.models.trunk", "TrunkAdapter"),
    "protenix": ("protein_interpretability.collection.models.trunk", "TrunkAdapter"),
}


def available() -> list[str]:
    """Models that can actually be collected, which is narrower than the
    capability registry: af2 is described there and has no adapter here."""
    return sorted(_ADAPTERS)


def adapter_for(name: str, spec) -> ModelAdapter:
    """Build the adapter for `name`. THIS is where a backend gets imported."""
    if name not in _ADAPTERS:
        from protein_interpretability.collection import capabilities as caps

        known = ", ".join(caps.available())
        raise AdapterError(
            f"no adapter for {name!r}; collectable models are {available()}. "
            f"The capability registry describes {known}, which is a wider set: "
            f"being describable is not being runnable.")
    module_name, attr = _ADAPTERS[name]
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)(spec)
