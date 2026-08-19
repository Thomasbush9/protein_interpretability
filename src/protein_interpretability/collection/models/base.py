"""The seam a model sits behind, and what it must be able to say about itself.

An adapter is the only thing in this package that imports a backend, and it does
so inside its methods rather than at module level, so the registry can be listed,
a task inspected and a job rendered on a login node without initialising CUDA.

WHAT AN ADAPTER OWES THE ARTIFACT. Not the values it was ASKED for -- the values
it actually used. A spec that says `recycles=3` and an artifact that records
`recycles=3` prove nothing if the second was copied from the first; the whole
apparatus is worth having only when the record comes from the loaded model. So
`identity()` reads the trunk depth off the real object and the MSA regime off the
built wrapper, and `verify()` compares both against the capability registry and
raises when they disagree.

WHAT IS DELIBERATELY NOT ABSTRACTED. There is no common `Layer` type across
Pairformer, OpenFold3 and Protenix, and no attempt to make their pLDDT mean the
same thing -- it is per-ATOM in OpenFold3 and per-TOKEN in the other two, and the
arrays are different lengths for the same protein. Those differences are carried
as recorded properties and enforced at the point of comparison by
`records.assert_comparable`, not smoothed away here. An adapter's job is to make
the three models runnable through one declaration, not to pretend they are one
model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelIdentity:
    """What was actually loaded, read off the object rather than the request.

    Every field here is a measurement. Where one cannot be measured it stays
    `None` and says so, following the same rule as the capability registry: a
    plausible default recorded as a fact is worse than a gap.
    """

    model: str
    architecture: str
    backend: str
    backend_version: str | None = None
    checkpoint: str | None = None
    trunk_depth: int | None = None
    pair_width: int | None = None
    single_width: int | None = None
    plddt_granularity: str | None = None
    msa_regime: str | None = None
    msa_rows_used: int | None = None
    recycles: int | None = None
    sampling_steps: int | None = None
    seed: int | None = None
    network_blocked: bool | None = None
    device: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {k: v for k, v in vars(self).items() if k != "extra" and v is not None}
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class ResolvedInput:
    """One materialized model input, and the checksums that identify it.

    The point of separating this from the assay record is that an assay names a
    sequence and an alignment, while a model consumes FILES generated from them.
    A run that cannot say which files it consumed cannot be reproduced, and the
    generated variant alignment is exactly where this project has already lost a
    row to a race between two jobs sharing a work directory.
    """

    input_id: str
    reference_id: str | None
    sequence: str
    work_dir: str
    files: dict = field(default_factory=dict)      # role -> path
    checksums: dict = field(default_factory=dict)  # role -> sha256
    n_tokens: int | None = None
    msa_rows: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if v is not None}


@runtime_checkable
class ModelAdapter(Protocol):
    """The minimum a model must support to be collected through a task."""

    def identity(self) -> ModelIdentity:
        """What is loaded. Compute-time: this may touch a backend."""
        ...

    def capabilities(self):
        """The registry's declaration for this model. Login-node safe."""
        ...

    def verify(self) -> dict:
        """Check the declaration against the loaded model. Raises on drift."""
        ...

    def collect_cohort(self, task, resolved, **kwargs) -> list:
        """Run the task and write one artifact per assay. Returns their paths."""
        ...


class AdapterError(RuntimeError):
    """An adapter cannot run what it was asked to run."""
