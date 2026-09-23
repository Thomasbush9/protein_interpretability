"""What to run, on what, capturing what -- decided before a backend is imported.

A `CollectionTask` is the whole scientific declaration of a collection run:
which cohort, which model in which regime, which fields at which layers, where
the result goes. `inspect()` resolves and checks all of it on a login node and
returns a `ResolvedTask` that is serializable, so the object the compute job
consumes is the same object that was inspected rather than a second set of
choices rebuilt from CLI defaults.

    task = CollectionTask(
        name="xmodel_depth",
        cohort=Cohort.load("heldout_assays"),
        model=ModelSpec(name="boltz2", recycles=3, seed=0, msa="full"),
        capture=CaptureSpec(model="boltz2", fields=("dz_vec", "kl_site"),
                            layers="all", reduction="vector"),
        output="runs/xmodel_depth",
    )
    resolved = task.inspect()      # raises, or tells you what it will cost
    task.run()                     # imports a backend; only here

WHY A SEPARATE `ModelSpec`. The capture spec says what to record; the model spec
says what produced it. They are different things that were previously the same
thing: `CaptureSpec.recycles` is a property of the RUN, not of the recording,
and it sat in the capture spec because that was the only object that existed.
Both still carry it, and `inspect()` refuses if they disagree -- a task that
declares one recycle count twice, differently, is exactly the kind of silent
contradiction this layer exists to catch.

WHY LAYERS ARE RESOLVED HERE. `layers="final"` means block 63 in Boltz-2, 47 in
OpenFold3 and 15 in Protenix. Until it is resolved against a specific model's
depth it is not an answer, and the archived cross-model work compares at matched
RELATIVE depth for exactly this reason. `resolve_layers` turns the request into
absolute indices once, at inspection, and the resolved list is what the artifact
records and what the runner is checked against.

Nothing here imports a backend, numpy, or jax: `inspect` and `render` must run
on a login node without initialising CUDA.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from protein_interpretability.collection import capabilities as caps
from protein_interpretability.collection.capture_spec import (
    CaptureSpec,
    CaptureSpecError,
)
from protein_interpretability.collection.cohorts import Cohort

SCHEMA_VERSION = 1          # of the resolved-task document, not of the package

MSA_REGIMES = ("full", "subsample", "none")
NETWORK_POLICIES = ("blocked", "allowed")
RESUME_POLICIES = ("refuse", "resume", "overwrite")


class TaskError(ValueError):
    """A task cannot be resolved, or contradicts itself."""


# ---- layers ---------------------------------------------------------------

def resolve_layers(layers, depth: int) -> tuple[int, ...]:
    """Turn a layer request into absolute, non-negative block indices.

    Accepts `"all"`, `"final"`, or explicit indices, negative counted from the
    end. Returns them in the order they will be stored, which is the order the
    artifact's layer axis is labelled with.

    Refuses duplicates and out-of-order requests. Both are almost always a typo,
    and neither has a meaning this project supports: a duplicated layer weights
    that layer twice in anything pooled over the axis, and an out-of-order list
    produces an artifact whose layer axis is not monotonic in depth, which every
    depth curve in this project assumes without checking.
    """
    if isinstance(layers, str):
        if layers == "all":
            return tuple(range(depth))
        if layers == "final":
            return (depth - 1,)
        raise TaskError(
            f"layers must be 'all', 'final' or explicit indices, got {layers!r}")

    if not layers:
        raise TaskError("an empty layer list captures nothing")

    resolved = []
    for want in layers:
        index = int(want)
        if not (-depth <= index < depth):
            raise TaskError(
                f"layer {want} is outside a {depth}-block trunk. Depths differ "
                f"across models (boltz2 64, of3 48, protenix 16), so an index "
                f"valid for one is not valid for another -- compare at matched "
                f"relative depth instead of assuming absolute indices "
                f"correspond.")
        resolved.append(index if index >= 0 else depth + index)

    dupes = sorted({i for i in resolved if resolved.count(i) > 1})
    if dupes:
        raise TaskError(
            f"layer(s) {dupes} requested more than once. After resolution "
            f"{list(layers)} is {resolved}, and a repeated layer is counted "
            f"twice by anything that pools over the layer axis.")
    if resolved != sorted(resolved):
        raise TaskError(
            f"layers {list(layers)} resolve to {resolved}, which is not in "
            f"increasing depth order. Every depth curve here reads the layer "
            f"axis as monotonic; storing it shuffled would not fail, it would "
            f"plot wrong.")
    return tuple(resolved)


# ---- the model ------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Which model, in which regime. Model-free: this imports no backend.

    Everything here changes the scientific result, which is the test for whether
    a setting belongs in this object rather than in a CLI flag.
    """

    name: str
    backend: str | None = None      # resolved from the registry when omitted
    checkpoint: str | None = None   # logical id; the run records what it loaded
    recycles: int = 3
    seed: int = 0
    msa: str = "full"
    msa_cap: int | None = 2048
    network: str = "blocked"
    options: dict = field(default_factory=dict)

    def validate(self) -> "ModelSpec":
        cap = caps.capabilities(self.name)       # raises on an unknown model

        if self.msa not in MSA_REGIMES:
            raise TaskError(
                f"msa must be one of {MSA_REGIMES}, got {self.msa!r}")
        caps.check_msa(self.name, use_msa=self.msa != "none")

        if self.backend is not None and self.backend not in cap.backend:
            raise TaskError(
                f"{self.name} is wrapped here by {cap.backend!r}; "
                f"{self.backend!r} is not one of them. {cap.evidence}")

        if self.recycles < 1:
            raise TaskError(
                "recycles must be >= 1: the trunk runs recycles-1 refinement "
                "iterations and then one final pass, so 0 would run no trunk")

        if self.network not in NETWORK_POLICIES:
            raise TaskError(
                f"network must be one of {NETWORK_POLICIES}, got "
                f"{self.network!r}")

        if self.msa != "none" and not self.msa_cap:
            raise TaskError(
                "an MSA regime needs an msa_cap: alignment depth is the "
                "quantity the archived captures control exactly across "
                "variants, and 'whatever the file happened to hold' is not a "
                "controlled quantity")
        return self

    @property
    def resolved_backend(self) -> str:
        """The backend actually used. `None` means the registry's first."""
        if self.backend:
            return self.backend
        return caps.capabilities(self.name).backend.split()[0]

    def declaration(self) -> dict:
        """What was ASKED for. The run records separately what it loaded."""
        self.validate()
        out = {
            "model": self.name,
            "backend": self.resolved_backend,
            "recycles": self.recycles,
            "seed": self.seed,
            "msa": self.msa,
            "network": self.network,
        }
        if self.checkpoint:
            out["checkpoint"] = self.checkpoint
        if self.msa_cap:
            out["msa_cap"] = self.msa_cap
        if self.options:
            out["options"] = dict(self.options)
        return out


# ---- the resolved task ----------------------------------------------------

@dataclass(frozen=True)
class ResolvedTask:
    """A task with every choice made, and nothing left to a default.

    This is what the compute job consumes and what the artifact embeds. It is
    plain data so that it serializes, compares and hashes -- `task_id` is the
    hash, which is what lets a later run say "this output was produced by a
    different task" rather than silently overwriting it.
    """

    schema_version: int
    name: str
    cohort: str
    assays: tuple[str, ...]
    model: dict
    capture: dict
    layers: tuple[int, ...]
    trunk_depth: int
    output: str
    resume: str
    resources: dict
    estimated_bytes: int

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task": self.name,
            "cohort": self.cohort,
            "assays": list(self.assays),
            "model": self.model,
            "capture": self.capture,
            "layers": list(self.layers),
            "trunk_depth": self.trunk_depth,
            "output": self.output,
            "resume": self.resume,
            "resources": self.resources,
            "estimated_bytes": self.estimated_bytes,
            "task_id": self.task_id,
        }

    @property
    def task_id(self) -> str:
        """A stable hash of every scientific choice in this task.

        Deliberately excludes `output`, `resume` and `resources`: writing the
        same measurement to a second path, or resuming it, does not make it a
        different measurement. Everything that WOULD change a number is in here.
        """
        body = {
            "schema_version": self.schema_version,
            "cohort": self.cohort,
            "assays": list(self.assays),
            "model": self.model,
            "capture": self.capture,
            "layers": list(self.layers),
            "trunk_depth": self.trunk_depth,
        }
        blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def output_for(self, assay_id: str) -> Path:
        return Path(self.output) / f"{self.name}_{assay_id}.npz"

    def describe(self) -> str:
        gb = self.estimated_bytes / 1e9
        return "\n".join([
            f"task     {self.name}  [{self.task_id}]",
            f"cohort   {self.cohort}: {len(self.assays)} assays",
            f"model    {self.model['model']} via {self.model['backend']}, "
            f"recycles={self.model['recycles']}, seed={self.model['seed']}, "
            f"msa={self.model['msa']}, network={self.model['network']}",
            f"layers   {len(self.layers)} of {self.trunk_depth}"
            + (f": {list(self.layers)}" if len(self.layers) <= 8
               else f": {list(self.layers[:4])} … {list(self.layers[-2:])}"),
            f"fields   {', '.join(self.capture['capture_fields'])} "
            f"({self.capture['reduction']}, {self.capture['dtype']})",
            f"output   {self.output}  ~{gb:.2f} GB, resume={self.resume}",
        ])


# ---- the task -------------------------------------------------------------

@dataclass(frozen=True)
class CollectionTask:
    """One collection run, declared in full and checkable before it starts."""

    name: str
    cohort: Cohort
    model: ModelSpec
    capture: CaptureSpec
    output: str
    resume: str = "refuse"
    resources: dict = field(default_factory=dict)
    n_variants: int | None = None       # None means every single-mutant row

    def inspect(self, *, verify_inputs: bool = True) -> ResolvedTask:
        """Resolve and check everything knowable without a backend.

        Raises on the first thing that would make the run wrong or wasteful.
        Loads no model, opens no checkpoint and initialises no accelerator, so
        it is safe on a login node -- which is the point: the failures it
        catches otherwise arrive forty minutes into a queued job.
        """
        if self.resume not in RESUME_POLICIES:
            raise TaskError(
                f"resume must be one of {RESUME_POLICIES}, got {self.resume!r}")

        self.model.validate()
        self.capture.validate()

        # The contradiction check. Two objects carry the model name and the
        # recycle count; a task that states either of them twice, differently,
        # is a task whose artifact cannot say which one it ran.
        if self.capture.model != self.model.name:
            raise TaskError(
                f"the capture spec is for {self.capture.model!r} but the model "
                f"spec is {self.model.name!r}. One of them is left over from "
                f"another experiment.")
        if self.capture.recycles != self.model.recycles:
            raise TaskError(
                f"recycles is declared twice and disagrees: capture says "
                f"{self.capture.recycles}, model says {self.model.recycles}. "
                f"The recycle count changes the representation, so an artifact "
                f"recording both would not say which produced it.")

        if not self.cohort.assays:
            raise TaskError(
                f"cohort {self.cohort.name!r} holds no assays; the run would "
                f"produce nothing and report success")
        if verify_inputs:
            self.cohort.verify()

        depth = caps.capabilities(self.model.name).require("n_trunk_blocks")
        layers = resolve_layers(self.capture.layers, depth)

        missing_length = [a.id for a in self.cohort if not a.wt_length]
        if missing_length:
            raise TaskError(
                f"{missing_length} record no wt_length, so this task cannot be "
                f"priced or its captures shape-checked. Regenerate the cohort "
                f"manifest with build_cohort_manifests.py.")

        total = 0
        for assay in self.cohort:
            n = self.n_variants or assay.n_single_variants or 0
            if not n:
                raise TaskError(
                    f"{assay.id} records no single-variant count and the task "
                    f"sets no n_variants, so its size is unknown")
            total += _estimate(self.capture, layers, n, assay.wt_length)

        return ResolvedTask(
            schema_version=SCHEMA_VERSION,
            name=self.name,
            cohort=self.cohort.name,
            assays=tuple(self.cohort.ids),
            model=self.model.declaration(),
            capture=self.capture.protocol(),
            layers=layers,
            trunk_depth=depth,
            output=str(self.output),
            resume=self.resume,
            resources=dict(self.resources),
            estimated_bytes=total,
        )

    def run(self, **kwargs):
        """Execute the task. This is the first thing here that loads a model.

        Delegates to the adapter registered for the model, so the CLI and a
        Python caller reach the same code by the same route.
        """
        from protein_interpretability.collection.models import adapter_for

        resolved = self.inspect()
        adapter = adapter_for(self.model.name, self.model)
        return adapter.collect_cohort(self, resolved, **kwargs)


def _estimate(capture: CaptureSpec, layers, n_variants: int, n_tokens: int) -> int:
    """Bytes for one assay at the RESOLVED layer count.

    `CaptureSpec.estimate_bytes` prices `spec.n_layers`, which is the count the
    spec asked for; after resolution the two agree, and this keeps the estimate
    honest if they ever stop agreeing.
    """
    from dataclasses import replace

    priced = replace(capture, layers=tuple(layers)) if len(layers) != capture.n_layers \
        else capture
    try:
        return priced.estimate_bytes(n_variants=n_variants, n_tokens=n_tokens)
    except CaptureSpecError:                                # pragma: no cover
        return capture.estimate_bytes(n_variants=n_variants, n_tokens=n_tokens)
