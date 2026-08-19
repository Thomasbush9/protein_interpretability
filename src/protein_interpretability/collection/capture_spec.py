"""What a capture will contain, decided and checked before a GPU is touched.

A `CaptureSpec` names the fields, the layers, the reduction and the precision,
and can say how large the result will be — all without importing a backend, so
it resolves on a login node.

WHY THE REDUCTION IS THE FIRST-CLASS FIELD. `dz_site` is a VECTOR of 128 pair
channels in the `gym2s_*` captures and a per-layer NORM in the `xm_*` ones. Same
name, two shapes, both live in `runs/` right now:

    gym2s_ARGR…npz   dz_site  (250, 64, 128)   the direction the row moved
    xm_boltz2_r1…npz dz_site  (100, 64)        how far it moved
                     dz_vec   (100, 64, 128)

`Capture` survives this by checking `ndim` instead of trusting the name, which
is the read side. This is the write side: a spec states which one it is, so an
archive can be checked against what it promised rather than against what a
later consumer assumes. The costliest error in this project was exactly that
gap — `deep2_*` stored a norm where consumers expected a vector, and the probe
returned +0.468 instead of raising.

WHAT IS ENCODED HERE IS MEASURED, NOT ASSUMED. Layer counts and widths come from
the archives themselves (`xm_{model}_r1_*.npz`) and from `depth_v1`'s protocol
block. AlphaFold2 has no entry because this project has no capture to read one
from, and a plausible guess is worse than a refusal.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from protein_interpretability.collection.capabilities import (
    PAIR_WIDTH,
    SINGLE_WIDTH,
    CapabilityError,
    available as _available,
    capabilities,
)

# Trunk depth per model comes from the capability registry rather than a second
# table here. Two tables describing the same models is how one of them ends up
# quietly wrong; there is exactly one, and it records where its numbers came
# from.
MODEL_LAYERS: dict[str, int | None] = {
    name: capabilities(name).n_trunk_blocks for name in _available()
}

REDUCTIONS = ("vector", "norm")

# field -> (axes, note). Axes are symbolic:
#   V variants · L captured layers · C channel width · P sampled pairs
#   B distogram bins · T tokens
FIELDS: dict[str, tuple[tuple, str]] = {
    # The unambiguous spelling. The cross-model captures write BOTH -- `dz_vec`
    # for the direction and `dz_site` for its norm -- which is why those archives
    # can be read without guessing and the gym2s ones cannot. A `_vec` field is
    # a vector by name, so it is never reducible: a norm stored under `dz_vec`
    # is a contradiction, not a configuration.
    "dz_vec":    (("V", "L", "C_pair"), "pair row at the mutated position"),
    "ds_vec":    (("V", "L", "C_single"), "single row at the mutated position"),
    "dz_site":   (("V", "L", "C_pair"), "pair row at the mutated position"),
    "ds_site":   (("V", "L", "C_single"), "single row at the mutated position"),
    "kl_site":   (("V", "L"), "KL at the mutated position"),
    "kl_glob":   (("V", "L"), "KL over the whole map"),
    "shift_site": (("V", "L"), "mean distance shift at the site"),
    "shift_glob": (("V", "L"), "mean distance shift, global"),
    "spread_site": (("V", "L"), "distogram spread at the site"),
    "spread_glob": (("V", "L"), "distogram spread, global"),
    "disto":     (("V", "P", "B"), "distogram logits over sampled pairs"),
    "ca":        (("V", "T", 3), "CA coordinates"),
    # Two spellings for the chain mean, because two capture families chose
    # different names for it: `gym2s_*` writes `plddt`, `xm_*` writes
    # `plddt_mean`. Both are declared rather than one being renamed, because
    # renaming would make one of the two families unreadable, and the whole
    # point of the registry is to describe what is on disk.
    "plddt":     (("V",), "mean pLDDT (gym2/gym3 spelling)"),
    "plddt_mean": (("V",), "mean pLDDT (xm spelling)"),
    "plddt_site": (("V",), "pLDDT at the mutated position"),
    "score":     (("V",), "assay score"),
    "pos":       (("V",), "mutated position, 0-based"),
}

# Fields whose shape depends on the reduction. Everything else is what it is.
REDUCIBLE = {"dz_site": "C_pair", "ds_site": "C_single"}

DTYPE_BYTES = {"float16": 2, "float32": 4, "float64": 8, "int64": 8}


class CaptureSpecError(ValueError):
    """A capture spec is not satisfiable, or an archive does not match one."""


@dataclass(frozen=True)
class CaptureSpec:
    """A declaration of what a collection run will write.

        spec = CaptureSpec(model="boltz2", fields=("dz_site", "kl_site"),
                           layers="all", reduction="vector", recycles=3)
        spec.validate()
        spec.estimate_bytes(n_variants=250, n_tokens=69)
    """

    model: str
    fields: tuple[str, ...]
    layers: str | tuple[int, ...] = "all"
    reduction: str = "vector"
    recycles: int = 3
    dtype: str = "float32"
    n_pairs: int | None = None          # required if `disto` is captured
    n_bins: int = 64
    notes: str = ""
    _validated: bool = dc_field(default=False, repr=False, compare=False)

    # ---- validation -------------------------------------------------------
    def validate(self) -> "CaptureSpec":
        """Raise unless this spec can be satisfied. Returns self, so it chains."""
        if self.model not in MODEL_LAYERS:
            raise CaptureSpecError(
                f"unknown model {self.model!r}; have {sorted(MODEL_LAYERS)}")
        depth = MODEL_LAYERS[self.model]
        if depth is None:
            raise CaptureSpecError(
                f"{self.model!r} has no recorded trunk depth in this project, so "
                f"a layer selection cannot be checked. Record one from a real "
                f"capture before specifying a capture against it -- a guessed "
                f"depth silently changes which layer 'final' means.")

        if not self.fields:
            raise CaptureSpecError(
                "a capture with no fields would run a model and write nothing")
        unknown = [f for f in self.fields if f not in FIELDS]
        if unknown:
            raise CaptureSpecError(
                f"unknown field(s) {unknown}; have {sorted(FIELDS)}")

        if self.reduction not in REDUCTIONS:
            raise CaptureSpecError(
                f"reduction must be one of {REDUCTIONS}, got {self.reduction!r}")

        if isinstance(self.layers, str):
            if self.layers not in ("all", "final"):
                raise CaptureSpecError(
                    f"layers must be 'all', 'final' or explicit indices, got "
                    f"{self.layers!r}")
        else:
            if not self.layers:
                raise CaptureSpecError("an empty layer list captures nothing")
            bad = [i for i in self.layers if not (-depth <= i < depth)]
            if bad:
                raise CaptureSpecError(
                    f"layer(s) {bad} are outside {self.model}'s {depth} blocks. "
                    f"Trunk depths differ across models (boltz2 64, of3 48, "
                    f"protenix 16), so an index valid for one is not for "
                    f"another -- that is why depth_v1 compares at matched "
                    f"RELATIVE depth rather than at the last layer.")

        if "disto" in self.fields and not self.n_pairs:
            raise CaptureSpecError(
                "capturing `disto` needs n_pairs: the full N x N map is not "
                "stored, a fixed sample of pairs is, and the sample size is "
                "part of what makes two captures comparable")

        if self.dtype not in DTYPE_BYTES:
            raise CaptureSpecError(
                f"unsupported dtype {self.dtype!r}; have {sorted(DTYPE_BYTES)}")

        if self.recycles < 0:
            raise CaptureSpecError("recycles must be >= 0")

        # The `_vec` case first: it is the more specific diagnosis, and both
        # rules fire on the same spec.
        vec_named = [f for f in self.fields if f.endswith("_vec")]
        if self.reduction == "norm" and vec_named:
            raise CaptureSpecError(
                f"reduction='norm' with {vec_named}: a field named `_vec` is a "
                f"vector by name. Storing a norm under it recreates exactly the "
                f"ambiguity those names exist to remove -- use the `_site` "
                f"spelling for a norm, or keep the vector.")

        if self.reduction == "norm" and not (set(self.fields) & set(REDUCIBLE)):
            raise CaptureSpecError(
                f"reduction='norm' applies to {sorted(REDUCIBLE)}, and none is "
                f"captured; the setting would silently do nothing")

        # Internally coherent is not the same as askable. The registry knows
        # what each wrapper actually emits.
        from protein_interpretability.collection import capabilities as _caps
        try:
            _caps.check_spec(self)
        except CapabilityError as exc:
            raise CaptureSpecError(str(exc)) from exc
        return self

    # ---- what it will produce ---------------------------------------------
    @property
    def n_layers(self) -> int:
        depth = MODEL_LAYERS[self.model]
        if self.layers == "all":
            return depth
        if self.layers == "final":
            return 1
        return len(self.layers)

    def expected_shapes(self, *, n_variants: int, n_tokens: int) -> dict:
        """Concrete shape per field. This is the contract an archive must meet."""
        self.validate()
        dims = {"V": n_variants, "L": self.n_layers, "T": n_tokens,
                "C_pair": PAIR_WIDTH, "C_single": SINGLE_WIDTH,
                "P": self.n_pairs, "B": self.n_bins}
        out = {}
        for name in self.fields:
            axes = FIELDS[name][0]
            if self.reduction == "norm" and name in REDUCIBLE:
                # The whole point: a norm drops the channel axis, and the
                # resulting array is a legal shape for a DIFFERENT quantity.
                axes = tuple(a for a in axes if a != REDUCIBLE[name])
            out[name] = tuple(dims[a] if isinstance(a, str) else a for a in axes)
        return out

    def estimate_bytes(self, *, n_variants: int, n_tokens: int) -> int:
        shapes = self.expected_shapes(n_variants=n_variants, n_tokens=n_tokens)
        width = DTYPE_BYTES[self.dtype]
        return sum(int(np_prod(s)) * width for s in shapes.values())

    def full_pair_tensor_bytes(self, *, n_variants: int, n_tokens: int) -> int:
        """What the unreduced N x N x C x L pair tensor would cost.

        The plan asks for reduced mutation-row captures to be preferred over the
        full tensor; this makes the comparison a number rather than an opinion.
        """
        depth = self.n_layers
        return (n_variants * depth * n_tokens * n_tokens * PAIR_WIDTH
                * DTYPE_BYTES[self.dtype])

    # ---- checking a written archive ---------------------------------------
    def validate_capture(self, capture, *, n_variants: int, n_tokens: int) -> None:
        """Raise unless an archive matches what this spec promised.

        Checks the SHAPE, not the name, because the name is the part that has
        already proven unreliable: `dz_site` is a vector in one family and a
        norm in another.
        """
        want = self.expected_shapes(n_variants=n_variants, n_tokens=n_tokens)
        problems = []
        for name, shape in want.items():
            if name not in capture.files:
                problems.append(f"{name}: promised but absent")
                continue
            got = tuple(capture._arr(name).shape)
            if got != shape:
                extra = ""
                if name in REDUCIBLE and len(got) == len(shape) - 1:
                    extra = (" -- this is the NORM where the spec promised the "
                             "vector, the shape that returned +0.468")
                elif name in REDUCIBLE and len(got) == len(shape) + 1:
                    extra = " -- this is the vector where the spec promised a norm"
                problems.append(f"{name}: {got} but the spec says {shape}{extra}")
        if problems:
            raise CaptureSpecError(
                f"{getattr(capture, 'path', '<capture>')} does not match its "
                f"spec:\n  " + "\n  ".join(problems))

    # ---- provenance -------------------------------------------------------
    def protocol(self) -> dict:
        """Protocol fields describing this capture, for the archive's block."""
        self.validate()
        return {
            "model": self.model,
            "capture_fields": list(self.fields),
            "layers": (self.layers if isinstance(self.layers, str)
                       else list(self.layers)),
            "n_layers_captured": self.n_layers,
            "reduction": self.reduction,
            "recycles": self.recycles,
            "dtype": self.dtype,
            **({"n_pairs": self.n_pairs} if self.n_pairs else {}),
            **({"notes": self.notes} if self.notes else {}),
        }


def np_prod(shape) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out
