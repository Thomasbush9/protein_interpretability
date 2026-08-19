"""What each model is, and what it can be asked for — without loading it.

A capture is planned, submitted and queued long before a GPU is reached, and
almost everything that makes a plan invalid is knowable at the start: a layer
index deeper than the trunk, a field the wrapper does not produce, an MSA asked
of a model that only takes single sequences. Discovering those from an exception
forty minutes into a job is the cost this table exists to avoid.

THE TABLE IS EVIDENCE, AND SAYS SO. Every number carries where it came from,
because a static description of a model is exactly the kind of thing that goes
quietly wrong when the wrapper is upgraded. Two consequences:

  * Anything this project has not measured is `None` and RAISES when asked,
    rather than being filled with a plausible default. AlphaFold2's trunk depth
    is the live example.
  * `verify_against_model()` compares the declaration to a real loaded model, so
    drift is detectable rather than assumed. Call it from a GPU job; it is the
    only function here that touches a backend, and it takes the model as an
    argument rather than importing one.

WHAT IS DELIBERATELY NOT NORMALISED. pLDDT is per-ATOM in OpenFold3 and
per-TOKEN in Protenix and Boltz-2, and the distogram grids are not guaranteed to
match. The plan asks for common outputs to be normalised "without normalising
away architecture-specific semantics" — so those differences are recorded here
as properties rather than smoothed over, and `records.assert_comparable` is what
enforces them at the point two models are actually compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Representation widths. Identical across the three trunks in every capture this
# project holds: xm_{boltz2,of3,protenix}_r1_*.npz all give dz_vec [., ., 128]
# and ds_vec [., ., 384].
PAIR_WIDTH = 128
SINGLE_WIDTH = 384

# Boltz-2's true distogram bin CENTRES: 2-22 A over 64 bins, so width 0.3125 and
# centres 2.15625 … 21.84375. The mosaic wrapper reports linspace(2, 22, 64)
# instead, which differs by up to ~0.16 A -- small, systematic, and enough to
# make new E[d] values disagree with every archived one. pi_models.BIN_OVERRIDE
# exists for this reason; the number is repeated here as a property of the
# MODEL, and `verify_against_model` checks the two still agree.
_B_MIN, _B_MAX, _B_N = 2.0, 22.0, 64
_B_W = (_B_MAX - _B_MIN) / _B_N
BOLTZ2_CENTRES = tuple(_B_MIN + _B_W / 2 + _B_W * i for i in range(_B_N))


class CapabilityError(ValueError):
    """A model cannot do what was asked, or its declaration no longer holds."""


@dataclass(frozen=True)
class ModelCapabilities:
    """A model's fixed properties, as far as this project has measured them."""

    name: str
    architecture: str
    backend: str
    evidence: str
    n_trunk_blocks: int | None = None
    pair_width: int | None = None
    single_width: int | None = None
    distogram_bins: int | None = None
    distogram_centres: tuple[float, ...] | None = None
    plddt_granularity: str | None = None          # "token" | "atom"
    supports_msa: bool = True
    subsamples_msa_by_default: bool | None = None
    capture_fields: tuple[str, ...] = ()
    unknown: tuple[str, ...] = field(default_factory=tuple)

    def require(self, attr: str):
        """Return an attribute, or raise if this project has not measured it."""
        value = getattr(self, attr)
        if value is None:
            raise CapabilityError(
                f"{self.name}: {attr} is not recorded in this project. "
                f"{self.evidence} A plausible default here would change results "
                f"silently -- measure it from a real run and record it instead.")
        return value


REGISTRY: dict[str, ModelCapabilities] = {
    "boltz2": ModelCapabilities(
        name="boltz2",
        architecture="Boltz-2",
        backend="joltz (pi_core) or mosaic (pi_models)",
        evidence=("depths and widths from xm_boltz2_r1_*.npz and depth_v1's "
                  "n_layers_per_model; bin centres from pi_core.BIN_CENTRES, "
                  "which every archived Boltz-2 number was computed with; "
                  "capture fields are the UNION of what gym2s_* and xm_* "
                  "actually hold, checked against the archives by "
                  "tests/test_capture_fields_match_archives.py."),
        n_trunk_blocks=64,
        pair_width=PAIR_WIDTH,
        single_width=SINGLE_WIDTH,
        distogram_bins=_B_N,
        distogram_centres=BOLTZ2_CENTRES,
        plddt_granularity="token",
        supports_msa=True,
        subsamples_msa_by_default=True,     # measured: mosaic builds it True/1024
        capture_fields=("dz_vec", "ds_vec", "dz_site", "ds_site",
                        "kl_site", "kl_glob", "shift_site", "shift_glob",
                        "spread_site", "spread_glob", "disto", "ca",
                        "plddt", "plddt_mean", "plddt_site", "score", "pos"),
    ),
    "of3": ModelCapabilities(
        name="of3",
        architecture="OpenFold3",
        backend="mosaic",
        evidence=("depth and widths from xm_of3_r1_*.npz and depth_v1; the "
                  "distogram grid is NOT recorded here -- read it from the "
                  "model and check it with records.assert_comparable. Capture "
                  "fields are read off xm_of3_r1_*.npz itself."),
        n_trunk_blocks=48,
        pair_width=PAIR_WIDTH,
        single_width=SINGLE_WIDTH,
        plddt_granularity="atom",           # per-ATOM here, per-token elsewhere
        supports_msa=True,
        # kl_site and kl_glob were missing from this list while every
        # xm_of3_r1_*.npz on disk carried both, so a spec asking for what this
        # model demonstrably produces was refused before it ran. The KL is
        # computed from OF3's OWN distogram head per layer, which is why it is
        # available without the bin grid being recorded: it never crosses
        # models. A cross-model KL still needs records.assert_comparable.
        capture_fields=("dz_vec", "ds_vec", "dz_site", "ds_site",
                        "kl_site", "kl_glob", "ca", "plddt", "plddt_mean",
                        "plddt_site", "score", "pos"),
        unknown=("distogram_bins", "distogram_centres"),
    ),
    "protenix": ModelCapabilities(
        name="protenix",
        architecture="Protenix",
        backend="mosaic",
        evidence=("depth and widths from xm_protenix_r1_*.npz and depth_v1; the "
                  "distogram grid is NOT recorded here. Capture fields are read "
                  "off xm_protenix_r1_*.npz itself."),
        n_trunk_blocks=16,
        pair_width=PAIR_WIDTH,
        single_width=SINGLE_WIDTH,
        plddt_granularity="token",
        supports_msa=True,
        capture_fields=("dz_vec", "ds_vec", "dz_site", "ds_site",
                        "kl_site", "kl_glob", "ca", "plddt", "plddt_mean",
                        "plddt_site", "score", "pos"),
        unknown=("distogram_bins", "distogram_centres"),
    ),
    "af2": ModelCapabilities(
        name="af2",
        architecture="AlphaFold2",
        backend="mosaic",
        evidence=("single-sequence constraint from the mosaic wrapper, which "
                  "asserts `not use_msa` and pins max_msa_clusters=1. No "
                  "capture in this project records its depth or widths."),
        supports_msa=False,                 # the one hard constraint we know
        capture_fields=(),
        unknown=("n_trunk_blocks", "pair_width", "single_width",
                 "distogram_bins", "distogram_centres", "plddt_granularity"),
    ),
}


def available() -> list[str]:
    return sorted(REGISTRY)


def capabilities(name: str) -> ModelCapabilities:
    """A model's declared properties. Imports no backend."""
    if name not in REGISTRY:
        raise CapabilityError(f"unknown model {name!r}; have {available()}")
    return REGISTRY[name]


def check_spec(spec) -> None:
    """Raise unless `spec` is something this model can actually be asked for.

    Complements `CaptureSpec.validate()`, which checks the spec is internally
    coherent. This checks it against the MODEL: fields the wrapper does not
    produce, and an MSA asked of a single-sequence-only model.
    """
    cap = capabilities(spec.model)

    if cap.capture_fields:
        unsupported = [f for f in spec.fields if f not in cap.capture_fields]
        if unsupported:
            raise CapabilityError(
                f"{cap.name} does not produce {unsupported}; it produces "
                f"{list(cap.capture_fields)}. Asking for a field a wrapper does "
                f"not emit fails inside the run, after the model is loaded.")
    elif spec.fields:
        raise CapabilityError(
            f"{cap.name} has no recorded capture fields in this project, so a "
            f"capture against it cannot be checked. {cap.evidence}")


def check_msa(name: str, *, use_msa: bool) -> None:
    """Raise if an alignment is asked of a model that cannot take one."""
    cap = capabilities(name)
    if use_msa and not cap.supports_msa:
        raise CapabilityError(
            f"{cap.name} is single-sequence only here, so it cannot be given an "
            f"alignment. Single-sequence input is a different operating point, "
            f"not a variant of the MSA one -- it was measured ~4.4x more "
            f"mutation-sensitive at full depth -- so any comparison involving "
            f"{cap.name} must put the other models in the same mode.")


# Where each wrapper keeps its stacked Pairformer parameters, from the wrapper
# down. Three names differ across three models, and the INNER field differs too
# -- Protenix holds its network at `.protenix` where the others use `.model`,
# which pi_models.INNER_FIELD records for the same reason. A single generic
# accessor found only Boltz-2 and reported the rest as agreement.
#
# Paths are tried in order, so a bare network (no wrapper) also resolves.
TRUNK_STACK: dict[str, tuple[tuple[str, ...], ...]] = {
    "boltz2": (("model", "pairformer_module", "stacked_parameters"),
               ("pairformer_module", "stacked_parameters")),
    "of3": (("model", "pairformer_stack", "stacked_params"),
            ("pairformer_stack", "stacked_params")),
    "protenix": (("protenix", "pairformer_stack", "stacked_parameters"),
                 ("model", "pairformer_stack", "stacked_parameters"),
                 ("pairformer_stack", "stacked_parameters")),
}


def _leading_axis(node, _depth=0):
    """The stack axis of a stacked-parameter pytree, found without importing jax.

    Every leaf under a stacked block carries the same leading axis -- that is
    what `jax.lax.scan` scans over -- so the first array-like found gives the
    block count. Walking the structure by hand rather than with
    `jax.tree_util.tree_leaves` is what keeps this module importable on a login
    node, which is the property the whole registry exists to have.
    """
    if _depth > 6:
        return None
    shape = getattr(node, "shape", None)
    if isinstance(shape, tuple) and shape:
        return int(shape[0])
    children = []
    if hasattr(node, "__dict__"):
        children = list(vars(node).values())
    elif isinstance(node, dict):
        children = list(node.values())
    elif isinstance(node, (list, tuple)):
        children = list(node)
    for child in children:
        if child is None or isinstance(child, (str, bytes, int, float, bool)):
            continue
        found = _leading_axis(child, _depth + 1)
        if found is not None:
            return found
    return None


def observed_trunk_depth(name: str, model) -> int | None:
    """Read a loaded model's trunk depth, or None if this wrapper does not expose
    it where the table expects."""
    for path in TRUNK_STACK.get(name, ()):
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        else:
            found = _leading_axis(node)
            if found is not None:
                return found
    return None


def verify_against_model(name: str, model) -> dict:
    """Compare this table to a real loaded model. Call from a GPU job.

    The only function here that touches a backend, and it takes the model rather
    than importing one, so the module stays login-node safe. Raises on a
    contradiction, because a table that quietly stops describing the model is
    worse than no table.

    Returns `{"checked": {...}, "unverified": [...]}`. The second key exists
    because the first version of this returned only what it could read, and the
    first real run reported OpenFold3 and Protenix as agreeing on the strength
    of having read nothing from either -- their wrappers do not expose the trunk
    the way Boltz-2's does. "Nothing contradicted me" is not agreement, and a
    checker that cannot tell the two apart is the vacuous guard this project
    tests against everywhere else.
    """
    cap = capabilities(name)
    checked: dict[str, object] = {}
    problems = []
    declared = {a for a in ("n_trunk_blocks",) if getattr(cap, a) is not None}

    inner = getattr(model, "model", model)
    depth = observed_trunk_depth(name, model)
    if depth is not None:
        checked["n_trunk_blocks"] = depth
        if cap.n_trunk_blocks is not None and depth != cap.n_trunk_blocks:
            problems.append(
                f"trunk depth: table says {cap.n_trunk_blocks}, model has "
                f"{depth}")

    mm = getattr(inner, "msa_module", None)
    if mm is not None and hasattr(mm, "subsample_msa"):
        flag = bool(mm.subsample_msa)
        checked["subsamples_msa"] = flag
        if (cap.subsamples_msa_by_default is not None
                and flag != cap.subsamples_msa_by_default):
            # Not a contradiction: the caller may have set it deliberately, and
            # pi_models.load(msa="full") does exactly that.
            checked["subsample_differs_from_default"] = True

    if problems:
        raise CapabilityError(
            f"the recorded capabilities for {name!r} no longer describe the "
            f"model:\n  " + "\n  ".join(problems)
            + f"\n\n{cap.evidence}\nUpdate the table from this run rather than "
              "working around it.")
    return {"checked": checked,
            "unverified": sorted(declared - set(checked))}


def describe(name: str) -> str:
    """A human-readable summary, for `pi models`."""
    c = capabilities(name)
    lines = [f"{c.name}  ({c.architecture}, via {c.backend})"]
    for attr, label in (("n_trunk_blocks", "trunk blocks"),
                        ("pair_width", "pair width"),
                        ("single_width", "single width"),
                        ("distogram_bins", "distogram bins"),
                        ("plddt_granularity", "pLDDT")):
        value = getattr(c, attr)
        lines.append(f"    {label:16s} {value if value is not None else '— not recorded'}")
    lines.append(f"    {'MSA':16s} "
                 + ("single-sequence only" if not c.supports_msa
                    else "supported" + (", subsampled by default"
                                        if c.subsamples_msa_by_default else "")))
    if c.capture_fields:
        lines.append(f"    {'fields':16s} {', '.join(c.capture_fields)}")
    if c.unknown:
        lines.append(f"    {'unknown':16s} {', '.join(c.unknown)}")
    lines.append(f"    evidence: {c.evidence}")
    return "\n".join(lines)
