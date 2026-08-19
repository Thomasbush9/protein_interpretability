"""The trunk adapter: Boltz-2, OpenFold3 and Protenix through one declaration.

The numerics are `jax_harness/exp_gym_deep.collect_assay`, imported and not
reimplemented. That function produced every archived `xm_*` capture; the
adapter's contribution is everything around it that the script did not do --
resolving the task, honouring a layer selection, recording what was actually
loaded, and writing through the artifact seam so the result carries its own
protocol.

WHY THE LAYER SLICE IS HERE AND NOT IN THE KERNEL. Each model's Pairformer runs
as a `jax.lax.scan` over stacked parameters, so the trunk is traversed in full
whichever layers are wanted -- there is no cheaper path through 64 blocks that
visits three of them. What a selection can save is STORAGE, and that saving is
real: `dz_vec` at all 64 layers is 3.3 MB per 100 variants against 0.15 MB for
three. So the request is honoured at the artifact, the traversal is not
pretended to be cheaper than it is, and the artifact records both the expression
that was asked for and the absolute indices it resolved to.

THE FIDELITY EVIDENCE TRAVELS WITH THE FEATURES. `collect_assay` already checks
its capture against the model's own trunk output and refuses a run whose
mutation signal is not far above that drift. Those two numbers are written into
the artifact beside the arrays they license, which is where the archived
captures put them and is the reason a reader can tell a real effect from capture
noise without rerunning anything.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from protein_interpretability.collection import capabilities as caps
from protein_interpretability.collection.models.base import (
    AdapterError,
    ModelIdentity,
)

# Fields whose second axis is the trunk layer. Everything else in the capture is
# per-variant or bookkeeping and must not be sliced.
LAYER_AXIS_FIELDS = ("kl_glob", "kl_site", "dz_site", "ds_site",
                     "dz_vec", "ds_vec")

REPO = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
            "protein_interpretability")


def _harness() -> Path:
    """Locate `jax_harness`, from a checkout or from the deployed mirror.

    The same fallback `pi_archive` uses, and for the same reason: jobs execute
    from a plain copy of the harness rather than a checkout, so a path relative
    to `__file__` resolves to somewhere that does not exist.
    """
    here = Path(__file__).resolve().parents[4] / "jax_harness"
    if (here / "exp_gym_deep.py").exists():
        return here
    if (REPO / "jax_harness" / "exp_gym_deep.py").exists():
        return REPO / "jax_harness"
    raise AdapterError(
        f"cannot find jax_harness from {Path(__file__).resolve()} or {REPO}. "
        f"The capture kernel lives there and this adapter runs it rather than "
        f"reimplementing it.")


class TrunkAdapter:
    """Per-layer trunk capture for the three models that support it."""

    def __init__(self, spec):
        self.spec = spec.validate()
        self.name = spec.name
        self._wrapper = None
        self._inner = None

    # ---- login-node safe ---------------------------------------------------
    def capabilities(self):
        return caps.capabilities(self.name)

    # ---- compute time ------------------------------------------------------
    def _load(self):
        """Import the backend and build the model. The first expensive call."""
        if self._wrapper is not None:
            return self._wrapper
        sys.path.insert(0, str(_harness()))
        import pi_models                                       # noqa: E402

        blocked = pi_models.block_network()
        if self.spec.network == "blocked" and not blocked:
            raise AdapterError(
                "the task asks for network access to be blocked and "
                "pi_models.block_network() reports nothing patched. A run that "
                "can reach the MSA server can silently substitute a remote "
                "alignment for the one this task declares.")
        self._wrapper = pi_models.load(self.name, msa=self.spec.msa)
        self._inner = pi_models.inner(self.name, self._wrapper)
        self._blocked = bool(blocked)
        return self._wrapper

    def identity(self) -> ModelIdentity:
        """Read off the loaded model. Not an echo of what was requested."""
        wrapper = self._load()
        sys.path.insert(0, str(_harness()))
        import pi_models                                       # noqa: E402

        cap = self.capabilities()
        depth = caps.observed_trunk_depth(self.name, self._inner)
        if depth is None:
            depth = caps.observed_trunk_depth(self.name, wrapper)
        regime = pi_models.regime_block(self.name, wrapper)
        device = None
        try:
            import jax                                          # noqa: E402
            device = str(jax.devices()[0])
        except Exception:                                       # pragma: no cover
            pass
        return ModelIdentity(
            model=self.name,
            architecture=cap.architecture,
            backend=self.spec.resolved_backend,
            backend_version=_version_of(self.name),
            checkpoint=os.environ.get("MOSAIC_WEIGHTS"),
            trunk_depth=depth,
            pair_width=cap.pair_width,
            single_width=cap.single_width,
            plddt_granularity=cap.plddt_granularity,
            msa_regime=str(regime.get("msa_regime", self.spec.msa)),
            recycles=self.spec.recycles,
            seed=self.spec.seed,
            network_blocked=self._blocked,
            device=device,
            extra={k: str(v) for k, v in regime.items() if k != "msa_regime"},
        )

    def verify(self) -> dict:
        """Compare the registry against the model in front of us."""
        self._load()
        report = caps.verify_against_model(self.name, self._inner)
        if report["unverified"]:
            print(f"  capabilities unverified for {self.name}: "
                  f"{report['unverified']} -- this wrapper does not expose them "
                  f"where the registry expects", flush=True)
        return report

    # ---- the run -----------------------------------------------------------
    def collect_cohort(self, task, resolved, *, assays=None, dry_run=False):
        """Run every assay in the task, writing one artifact each.

        Returns the paths written. An assay whose artifact already exists is
        handled by the task's resume policy rather than by a flag here, so the
        same decision applies whether the run came from Python or the CLI.
        """
        import numpy as np

        from protein_interpretability import artifacts
        from protein_interpretability.experiments import protocol as P

        out_dir = Path(resolved.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        wanted = set(assays) if assays else None

        # Decide what is left to do BEFORE loading a model. A relaunch after a
        # partial sweep is the normal case, not the exception -- that is what
        # the resume policy is for -- and loading a backend only to discover
        # there is nothing to do costs two to three minutes of GPU allocation
        # per job. It also means a task-id mismatch, which `_resume` raises on,
        # fails in seconds rather than after a model load.
        todo, written = [], []
        for assay in task.cohort:
            if wanted and assay.id not in wanted:
                continue
            path = resolved.output_for(assay.id)
            if path.exists() and _resume(path, resolved) == "skip":
                print(f"  {assay.id}: already collected by this task, skipping",
                      flush=True)
                written.append(path)
                continue
            todo.append((assay, path))

        if dry_run:
            for assay, path in todo:
                print(f"  would collect {assay.id} -> {path}")
            return written
        if not todo:
            print("  nothing left to collect; no model loaded", flush=True)
            return written

        # Past this point a backend is required, so the imports live here and
        # not at the top of the method: everything above runs on a login node,
        # which is what makes "what is left to collect?" an answerable question
        # without a GPU allocation.
        sys.path.insert(0, str(_harness()))
        import exp_gym_deep                                     # noqa: E402

        identity = self.identity()
        self.verify()
        if identity.trunk_depth not in (None, resolved.trunk_depth):
            raise AdapterError(
                f"the task resolved its layers against a depth of "
                f"{resolved.trunk_depth} but the loaded {self.name} has "
                f"{identity.trunk_depth} blocks. Every layer index in this task "
                f"means a different block than intended.")

        for assay, path in todo:
            work = _work_dir(resolved, assay.id)
            arrays = exp_gym_deep.collect_assay(
                self.name, assay.id, str(Path(assay.assay_csv).parent),
                assay.msa_path, str(work),
                n_variants=task.n_variants or 100,
                recycles=self.spec.recycles,
                sampling_steps=int(self.spec.options.get("sampling_steps", 200)),
                msa_cap=self.spec.msa_cap,
                msa=self.spec.msa,
                out_path=str(path))

            arrays = _select_layers(arrays, resolved, np)
            arrays["layer_index"] = np.asarray(resolved.layers, np.int32)

            proto = P.protocol(
                script="collection.models.trunk.TrunkAdapter",
                design=f"per-layer trunk capture of {self.name} at the mutated "
                       f"position; kl from the model's own distogram head "
                       f"(logit lens), dz/ds against the wild type",
                layer=P.layers(
                    resolved.capture["layers"],
                    n_layers=len(resolved.layers)),
                features=P.features(
                    f"{self.name} pair row", identity.pair_width or 0),
                source=str(assay.msa_path),
                n_assays=1,
                cohort=resolved.cohort,
                assay=assay.id,
                task=resolved.name,
                task_id=resolved.task_id,
                schema_version=resolved.schema_version,
                resolved_task=resolved.to_dict(),
                model_identity=identity.to_dict(),
                layers_requested=resolved.capture["layers"],
                layers_resolved=list(resolved.layers),
                assay_csv_sha256=assay.assay_csv_sha256,
                msa_sha256=assay.msa_sha256,
            )
            artifacts.write_npz(path, arrays, protocol=proto)
            print(f"  wrote {path.name}  "
                  f"({arrays['dz_vec'].shape[0]} variants x "
                  f"{len(resolved.layers)} layers)", flush=True)
            written.append(path)
        return written


def _version_of(name: str) -> str | None:
    """The installed version of the package that actually holds the weights."""
    import importlib.metadata as md

    for dist in ("mosaic", "joltz", "jopenfold3", "protenij", "boltz"):
        try:
            return f"{dist} {md.version(dist)}"
        except Exception:
            continue
    return None


def _work_dir(resolved, assay_id: str) -> Path:
    """A job-local, run-unique scratch directory.

    Two jobs sharing one work directory race on the per-variant alignment file,
    and this project has already lost a row to exactly that: A10E in the ARGR
    archive differs from a fresh capture by 7.8% with cosine 0.997, the
    signature of a partially written a3m. The SLURM job id is in the path so
    two jobs cannot collide even if they are launched with the same output.
    """
    job = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
    return Path(resolved.output) / "work" / f"{assay_id}_{job}"


def _resume(path: Path, resolved) -> str:
    """What to do about an artifact that is already there.

    An existing file is only resumable work if it was produced by THIS task;
    otherwise it is someone else's result wearing the name this run wants.
    """
    from protein_interpretability import artifacts

    if resolved.resume == "overwrite":
        return "write"
    try:
        meta = artifacts.npz_meta(path) or {}
        theirs = (meta.get("protocol") or {}).get("task_id")
    except Exception:
        theirs = None

    if theirs == resolved.task_id:
        if resolved.resume in ("resume", "refuse"):
            return "skip"
        return "write"
    raise AdapterError(
        f"{path.name} already exists and was produced by task "
        f"{theirs or '<unrecorded>'}, not {resolved.task_id}. Overwriting it "
        f"would replace one measurement with a different one under the same "
        f"name. Choose a new output, or set resume='overwrite' deliberately.")


def _select_layers(arrays: dict, resolved, np) -> dict:
    """Keep only the requested layers, and refuse a backend that ignored them.

    The check is the point. A wrapper that returns all 64 blocks against a
    three-layer request produces an artifact whose layer axis is silently the
    wrong length, and every depth curve read off it would be wrong in a way no
    shape assertion downstream could catch -- because the shape would be
    perfectly self-consistent.
    """
    depth = int(arrays.get("n_layers", 0)) or None
    if depth is not None and depth != resolved.trunk_depth:
        raise AdapterError(
            f"the capture returned {depth} layers but the task resolved against "
            f"a {resolved.trunk_depth}-block trunk")

    layers = list(resolved.layers)
    if depth is not None and len(layers) == depth:
        return arrays                       # everything was asked for

    out = dict(arrays)
    for key in LAYER_AXIS_FIELDS:
        if key in out:
            arr = np.asarray(out[key])
            if arr.ndim < 2 or arr.shape[1] != depth:
                raise AdapterError(
                    f"{key} has shape {arr.shape}; its layer axis is not "
                    f"{depth} long, so a layer selection cannot be applied to "
                    f"it honestly")
            out[key] = arr[:, layers]
    out["n_layers"] = np.array(len(layers))
    return out
