"""The only sanctioned way to write a result, and the guard that reads it back.

`pi_protocol` states the rule an archived number has to satisfy: it must carry,
in the same file, everything needed to know what it is comparable to. That rule
was right and it did not hold, because `protocol()` raises only if you call it,
from the bottom of a 300-line main(), after every number has already been
computed. Seven scripts of fifty-six call it. Twenty-seven archives of a hundred
and thirteen carry a block.

A guard you have to remember to invoke is documentation wearing a `raise`. This
module moves it to the seam it should always have been at: you cannot write a
result file without one, because `protocol` is a required keyword of the only
function that writes them.

WHY `argv` IS HERE, AND WHY IT MATTERS MORE THAN IT LOOKS.

Migrating the PC basis meant rerunning ten analyses and diffing each against its
archive. Five of those runs disagreed with the archive for reasons that had
nothing to do with the change:

    gym2s_*.npz where the documented glob was gym2_*.npz
    --k 16 where the archive used --k 256
    --n-perm 200 where the archive used 1000
    a bare --k-sweep, which argparse rejected
    a missing --features

Three produced plausible, wrong diffs -- one of them 419 differences and another
131 -- and each cost a round trip to diagnose. The three that WERE diagnosable
were diagnosable for exactly one reason: the differing parameter happened to be
recorded in the output, so it turned up in the diff as a number. `/protocol/k`
appeared with d=2.400e+02 and the mystery evaporated.

`sys.argv` generalises that to every parameter at once. It is one line, it costs
nothing, and it is the difference between "this analysis changed" and "you ran
it differently".

WHAT IS DELIBERATELY NOT RECONSTRUCTED. `reconstruct_provenance` records facts
ABOUT a file -- mtime, size, sha256, the commit that was checked out then, array
shapes and dtypes read from the arrays themselves. It never guesses an
invocation. An inferred command line written into a file is indistinguishable
from a recorded one afterwards, and this project's entire provenance apparatus
exists because that distinction was lost once. Its output goes under
`provenance_reconstructed`, never `protocol`, so measured and inferred can never
be confused.

  from pi_archive import write_result
  write_result(out_path, payload, protocol=pi_protocol.protocol(...))
"""

from __future__ import annotations

import dataclasses  # noqa: F401
import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Jobs execute from prot_interp_files/harness/, a plain COPY of this directory
# and not a git checkout, so git commands run there return nothing. The first
# batch of archives written through this module recorded git_commit: null for
# exactly that reason -- a silent None, which is the failure this module exists
# to stop. The repo is located explicitly, and `mirrored` records whether the
# code that ran was the checkout or a copy of it.
REPO = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
            "protein_interpretability")
PROTOCOL_KEY = "protocol"
PROVENANCE_KEY = "provenance"
RECONSTRUCTED_KEY = "provenance_reconstructed"
NPZ_META_KEY = "_pi_meta"


# ---- facts about the run --------------------------------------------------
def _git_root():
    """The checkout, whether or not this file is running from inside it."""
    for cand in (HERE, REPO):
        try:
            r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                               cwd=str(cand), capture_output=True, text=True,
                               timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                return Path(r.stdout.strip())
        except Exception:
            pass
    return None


def _git(*args, cwd=None):
    root = cwd or _git_root()
    if root is None:
        return None
    try:
        return subprocess.run(["git", *args], cwd=str(root),
                              capture_output=True, text=True,
                              timeout=15).stdout.strip() or None
    except Exception:
        return None


def run_provenance():
    """Everything about THIS invocation that a later reader would need.

    `argv` is the field that matters. A protocol block says what the analysis
    is; argv says what was actually asked of it, which is the axis five
    verification runs drifted on in a single afternoon.
    """
    dirty = _git("status", "--porcelain")
    return {
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "git_root": str(_git_root() or ""),
        "mirrored": not str(HERE).startswith(str(REPO)),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- writing --------------------------------------------------------------
def write_result(path, payload, *, protocol, indent=2):
    """Write a result JSON. `protocol` is required; omitting it is a TypeError.

    The block is placed here rather than by the caller. Scripts used to nest it
    three different ways -- bare, spread, or merged with extra keys -- so a
    reader could not rely on one path, and neither could a guard.
    """
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError("protocol must be a non-empty dict from "
                         "pi_protocol.protocol(); see pi_protocol.__doc__")
    body = dict(payload)
    if PROTOCOL_KEY in body and body[PROTOCOL_KEY] != protocol:
        # A script that already built its own block: merge, protocol wins on
        # conflict, and nothing is silently dropped.
        body[PROTOCOL_KEY] = {**body[PROTOCOL_KEY], **protocol}
    else:
        body[PROTOCOL_KEY] = dict(protocol)
    body[PROVENANCE_KEY] = run_provenance()
    Path(path).write_text(json.dumps(body, indent=indent, default=float))
    return path


def write_npz(path, arrays, *, protocol, compressed=True):
    """Write an npz carrying its own protocol block.

    The costliest error in this project's history was an npz: `deep2_*` stored
    `dz_site` as a per-layer NORM where every consumer assumed a vector, so a
    probe that could not answer the question returned +0.468 instead of raising.
    An archive that records its own block shapes turns that into a load error.
    """
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError("protocol must be a non-empty dict from "
                         "pi_protocol.protocol()")
    meta = {PROTOCOL_KEY: protocol, PROVENANCE_KEY: run_provenance(),
            "arrays": {k: {"shape": list(np.shape(v)),
                           "dtype": str(np.asarray(v).dtype)}
                       for k, v in arrays.items()}}
    payload = dict(arrays)
    payload[NPZ_META_KEY] = np.array(json.dumps(meta, default=float))
    (np.savez_compressed if compressed else np.savez)(path, **payload)
    return path


# ---- reading, and the guard ----------------------------------------------
def read_result(path, *, require_protocol=False, quoted=False):
    """Load a result JSON, optionally refusing one that cannot be interpreted.

    `quoted` marks an input that supplies a number a page states. Those fail
    hard: a figure or a sentence built on an archive whose protocol is unknown
    is exactly the situation that cost a session to untangle. Everything else
    loads and is flagged, because a build that cannot run is a build someone
    turns the guard off for.
    """
    d = json.loads(Path(path).read_text())
    has = isinstance(d, dict) and isinstance(d.get(PROTOCOL_KEY), dict) \
        and bool(d[PROTOCOL_KEY])
    if not has and (require_protocol or quoted):
        raise ValueError(
            f"{Path(path).name} has no protocol block and supplies a quoted "
            f"number. Rerun it through pi_archive.write_result, or drop the "
            f"claim that reads it. Do NOT synthesise a block: an inferred "
            f"protocol is indistinguishable from a recorded one afterwards.")
    return d, has


def npz_meta(path):
    """The block an npz carries, or None if it predates the convention."""
    with np.load(path, allow_pickle=True) as z:
        if NPZ_META_KEY not in z.files:
            return None
        return json.loads(str(z[NPZ_META_KEY]))


# ---- reconstruction, strictly of what can be measured ---------------------
def reconstruct_provenance(path):
    """Facts about an existing archive. Never an inferred invocation.

    Recorded: mtime, size, sha256, the commit checked out at that mtime, and --
    for an npz -- the shape and dtype of every array, read from the arrays.
    Those are measurements.

    NOT recorded: which glob, which --k, which --n-perm. Those are guesses, and
    this session produced two archives whose real invocation differed from the
    one its own docstring documented. A guess written here would look identical
    to a fact.
    """
    p = Path(path)
    st = p.stat()
    when = datetime.fromtimestamp(st.st_mtime, timezone.utc)
    out = {
        "reconstructed_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"),
        "inferred": True,
        "note": "Facts about the file, not a record of how it was produced. "
                "No invocation is reconstructed here by design.",
        "file": {"name": p.name, "bytes": st.st_size,
                 "mtime_utc": when.strftime("%Y-%m-%d %H:%M UTC"),
                 "sha256": sha256(p)},
        "git_commit_at_mtime": _git(
            "rev-list", "-1", f"--before={when.isoformat()}", "HEAD"),
    }
    if p.suffix == ".npz":
        try:
            with np.load(p, allow_pickle=True) as z:
                out["arrays"] = {k: {"shape": list(z[k].shape),
                                     "dtype": str(z[k].dtype)}
                                 for k in z.files if k != NPZ_META_KEY}
        except Exception as e:                       # pragma: no cover
            out["arrays_error"] = str(e)
    return out


def stamp_reconstructed(path, *, dry_run=True):
    """Add `provenance_reconstructed` to an existing JSON. Never `protocol`."""
    p = Path(path)
    d = json.loads(p.read_text())
    if not isinstance(d, dict):
        return False
    d[RECONSTRUCTED_KEY] = reconstruct_provenance(p)
    if not dry_run:
        p.write_text(json.dumps(d, indent=2, default=float))
    return True


# ==========================================================================
# Reading a capture: what the archive IS, not what its filename suggests
# ==========================================================================
# `dz_site` is (n, L, 128) pair-row VECTORS in gym2/gym2s/gym3/gym3p3 and
# (n, L) per-layer NORMS in deep2/xm. In xm the norms sit in the same file as
# the vectors, under `dz_vec`. Nothing checked, and 96 raw reads across 35
# scripts each carried an unstated assumption about which one they had.
#
# The cost is on record. The multi-model section of the master report was built
# on the norm family and was STRUCTURALLY INCAPABLE of the comparison it
# claimed: a probe on norms sees how far the pair row moved and never which
# way. It returned +0.468 against the +0.61 the same model reaches when the
# direction is available -- a plausible number, low enough to read as an honest
# negative result about Boltz-2. It was caught by reasoning about the archive,
# not by anything the code did.
#
# One correction to how this was first described. `analyze_gym_deep.py:115`
#     v = np.linalg.norm(v, axis=-1) if v.ndim == 3 else v
# looks like that failure and is NOT: its comment states the reason -- the
# models store different things, and reducing both to a magnitude stops one
# contributing C times as many columns. It is a magnitude comparison on
# purpose. The danger is only that the expression is silent about intent, so
# the same line copied without its comment would be indistinguishable from an
# accident. `magnitudes()` says the intent out loud and does the same thing.
#
# So `pair_row()` RAISES on a norm-only archive rather than returning norms,
# and `magnitudes()` is always available because it is derivable from either.

VECTOR_FAMILIES = ("gym2", "gym2s", "gym3", "gym3p3")
NORM_FAMILIES = ("deep2",)
BOTH_FAMILIES = ("xm",)


def _family_of(path):
    stem = Path(path).stem
    for fam in ("gym3p3", "gym2s", "gym3", "gym2", "deep2", "xm"):
        if stem.startswith(fam + "_"):
            return fam
    return None


@dataclass
class Capture:
    """A capture archive that knows what it holds.

    The filename is recorded as `family` but never trusted: every accessor
    checks the array it actually loaded, so a file renamed into the wrong
    family fails on the shape rather than on the name.
    """

    path: Path
    family: str
    files: tuple
    _z: object = field(repr=False, default=None)

    def has_vectors(self):
        for k in ("dz_vec", "dz_site"):
            if k in self.files and self._arr(k).ndim == 3:
                return True
        return False

    def pair_row(self, layer, key="dz"):
        """(n, 128) at `layer`. Raises if this archive holds only norms.

        float64 on the way out. The archives are float32 and analyze_pc2 fed
        that straight to numpy, so pc2_v2.npz -- the basis six analyses
        inherit -- was computed in single precision for a week.
        """
        for k in (f"{key}_vec", f"{key}_site"):
            if k in self.files and self._arr(k).ndim == 3:
                return np.asarray(self._arr(k)[:, layer, :], np.float64)
        shape = self._arr(f"{key}_site").shape if f"{key}_site" in self.files else "?"
        raise ValueError(
            f"{self.path.name} (family {self.family!r}) stores {key}_site as "
            f"{shape} -- per-layer NORMS, not vectors. A direction cannot be "
            f"recovered from a magnitude. Use magnitudes(), or read a family "
            f"that carries {key}_vec: {', '.join(BOTH_FAMILIES + VECTOR_FAMILIES)}.")

    def magnitudes(self, key="dz"):
        """(n, L) per-layer norms, from whichever representation exists."""
        a = self._arr(f"{key}_site")
        return np.asarray(np.linalg.norm(a, axis=-1) if a.ndim == 3 else a,
                          np.float64)

    def rows(self, layer_index, key="dz"):
        """(n, R, 128) per-residue rows, for the families that captured them."""
        if f"{key}_row" not in self.files:
            raise ValueError(f"{self.path.name} has no {key}_row; only "
                             f"gym3/gym3p3 captured per-residue rows.")
        return np.asarray(self._arr(f"{key}_row")[:, layer_index], np.float64)

    def _arr(self, k):
        if k not in self.files:
            raise KeyError(f"{self.path.name} has no {k!r}; it holds "
                           f"{', '.join(sorted(self.files)[:8])}")
        return self._z[k]

    def field(self, k, dtype=None):
        """A field this type does not interpret. Raises if absent."""
        a = self._arr(k)
        return np.asarray(a, dtype) if dtype is not None else a

    def get(self, k, default=None, dtype=None):
        """An OPTIONAL field. Captures differ in what they carry -- model,
        capture_drift and signal_to_drift are present in some families and not
        others -- and `k in cap` is the honest way to ask rather than reaching
        past the interface into the npz."""
        if k not in self.files:
            return default
        a = self._arr(k)
        return np.asarray(a, dtype) if dtype is not None else a

    def __contains__(self, k):
        return k in self.files

    @property
    def n_layers(self):
        return int(self._arr("dz_site").shape[1])

    def describe(self):
        return {"path": str(self.path), "family": self.family,
                "vectors": self.has_vectors(),
                "arrays": {k: {"shape": list(self._arr(k).shape),
                               "dtype": str(self._arr(k).dtype)}
                           for k in self.files
                           if k.startswith(("dz", "ds"))}}


def load_capture(path, *, require_vectors=False):
    """Open a capture, optionally refusing a norm-only archive up front."""
    p = Path(path)
    z = np.load(p, allow_pickle=True)
    cap = Capture(path=p, family=_family_of(p), files=tuple(z.files), _z=z)
    if require_vectors and not cap.has_vectors():
        raise ValueError(
            f"{p.name} holds per-layer norms only, and this analysis needs "
            f"directions. Running it on norms is what produced the +0.468 "
            f"cross-model result no probe on that archive could have improved.")
    side = p.with_suffix(".meta.json")
    if side.exists():
        rec = json.loads(side.read_text()).get("arrays", {})
        for k, want in rec.items():
            if k in cap.files and list(cap._arr(k).shape) != want["shape"]:
                raise ValueError(
                    f"{p.name}: {k} is {list(cap._arr(k).shape)} but its "
                    f"sidecar records {want['shape']}. The file and its "
                    f"metadata disagree; do not use either until you know why.")
    return cap


def write_capture_sidecar(path, *, dry_run=False):
    """Record an existing capture's shapes beside it, without rewriting 12 GB.

    The captures cost GPU-hours each and are the one thing here that cannot be
    regenerated cheaply, so they are not touched. The sidecar carries what
    `write_npz` would have embedded, and `load_capture` cross-checks it, so a
    later mismatch between file and metadata is detectable rather than silent.
    """
    p = Path(path)
    cap = load_capture(p)
    meta = {**cap.describe(), "inferred": True,
            "note": "Written after the fact from the arrays themselves. "
                    "Measurements, not a record of how the capture was made.",
            "file": {"bytes": p.stat().st_size, "sha256": sha256(p)},
            "recorded": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    if not dry_run:
        p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta
