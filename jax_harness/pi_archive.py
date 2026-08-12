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

import hashlib
import json
import os
import socket
import subprocess
import sys
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
