"""Versioned save/load schema for attribution results.

One ``.pt`` per (record, recycling_step). Schema is opaque dict on disk; the
:class:`AttributionResult` dataclass is a typed mirror with helpers.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = "attribution_v2"
_SUPPORTED_SCHEMA_VERSIONS = ("attribution_v1", "attribution_v2")


@dataclass
class AttributionResult:
    """One backward-pass worth of gradients on the captured surfaces.

    Tensors are stored CPU-side, fp32 unless the caller downcasts. ``*_input``
    fields hold the surface activations themselves (post-forward, pre-backward)
    so downstream analysis can compute ``input × gradient`` without re-running
    the model.

    Pair-rep mode (default in v2): only ``pair_grad`` / ``pair_input`` are
    populated (gradient at distogram_module's input — the final pair
    representation). ``query_*`` and ``msa_*`` are ``None`` because the trunk
    has no autograd graph in this mode.

    Legacy mode (v1, mock-model tests only): ``query_*`` and ``msa_*`` are
    populated; ``pair_*`` are ``None``.
    """

    record_id: str
    recycling_steps: int
    target_value: float
    target_spec: dict
    token_mask: torch.Tensor                # (B, N) bool
    query_grad: torch.Tensor | None = None  # (B, N, D_s) or None
    query_input: torch.Tensor | None = None
    msa_grad: torch.Tensor | None = None    # (B, S, N, D_m) or None
    msa_input: torch.Tensor | None = None
    pair_grad: torch.Tensor | None = None   # (B, N, N, D_z) or None
    pair_input: torch.Tensor | None = None
    provenance: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def input_x_grad(self, surface: str = "pair") -> torch.Tensor:
        """Element-wise input×gradient on a chosen surface."""
        if surface == "pair":
            if self.pair_input is None or self.pair_grad is None:
                raise ValueError("pair surface not captured for this result")
            return self.pair_input * self.pair_grad
        if surface == "query":
            if self.query_input is None or self.query_grad is None:
                raise ValueError("query surface not captured for this result")
            return self.query_input * self.query_grad
        if surface == "msa":
            if self.msa_input is None or self.msa_grad is None:
                raise ValueError("msa surface not captured for this result")
            return self.msa_input * self.msa_grad
        raise ValueError(f"unknown surface: {surface!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AttributionResult":
        version = d.get("schema_version", "unknown")
        if version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"schema version {version!r} not supported. Code expects one "
                f"of {_SUPPORTED_SCHEMA_VERSIONS}."
            )
        # v1 files lack pair_* fields; default them to None and pass through.
        d = {**d}
        d.setdefault("pair_grad", None)
        d.setdefault("pair_input", None)
        return cls(**d)


def save_result(result: AttributionResult, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.to_dict(), path)
    return path


def load_result(path: Path | str) -> AttributionResult:
    d = torch.load(Path(path), map_location="cpu", weights_only=False)
    return AttributionResult.from_dict(d)


# ----------------------------------------------------------------------
# Provenance helpers
# ----------------------------------------------------------------------


def collect_provenance(
    config: dict | None = None,
    checkpoint_path: str | os.PathLike | None = None,
    extra: dict | None = None,
) -> dict:
    """Best-effort provenance dict — git SHA, platform, optional ckpt hash."""
    prov: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    sha = _git_head()
    if sha is not None:
        prov["git_sha"] = sha
    if checkpoint_path is not None:
        try:
            prov["checkpoint_sha1"] = _file_sha1(Path(checkpoint_path))
            prov["checkpoint_path"] = str(checkpoint_path)
        except OSError:
            prov["checkpoint_path"] = str(checkpoint_path)
    if config is not None:
        prov["config"] = config
    if extra is not None:
        prov.update(extra)
    return prov


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()
