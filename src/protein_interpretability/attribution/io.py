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

SCHEMA_VERSION = "attribution_v1"


@dataclass
class AttributionResult:
    """One backward-pass worth of gradients on the captured surfaces.

    Tensors are stored CPU-side, fp32 unless the caller downcasts. ``*_input``
    fields hold the surface activations themselves (post-forward, pre-backward)
    so downstream analysis can compute ``input × gradient`` without re-running
    the model.
    """

    record_id: str
    recycling_steps: int
    target_value: float
    target_spec: dict
    query_grad: torch.Tensor                # (B, N, D_s)
    query_input: torch.Tensor               # (B, N, D_s)
    msa_grad: torch.Tensor | None           # (B, S, N, D_m) or None
    msa_input: torch.Tensor | None
    token_mask: torch.Tensor                # (B, N) bool
    provenance: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def input_x_grad(self, surface: str = "query") -> torch.Tensor:
        """Element-wise input×gradient on a chosen surface."""
        if surface == "query":
            return self.query_input * self.query_grad
        if surface == "msa":
            if self.msa_input is None or self.msa_grad is None:
                raise ValueError("msa surface not captured for this result")
            return self.msa_input * self.msa_grad
        raise ValueError(f"unknown surface: {surface!r}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["query_grad"] = self.query_grad
        d["query_input"] = self.query_input
        d["msa_grad"] = self.msa_grad
        d["msa_input"] = self.msa_input
        d["token_mask"] = self.token_mask
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AttributionResult":
        version = d.get("schema_version", "unknown")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"schema version mismatch: file has {version!r}, code expects "
                f"{SCHEMA_VERSION!r}"
            )
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
