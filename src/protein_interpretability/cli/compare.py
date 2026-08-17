"""Numeric comparison of two result archives, with the bands that are known.

Two results of the same producer are almost never bit-identical, and the useful
question is never "are they equal" but "did they move by more than this producer
moves on its own". That distinction was established the hard way: three separate
differences this project has investigated as regressions turned out to be
run-to-run variation, each identified only by running the unchanged code twice.

The bands below are measured, not guessed, and each records how. A producer with
no entry is expected to be exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# producer stem -> (tolerance, how it was established)
KNOWN_BANDS: dict[str, tuple[float, str]] = {
    "svd_ds_v1": (2e-3, "prediction-ordered curves; two identical runs differ "
                        "by 8.96e-04 (2026-08-17)"),
    "svd_dz_v3": (1e-12, "float noise in a batched SVD; 3.01e-14 between "
                         "identical runs"),
    "gate_probe": (5e-6, "live_by_layer reduces jax.vmap over the Pairformer "
                         "transitions in explicit float32; 8.90e-07 between "
                         "identical runs"),
}

# Keys that legitimately differ between any two runs.
VOLATILE = {"provenance"}


@dataclass
class Diff:
    n_numbers: int = 0
    max_abs: float = 0.0
    max_abs_at: str = ""
    max_rel: float = 0.0
    issues: list[str] = field(default_factory=list)

    def note(self, msg: str, cap: int = 8) -> None:
        if len(self.issues) < cap:
            self.issues.append(msg)
        elif len(self.issues) == cap:
            self.issues.append("... further issues suppressed")


def _walk(old, new, path: str, d: Diff) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for k in old:
            if k in VOLATILE or (k == "recorded" and path.endswith("protocol")):
                continue
            if k in new:
                _walk(old[k], new[k], f"{path}.{k}", d)
            else:
                d.note(f"{path}.{k}: missing from the new result")
        for k in new:
            if k not in VOLATILE and k not in old:
                d.note(f"{path}.{k}: present only in the new result")
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            d.note(f"{path}: length {len(old)} -> {len(new)}")
        for i, (a, b) in enumerate(zip(old, new)):
            _walk(a, b, f"{path}[{i}]", d)
    elif isinstance(old, bool) or isinstance(new, bool):
        if old is not new:
            d.note(f"{path}: {old!r} -> {new!r}")
    elif isinstance(old, (int, float)) and isinstance(new, (int, float)):
        d.n_numbers += 1
        if math.isnan(old) and math.isnan(new):
            return
        delta = abs(old - new)
        if delta > d.max_abs:
            d.max_abs, d.max_abs_at = delta, path
        scale = max(abs(old), abs(new))
        if scale > 0:
            d.max_rel = max(d.max_rel, delta / scale)
    elif old != new:
        d.note(f"{path}: {str(old)[:50]!r} -> {str(new)[:50]!r}")


def compare(old: dict, new: dict) -> Diff:
    """Every number and every string, excluding provenance and its timestamp."""
    d = Diff()
    _walk(old, new, "$", d)
    return d


def band_for(name: str) -> tuple[float, str]:
    """The tolerance for a producer, and why it is what it is."""
    return KNOWN_BANDS.get(name, (0.0, "expected exact"))


def verdict(name: str, d: Diff) -> tuple[bool, str]:
    """Does this diff pass, given what the producer is known to do on its own?"""
    tol, why = band_for(name)
    if d.issues:
        return False, f"structural differences: {d.issues[0]}"
    if d.max_abs <= max(tol, 1e-12):
        if tol:
            return True, f"within the known band ({tol:.0e}: {why})"
        return True, "exact"
    if tol:
        return False, (f"{d.max_abs:.3e} at {d.max_abs_at}, above the known "
                       f"band of {tol:.0e} ({why})")
    return False, (f"{d.max_abs:.3e} at {d.max_abs_at}; this producer is "
                   "expected to reproduce exactly. Before treating it as a "
                   "regression, run the unchanged code twice -- that is how "
                   "every previous difference here turned out to be noise.")
