"""Where this runs: roots, scheduler, container, caches — resolved once.

The three submitters in `jax_harness/` are nearly identical files whose real
differences are memory, time and which directory they exec from. Everything else
— account, partition, image, weights, scratch, the two SINGULARITYENV settings —
is the same three times over, which is how one of them ends up subtly different
from the others without anyone noticing.

This is that shared part. `configs/site/default.yaml` is committed and holds
logical names and ${VARIABLES}; `configs/site/local.yaml` beside it is ignored
by git and overrides key by key, which is what lets a checkout be moved to
another cluster or another user without editing tracked files.

Nothing here imports a backend or touches CUDA, so a profile resolves and a job
renders on a login node — which is the point, since the entire reason to render
a job is to see it before it is queued.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SITE_DIR = REPO / "configs" / "site"

def _split_refs(text: str):
    """Yield ('lit', s) and ('ref', name, default) across ${...} with NESTING.

    A regex cannot do this: the default of `${MOSAIC_SCRATCH:-/n/.../${USER}/m}`
    contains its own `${...}`, and any `[^}]*` stops at the inner closing brace.
    The first version of this produced `/n/.../${USER/mosaic}` -- a path that is
    wrong in a way a reader skims straight past.
    """
    i, n = 0, len(text)
    while i < n:
        start = text.find("${", i)
        if start < 0:
            yield ("lit", text[i:])
            return
        if start > i:
            yield ("lit", text[i:start])
        depth, j = 1, start + 2
        while j < n and depth:
            if text.startswith("${", j):
                depth += 1
                j += 2
                continue
            if text[j] == "}":
                depth -= 1
                if not depth:
                    break
            j += 1
        if depth:                       # unbalanced; treat the rest as literal
            yield ("lit", text[start:])
            return
        body = text[start + 2:j]
        name, sep, default = body.partition(":-")
        yield ("ref", name.strip(), default if sep else None)
        i = j + 1


class SiteError(RuntimeError):
    """The site profile is missing or does not describe a usable site."""


def _scalar(text: str):
    text = text.strip()
    if text in ("null", ""):
        return None
    if text in ("true", "false"):
        return text == "true"
    if len(text) >= 2 and text[0] == text[-1] in "\"'":
        return text[1:-1]
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text


def _parse(text: str) -> dict:
    """Two-level `key:` / `  key: value` YAML, which is all a site profile is."""
    doc: dict = {}
    section = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        key, _, value = raw.strip().partition(":")
        key = key.strip()
        if indent == 0:
            if value.strip():
                doc[key] = _scalar(value)
                section = None
            else:
                section = doc.setdefault(key, {})
                if not isinstance(section, dict):
                    section = doc[key] = {}
        elif section is not None:
            section[key] = _scalar(value)
    return doc


def _merge(base: dict, over: dict) -> dict:
    """Key-by-key, one level deep. An override should be able to set a single
    partition without restating the whole scheduler block."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


class Site:
    """A resolved site profile."""

    def __init__(self, raw: dict, sources: list[Path]):
        self._raw = raw
        self.sources = sources
        self._cache: dict[str, str] = {}

    # ---- construction -----------------------------------------------------
    @classmethod
    def load(cls, path=None) -> "Site":
        """default.yaml, then local.yaml, then $PROT_INTERP_SITE, then `path`."""
        candidates = [SITE_DIR / "default.yaml", SITE_DIR / "local.yaml"]
        env_profile = os.environ.get("PROT_INTERP_SITE")
        if env_profile:
            candidates.append(Path(env_profile))
        if path:
            candidates.append(Path(path))

        raw: dict = {}
        used: list[Path] = []
        for cand in candidates:
            if cand.exists():
                raw = _merge(raw, _parse(cand.read_text()))
                used.append(cand)
        if not used:
            raise SiteError(
                f"no site profile found. Expected {SITE_DIR / 'default.yaml'} "
                f"to be committed; add configs/site/local.yaml for anything "
                f"specific to this cluster or user.")
        return cls(raw, used)

    # ---- resolution -------------------------------------------------------
    def _expand(self, value: str, _seen=()) -> str:
        """Expand ${VAR}, ${VAR:-default} and ${section.key}, nesting included."""
        out = []
        for piece in _split_refs(value):
            if piece[0] == "lit":
                out.append(piece[1])
                continue
            _, name, default = piece
            if "." in name:                      # a reference to another key
                if name in _seen:
                    raise SiteError(
                        f"site profile has a circular reference through {name}")
                section, _, key = name.partition(".")
                inner = self._raw.get(section, {})
                if not isinstance(inner, dict) or key not in inner:
                    raise SiteError(f"site profile has no {name!r} to expand")
                out.append(self._expand(str(inner[key]), _seen + (name,)))
                continue
            got = os.environ.get(name)
            if got:
                out.append(got)
            elif default is not None:
                out.append(self._expand(default, _seen))
            else:
                raise SiteError(
                    f"site profile needs ${{{name}}}, which is not set and has "
                    f"no default. Set it, or give it one in "
                    f"configs/site/local.yaml.")
        return "".join(out)

    def get(self, dotted: str, default=None):
        """A value by `section.key`, with variables expanded."""
        section, _, key = dotted.partition(".")
        block = self._raw.get(section)
        if not isinstance(block, dict) or key not in block:
            if default is not None:
                return default
            raise SiteError(
                f"site profile has no {dotted!r}. Profiles read: "
                f"{[str(p) for p in self.sources]}")
        value = block[key]
        return self._expand(value) if isinstance(value, str) else value

    def path(self, dotted: str) -> Path:
        return Path(self.get(dotted))

    @property
    def env(self) -> dict[str, str]:
        return {k: self._expand(str(v))
                for k, v in (self._raw.get("env") or {}).items()}

    # ---- checks -----------------------------------------------------------
    def verify(self, *, require_roots=True) -> None:
        """Raise unless the profile describes a site that exists.

        Cheap, and worth doing before rendering: a job whose work root is a
        typo fails after it is queued, scheduled and started.
        """
        problems = []
        for key in ("scheduler.account", "scheduler.partition"):
            try:
                if not self.get(key):
                    problems.append(f"{key} is empty")
            except SiteError as exc:
                problems.append(str(exc))
        if require_roots:
            for key in ("roots.work", "roots.repo"):
                try:
                    p = self.path(key)
                except SiteError as exc:
                    problems.append(str(exc))
                    continue
                if not p.is_dir():
                    problems.append(f"{key} is {p}, which is not a directory")
        if problems:
            raise SiteError(
                "site profile does not describe a usable site:\n  "
                + "\n  ".join(problems)
                + f"\n\nprofiles read: {[str(p) for p in self.sources]}")

    def describe(self) -> str:
        lines = [f"site: {self._raw.get('site', '?')}",
                 f"profiles: {', '.join(str(p.name) for p in self.sources)}"]
        for key in ("roots.work", "roots.repo", "roots.scratch", "roots.logs",
                    "scheduler.account", "scheduler.partition",
                    "container.image", "caches.uv"):
            try:
                lines.append(f"  {key:20s} {self.get(key)}")
            except SiteError as exc:
                lines.append(f"  {key:20s} UNRESOLVED ({exc})")
        return "\n".join(lines)
