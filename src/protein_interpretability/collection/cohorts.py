"""A cohort is a named, checksummed list of assays -- not whatever a glob matched.

    from protein_interpretability.collection import Cohort

    basis = Cohort.load("basis_assays")
    basis.verify()                       # raises if an input moved
    for assay in basis:
        print(assay.id, assay.msa_path, assay.msa_rows)

`verify()` is the point. Every input carries a sha256 taken when the manifest
was written, and the alignment underneath a cohort has changed underfoot in this
project before. A run that reads a different alignment than the one its numbers
were computed with does not fail -- it returns a plausible result -- so the check
has to be explicit and has to happen before the GPU is touched.

Deliberately dependency-free: this module parses the small, fixed YAML subset the
manifests are written in rather than importing a YAML library, so a cohort can be
inspected in any environment, including one with no scientific stack at all.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
COHORT_DIR = REPO / "configs" / "cohorts"


class CohortError(RuntimeError):
    """A cohort does not match what is on disk."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scalar(text: str):
    text = text.strip()
    if text in ("null", ""):
        return None
    if text in ("true", "false"):
        return text == "true"
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1]
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d*\.\d+", text):
        return float(text)
    return text


def _parse(text: str) -> dict:
    """The fixed shape `build_cohort_manifests.py` writes, and nothing else."""
    doc: dict = {}
    assays: list[dict] = []
    cur: dict | None = None
    nested: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if indent == 0:
            if line == "assays:":
                doc["assays"] = assays
                cur = None
            elif ":" in line:
                k, _, v = line.partition(":")
                doc[k.strip()] = _scalar(v)
            continue
        if line.startswith("- "):
            cur = {}
            assays.append(cur)
            line = line[2:].strip()
            nested = None
        if cur is None:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        if v.strip() == "" and indent >= 4:
            if indent >= 6 and nested:
                cur[nested][k] = None
            else:
                nested = k
                cur[k] = {}
            continue
        if indent >= 6 and nested:
            cur[nested][k] = _scalar(v)
        else:
            nested = None
            cur[k] = _scalar(v)
    doc.setdefault("assays", assays)
    return doc


@dataclass(frozen=True)
class Assay:
    id: str
    assay_csv: str | None
    assay_csv_sha256: str | None
    msa_path: str | None
    msa_sha256: str | None
    msa_rows: int | None
    n_single_variants: int | None
    wt_length: int | None

    def inputs(self) -> list[tuple[str, str, str]]:
        """(role, path, expected sha256) for everything this assay needs."""
        out = []
        if self.assay_csv and self.assay_csv_sha256:
            out.append(("assay_csv", self.assay_csv, self.assay_csv_sha256))
        if self.msa_path and self.msa_sha256:
            out.append(("msa", self.msa_path, self.msa_sha256))
        return out


class Cohort:
    """A named list of assays, loaded from a manifest."""

    def __init__(self, name: str, description: str, assays: list[Assay],
                 source: Path | None = None):
        self.name = name
        self.description = description
        self.assays = assays
        self.source = source

    # ---- construction -----------------------------------------------------
    @staticmethod
    def _check_document(doc: dict, where: str) -> None:
        """Refuse a manifest that does not describe a usable cohort.

        The parser is lenient by design — it accepts the small fixed shape the
        generator writes — so this is where a hand-edited or truncated manifest
        has to be caught. The dangerous case is not a malformed file, which
        fails loudly on its own, but an EMPTY one: a cohort with no assays
        loads, iterates zero times, and `verify()` passes because there is
        nothing to check. That is a guard reporting success for having done
        nothing, which is the failure mode this project tests against
        everywhere else.
        """
        problems = []
        assays = doc.get("assays") or []
        if not assays:
            problems.append(
                "no assays. An empty cohort verifies clean and analyses to "
                "nothing, so it is refused rather than loaded")

        ids = [e.get("id") for e in assays]
        if any(not i for i in ids):
            problems.append(f"{sum(1 for i in ids if not i)} assay(s) with no id")
        dupes = sorted({i for i in ids if i and ids.count(i) > 1})
        if dupes:
            problems.append(
                f"duplicate ids {dupes}; an assay counted twice weights it "
                "twice in every pooled statistic")

        declared = doc.get("n_assays")
        if isinstance(declared, int) and declared != len(assays):
            problems.append(
                f"header says n_assays: {declared} but the list holds "
                f"{len(assays)}. The manifest disagrees with itself, which "
                "means it was edited by hand rather than regenerated")

        no_inputs = [e.get("id") for e in assays
                     if not (e.get("assay_csv") or e.get("msa"))]
        if no_inputs:
            problems.append(
                f"{len(no_inputs)} assay(s) name no inputs at all ({no_inputs[:3]}"
                f"{'...' if len(no_inputs) > 3 else ''}); nothing about them can "
                "be verified")

        if problems:
            raise CohortError(
                f"{where} is not a usable cohort manifest:\n  "
                + "\n  ".join(problems)
                + "\n\nRegenerate it with jax_harness/build_cohort_manifests.py "
                  "rather than editing it.")

    @classmethod
    def from_manifest(cls, path) -> "Cohort":
        path = Path(path)
        doc = _parse(path.read_text())
        cls._check_document(doc, str(path))
        assays = [
            Assay(
                id=e.get("id"),
                assay_csv=(e.get("assay_csv") or {}).get("path"),
                assay_csv_sha256=(e.get("assay_csv") or {}).get("sha256"),
                msa_path=(e.get("msa") or {}).get("path"),
                msa_sha256=(e.get("msa") or {}).get("sha256"),
                msa_rows=(e.get("msa") or {}).get("rows"),
                n_single_variants=e.get("n_single_variants"),
                wt_length=e.get("wt_length"),
            )
            for e in doc.get("assays", [])
        ]
        return cls(doc.get("cohort", path.stem), doc.get("description", ""),
                   assays, source=path)

    @classmethod
    def load(cls, name: str) -> "Cohort":
        """Load by name from configs/cohorts."""
        path = COHORT_DIR / f"{name}.yaml"
        if not path.exists():
            have = sorted(p.stem for p in COHORT_DIR.glob("*.yaml"))
            raise KeyError(f"no cohort {name!r}; have {have}")
        return cls.from_manifest(path)

    @staticmethod
    def available() -> list[str]:
        return sorted(p.stem for p in COHORT_DIR.glob("*.yaml"))

    # ---- the checks -------------------------------------------------------
    def verify(self, *, checksums: bool = True) -> None:
        """Raise unless every input exists and still hashes to its manifest value.

        `checksums=False` checks existence only, which is the cheap version for
        a login-node sanity check; the default is the one that catches an
        alignment regenerated in place.
        """
        if not self.assays:
            raise CohortError(
                f"cohort {self.name!r} holds no assays, so verify() would pass "
                "for having checked nothing")
        problems = []
        for assay in self.assays:
            if not assay.inputs():
                problems.append(
                    f"{assay.id}: no checksummed inputs, so nothing about it "
                    "can be verified")
            for role, path_s, expected in assay.inputs():
                p = Path(path_s)
                if not p.exists():
                    problems.append(f"{assay.id}: {role} missing at {p}")
                elif checksums and _sha256(p) != expected:
                    problems.append(
                        f"{assay.id}: {role} at {p} has changed since the "
                        f"manifest was written (sha256 differs). Regenerate the "
                        f"manifest deliberately, or find out what rewrote it -- "
                        f"do not run against it.")
        if problems:
            raise CohortError(
                f"cohort {self.name!r} does not match disk:\n  "
                + "\n  ".join(problems))

    def assert_disjoint(self, other: "Cohort") -> None:
        """Raise if two cohorts share an assay.

        `heldout_v1`'s entire claim is that its cohort never touched the basis.
        That is checkable, so it should be checked rather than asserted in prose.
        """
        shared = {a.id for a in self.assays} & {a.id for a in other.assays}
        if shared:
            raise CohortError(
                f"{self.name} and {other.name} share {len(shared)} assays: "
                f"{sorted(shared)}. A frozen-basis result computed over these "
                f"is not held out.")

    # ---- conveniences -----------------------------------------------------
    @property
    def ids(self) -> list[str]:
        return [a.id for a in self.assays]

    def __iter__(self):
        return iter(self.assays)

    def __len__(self) -> int:
        return len(self.assays)

    def __repr__(self) -> str:
        return f"<Cohort {self.name!r} n={len(self.assays)}>"
