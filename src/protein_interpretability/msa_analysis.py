"""MSA analysis helpers comparing perturbed MSAs against a wild-type MSA.

Metrics computed per perturbed MSA:

* ``depth`` — number of aligned sequences (including the query row)
* ``length`` — number of reference columns
* ``neff_curve`` — Neff at multiple identity thresholds (50/62/80/90/95 %)
  plus per-hit identity to the query (``pid_to_query``)
* per-column amino-acid distribution and a Pearson correlation against the WT
  MSA's column distributions (``column_corr`` array + ``corr_mean`` scalar)
* set-overlap of homolog IDs vs the WT MSA: ``frac_pert_in_wt``,
  ``frac_wt_in_pert``, ``jaccard`` (+ raw counts), answering "how much of the
  perturbed MSA is just the WT MSA, and vice versa?"
* per-position conservation against the (mutant) query, useful for plots

The library is consumed by ``scripts/analyze_msa_perturbation.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


GAP_CHAR = "-"
ALPHABET = list("ACDEFGHIKLMNPQRSTVWY-")
ALPHABET_CODES = np.array([ord(c) for c in ALPHABET], dtype=np.uint8)
_PERTURB_RE = re.compile(r"(?:^|/)(p\d+)(?:/|$)")
_SEQDIR_RE = re.compile(r"^seq_0*(\d+)$")
_FILENAME_IDX_RE = re.compile(r"^0*(\d+)")


def parse_a3m_with_headers(path: str | Path) -> tuple[list[str], list[str]]:
    """Parse an A3M into ``(headers, sequences)``. ``headers[i]`` is the first
    whitespace-delimited token after ``>`` (typically a UniRef accession)."""
    headers: list[str] = []
    sequences: list[str] = []
    current: list[str] = []
    cur_header: str | None = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_header is not None:
                    headers.append(cur_header)
                    sequences.append("".join(current))
                cur_header = line[1:].split()[0] if len(line) > 1 else ""
                current = []
            else:
                current.append(line)
    if cur_header is not None:
        headers.append(cur_header)
        sequences.append("".join(current))
    return headers, sequences


def msa_to_array(
    msa_path: str | Path,
) -> tuple[np.ndarray, str, list[str]]:
    """Load an A3M as an aligned ``(n_seqs, L)`` ``uint8`` ascii array.

    The first sequence defines the reference columns (uppercase, non-gap).
    Lowercase insertions in other rows are dropped. Rows whose aligned-column
    count diverges from the reference are skipped (with their headers).

    Returns
    -------
    arr : (n_seqs, L) uint8 — row 0 is the query
    ref : query sequence (uppercase, gaps removed)
    kept_headers : header tokens aligned with ``arr`` rows
    """
    headers, sequences = parse_a3m_with_headers(msa_path)
    if not sequences:
        raise ValueError(f"Empty MSA: {msa_path}")

    ref_raw = sequences[0]
    ref_col_set = {i for i, c in enumerate(ref_raw) if c.isupper() and c != GAP_CHAR}
    ref_cols = sorted(ref_col_set)
    L = len(ref_cols)
    ref = "".join(ref_raw[i] for i in ref_cols)

    rows: list[list[int]] = [[ord(c) for c in ref]]
    kept_headers: list[str] = [headers[0]]
    for header, seq in zip(headers[1:], sequences[1:]):
        cols: list[int] = []
        col_idx = 0
        for c in seq:
            if c == GAP_CHAR or c.isupper():
                if col_idx in ref_col_set:
                    cols.append(ord(c.upper()))
                col_idx += 1
            # lowercase = insertion vs ref → skip
        if len(cols) == L:
            rows.append(cols)
            kept_headers.append(header)

    arr = np.asarray(rows, dtype=np.uint8)
    return arr, ref, kept_headers


def column_aa_distributions(arr: np.ndarray) -> np.ndarray:
    """Per-column amino-acid frequency over the standard 20 + gap alphabet.

    Counts use **all** rows (query + hits). Symbols outside ``ALPHABET`` (e.g.
    ``X``, ``B``, ``Z``) are ignored when normalising — columns with no
    in-alphabet observations get a uniform distribution to keep the math
    well-defined.
    """
    L = arr.shape[1]
    counts = np.zeros((L, len(ALPHABET)), dtype=np.float32)
    for k, code in enumerate(ALPHABET_CODES):
        counts[:, k] = (arr == code).sum(axis=0)
    total = counts.sum(axis=1, keepdims=True)
    safe_total = np.where(total > 0, total, 1.0)
    dist = counts / safe_total
    # Columns with zero in-alphabet counts → uniform (avoids NaN in correlations)
    empty = total.squeeze(-1) == 0
    if empty.any():
        dist[empty] = 1.0 / len(ALPHABET)
    return dist


def column_correlations(dist_a: np.ndarray, dist_b: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlation between two (L, 21) distribution arrays.

    Columns where one side has zero variance (e.g. fully conserved) yield 0.0.
    """
    if dist_a.shape != dist_b.shape:
        raise ValueError(
            f"Shape mismatch for column correlation: {dist_a.shape} vs {dist_b.shape}"
        )
    a = dist_a - dist_a.mean(axis=1, keepdims=True)
    b = dist_b - dist_b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    denom = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    out = np.zeros(dist_a.shape[0], dtype=np.float32)
    mask = denom > 0
    out[mask] = num[mask] / denom[mask]
    return out


def homolog_set_overlap(
    pert_headers: list[str],
    wt_id_set: set[str],
    wt_n_hits: int,
) -> dict[str, float]:
    """Set-overlap metrics between the perturbed MSA's hits and the WT hits.

    Returns
    -------
    dict with keys:
        frac_pert_in_wt — fraction of perturbed hits whose ID is also a WT hit
        frac_wt_in_pert — fraction of WT hits that are also in the perturbed MSA
        jaccard         — |∩| / |∪|
        n_shared        — count of shared hit IDs
        n_pert_unique   — perturbed hits not present in WT (potential novelty)
        n_wt_unique     — WT hits absent from the perturbed MSA
    """
    pert_hits = pert_headers[1:] if len(pert_headers) > 1 else []
    pert_set = set(pert_hits)
    shared = pert_set & wt_id_set
    union = pert_set | wt_id_set
    n_shared = len(shared)
    return {
        "frac_pert_in_wt": (n_shared / len(pert_set)) if pert_set else float("nan"),
        "frac_wt_in_pert": (n_shared / wt_n_hits) if wt_n_hits else float("nan"),
        "jaccard": (n_shared / len(union)) if union else float("nan"),
        "n_shared": float(n_shared),
        "n_pert_unique": float(len(pert_set - wt_id_set)),
        "n_wt_unique": float(len(wt_id_set - pert_set)),
    }


DEFAULT_NEFF_THRESHOLDS: tuple[float, ...] = (0.5, 0.62, 0.8, 0.9, 0.95)


def neff_at_thresholds(
    arr: np.ndarray,
    thresholds: tuple[float, ...] | list[float] = DEFAULT_NEFF_THRESHOLDS,
    *,
    max_seqs: int | None = 5000,
    chunk: int = 128,
    seed: int = 0,
) -> dict[float, float]:
    """Effective number of sequences at multiple redundancy ``thresholds``.

    Each sequence gets weight ``1 / |{j : id(i, j) >= t}|`` and Neff(t) is the
    sum of weights (HHsuite/AlphaFold convention). The pairwise identity matrix
    is built in row chunks to bound memory; for very deep MSAs we subsample to
    ``max_seqs`` rows (query always kept) and rescale by ``n / max_seqs``.
    """
    n = arr.shape[0]
    thresholds = tuple(float(t) for t in thresholds)
    if n == 0:
        return {t: 0.0 for t in thresholds}
    if n == 1:
        return {t: 1.0 for t in thresholds}

    sample = arr
    scale = 1.0
    if max_seqs is not None and n > max_seqs:
        rng = np.random.default_rng(seed)
        keep = np.concatenate(
            ([0], rng.choice(np.arange(1, n), size=max_seqs - 1, replace=False))
        )
        sample = arr[keep]
        scale = n / max_seqs

    n_s, L = sample.shape
    thr = np.asarray(thresholds, dtype=np.float32)
    cluster_sizes = np.zeros((n_s, thr.size), dtype=np.int32)
    for i in range(0, n_s, chunk):
        block = sample[i : i + chunk]
        eq = block[:, None, :] == sample[None, :, :]
        sim = eq.sum(axis=-1, dtype=np.int32).astype(np.float32) / L
        for t_idx, t_val in enumerate(thr):
            cluster_sizes[i : i + chunk, t_idx] = (sim >= t_val).sum(axis=1)

    weights = 1.0 / np.maximum(cluster_sizes, 1)
    neffs = weights.sum(axis=0) * scale
    return {float(t): float(v) for t, v in zip(thresholds, neffs)}


def pairwise_identity_to_query(arr: np.ndarray) -> np.ndarray:
    """Per-sequence fraction of columns matching the query (row 0).

    Returns a ``(n_seqs - 1,)`` array — the query itself is excluded.
    """
    if arr.shape[0] < 2:
        return np.empty(0, dtype=np.float32)
    return (arr[1:] == arr[0]).mean(axis=1, dtype=np.float64).astype(np.float32)


def conservation_per_position(arr: np.ndarray) -> np.ndarray:
    """Per-position fraction of non-query rows matching the query residue."""
    if arr.shape[0] < 2:
        return np.ones(arr.shape[1], dtype=np.float32)
    return (arr[1:] == arr[0]).mean(axis=0).astype(np.float32)


@dataclass
class WTReference:
    """Pre-computed wild-type MSA summary used for cross-MSA comparison."""

    path: Path
    length: int
    depth: int
    neff_curve: dict[float, float]
    pid_to_query: np.ndarray = field(repr=False)
    column_dist: np.ndarray = field(repr=False)
    hit_id_set: set[str] = field(repr=False)
    conservation: np.ndarray = field(repr=False)


def load_wt_reference(
    path: str | Path,
    *,
    neff_thresholds: tuple[float, ...] | list[float] = DEFAULT_NEFF_THRESHOLDS,
    neff_max_seqs: int | None = 5000,
) -> WTReference:
    arr, _ref, headers = msa_to_array(path)
    return WTReference(
        path=Path(path),
        length=int(arr.shape[1]),
        depth=int(arr.shape[0]),
        neff_curve=neff_at_thresholds(arr, neff_thresholds, max_seqs=neff_max_seqs),
        pid_to_query=pairwise_identity_to_query(arr),
        column_dist=column_aa_distributions(arr),
        hit_id_set=set(headers[1:]),
        conservation=conservation_per_position(arr),
    )


@dataclass
class MSARecord:
    """Per-MSA summary for downstream aggregation and plotting."""

    path: Path
    level: str
    seq_id: str
    sequence_idx: int | None
    depth: int
    length: int
    neff_curve: dict[float, float]
    corr_mean: float
    corr_median: float
    frac_pert_in_wt: float
    frac_wt_in_pert: float
    jaccard: float
    n_shared: float
    n_pert_unique: float
    n_wt_unique: float
    pid_to_query: np.ndarray = field(repr=False)
    column_corr: np.ndarray = field(repr=False)
    conservation: np.ndarray = field(repr=False)
    length_match: bool


def perturbation_level_from_path(path: Path) -> str:
    """Detect ``p<NN>`` in the path. Returns ``"original"`` if absent."""
    m = _PERTURB_RE.search(str(path))
    return m.group(1) if m else "original"


def seq_id_from_path(path: Path) -> str:
    """Best-effort sequence identifier — uses ``seq_XXXXX`` parent if present."""
    for part in reversed(path.parts):
        if part.startswith("seq_"):
            return part
    return path.stem


def sequence_idx_from_path(path: Path) -> int | None:
    """Integer sequence index parsed from the path.

    Tries, in order:

    * a ``seq_<digits>`` directory anywhere along the path
      (``seq_00318`` → ``318``)
    * a leading-digits filename (``318_protein_.a3m`` → ``318``)

    Returns ``None`` if neither pattern matches.
    """
    for part in reversed(path.parts):
        m = _SEQDIR_RE.match(part)
        if m:
            return int(m.group(1))
    m = _FILENAME_IDX_RE.match(path.stem)
    if m:
        return int(m.group(1))
    return None


def analyze_msa(
    msa_path: str | Path,
    wt: WTReference,
    *,
    level: str | None = None,
    neff_thresholds: tuple[float, ...] | list[float] = DEFAULT_NEFF_THRESHOLDS,
    neff_max_seqs: int | None = 5000,
) -> MSARecord:
    """Compute per-MSA metrics and comparisons against ``wt``."""
    msa_path = Path(msa_path)
    arr, _ref, headers = msa_to_array(msa_path)

    depth = int(arr.shape[0])
    length = int(arr.shape[1])
    cons = conservation_per_position(arr)
    neff_curve = neff_at_thresholds(arr, neff_thresholds, max_seqs=neff_max_seqs)
    pid = pairwise_identity_to_query(arr)
    overlap = homolog_set_overlap(headers, wt.hit_id_set, len(wt.hit_id_set))

    length_match = length == wt.length
    if length_match:
        dist = column_aa_distributions(arr)
        col_corr = column_correlations(dist, wt.column_dist)
        corr_mean = float(col_corr.mean()) if col_corr.size else float("nan")
        corr_median = float(np.median(col_corr)) if col_corr.size else float("nan")
    else:
        col_corr = np.full(length, np.nan, dtype=np.float32)
        corr_mean = float("nan")
        corr_median = float("nan")

    return MSARecord(
        path=msa_path,
        level=level if level is not None else perturbation_level_from_path(msa_path),
        seq_id=seq_id_from_path(msa_path),
        sequence_idx=sequence_idx_from_path(msa_path),
        depth=depth,
        length=length,
        neff_curve=neff_curve,
        corr_mean=corr_mean,
        corr_median=corr_median,
        frac_pert_in_wt=overlap["frac_pert_in_wt"],
        frac_wt_in_pert=overlap["frac_wt_in_pert"],
        jaccard=overlap["jaccard"],
        n_shared=overlap["n_shared"],
        n_pert_unique=overlap["n_pert_unique"],
        n_wt_unique=overlap["n_wt_unique"],
        pid_to_query=pid,
        column_corr=col_corr,
        conservation=cons,
        length_match=length_match,
    )


def discover_msas(
    input_dir: str | Path,
    *,
    pattern: str = "**/*.a3m",
) -> list[Path]:
    """Recursively glob ``.a3m`` files under ``input_dir`` (sorted, deduped)."""
    input_dir = Path(input_dir)
    return sorted({p.resolve() for p in input_dir.glob(pattern) if p.is_file()})
