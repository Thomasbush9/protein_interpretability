"""Fold-switching region scoring.

Porter's benchmark scores predictions against the actual fold-switching
region only, not the whole chain. The whole-chain TM is inflated by
scaffold overlap; the region-restricted metric is the apples-to-apples
comparison to AF-vs-fold-switch literature.

Workflow:
    1. ``align_region(region_seq, target_seq)`` finds where the fold-switch
       region sits in a sequence (predicted, ref1, ref2) via Biopython local
       alignment.
    2. ``slice_region(coords, seq, start, end)`` cuts the per-residue arrays
       to that range.
    3. Pass the sliced ref/pred pairs to the standard ``tm_score`` / ``rmsd``
       in :mod:`scoring.utils`.

Failure modes returned by ``align_region``:
- ``None`` if the alignment quality is below ``min_score_per_aa`` or the
  matched span is shorter than ``min_len``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices


@dataclass(frozen=True)
class RegionMatch:
    """Where a region sequence aligned to a target sequence."""

    start: int   # 0-based, inclusive (on the target sequence)
    end: int     # 0-based, exclusive
    score: float
    matched_target: str
    matched_region: str

    @property
    def length(self) -> int:
        return self.end - self.start


_BLOSUM62 = substitution_matrices.load("BLOSUM62")


def _make_aligner() -> PairwiseAligner:
    a = PairwiseAligner()
    a.mode = "local"
    a.substitution_matrix = _BLOSUM62
    a.open_gap_score = -10
    a.extend_gap_score = -0.5
    return a


_ALIGNER = _make_aligner()


def align_region(
    region_seq: str,
    target_seq: str,
    *,
    min_len: int = 10,
    min_score_per_aa: float = 1.0,
) -> RegionMatch | None:
    """Locate ``region_seq`` within ``target_seq`` by local protein alignment.

    Returns the start/end indices on the *target*. Returns ``None`` if the
    best local alignment is shorter than ``min_len`` residues or has an
    average BLOSUM62 score below ``min_score_per_aa``.
    """
    region_seq = region_seq.strip().upper()
    target_seq = target_seq.strip().upper()
    if not region_seq or not target_seq:
        return None

    # BLOSUM62 only knows standard residues; replace anything else with X
    safe = "".join(c if c in _BLOSUM62.alphabet else "X" for c in target_seq)
    safe_region = "".join(c if c in _BLOSUM62.alphabet else "X" for c in region_seq)

    try:
        aln = _ALIGNER.align(safe, safe_region)[0]
    except (ValueError, IndexError):
        return None

    # aln.aligned is a pair of arrays of (start, end) blocks on (target, region)
    target_blocks, _ = aln.aligned
    if len(target_blocks) == 0:
        return None
    start = int(target_blocks[0][0])
    end = int(target_blocks[-1][1])
    span_len = end - start
    if span_len < min_len:
        return None

    score = float(aln.score)
    if score / span_len < min_score_per_aa:
        return None

    return RegionMatch(
        start=start,
        end=end,
        score=score,
        matched_target=target_seq[start:end],
        matched_region=region_seq,
    )


def slice_region(
    coords: np.ndarray, seq: str, start: int, end: int
) -> tuple[np.ndarray, str]:
    """Return the (coords, seq) restricted to ``[start, end)`` residues.

    ``coords`` is the (N, 3) array from ``extract_residue_coordinates``;
    ``seq`` is the matching residue string from ``extract_residue_sequence``.
    """
    if len(coords) != len(seq):
        raise ValueError(
            f"coords/seq length mismatch: {len(coords)} vs {len(seq)}"
        )
    if start < 0 or end > len(seq) or start >= end:
        raise ValueError(
            f"slice [{start}, {end}) out of bounds for length {len(seq)}"
        )
    return coords[start:end], seq[start:end]
