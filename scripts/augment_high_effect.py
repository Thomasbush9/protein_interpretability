#!/usr/bin/env python3
"""Augment high-effect mutations with random extra mutations.

Picks ``--n`` rows from the Sarkisyan avGFP TSV whose brightness is
below ``--high-max`` (high-effect) and, for each one, adds a fixed
percentage of *new* random single-residue mutations at positions that
are NOT already mutated in that row. Wild-type identities at each
position are derived directly from the TSV (every mutation token encodes
its source AA, e.g. ``SA108D`` implies WT A at position 108).

Output
------
``<out_dir>/augmented.tsv`` with columns::

    orig_idx  orig_mutations  orig_brightness  n_extra
    extra_mutations  combined_mutations

If ``--wt-fasta`` is provided, also writes one
``seq_{orig_idx}_aug.fasta`` per picked row containing the mutated
sequence (original row mutations + extra mutations applied to the WT).

Usage
-----
::

    python scripts/augment_high_effect.py \\
        --out-dir ./augmented \\
        --pct 5 \\
        --wt-fasta /path/to/avGFP_wt.fasta
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

MUT_RE = re.compile(r"^([A-Z])([A-Z*])(\d+)([A-Z*])$")
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids (no stops)

DEFAULT_TSV = (
    "/n/home06/tbush/gfp_function_prediction/data/raw_data/"
    "amino_acid_genotypes_to_brightness.tsv"
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_token(tok: str) -> tuple[str, str, int, str]:
    """Parse a mutation token like ``SA108D`` into ``(prefix, wt, pos, to)``."""
    m = MUT_RE.match(tok)
    if m is None:
        raise ValueError(f"Unparseable mutation token: {tok!r}")
    prefix, fr, pos, to = m.groups()
    return prefix, fr, int(pos), to


def derive_wt_from_tsv(tsv: Path) -> dict[int, tuple[str, str]]:
    """Infer ``{position: (prefix, wt_aa)}`` from every token in the TSV.

    Raises if the same position is annotated with two different WT AAs.
    """
    wt: dict[int, tuple[str, str]] = {}
    with open(tsv) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)  # header
        for row in r:
            if not row or not row[0]:
                continue
            for tok in row[0].split(":"):
                prefix, fr, pos, _to = parse_token(tok)
                prev = wt.get(pos)
                if prev is None:
                    wt[pos] = (prefix, fr)
                elif prev != (prefix, fr):
                    raise ValueError(
                        f"Inconsistent WT at position {pos}: {prev} vs {(prefix, fr)}"
                    )
    return wt


def load_rows(tsv: Path) -> list[dict]:
    """Load all non-WT rows with parsed mutation list and brightness."""
    rows = []
    with open(tsv) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for i, row in enumerate(r):
            if len(row) < 3 or not row[0]:
                continue
            try:
                b = float(row[2])
            except ValueError:
                continue
            rows.append({
                "idx": i,
                "muts": row[0],
                "mut_list": row[0].split(":"),
                "b": b,
            })
    return rows


# ---------------------------------------------------------------------------
# Sampling / augmentation
# ---------------------------------------------------------------------------

def sample_extra_mutations(
    mutated_positions: set[int],
    wt_map: dict[int, tuple[str, str]],
    n_extra: int,
    rng: random.Random,
) -> list[str]:
    """Pick ``n_extra`` new tokens at positions NOT in ``mutated_positions``.

    The replacement AA is drawn uniformly from the 19 standard AAs that
    differ from WT at the chosen position (stop codons are never used).
    """
    available = [p for p in wt_map if p not in mutated_positions]
    if n_extra > len(available):
        raise ValueError(
            f"Cannot sample {n_extra} new positions from {len(available)} available."
        )
    chosen = rng.sample(available, n_extra)
    tokens: list[str] = []
    for pos in chosen:
        prefix, wt_aa = wt_map[pos]
        candidates = [a for a in STANDARD_AA if a != wt_aa]
        new_aa = rng.choice(candidates)
        tokens.append(f"{prefix}{wt_aa}{pos}{new_aa}")
    return tokens


def sort_tokens(tokens: list[str]) -> list[str]:
    """Sort mutation tokens by residue position."""
    return sorted(tokens, key=lambda t: parse_token(t)[2])


def diversify_pick(
    pool: list[dict],
    n: int,
    max_per_mut: int,
    rng: random.Random,
) -> list[dict]:
    """Greedy shuffle + reject so each original mutation token appears
    in at most ``max_per_mut`` picked rows."""
    pool = list(pool)
    rng.shuffle(pool)
    mut_count: dict[str, int] = {}
    picked: list[dict] = []
    for row in pool:
        if any(mut_count.get(m, 0) >= max_per_mut for m in row["mut_list"]):
            continue
        picked.append(row)
        for m in row["mut_list"]:
            mut_count[m] = mut_count.get(m, 0) + 1
        if len(picked) == n:
            break
    return picked


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------

def read_fasta(path: Path) -> str:
    """Read a single-record FASTA and return the raw sequence string."""
    header = None
    parts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    raise ValueError("Multi-record FASTA not supported for --wt-fasta")
                header = line
            else:
                parts.append(line.strip())
    if header is None:
        raise ValueError(f"No FASTA header found in {path}")
    return "".join(parts)


def apply_mutations(wt_seq: str, tokens: list[str]) -> str:
    """Apply every mutation token to ``wt_seq`` (1-indexed positions)."""
    seq = list(wt_seq)
    for tok in tokens:
        _pre, fr, pos, to = parse_token(tok)
        i = pos - 1
        if i < 0 or i >= len(seq):
            raise ValueError(
                f"Position {pos} out of range for WT sequence of length {len(seq)}"
            )
        if seq[i] != fr:
            raise ValueError(
                f"WT mismatch at position {pos}: FASTA has {seq[i]!r} "
                f"but token {tok!r} expects {fr!r}"
            )
        seq[i] = to
    return "".join(seq)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tsv", default=DEFAULT_TSV, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=100,
                    help="number of high-effect rows to sample (default 100)")
    ap.add_argument("--high-max", type=float, default=2.8,
                    help="brightness <= this is high-effect (default 2.8)")
    ap.add_argument("--pct", type=float, default=5.0,
                    help="percentage of protein length to add as random extra "
                         "mutations (default 5%%)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-mut", type=int, default=5,
                    help="diversity cap on original mutation tokens (default 5)")
    ap.add_argument("--wt-fasta", type=Path, default=None,
                    help="if given, also write seq_<idx>_aug.fasta for each row")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    wt_map = derive_wt_from_tsv(args.tsv)
    L = len(wt_map)
    print(f"[INFO] WT map: {L} positions "
          f"(min={min(wt_map)}, max={max(wt_map)})")

    n_extra_per_row = max(1, round(args.pct / 100 * L))
    print(f"[INFO] pct={args.pct}% of L={L} -> "
          f"{n_extra_per_row} extra mutations per row")

    rows = load_rows(args.tsv)
    pool = [r for r in rows if r["b"] <= args.high_max]
    print(f"[INFO] high-effect pool (brightness <= {args.high_max}): {len(pool)}")

    picked = diversify_pick(pool, args.n, args.max_per_mut, rng)
    if len(picked) < args.n:
        print(
            f"[WARN] only picked {len(picked)}/{args.n} rows "
            f"after diversity cap (max_per_mut={args.max_per_mut})",
            file=sys.stderr,
        )
    n_unique = len({m for p in picked for m in p["mut_list"]})
    print(f"[INFO] picked {len(picked)} rows spanning "
          f"{n_unique} distinct original tokens")

    wt_seq: str | None = None
    if args.wt_fasta is not None:
        wt_seq = read_fasta(args.wt_fasta)
        print(f"[INFO] loaded WT sequence: length={len(wt_seq)}")

    out_tsv = args.out_dir / "augmented.tsv"
    with open(out_tsv, "w") as f:
        f.write(
            "orig_idx\torig_mutations\torig_brightness\t"
            "n_extra\textra_mutations\tcombined_mutations\n"
        )
        for row in picked:
            mutated_positions = {parse_token(t)[2] for t in row["mut_list"]}
            extra = sample_extra_mutations(
                mutated_positions, wt_map, n_extra_per_row, rng,
            )
            combined = sort_tokens(row["mut_list"] + extra)
            combined_str = ":".join(combined)
            extra_str = ":".join(extra)
            f.write(
                f"{row['idx']}\t{row['muts']}\t{row['b']}\t"
                f"{len(extra)}\t{extra_str}\t{combined_str}\n"
            )

            if wt_seq is not None:
                mutated = apply_mutations(wt_seq, combined)
                out_fa = args.out_dir / f"seq_{row['idx']:05d}_aug.fasta"
                with open(out_fa, "w") as ff:
                    ff.write(
                        f">seq_{row['idx']:05d}_aug | "
                        f"orig={row['muts']} | extra={extra_str}\n"
                    )
                    ff.write(mutated + "\n")

    print(f"[DONE] wrote {out_tsv}")
    if wt_seq is not None:
        print(f"[DONE] wrote {len(picked)} FASTA files to {args.out_dir}")


if __name__ == "__main__":
    main()
