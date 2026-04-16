#!/usr/bin/env python3
"""Add random mutations to existing FASTA files, skipping already-mutated positions.

Given a directory of mutant ``seq_XXXXX.fasta`` files and the
``selected_*.tsv`` that defines which positions are already mutated in each
sequence, this script adds ``proportion * L`` random amino-acid substitutions
per sequence at positions NOT in the already-mutated set, and writes the
augmented sequences to an output directory with the same filenames.

Usage
-----
::

    python scripts/augment_fasta_random.py \\
        --in-dir /path/to/multi_mut/high_effect \\
        --tsv   /path/to/multi_mut/selected_high_effect.tsv \\
        --proportion 0.2 \\
        --out-dir /path/to/multi_mut/high_effect_aug_20pct \\
        --seed 42

Output layout
-------------
::

    <out-dir>/
      seq_00132.fasta              # sequences with extra random mutations
      seq_00318.fasta
      ...
      augmented.tsv                # idx, orig_mutations, extra_mutations, combined_mutations

The new ``augmented.tsv`` is compatible with ``compute_divergence.py`` if you
pass its ``extra_mutations`` column (so mutation positions for the windowed
metrics reflect just the RANDOMLY added sites).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

MUT_RE = re.compile(r"^([A-Z])([A-Z*])(\d+)([A-Z*])$")
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
SEQ_IDX_RE = re.compile(r"seq_(\d+)")


def parse_token(tok: str) -> tuple[str, str, int, str]:
    m = MUT_RE.match(tok)
    if m is None:
        raise ValueError(f"Unparseable mutation token: {tok!r}")
    prefix, fr, pos, to = m.groups()
    return prefix, fr, int(pos), to


def read_fasta(path: Path) -> tuple[str, str]:
    """Return (header, sequence)."""
    header = None
    parts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                header = line
            else:
                parts.append(line.strip())
    if header is None:
        raise ValueError(f"No header in {path}")
    return header, "".join(parts)


def load_selected_tsv(path: Path) -> dict[int, list[str]]:
    """Return ``{idx: [mutation tokens]}`` from a selected_*.tsv."""
    out: dict[int, list[str]] = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = int(row["idx"])
            tokens = [t.strip() for t in row["mutations"].split(":") if t.strip()]
            out[idx] = tokens
    return out


def extract_seq_idx(path: Path) -> int:
    m = SEQ_IDX_RE.search(path.stem)
    if m is None:
        raise ValueError(f"Could not extract sequence index from {path.name}")
    return int(m.group(1))


def sample_random_mutations(
    seq: str,
    forbidden_positions: set[int],  # 1-indexed
    n_extra: int,
    rng: random.Random,
) -> list[str]:
    """Pick ``n_extra`` positions not in ``forbidden_positions`` and randomise them.

    Positions are 1-indexed (to match the token format). Stop codons and
    the identity AA are excluded from the replacement alphabet.
    """
    L = len(seq)
    available = [p for p in range(1, L + 1) if p not in forbidden_positions]
    if n_extra > len(available):
        raise ValueError(
            f"Cannot sample {n_extra} positions from {len(available)} available "
            f"(forbidden={len(forbidden_positions)}, L={L})."
        )
    chosen = rng.sample(available, n_extra)
    tokens: list[str] = []
    for pos in chosen:
        wt_aa = seq[pos - 1]
        if wt_aa not in STANDARD_AA:
            # Skip non-standard residues (shouldn't happen for GFP, but be safe)
            continue
        candidates = [a for a in STANDARD_AA if a != wt_aa]
        new_aa = rng.choice(candidates)
        # Prefix "S" to match the existing token convention
        tokens.append(f"S{wt_aa}{pos}{new_aa}")
    return sorted(tokens, key=lambda t: parse_token(t)[2])


def apply_tokens(seq: str, tokens: list[str]) -> str:
    out = list(seq)
    for tok in tokens:
        _pre, fr, pos, to = parse_token(tok)
        i = pos - 1
        if not (0 <= i < len(out)):
            raise ValueError(f"Position {pos} out of range (L={len(out)})")
        if out[i] != fr:
            raise ValueError(
                f"Sequence mismatch at pos {pos}: FASTA has {out[i]!r} "
                f"but token {tok!r} expects {fr!r}"
            )
        out[i] = to
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in-dir", required=True, type=Path,
                    help="Directory containing seq_XXXXX.fasta files.")
    ap.add_argument("--tsv", required=True, type=Path,
                    help="selected_*.tsv with columns idx, mutations, brightness.")
    ap.add_argument("--proportion", required=True, type=float,
                    help="Fraction of protein length to add as random mutations (e.g. 0.2 = 20%%).")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory for augmented FASTA files.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (0 < args.proportion <= 1):
        print("[ERROR] --proportion must be in (0, 1]", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Load mutation map from TSV
    selected = load_selected_tsv(args.tsv)
    print(f"[INFO] loaded {len(selected)} rows from {args.tsv}")

    # Discover input FASTAs
    fasta_paths = sorted(args.in_dir.glob("seq_*.fasta"))
    print(f"[INFO] found {len(fasta_paths)} FASTA files in {args.in_dir}")
    if not fasta_paths:
        print("[ERROR] no seq_*.fasta found", file=sys.stderr)
        sys.exit(1)

    out_tsv = args.out_dir / "augmented.tsv"
    with open(out_tsv, "w") as tf:
        tf.write(
            "idx\torig_mutations\textra_mutations\tcombined_mutations\t"
            "n_extra\tn_orig\tlength\n"
        )

        n_written = 0
        for fa_path in fasta_paths:
            idx = extract_seq_idx(fa_path)
            header, seq = read_fasta(fa_path)
            L = len(seq)

            orig_tokens = selected.get(idx, [])
            if not orig_tokens:
                print(f"[WARN] no TSV entry for idx={idx} ({fa_path.name}), skipping",
                      file=sys.stderr)
                continue
            forbidden_positions = {parse_token(t)[2] for t in orig_tokens}

            n_extra = max(1, round(args.proportion * L))
            extra = sample_random_mutations(seq, forbidden_positions, n_extra, rng)

            mutated_seq = apply_tokens(seq, extra)
            combined = sorted(orig_tokens + extra, key=lambda t: parse_token(t)[2])

            # Write augmented FASTA (preserve original filename + short header).
            # Mutation summary goes to a sidecar .txt — long headers break
            # downstream pipelines that derive filenames from the header.
            out_fa = args.out_dir / fa_path.name
            seq_name = f"seq_{idx:05d}"
            fasta_header = f"{idx}|protein|"
            with open(out_fa, "w") as ff:
                ff.write(f">{fasta_header}\n")
                ff.write(mutated_seq + "\n")

            out_txt = args.out_dir / f"{seq_name}.txt"
            with open(out_txt, "w") as tx:
                tx.write(f"name\t{seq_name}\n")
                tx.write(f"orig\t{':'.join(orig_tokens)}\n")
                tx.write(f"extra\t{':'.join(extra)}\n")
                tx.write(f"combined\t{':'.join(combined)}\n")
                tx.write(f"n_orig\t{len(orig_tokens)}\n")
                tx.write(f"n_extra\t{len(extra)}\n")
                tx.write(f"length\t{L}\n")

            tf.write(
                f"{idx}\t{':'.join(orig_tokens)}\t{':'.join(extra)}\t"
                f"{':'.join(combined)}\t{len(extra)}\t{len(orig_tokens)}\t{L}\n"
            )
            n_written += 1

    print(f"[DONE] wrote {n_written} augmented FASTAs + {out_tsv}")


if __name__ == "__main__":
    main()
