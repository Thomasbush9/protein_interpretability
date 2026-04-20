#!/usr/bin/env python3
"""Generate N random-mutant copies of a single FASTA.

Given one input FASTA (e.g. a wild-type sequence), produce ``--n`` perturbed
sequences, each with ``proportion * L`` random amino-acid substitutions drawn
uniformly across the full length (no forbidden positions, since there is no
prior mutation annotation).

Output layout
-------------
::

    <out-dir>/
      seq_00000.fasta              # mutated sequence 0
      seq_00000.txt                # mutation sidecar
      seq_00001.fasta
      seq_00001.txt
      ...
      augmented.tsv                # idx, extra_mutations, length

Compatible with ``compute_divergence.py`` via the ``extra_mutations`` column.

Usage
-----
::

    python scripts/augment_single_fasta_random.py \\
        --in-fasta /path/to/wt.fasta \\
        --n 50 \\
        --proportion 0.2 \\
        --out-dir /path/to/wt_aug_p20 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

MUT_RE = re.compile(r"^([A-Z])([A-Z*])(\d+)([A-Z*])$")
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def parse_token(tok: str) -> tuple[str, str, int, str]:
    m = MUT_RE.match(tok)
    if m is None:
        raise ValueError(f"Unparseable mutation token: {tok!r}")
    prefix, fr, pos, to = m.groups()
    return prefix, fr, int(pos), to


def read_fasta(path: Path) -> tuple[str, str]:
    header = None
    parts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    raise ValueError("Multi-record FASTA not supported")
                header = line
            else:
                parts.append(line.strip())
    if header is None:
        raise ValueError(f"No header in {path}")
    return header, "".join(parts)


def sample_random_mutations(
    seq: str,
    n_extra: int,
    rng: random.Random,
) -> list[str]:
    L = len(seq)
    available = list(range(1, L + 1))
    if n_extra > len(available):
        raise ValueError(f"Cannot sample {n_extra} positions from L={L}")
    chosen = rng.sample(available, n_extra)
    tokens: list[str] = []
    for pos in chosen:
        wt_aa = seq[pos - 1]
        if wt_aa not in STANDARD_AA:
            continue
        candidates = [a for a in STANDARD_AA if a != wt_aa]
        new_aa = rng.choice(candidates)
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
    ap.add_argument("--in-fasta", required=True, type=Path,
                    help="Single input FASTA to perturb.")
    ap.add_argument("--n", required=True, type=int,
                    help="Number of perturbed copies to generate.")
    ap.add_argument("--proportion", required=True, type=float,
                    help="Fraction of protein length to mutate per copy (e.g. 0.2).")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory for seq_XXXXX.fasta + seq_XXXXX.txt.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--idx-width", type=int, default=5,
                    help="Zero-pad width for seq_XXXXX filenames (default 5).")
    args = ap.parse_args()

    if not (0 < args.proportion <= 1):
        print("[ERROR] --proportion must be in (0, 1]", file=sys.stderr)
        sys.exit(1)
    if args.n <= 0:
        print("[ERROR] --n must be > 0", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    _header, seq = read_fasta(args.in_fasta)
    L = len(seq)
    n_extra = max(1, round(args.proportion * L))
    print(f"[INFO] input L={L}, proportion={args.proportion} -> "
          f"{n_extra} mutations per copy, n_copies={args.n}")

    out_tsv = args.out_dir / "augmented.tsv"
    with open(out_tsv, "w") as tf:
        tf.write("idx\textra_mutations\tn_extra\tlength\n")

        for idx in range(args.n):
            extra = sample_random_mutations(seq, n_extra, rng)
            mutated_seq = apply_tokens(seq, extra)

            seq_name = f"seq_{idx:0{args.idx_width}d}"
            out_fa = args.out_dir / f"{seq_name}.fasta"
            with open(out_fa, "w") as ff:
                ff.write(f">{idx}|protein|\n")
                ff.write(mutated_seq + "\n")

            out_txt = args.out_dir / f"{seq_name}.txt"
            with open(out_txt, "w") as tx:
                tx.write(f"name\t{seq_name}\n")
                tx.write(f"source\t{args.in_fasta.name}\n")
                tx.write(f"extra\t{':'.join(extra)}\n")
                tx.write(f"n_extra\t{len(extra)}\n")
                tx.write(f"length\t{L}\n")

            tf.write(f"{idx}\t{':'.join(extra)}\t{len(extra)}\t{L}\n")

    print(f"[DONE] wrote {args.n} FASTAs to {args.out_dir} + {out_tsv}")


if __name__ == "__main__":
    main()
