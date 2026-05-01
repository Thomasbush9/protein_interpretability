#!/usr/bin/env python3
"""Build the *Sequence-perturb / MSA-constant* dataset for the MSA sanity check.

For each perturbation level ``p`` and each sample ``i``:

1. Sample ``round(p * L)`` random AA substitutions on the WT sequence.
2. Copy the WT MSA into ``<sample>/msa/A_protein_.a3m``, replacing the query
   row (line 2) with the perturbed sequence — keeps the WT homolog rows.
3. Emit a Boltz-ready YAML (``<sample>/<sample>.yaml``) pointing at the absolute
   path of the swapped MSA.

Output layout
-------------
::

    <out-dir>/
      p05/
        sequences/
          seq_00000/
            seq_00000.yaml
            seq_00000.txt        # mutation sidecar
            msa/A_protein_.a3m
          seq_00001/...
        manifest.tsv             # one row per sample at this level
      p10/...
      ...

Each ``<level>/sequences`` directory is a self-contained drop-in for the
external Boltz prediction pipeline.

Usage
-----
::

    python scripts/build_seq_perturb_dataset.py \\
        --wt-yaml /path/to/original.yaml \\
        --wt-msa  /path/to/original/msa/A_protein_.a3m \\
        --levels  0.05,0.1,0.2,0.3,0.5 \\
        --n-per-level 10 \\
        --out-dir /path/to/seq_perturb \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from pathlib import Path

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
_YAML_ID_RE = re.compile(r'^\s*id:\s*"?([^"\s]+)"?\s*$')
_YAML_SEQ_RE = re.compile(r"^\s*sequence:\s*(\S+)\s*$")


def parse_wt_yaml(path: Path) -> tuple[str, str]:
    """Return ``(chain_id, sequence)`` from a Boltz input YAML.

    Uses a line scanner instead of pulling in PyYAML — the file format here is
    a fixed two-field shape (``id`` then ``sequence``) under a single protein
    block, which doesn't justify a new dependency.
    """
    chain_id: str | None = None
    sequence: str | None = None
    for line in path.read_text().splitlines():
        if chain_id is None:
            m = _YAML_ID_RE.match(line)
            if m:
                chain_id = m.group(1)
                continue
        if sequence is None:
            m = _YAML_SEQ_RE.match(line)
            if m:
                sequence = m.group(1)
        if chain_id is not None and sequence is not None:
            break
    if chain_id is None or sequence is None:
        raise ValueError(f"Could not parse id/sequence from {path}")
    return chain_id, sequence


def sample_perturbed_sequence(
    seq: str, n_edits: int, rng: random.Random
) -> tuple[str, list[str]]:
    """Return ``(mutated_seq, tokens)`` after ``n_edits`` random AA substitutions."""
    L = len(seq)
    if n_edits > L:
        raise ValueError(f"n_edits={n_edits} > L={L}")
    positions = rng.sample(range(L), n_edits)
    out = list(seq)
    tokens: list[str] = []
    for pos in sorted(positions):
        wt_aa = seq[pos]
        if wt_aa not in STANDARD_AA:
            continue
        new_aa = rng.choice([a for a in STANDARD_AA if a != wt_aa])
        out[pos] = new_aa
        tokens.append(f"S{wt_aa}{pos + 1}{new_aa}")
    return "".join(out), tokens


def write_swapped_msa(wt_msa_lines: list[str], query_seq: str, dst: Path) -> None:
    """Write ``dst`` = WT MSA with line 2 (query row) replaced by ``query_seq``.

    Preserves the rest of the file byte-for-byte (homolog rows + their headers).
    Mirrors ``scripts/swap_msa_query.py`` but operates from in-memory lines.
    """
    if len(wt_msa_lines) < 2 or not wt_msa_lines[0].startswith(">"):
        raise ValueError("WT MSA must start with a '>' header followed by the query")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        f.write(wt_msa_lines[0])           # query header (e.g. ">101")
        f.write(query_seq + "\n")          # perturbed query row
        f.writelines(wt_msa_lines[2:])     # all WT homolog rows untouched


def write_yaml(dst: Path, chain_id: str, sequence: str, msa_abs_path: Path) -> None:
    dst.write_text(
        "version: 1\n"
        "\n"
        "sequences:\n"
        "  - protein:\n"
        f'      id: "{chain_id}"\n'
        f"      sequence: {sequence}\n"
        f"      msa: {msa_abs_path}\n"
    )


def level_dirname(p: float) -> str:
    """``0.05 -> 'p05'``, ``0.5 -> 'p50'``, ``1.0 -> 'p100'``."""
    return f"p{int(round(p * 100)):02d}"


def parse_levels(s: str) -> list[float]:
    out: list[float] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        v = float(chunk)
        if not (0 < v <= 1):
            raise argparse.ArgumentTypeError(f"level {v} not in (0, 1]")
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("no levels parsed")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wt-yaml", required=True, type=Path,
                    help="WT Boltz YAML — used to read chain id + sequence.")
    ap.add_argument("--wt-msa", required=True, type=Path,
                    help="WT MSA (.a3m) — copied verbatim into every sample.")
    ap.add_argument("--levels", required=True, type=parse_levels,
                    help="Comma-separated perturbation fractions, e.g. 0.05,0.1,0.2,0.5")
    ap.add_argument("--n-per-level", required=True, type=int,
                    help="Number of perturbed samples per level.")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output root.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--idx-width", type=int, default=5)
    ap.add_argument("--msa-filename", default="A_protein_.a3m",
                    help="Filename for the per-sample MSA (default: A_protein_.a3m).")
    args = ap.parse_args()

    if args.n_per_level <= 0:
        print("[ERROR] --n-per-level must be > 0", file=sys.stderr)
        sys.exit(1)

    chain_id, wt_seq = parse_wt_yaml(args.wt_yaml)
    L = len(wt_seq)
    wt_msa_lines = args.wt_msa.read_text().splitlines(keepends=True)
    if not wt_msa_lines or not wt_msa_lines[0].startswith(">"):
        print(f"[ERROR] {args.wt_msa} is not a valid a3m", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] WT chain={chain_id!r} L={L} levels={args.levels} "
          f"n_per_level={args.n_per_level}")

    rng = random.Random(args.seed)
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for p in args.levels:
        level_dir = out_root / level_dirname(p)
        seqs_dir = level_dir / "sequences"
        seqs_dir.mkdir(parents=True, exist_ok=True)
        n_edits = max(1, round(p * L))

        manifest = level_dir / "manifest.tsv"
        with manifest.open("w") as mf:
            mf.write("idx\tlevel\tn_edits\tlength\tmutations\tsample_dir\n")

            for i in range(args.n_per_level):
                seq_name = f"seq_{i:0{args.idx_width}d}"
                sample_dir = seqs_dir / seq_name
                sample_dir.mkdir(parents=True, exist_ok=True)

                mutated, tokens = sample_perturbed_sequence(wt_seq, n_edits, rng)

                msa_dst = sample_dir / "msa" / args.msa_filename
                write_swapped_msa(wt_msa_lines, mutated, msa_dst)

                yaml_dst = sample_dir / f"{seq_name}.yaml"
                write_yaml(yaml_dst, chain_id, mutated, msa_dst.resolve())

                txt_dst = sample_dir / f"{seq_name}.txt"
                txt_dst.write_text(
                    f"name\t{seq_name}\n"
                    f"level\t{p}\n"
                    f"n_edits\t{len(tokens)}\n"
                    f"length\t{L}\n"
                    f"mutations\t{':'.join(tokens)}\n"
                    f"seed\t{args.seed}\n"
                )

                mf.write(
                    f"{i}\t{p}\t{len(tokens)}\t{L}\t{':'.join(tokens)}\t"
                    f"{sample_dir.resolve()}\n"
                )

        print(f"[DONE] {level_dir} — {args.n_per_level} samples "
              f"({n_edits} edits each)")


if __name__ == "__main__":
    main()
