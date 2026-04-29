"""Swap MSAs in place: replace each mutant a3m with the source MSA, keeping the
mutant's original query row (first header + first sequence).

Usage:
    python swap_msa_query.py \
        --source /path/to/original/A_protein_.a3m \
        --root   /path/to/sequences
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path,
                    help="dir containing seq_*/msa/*.a3m")
    args = ap.parse_args()

    source_lines = args.source.read_text().splitlines(keepends=True)
    if len(source_lines) < 2 or not source_lines[0].startswith(">"):
        raise SystemExit(f"source {args.source} is not a valid a3m")

    targets = sorted(args.root.glob("*/msa/*.a3m"))
    if not targets:
        raise SystemExit(f"no a3m files under {args.root}/*/msa/")

    for t in targets:
        with t.open() as f:
            header = f.readline()
            seq = f.readline()
        if not header.startswith(">") or not seq.strip():
            print(f"skip {t}: missing query header/sequence")
            continue
        with t.open("w") as f:
            f.write(header)
            f.write(seq)
            f.writelines(source_lines[2:])
        print(f"swapped {t}")


if __name__ == "__main__":
    main()
