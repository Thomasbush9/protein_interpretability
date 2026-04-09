#!/usr/bin/env bash
# sample_mutations.sh
#
# Sample a study set of mutations from the Sarkisyan avGFP TSV
# (amino_acid_genotypes_to_brightness.tsv) and copy the matching
# per-sequence FASTA files into a standard directory layout.
#
# Layout produced:
#     $OUT_DIR/
#       single_mut/
#         high_effect/     N rows with brightness <= HIGH_MAX,  exactly 1 mutation
#         neutral/         N rows with brightness >= NEUTRAL_MIN, exactly 1 mutation
#       multi_mut/
#         high_effect/     N rows with brightness <= HIGH_MAX,  1 or more mutations
#         neutral/         N rows with brightness >= NEUTRAL_MIN, 1 or more mutations
#       sampling_summary.json
#
# Sampling:
#   - Reproducible (SEED).
#   - Diversity-aware: for the multi_mut pool, each individual mutation token
#     (e.g. "SA108D") is allowed to appear in at most MAX_PER_MUT picked rows.
#     Singles are already unique so this is a no-op for them.
#
# Usage:
#     FASTA_DIR=/path/to/fastas OUT_DIR=./study_set bash sample_mutations.sh
# Or:
#     bash sample_mutations.sh --fasta-dir /path --out-dir ./study_set
#
# The source FASTA directory is expected to contain files named
# "seq_${idx}.fasta", where ${idx} is the 0-based data-row index in the TSV
# (row 0 is wild-type and is skipped).

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
TSV="/n/home06/tbush/gfp_function_prediction/data/raw_data/amino_acid_genotypes_to_brightness.tsv"
FASTA_DIR="${FASTA_DIR:-}"
OUT_DIR="${OUT_DIR:-}"
N_PER_GROUP="${N_PER_GROUP:-100}"
HIGH_MAX="${HIGH_MAX:-2.8}"       # brightness <= HIGH_MAX   -> high-effect
NEUTRAL_MIN="${NEUTRAL_MIN:-3.7}" # brightness >= NEUTRAL_MIN -> neutral
MAX_PER_MUT="${MAX_PER_MUT:-5}"   # each mutation token in at most N picked multi rows
SEED="${SEED:-42}"

# --- CLI override -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tsv)         TSV="$2"; shift 2 ;;
        --fasta-dir)   FASTA_DIR="$2"; shift 2 ;;
        --out-dir)     OUT_DIR="$2"; shift 2 ;;
        --n)           N_PER_GROUP="$2"; shift 2 ;;
        --high-max)    HIGH_MAX="$2"; shift 2 ;;
        --neutral-min) NEUTRAL_MIN="$2"; shift 2 ;;
        --max-per-mut) MAX_PER_MUT="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,35p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$FASTA_DIR" || -z "$OUT_DIR" ]]; then
    echo "ERROR: FASTA_DIR and OUT_DIR must be set." >&2
    echo "Usage: FASTA_DIR=... OUT_DIR=... bash $0" >&2
    exit 1
fi
if [[ ! -f "$TSV" ]]; then
    echo "ERROR: TSV not found: $TSV" >&2
    exit 1
fi
if [[ ! -d "$FASTA_DIR" ]]; then
    echo "ERROR: FASTA_DIR not found: $FASTA_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
for variant in single_mut multi_mut; do
    for group in high_effect neutral; do
        mkdir -p "$OUT_DIR/$variant/$group"
    done
done

echo "[INFO] TSV         = $TSV"
echo "[INFO] FASTA_DIR   = $FASTA_DIR"
echo "[INFO] OUT_DIR     = $OUT_DIR"
echo "[INFO] N_PER_GROUP = $N_PER_GROUP"
echo "[INFO] HIGH_MAX    = $HIGH_MAX    (brightness <= this = high-effect)"
echo "[INFO] NEUTRAL_MIN = $NEUTRAL_MIN (brightness >= this = neutral)"
echo "[INFO] MAX_PER_MUT = $MAX_PER_MUT (diversity cap for multi_mut)"
echo "[INFO] SEED        = $SEED"

# --- Sampling (Python) ------------------------------------------------------
python3 - \
    "$TSV" "$OUT_DIR" "$N_PER_GROUP" "$HIGH_MAX" "$NEUTRAL_MIN" "$MAX_PER_MUT" "$SEED" \
    <<'PY_EOF'
import csv, json, random, sys
from pathlib import Path

tsv, out_dir, n, high_max, neutral_min, max_per_mut, seed = sys.argv[1:]
n = int(n)
high_max = float(high_max)
neutral_min = float(neutral_min)
max_per_mut = int(max_per_mut)
seed = int(seed)
out_dir = Path(out_dir)

rows = []
with open(tsv) as f:
    r = csv.reader(f, delimiter="\t")
    next(r)  # header
    for i, row in enumerate(r):
        # i = 0 is wild-type (empty mutation string), skipped below.
        if len(row) < 3:
            continue
        muts = row[0]
        if not muts:
            continue
        try:
            b = float(row[2])
        except ValueError:
            continue
        rows.append({"idx": i, "muts": muts, "mut_list": muts.split(":"), "b": b})

def diversify_sample(pool, n, max_per_mut, rng):
    """Greedy diversified sampling.

    Shuffles the pool and accepts a row only if none of its mutation tokens
    has already been used in >= max_per_mut previously-picked rows.
    """
    rng.shuffle(pool)
    mut_count = {}
    picked = []
    for row in pool:
        if any(mut_count.get(m, 0) >= max_per_mut for m in row["mut_list"]):
            continue
        picked.append(row)
        for m in row["mut_list"]:
            mut_count[m] = mut_count.get(m, 0) + 1
        if len(picked) == n:
            break
    return picked

rng = random.Random(seed)

pools = {
    ("single_mut", "high_effect"):
        [r for r in rows if len(r["mut_list"]) == 1 and r["b"] <= high_max],
    ("single_mut", "neutral"):
        [r for r in rows if len(r["mut_list"]) == 1 and r["b"] >= neutral_min],
    ("multi_mut",  "high_effect"):
        [r for r in rows if r["b"] <= high_max],
    ("multi_mut",  "neutral"):
        [r for r in rows if r["b"] >= neutral_min],
}

summary = {}
for (variant, group), pool in pools.items():
    # Each row in single_mut contributes exactly 1 mutation token and the
    # tokens are unique, so the diversity cap is effectively 1.
    cap = 1 if variant == "single_mut" else max_per_mut
    picks = diversify_sample(list(pool), n, cap, rng)
    target = out_dir / variant / group / "selected.tsv"
    with open(target, "w") as f:
        f.write("idx\tmutations\tbrightness\n")
        for p in picks:
            f.write(f"{p['idx']}\t{p['muts']}\t{p['b']}\n")
    summary[f"{variant}/{group}"] = {
        "pool_size": len(pool),
        "picked": len(picks),
        "unique_mutation_tokens": len({m for p in picks for m in p["mut_list"]}),
    }
    if len(picks) < n:
        print(
            f"[WARN] {variant}/{group}: wanted {n}, got {len(picks)} "
            f"(pool={len(pool)}, cap={cap})",
            file=sys.stderr,
        )
    print(
        f"[INFO] {variant}/{group}: pool={len(pool)} picked={len(picks)} "
        f"unique_tokens={summary[f'{variant}/{group}']['unique_mutation_tokens']}"
    )

with open(out_dir / "sampling_summary.json", "w") as f:
    json.dump(
        {
            "tsv": tsv,
            "seed": seed,
            "n_per_group": n,
            "high_max": high_max,
            "neutral_min": neutral_min,
            "max_per_mut_multi": max_per_mut,
            "counts": summary,
        },
        f,
        indent=2,
    )
PY_EOF

# --- Copy FASTA files -------------------------------------------------------
echo "[INFO] Copying FASTA files from $FASTA_DIR"
total_copied=0
total_missing=0
for variant in single_mut multi_mut; do
    for group in high_effect neutral; do
        sel="$OUT_DIR/$variant/$group/selected.tsv"
        dest="$OUT_DIR/$variant/$group"
        [[ -f "$sel" ]] || continue
        while IFS=$'\t' read -r idx muts bright; do
            [[ "$idx" == "idx" ]] && continue  # header
            [[ -z "$idx" ]] && continue
            src="$FASTA_DIR/seq_${idx}.fasta"
            if [[ -f "$src" ]]; then
                cp "$src" "$dest/"
                total_copied=$((total_copied + 1))
            else
                echo "[WARN] missing $src" >&2
                total_missing=$((total_missing + 1))
            fi
        done < "$sel"
    done
done

echo "[DONE] copied=$total_copied missing=$total_missing"
echo "[DONE] summary: $OUT_DIR/sampling_summary.json"
