#!/bin/bash
#SBATCH --job-name=msa_real
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=240000
#SBATCH --time=240
#SBATCH --exclusive
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%j.out
#
# Genuinely re-searched MSAs, one per MUTANT sequence.
#
# Every experiment so far grafts the wild type's alignment onto each variant.
# That is a deliberate control -- it fixes the alignment as an exactly known
# variable -- but it also guarantees the alignment cannot carry mutation-specific
# information, which makes "the mutation does not enter through the MSA routes"
# partly circular. This produces the alignments a user would actually get, so
# the route decomposition can be rerun without that circularity.
#
# One batched colabfold_search call: cost is a fixed ~30 min per invocation plus
# ~9 s/sequence, so 39 sequences in one call is ~36 min and 39 calls would be
# ~20 hours.
set -euo pipefail
W=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files
DB=/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db
SIF=/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge-assets/sifs/msa.sif
FASTA=$W/data/msa_real/mutants.fasta
OUT=$W/data/msa_real/a3m
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0 NUM_GPU_DEVICES=1
echo "host=$(hostname)  seqs=$(grep -c '^>' $FASTA)"
singularity exec --nv -B "$FASTA":"$FASTA" -B "$DB":"$DB" -B "$OUT":"$OUT" \
    "$SIF" colabfold_search "$FASTA" "$DB" "$OUT" --threads 15 --gpu 1
echo "=== done ==="
ls "$OUT" | head -20
echo "a3m count: $(ls $OUT/*.a3m 2>/dev/null | wc -l)"
