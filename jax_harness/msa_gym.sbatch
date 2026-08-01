#!/bin/bash
#SBATCH --job-name=gym_msa
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
# MSA generation for the protein panel, using ProtForge's own MSA container and
# shared MMseqs2 DB, invoked directly rather than through the Snakemake wrapper
# (one chunk, one call -- the wrapper's chunking/scatter machinery buys nothing
# for 8 sequences and would need a full config).
#
# Per the msa_bench DECISIONS.md: MSA is I/O-bound with a fixed ~30 min cost per
# colabfold_search call plus ~9 s/sequence, and co-locating jobs on one node
# costs 1.19x -- hence --exclusive and a single batched call.
set -euo pipefail
W=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files
DB=/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db
SIF=/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge-assets/sifs/msa.sif
FASTA=$W/data/gym/panel/combined.fasta
OUT=$W/data/gym/panel/colabfold_output
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0 NUM_GPU_DEVICES=1
echo "host=$(hostname)  seqs=$(grep -c '^>' $FASTA)"
singularity exec --nv -B "$FASTA":"$FASTA" -B "$DB":"$DB" -B "$OUT":"$OUT" \
    "$SIF" colabfold_search "$FASTA" "$DB" "$OUT" --threads 15 --gpu 1
echo "=== done; a3m files: ==="
ls -la "$OUT" | head -20
