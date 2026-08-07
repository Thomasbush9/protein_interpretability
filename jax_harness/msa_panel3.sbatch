#!/bin/bash
#SBATCH --job-name=msa_panel3
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
# Alignments for the fold-switch targets.
#
# One search per WILD TYPE only. Every variant in the conformational experiment
# is then grafted onto its wild type's alignment, exactly as `exp_gym.graft_a3m`
# does for ProteinGym: the alignment is held identical across the variants of a
# system so that the only thing differing between two forward passes is the
# sequence itself. Searching each variant separately would let alignment
# composition move with the mutation and there would be no way to tell which one
# moved the internal state.
set -euo pipefail
W=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files
DB=/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db
SIF=/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge-assets/sifs/msa.sif
FASTA=$W/data/gym/panel3/targets.fasta
OUT=$W/data/gym/panel3/colabfold_output
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0 NUM_GPU_DEVICES=1
echo "host=$(hostname)  seqs=$(grep -c '^>' $FASTA)"
singularity exec --nv -B "$FASTA":"$FASTA" -B "$DB":"$DB" -B "$OUT":"$OUT" \
    "$SIF" colabfold_search "$FASTA" "$DB" "$OUT" --threads 15 --gpu 1
echo "=== done ==="
ls -la "$OUT"
for f in "$OUT"/*.a3m; do echo "$(basename $f): $(grep -c '^>' $f) sequences"; done
