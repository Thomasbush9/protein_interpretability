#!/bin/bash
#SBATCH --job-name=pi
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000
#SBATCH --time=120
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%j.out
#
# Run any harness script inside the mosaic container on a GPU node.
#   sbatch run.sbatch smoke_test.py [args...]
set -euo pipefail

SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup
WORK=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files

export MOSAIC_SIF=$SETUP/images/mosaic.sif
export MOSAIC_WEIGHTS=$SETUP/weights
export MOSAIC_SCRATCH=/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic

# Keep JAX from grabbing the whole card up front; the trunk captures are spiky.
export SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export SINGULARITYENV_JAX_PLATFORMS=cuda

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} script=$1"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec "$SETUP/mosaic/singularity/mosaic-exec.sh" python "$WORK/harness/$@"
