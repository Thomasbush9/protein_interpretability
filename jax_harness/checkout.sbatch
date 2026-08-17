#!/bin/bash
#SBATCH --job-name=pi_ckt
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180000
#SBATCH --time=60
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%j.out
#
# analysis.sbatch, but running the GIT CHECKOUT instead of the deployed mirror.
#
# `analysis.sbatch` execs $WORK/harness/, a copy of jax_harness that is not a
# checkout. That copy is what has produced every archived result, and it should
# keep doing so. But the refactor moves the library into src/, which the mirror
# does not carry, and the plan's acceptance test is explicitly about jobs
# "submitted from a clean checkout" -- so verifying the promotion needs a way to
# run the repository itself.
#
# No PYTHONPATH is set on purpose. The pi_* shims locate src/ themselves, so if
# this script had to export it, that would hide a broken shim rather than
# reveal it.
#
#   sbatch checkout.sbatch analyze_svd.py --out ... [args]
set -euo pipefail

SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup
REPO=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability

export MOSAIC_SIF=$SETUP/images/mosaic.sif
export MOSAIC_WEIGHTS=$SETUP/weights
export MOSAIC_SCRATCH=/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic

export SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export SINGULARITYENV_JAX_PLATFORMS=cuda

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} script=$1"
echo "commit=$(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec "$SETUP/mosaic/singularity/mosaic-exec.sh" python "$REPO/jax_harness/$@"
