#!/bin/bash
#SBATCH --job-name=pi_ana
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180000
#SBATCH --time=60
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%j.out
#
# Analysis jobs that never load model weights but are still real compute.
#
# There is no CPU partition available on this account, so these run on a GPU
# node like everything else. They are written to actually use the device: the
# SVD study is ~7700 batched decompositions plus millions of small ridge
# solves over (layer x lambda x component-count x permutation), and the drift
# calibration softmaxes ~570M distogram logits. Both are batched `jnp`, so the
# card is doing the work rather than idling next to a numpy loop.
#
# Memory is set high because the archives are ~114 MB each and twelve of them
# are held open at once, twice over where a repeat run is compared.
#
#   sbatch analysis.sbatch analyze_svd.py --out ... [args]
set -euo pipefail

SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup
WORK=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files

export MOSAIC_SIF=$SETUP/images/mosaic.sif
export MOSAIC_WEIGHTS=$SETUP/weights
export MOSAIC_SCRATCH=/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic

export SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export SINGULARITYENV_JAX_PLATFORMS=cuda

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} script=$1"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec "$SETUP/mosaic/singularity/mosaic-exec.sh" python "$WORK/harness/$@"
