#!/bin/bash
#SBATCH --job-name=pi_arr
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180000
#SBATCH --time=120
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%A_%a.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%A_%a.out
#
# checkout.sbatch as a JOB ARRAY, so a sweep is one queue entry with a
# concurrency cap instead of N independent submissions.
#
# WHY. Submitting 48 separate jobs put 48 entries in the queue at once. The
# scheduler ran 16 and held the rest, which works, but it floods the queue for
# everyone sharing the allocation and there is no way to say "at most ten of
# mine at a time". `--array=1-48%10` says exactly that, in one entry, and the
# cap is enforced by SLURM rather than by remembering to submit in waves.
#
# Each array index runs ONE line of a manifest: a tab- or space-separated list
# of argument strings, one job per line, passed verbatim to the script. Blank
# lines and #-comments are skipped when the manifest is built, not here, so the
# index-to-line mapping stays stable if the file is regenerated.
#
#   sbatch --array=1-48%10 checkout_array.sbatch jobs.tsv
#
# The per-job wall clock comes from --time on the submit line, so a sweep of
# mixed models can still ask for what its slowest member needs.
set -euo pipefail

SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup
REPO=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability

MANIFEST=${1:?usage: sbatch --array=1-N%10 checkout_array.sbatch MANIFEST}
IDX=${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array}

LINE=$(sed -n "${IDX}p" "$MANIFEST")
if [ -z "$LINE" ]; then
    echo "manifest $MANIFEST has no line $IDX; refusing to run an empty job" >&2
    exit 2
fi

export MOSAIC_SIF=$SETUP/images/mosaic.sif
export MOSAIC_WEIGHTS=$SETUP/weights
export MOSAIC_SCRATCH=/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic

export SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export SINGULARITYENV_JAX_PLATFORMS=cuda

echo "host=$(hostname) array=${SLURM_ARRAY_JOB_ID}_${IDX} manifest=$MANIFEST"
echo "line=$LINE"
echo "commit=$(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Deliberately unquoted: the manifest line IS the argument list.
# shellcheck disable=SC2086
exec "$SETUP/mosaic/singularity/mosaic-exec.sh" python "$REPO/jax_harness/"$LINE
