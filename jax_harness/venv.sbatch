#!/bin/bash
#SBATCH --job-name=pi_venv
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180000
#SBATCH --time=120
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/logs/%x_%j.out
#
# The third submitter: the git checkout in the PROJECT VENV, not the container.
#
# `checkout.sbatch` and `analysis.sbatch` both exec inside mosaic.sif, which is
# what anything touching a model needs. But the container is not a superset of
# the analysis environment -- `tmtools` (TM-align) is installed in .venv and is
# NOT in the sif, so a pure-analysis script that computes a TM score runs fine
# under `uv run` and dies inside the container with ModuleNotFoundError. That is
# how this file came to exist, half an hour into a job that had already done the
# expensive part.
#
# Use this for analysis that needs the venv and no model. Use checkout.sbatch
# for anything that loads one. If a script needs both, it is two scripts.
#
#   sbatch venv.sbatch ../experiments/analysis/ensemble_baseline.py --out ...
#   sbatch venv.sbatch analyze_svd.py --out ...
#
# Paths are resolved relative to jax_harness/, matching checkout.sbatch, so the
# `../experiments/...` form is the same in both.
set -euo pipefail

REPO=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} script=$1"
echo "commit=$(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
exec uv run --no-sync python "$REPO/jax_harness/$@"
