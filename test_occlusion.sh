#!/usr/bin/env bash
# Interactive smoke test for per-position query occlusion on a single p40 mutant.
# Run from an interactive SLURM session inside the boltz env.

set -euo pipefail

REPO_ROOT="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability"

# Idempotent — does nothing if the boltz env is already active.
source "$REPO_ROOT/scripts/prepare_env.sh"

# Same runtime tuning as run_boltz_attention.slrm.
export CUEQ_DEFAULT_CONFIG="${CUEQ_DEFAULT_CONFIG:-1}"
export CUEQ_DISABLE_AOT_TUNING="${CUEQ_DISABLE_AOT_TUNING:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/triton_cache_${USER}_$$}"
mkdir -p "$TRITON_CACHE_DIR"

# Per-position WT-reversion forward sweep. Pure inference (no autograd).
# `skip_run_structure=True` skips diffusion sampling — distogram is computed
# before the structure module, so we get the contact map at a fraction of the
# cost of a full inference run.
python "$REPO_ROOT/scripts/run_query_occlusion.py" \
    /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/multi_mut/augmented/p40/outputs/sequences/seq_00132/seq_00132.yaml \
    --wt_yaml /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/multi_mut/augmented/msa-swapping/sequences/original/original.yaml \
    --out_dir /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/occlusion \
    --cache /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db \
    --recycling_steps 3 \
    --keep_distograms
