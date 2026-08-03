#!/usr/bin/env bash
# Interactive smoke test for gradient attribution on a single p40 mutant.
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

# Mitigate H100 fragmentation under the heavier backward graph.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Pair-rep gradient attribution: backward only flows through distogram_module
# (one Linear), with all model parameters frozen. The trunk has no autograd
# graph at all, so memory is bounded regardless of K. Fused triangle attention
# kernels (trifast / cuequivariance_torch) are used — they have working
# backward (Boltz trains with them) and never materialise the softmax matrix.
python "$REPO_ROOT/scripts/run_attribution_single.py" \
    /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/multi_mut/augmented/p40/outputs/sequences/seq_00132/seq_00132.yaml \
    --out_dir /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/grad_att \
    --cache /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db \
    --target contact:65,202 \
    --target mean_contact \
    --recycling_steps 0,5,10
