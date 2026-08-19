#!/bin/bash
# Tonight's sweep: three models x the held-out cohort, per-layer trunk capture.
#
# One job per (model, assay) rather than one per model. The reason is recorded
# in launch_gym2s.sh and cost a sweep once: the archive is written only at the
# end, so a job killed at the wall clock produces NOTHING. Sixteen assays in one
# job is sixteen assays lost; one assay per job loses one, and the task's
# resume policy means a relaunch skips what is already on disk.
#
# Runtime comes from MEASURED marginal per-variant rates, not from a scaling
# argument. Measured 2026-08-19 on this cohort at 100 variants:
#
#     protenix   8.3 s/variant at  40 aa    10.1 s/variant at 118 aa
#     boltz2    20.7 s/variant at  40 aa
#     of3       21.7 s/variant at  40 aa
#
# PROTEIN LENGTH IS NOT THE DRIVER in this range, which is worth stating because
# the obvious argument says it should be. The Pairformer's triangle operations
# are O(N^3), so a first pass at these limits assumed the 118-residue assay would
# cost ~26x the 40-residue one; it costs 1.2x. The per-variant time is dominated
# by length-independent work -- the alignment is re-parsed for every variant, and
# the diffusion sampler runs a fixed 200 steps. exp_gym2's docstring says the
# same thing about the MSA: "Boltz re-parses it per variant and that, not the
# GPU, sets the wall-clock."
#
# So a job is ~40 min for boltz2/of3 and ~18 min for protenix, whatever the
# assay. The limits below are ~2.5x that: enough headroom for a slow node,
# little enough that 44 jobs are not asking the scheduler for a day each.
#
#   bash jax_harness/launch_xmodel_layers.sh            # submit everything
#   bash jax_harness/launch_xmodel_layers.sh --dry-run  # print, submit nothing
#   MODELS=protenix bash jax_harness/launch_xmodel_layers.sh
set -euo pipefail

REPO=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability
W=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files
EXP=../experiments/collection/collect_xmodel_layers.py
OUT=$W/runs/xmodel_layers

MODELS=${MODELS:-"boltz2 of3 protenix"}
COHORT=${COHORT:-heldout_assays}
N_VARIANTS=${N_VARIANTS:-100}
DRY=${1:-}

declare -A MINUTES=( [boltz2]=120 [of3]=120 [protenix]=60 )

mapfile -t ASSAYS < <(cd "$REPO" && uv run python -c "
from protein_interpretability.collection import Cohort
print('\n'.join(Cohort.load('$COHORT').ids))")

echo "cohort $COHORT: ${#ASSAYS[@]} assays"
echo "models $MODELS"
echo "output $OUT"
echo

# Resolve and price before submitting anything. This loads no model, verifies
# every input checksum, and refuses the whole sweep if one has moved -- which is
# the failure worth catching before 48 jobs queue behind it.
ONLY=""
[ "$(echo "$MODELS" | wc -w)" = "1" ] && ONLY="--model $MODELS"
# shellcheck disable=SC2086
(cd "$REPO" && uv run python "experiments/collection/collect_xmodel_layers.py" \
    --inspect $ONLY --cohort "$COHORT" --n-variants "$N_VARIANTS" --output "$OUT")
echo

n=0
for m in $MODELS; do
  mins=${MINUTES[$m]:-360}
  for a in "${ASSAYS[@]}"; do
    short=$(echo "$a" | cut -d_ -f1)
    cmd=(sbatch --time="$mins" --job-name="xml_${m}_${short}"
         "$REPO/jax_harness/checkout.sbatch" "$EXP"
         --model "$m" --assay "$a" --cohort "$COHORT"
         --n-variants "$N_VARIANTS" --output "$OUT")
    if [ "$DRY" = "--dry-run" ]; then
      echo "${cmd[*]}"
    else
      "${cmd[@]}"
    fi
    n=$((n + 1))
  done
done
echo
echo "$n job(s) ${DRY:+would be }submitted"
echo "artifacts land in $OUT; relaunching skips whatever this task already wrote"
