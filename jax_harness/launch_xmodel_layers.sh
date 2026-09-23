#!/bin/bash
# Tonight's sweep: three models x a cohort, per-layer trunk capture.
#
# Submitted as ONE job array with a concurrency cap, not as N independent jobs.
# The first version submitted 48 separate entries; the scheduler ran 16 and
# queued the rest, which works but floods a shared allocation and gives no way
# to say "at most ten of mine at a time". `--array=1-N%MAX` says exactly that,
# and SLURM enforces it rather than the launcher remembering to submit in waves.
#
# One array index per (model, assay) rather than per model. The reason is
# recorded in launch_gym2s.sh and cost a sweep once: the archive is written only
# at the end, so a job killed at the wall clock produces NOTHING. Sixteen assays
# in one job is sixteen assays lost; one assay per index loses one, and the
# task's resume policy means a relaunch skips what is already on disk -- now
# without loading a model to find out.
#
# Runtime comes from MEASURED marginal per-variant rates, not a scaling
# argument. Measured 2026-08-19 at 100 variants:
#
#     protenix   8.3 s/variant at  40 aa    10.1 s/variant at 118 aa   24 s at 403 aa
#     boltz2    20.7 s/variant at  40 aa                               38 s at 403 aa
#     of3       21.7 s/variant at  40 aa                               39 s at 403 aa
#
# PROTEIN LENGTH IS NOT THE DRIVER in this range, which is worth stating because
# the obvious argument says it should be. The Pairformer's triangle operations
# are O(N^3), so a first pass at these limits assumed the 118-residue assay would
# cost ~26x the 40-residue one; it costs 1.2x. Per-variant time is dominated by
# length-independent work -- the alignment is re-parsed for every variant, and
# the diffusion sampler runs a fixed 200 steps. exp_gym2's docstring says the
# same thing about the MSA: "Boltz re-parses it per variant and that, not the
# GPU, sets the wall-clock."
#
#   bash jax_harness/launch_xmodel_layers.sh            # submit, 10 at a time
#   bash jax_harness/launch_xmodel_layers.sh --dry-run  # print, submit nothing
#   MAX=4 MODELS=protenix bash jax_harness/launch_xmodel_layers.sh
set -euo pipefail

REPO=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability
W=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files
EXP=../experiments/collection/collect_xmodel_layers.py
OUT=${OUT:-$W/runs/xmodel_layers}

MODELS=${MODELS:-"boltz2 of3 protenix"}
COHORT=${COHORT:-heldout_assays}
N_VARIANTS=${N_VARIANTS:-100}
# At most this many of ours running at once. Ten is the standing request: it
# leaves the shared allocation usable while still finishing a 48-job sweep in
# about four hours at ~40 min a job.
MAX=${MAX:-10}
DRY=${1:-}

# One wall clock for the array, so it must suit the slowest member. boltz2 and
# of3 are ~40 min at 100 variants; 120 gives ~3x headroom for a slow node.
MINUTES=${MINUTES:-120}

mapfile -t ASSAYS < <(cd "$REPO" && uv run python -c "
from protein_interpretability.collection import Cohort
print('\n'.join(Cohort.load('$COHORT').ids))")

echo "cohort  $COHORT: ${#ASSAYS[@]} assays"
echo "models  $MODELS"
echo "output  $OUT"
echo "limit   $MAX concurrent, ${MINUTES} min each"
echo

ONLY=""
[ "$(echo "$MODELS" | wc -w)" = "1" ] && ONLY="--model $MODELS"
# Resolve and price before submitting anything. Loads no model, verifies every
# input checksum, and refuses the whole sweep if one has moved -- the failure
# worth catching before jobs queue behind it.
# shellcheck disable=SC2086
(cd "$REPO" && uv run python "experiments/collection/collect_xmodel_layers.py" \
    --inspect $ONLY --cohort "$COHORT" --n-variants "$N_VARIANTS" --output "$OUT")
echo

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST=$W/runs/manifests/xmodel_layers_${COHORT}_${STAMP}.tsv
mkdir -p "$(dirname "$MANIFEST")"
: > "$MANIFEST"
for m in $MODELS; do
  for a in "${ASSAYS[@]}"; do
    # One line per array index. Written to a file rather than passed inline so
    # the exact job list is recoverable afterwards -- an array's arguments are
    # otherwise nowhere on disk.
    echo "$EXP --model $m --assay $a --cohort $COHORT --n-variants $N_VARIANTS --output $OUT" \
        >> "$MANIFEST"
  done
done
N=$(wc -l < "$MANIFEST")

echo "manifest $MANIFEST  ($N jobs)"
if [ "$DRY" = "--dry-run" ]; then
  echo
  echo "would submit: sbatch --array=1-${N}%${MAX} --time=$MINUTES --job-name=xml \\"
  echo "                     $REPO/jax_harness/checkout_array.sbatch $MANIFEST"
  echo
  head -3 "$MANIFEST"
  echo "  ..."
  exit 0
fi

sbatch --array="1-${N}%${MAX}" --time="$MINUTES" --job-name=xml \
       "$REPO/jax_harness/checkout_array.sbatch" "$MANIFEST"

echo
echo "one array, at most $MAX running at a time; artifacts land in $OUT"
echo "relaunching skips whatever this task already wrote, without loading a model"
