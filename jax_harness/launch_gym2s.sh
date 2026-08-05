#!/bin/bash
# Rerun the 12 Boltz-2 ProteinGym assays with the shift/spread features added.
#
# Output goes to gym2s_*.npz, NOT gym2_*.npz: the original archives stay on disk
# so the reproduced kl_glob/kl_site can be checked against them. If the KL
# features do not come back identical, the rerun changed something it should not
# have and the new features cannot be trusted either.
set -euo pipefail
W=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files
A=$W/data/gym/assays/DMS_ProteinGym_substitutions
H=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability/jax_harness

# Work directories are stamped per submission. `exp_gym2` rewrites
# msa/wt.a3m and msa/mut.a3m for every variant, so two live jobs on the same
# assay sharing a directory can read each other's alignment mid-run and produce
# features for a sequence-alignment pair that never existed together. That
# happened once here when a resubmission overlapped the jobs it replaced, and it
# would not have raised anything -- exp_gym2 has no alignment-depth guard.
STAMP=$(date +%Y%m%d_%H%M%S)

for f in "$W"/runs/gym2_*.npz; do
    n=$(basename "$f" .npz); n=${n#gym2_}
    a3m=""
    for p in panel2 panel; do
        [ -f "$W/data/gym/$p/colabfold_output/$n.a3m" ] && a3m="$W/data/gym/$p/colabfold_output/$n.a3m" && break
    done
    if [ -z "$a3m" ]; then echo "SKIP $n: no a3m"; continue; fi
    short=$(echo "$n" | cut -d_ -f1)
    # run.sbatch defaults to --time=120, but a 250-variant gym2 assay takes
    # ~3.5 h (measured: RCRO finished at 12417 s) and the archive is written
    # only at the end -- a job killed at the wall clock produces NOTHING. The
    # first submission of this sweep was lost that way. Override it here.
    sbatch --time=360 --job-name="g2s_$short" "$H/run.sbatch" exp_gym2.py \
        --assay "$n" --assay-dir "$A" --a3m "$a3m" \
        --work "$W/runs/g2s_work/${short}_${STAMP}" \
        --out "$W/runs/gym2s_$n.npz"
done
