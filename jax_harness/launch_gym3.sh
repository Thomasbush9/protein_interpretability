#!/bin/bash
# Re-run the 12 Boltz-2 ProteinGym assays capturing per-residue pLDDT and the
# unaveraged pair row.
#
# Output goes to gym3_*.npz. The gym2s_* archives stay on disk: kl_glob is
# re-emitted by exp_gym3 precisely so the rerun can be checked against them
# variant-for-variant. If the divergences do not come back in rank agreement,
# something in the capture changed that should not have and the new arrays
# cannot be trusted either.
set -euo pipefail
W=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files
A=$W/data/gym/assays/DMS_ProteinGym_substitutions
H=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability/jax_harness

# Per-submission stamp. exp_gym3 rewrites msa/wt.a3m and msa/mut.a3m for every
# variant, so two live jobs on one assay sharing a work directory can read each
# other's alignment mid-run and emit features for a sequence-alignment pair that
# never existed together. That happened once on the gym2s sweep and nothing
# would have flagged it -- there is no alignment-depth guard.
STAMP=$(date +%Y%m%d_%H%M%S)

for f in "$W"/runs/gym2s_*.npz; do
    n=$(basename "$f" .npz); n=${n#gym2s_}
    a3m=""
    for p in panel2 panel; do
        [ -f "$W/data/gym/$p/colabfold_output/$n.a3m" ] && a3m="$W/data/gym/$p/colabfold_output/$n.a3m" && break
    done
    if [ -z "$a3m" ]; then echo "SKIP $n: no a3m"; continue; fi
    short=$(echo "$n" | cut -d_ -f1)
    # Measured on the smoke run: ~59 s/variant, so ~4.1 h for 250. gym2s needed
    # 720 min for CBPA2, and the archive is written only at the end -- a job
    # killed at the wall clock produces nothing at all. Ask for 12 h.
    sbatch --time=720 --job-name="g3_$short" "$H/run.sbatch" exp_gym3.py \
        --assay "$n" --assay-dir "$A" --a3m "$a3m" \
        --work "$W/runs/g3_work/${short}_${STAMP}" \
        --out "$W/runs/gym3_$n.npz"
done
