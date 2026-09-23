#!/bin/bash
# Jacobian of transition_z at the wild-type operating point, one job per assay.
#
# Unlike exp_gym2 this runs the WT ONCE -- no per-variant loop -- so it is
# minutes, not hours. The per-assay split still matters: the Jacobian depends on
# the operating point, and twelve folds is the only way to tell a property of
# the layer from a property of one protein.
#
# Work directories are stamped per submission for the same reason as
# launch_gym2s.sh: exp_jac rewrites msa/wt_jac.a3m, so two live jobs sharing a
# directory can read each other's alignment mid-run.
set -euo pipefail
W=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files
A=$W/data/gym/assays/DMS_ProteinGym_substitutions
H=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability/jax_harness
STAMP=$(date +%Y%m%d_%H%M%S)

for f in "$W"/runs/gym2_*.npz; do
    n=$(basename "$f" .npz); n=${n#gym2_}
    a3m=""
    for p in panel2 panel; do
        [ -f "$W/data/gym/$p/colabfold_output/$n.a3m" ] && a3m="$W/data/gym/$p/colabfold_output/$n.a3m" && break
    done
    if [ -z "$a3m" ]; then echo "SKIP $n: no a3m"; continue; fi
    short=$(echo "$n" | cut -d_ -f1)
    sbatch --time=180 --job-name="comp_$short" "$H/run.sbatch" exp_compose.py \
        --assay "$n" --assay-dir "$A" --a3m "$a3m" \
        --gym "$f" --pc "$W/runs/pc2_v2.npz" \
        --work "$W/runs/comp_work/${short}_${STAMP}" \
        --out "$W/runs/comp_$n.npz"
done
