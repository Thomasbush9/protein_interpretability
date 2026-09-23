#!/bin/bash
# Cross-model raw-vector capture: 4 assays x 3 models x 2 runs.
#
# The second run per cell is not redundancy, it is the measurement that makes
# the comparison interpretable. Boltz-2's gym2/gym2s pair happened to be an
# exact repeat and gave a subspace-reproducibility ceiling of 0.9998; that
# number is what turned "cross-assay agreement 0.750" into a result rather than
# a number. Without the same floor for OpenFold3 and Protenix, a LOW cross-model
# agreement cannot be told apart from "one of these models is simply noisier".
# No code change is needed for it -- the trunks are not bit-reproducible, so
# identical arguments and a different output path give the drift directly.
#
# Assays match the existing deep2_* set so the new vectors sit beside the
# scalar features already computed for the same variants.
set -euo pipefail
W=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files
A=$W/data/gym/assays/DMS_ProteinGym_substitutions
H=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability/jax_harness
STAMP=$(date +%Y%m%d_%H%M%S)

ASSAYS="NKX31_HUMAN_Tsuboyama_2023_2L9R PSAE_PICP2_Tsuboyama_2023_1PSE
        RCRO_LAMBD_Tsuboyama_2023_1ORC RS15_GEOSE_Tsuboyama_2023_1A32"

for n in $ASSAYS; do
    a3m=""
    for p in panel2 panel; do
        [ -f "$W/data/gym/$p/colabfold_output/$n.a3m" ] && a3m="$W/data/gym/$p/colabfold_output/$n.a3m" && break
    done
    if [ -z "$a3m" ]; then echo "SKIP $n: no a3m"; continue; fi
    short=$(echo "$n" | cut -d_ -f1)
    for m in boltz2 of3 protenix; do
        for run in r1 r2; do
            # Per-cell work directory. exp_gym_deep rewrites its alignment per
            # variant, so two live jobs sharing one directory can read each
            # other's mid-write -- the failure that cost a whole gym2s sweep.
            sbatch --time=480 --job-name="xm_${m}_${short}_${run}" \
                "$H/run.sbatch" exp_gym_deep.py \
                --model "$m" --assay "$n" --assay-dir "$A" --a3m "$a3m" \
                --work "$W/runs/xm_work/${m}_${short}_${run}_${STAMP}" \
                --out "$W/runs/xm_${m}_${run}_$n.npz"
        done
    done
done
