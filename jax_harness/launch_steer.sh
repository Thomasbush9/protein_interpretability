#!/bin/bash
# The causal leg, across all twelve assays instead of one.
#
# `steer_RCRO_v4` tested a single protein against four random directions.
# PC2's odd (sign-dependent) response came out largest of the seven directions
# tried, but only modestly above the best random control -- being top of seven
# once is suggestive and nothing more, and `analyze_steer.py` says so in its own
# docstring.
#
# The power is in the number of ASSAYS, not in more draws inside one. Twelve
# independent proteins each asking "does PC2 rank above its own randoms" turns a
# descriptive ordering into something a sign test can speak to, which is what
# `analyze_steer_pool.py` then does. --n-random is raised to 8 anyway so a
# single assay's ranking is not decided by four draws.
#
# Runtime scales with (3 + n_random) x alphas x sites x sampling steps. RCRO
# (63 residues) took 3060 s for 441 runs at n_random=4; this is ~11/7 of that
# per assay, more for the longer domains, hence --time=600.
set -euo pipefail
W=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files
A=$W/data/gym/assays/DMS_ProteinGym_substitutions
H=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability/jax_harness
STAMP=$(date +%Y%m%d_%H%M%S)

for f in "$W"/runs/gym2s_*.npz; do
    n=$(basename "$f" .npz); n=${n#gym2s_}
    a3m=""
    for p in panel2 panel; do
        [ -f "$W/data/gym/$p/colabfold_output/$n.a3m" ] && a3m="$W/data/gym/$p/colabfold_output/$n.a3m" && break
    done
    if [ -z "$a3m" ]; then echo "SKIP $n: no a3m"; continue; fi
    short=$(echo "$n" | cut -d_ -f1)
    sbatch --time=600 --job-name="st_$short" "$H/run.sbatch" exp_steer.py \
        --assay "$n" --assay-dir "$A" --a3m "$a3m" \
        --basis "$W/runs/pc2_v2.npz" --features "$f" \
        --n-random 8 \
        --work "$W/runs/steer_all_work/${short}_${STAMP}" \
        --out "$W/runs/steerall_$n.npz"
done
