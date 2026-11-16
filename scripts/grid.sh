#!/bin/bash
# Every arm of the experiments, 2 seeds each: 22 training runs, each followed by its eval. With SLURM each run is its
# own job (extra sbatch flags from $SBATCH_ARGS, e.g. SBATCH_ARGS="--partition=long"); without it, they run one after
# another on this machine's GPU (~30 GPU-hours in all).
#   bash scripts/grid.sh
# Then: bash scripts/monitor.sh; bash scripts/monitor.sh ladder; python tables.py
set -euo pipefail
run() {  # <world model> <tag> <arm> <seed>
    if command -v sbatch > /dev/null; then
        sbatch ${SBATCH_ARGS:-} --job-name=$3$2_s$4 --export=ALL,P=$1,TAG=$2 scripts/arm.sh $3 $4
    else
        P=$1 TAG=$2 bash scripts/arm.sh $3 $4
    fi
}
for seed in 0 1; do
    for arm in e r rsw rcl rcv rtrue rtb rtbsw rtbcl; do
        run p_ff "" $arm $seed
    done
    # the step checker again, from the world model whose solutions sometimes end with an identity step
    run p_ffpad2 pad2 rcv $seed
    run p_ffpad2 pad2 e $seed
done
