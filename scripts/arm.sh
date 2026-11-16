#!/bin/bash
# One arm on one seed: train q from the world model, then evaluate it (1 GPU; ~1 h for 300 steps, ~3 h for 1,000).
#   bash scripts/arm.sh <arm> <seed>                  e.g. bash scripts/arm.sh e 0
#   P=p_ffpad2 TAG=pad2 bash scripts/arm.sh rcv 0     (another world model; TAG goes into the run name)
#   RUNG=kc P=p_ffc bash scripts/arm.sh r 0           (coin problems)
# Run name: <arm>_<rung><tag>_s<seed> (e.g. e_kf_s0, rcl_kf_s1), under $SAI_DIR/runs. Reruns resume / skip finished runs.
# The arms (README.md): the Scientist AI (e), GRPO against each grader (r, rsw, rcl, rcv), GRPO on the truth (rtrue),
# and trajectory balance toward p * exp(r / tau) against the graders (rtb, rtbsw, rtbcl; tau = 0.1).
#SBATCH --gpus-per-task=1 --cpus-per-task=4 --mem=24G --time=12:00:00
set -euo pipefail
ARM=$1 SEED=$2
P=${P:-p_ff} RUNG=${RUNG:-kf}
POL="--n_items=16 --k=16"  # (policies: 16 problems x 16 samples per step)
TB="--objective=tb_vargrad --lr_scale=0.01 --explore_temp=1.2"
GRPO="--objective=grpo --lr_scale=0.05 $POL"
case $ARM in
    e)     steps=300;  args="--verdict=1 --alt_frac=0 $TB" ;;
    r)     steps=300;  args="--target=reward $GRPO" ;;
    rsw)   steps=300;  args="--target=reward --show_work=3 $GRPO" ;;
    rcl)   steps=1000; args="--target=reward --show_work=3 --check_last=1 $GRPO" ;;
    rcv)   steps=1000; args="--target=reward --show_work=3 --check_last=2 $GRPO" ;;
    rtrue) steps=300;  args="--target=truth $GRPO" ;;
    rtb)   steps=300;  args="--target=reward --tau=0.1 $POL $TB" ;;
    rtbsw) steps=300;  args="--target=reward --tau=0.1 --show_work=3 $POL $TB" ;;
    rtbcl) steps=300;  args="--target=reward --tau=0.1 --show_work=3 --check_last=1 $POL $TB" ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
esac
steps=${STEPS:-$steps}
name=${ARM}_${RUNG}${TAG:-}_s${SEED}
common="--init_from=$P --rung=$RUNG --rho=0.5 --forced_stop=1 --max_tokens=192"
python sai.py train --run_name=$name --seed=$SEED --steps=$steps $common $args
python sai.py eval --run_name=$name $common
