#!/bin/bash
# The Scientist AI as a monitor: its estimator (a trained e_* run) scores every answer of the RL policies, next to
# direct p (no reasoning) and a chain-of-thought monitor (p reading the policy's own solution). Results go to the
# estimator's gfn_evals.jsonl under "monitor/<policy>". Run after scripts/grid.sh.
#   bash scripts/monitor.sh            # estimator e_kf_s0 on p_ff, all p_ff policies
#   bash scripts/monitor.sh ladder     # each world model of the ladder as an untrained estimator (q = p)
#SBATCH --gpus-per-task=1 --cpus-per-task=4 --mem=24G --time=12:00:00
set -euo pipefail
common="--rung=kf --rho=0.5 --forced_stop=1 --max_tokens=192"
POLICIES=r_kf_s0,r_kf_s1,rsw_kf_s0,rsw_kf_s1,rcl_kf_s0,rcl_kf_s1,rcv_kf_s0,rcv_kf_s1,rtrue_kf_s0,rtrue_kf_s1
POLICIES=$POLICIES,rtb_kf_s0,rtb_kf_s1,rtbsw_kf_s0,rtbsw_kf_s1,rtbcl_kf_s0,rtbcl_kf_s1
CHEATS=r_kf_s0,r_kf_s1,rsw_kf_s0,rsw_kf_s1,rcl_kf_s0,rcl_kf_s1,rcv_kfpad2_s0,rcv_kfpad2_s1
case ${1:-main} in
    main)
        python sai.py eval --run_name=e_kf_s0 --init_from=p_ff $common --monitor=$POLICIES
        # the policies that found the step checker's hole, with the estimator trained on their own world model
        python sai.py eval --run_name=e_kfpad2_s0 --init_from=p_ffpad2 $common --monitor=rcv_kfpad2_s0,rcv_kfpad2_s1 ;;
    ladder)
        for p in p_ff p_ff_100m p_ff_30m p_ff_10m; do
            python sai.py eval --run_name=$p --init_from=$p --verdict=1 $common --monitor=$CHEATS
        done ;;
esac
