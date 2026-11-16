#!/bin/bash
# The world models: continued pretraining of sai_ctx2 (the released base model) on 0.5B tokens, half replay of its own
# pretraining mix and half new computed data (free-form solutions, their verdicts, grader logs). ~30 min each on 1 A100.
# Optional: the released checkpoints of all of these are what the experiments use (scripts/download.sh).
#   bash scripts/download.sh base replay      # sai_ctx2 (with its optimizer state) and the replay token streams
#   bash scripts/pretrain.sh p_ff             # or p_ffpad2, p_ffc, ladder (p_ff_100m, p_ff_30m, p_ff_10m)
#SBATCH --gpus-per-task=1 --cpus-per-task=8 --mem=48G --time=3:00:00
set -euo pipefail
WHAT=$1
mix() { python -c "import data, sys; print(data.doc_mix(sys.argv[1]))" "$1"; }
# token shares: 50% replay (where sai_ctx2 left off), 30% solutions, 12% verdicts, 8% grader logs
REPLAY=fineweb_edu_ctx:0.40,wikipedia_ctx:0.03,python_ctx:0.03,facts:0.01,fact_chains:0.03
cpt() {  # <run> <data prefix: ff | ffp | ffc> <tokens> [extra flags]
    run=$1 d=$2 tokens=$3; shift 3
    python data.py ${d}_chains ${d}_verdicts ${d}_grader_logs
    python pretrain.py --run_name=$run --init_from=sai_ctx2 --init_optimizer=True --continue_streams=False \
        --lr_scale=0.5 --num_tokens=$tokens --weight_decay=0 --warmdown_ratio=0.5 --final_lr_frac=0 --anneal_frac=0 \
        --train_mix=$(mix $REPLAY,${d}_chains:0.30,${d}_verdicts:0.12,${d}_grader_logs:0.08) --val_source=${d}_chains "$@"
    python sai.py precheck --run_name=$run --rung=$([ $d = ffc ] && echo kc || echo f)
}
case $WHAT in
    p_ff)     cpt p_ff ff 500_000_000 ;;
    p_ffpad2) cpt p_ffpad2 ffp 500_000_000 ;;  # 30% of solutions end with an identity step ("71 + 0 = 71.")
    p_ffc)    cpt p_ffc ffc 500_000_000 ;;     # half the problems contain a fair coin flip
    ladder)   for m in 100 30 10; do cpt p_ff_${m}m ff ${m}_000_000 --warmup_steps=$((m * 10 / 26)); done ;;
    *) echo "unknown world model $WHAT"; exit 1 ;;
esac
