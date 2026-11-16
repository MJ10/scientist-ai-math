#!/bin/bash
# The world models: continued pretraining of sai_ctx2 (the released base model) on 0.5B tokens, half replay of its own
# pretraining mix and half new computed data (free-form solutions, their verdicts, grader logs). ~30 min each on 1 A100.
# Optional: the released checkpoints of all of these are what the experiments use (scripts/download.sh).
#   bash scripts/download.sh base replay      # sai_ctx2 (with its optimizer state) and the replay token streams
#   bash scripts/pretrain.sh p_ff             # or p_ffpad2, p_ffc, ladder (p_ff_100m, p_ff_30m, p_ff_10m)
#SBATCH --gpus-per-task=1 --cpus-per-task=8 --mem=32G --time=3:00:00
set -euo pipefail
WHAT=$1
# Token shares 50% replay (fineweb_edu_ctx 0.40, wikipedia_ctx 0.03, python_ctx 0.03, facts 0.01, fact_chains 0.03) and
# 30% solutions, 12% verdicts, 8% grader logs, as document mixes: `python -c "import data; print(data.doc_mix(...))"`
# computes them; these are the exact ones of the original runs (doc_mix on the full token files).
REPLAY=fineweb_edu_ctx:0.24457,wikipedia_ctx:0.01854,python_ctx:0.02932,facts:0.05213,fact_chains:0.05364
FF=$REPLAY,ff_chains:0.34394,ff_verdicts:0.18457,ff_grader_logs:0.07328
FFP=fineweb_edu_ctx:0.24643,wikipedia_ctx:0.01869,python_ctx:0.02955,facts:0.05252,fact_chains:0.05405
FFP=$FFP,ffp_chains:0.34117,ffp_verdicts:0.18471,ffp_grader_logs:0.07289
FFC=fineweb_edu_ctx:0.26060,wikipedia_ctx:0.01976,python_ctx:0.03125,facts:0.05554,fact_chains:0.05716
FFC=$FFC,ffc_chains:0.32409,ffc_verdicts:0.18102,ffc_grader_logs:0.07058
cpt() {  # <run> <data prefix: ff | ffp | ffc> <mix> <tokens> [extra flags]
    run=$1 d=$2 mix=$3 tokens=$4; shift 4
    python data.py ${d}_chains ${d}_verdicts ${d}_grader_logs
    python pretrain.py --run_name=$run --init_from=sai_ctx2 --init_optimizer=True --continue_streams=False \
        --lr_scale=0.5 --num_tokens=$tokens --weight_decay=0 --warmdown_ratio=0.5 --final_lr_frac=0 --anneal_frac=0 \
        --train_mix=$mix --val_source=${d}_chains "$@"
    python sai.py precheck --run_name=$run --rung=$([ $d = ffc ] && echo kc || echo f)
}
case $WHAT in
    p_ff)     cpt p_ff ff $FF 500_000_000 ;;
    p_ffpad2) cpt p_ffpad2 ffp $FFP 500_000_000 ;;  # 30% of solutions end with an identity step ("71 + 0 = 71.")
    p_ffc)    cpt p_ffc ffc $FFC 500_000_000 ;;     # half the problems contain a fair coin flip
    ladder)   for m in 100 30 10; do cpt p_ff_${m}m ff $FF ${m}_000_000 --warmup_steps=$((m * 10 / 26)); done ;;
    *) echo "unknown world model $WHAT"; exit 1 ;;
esac
