#!/bin/bash
# Downloads the released checkpoints (and, for rebuilding the world models, the replay token streams) into $SAI_DIR.
#   bash scripts/download.sh                # the world models the experiments use: p_ff, p_ffpad2, p_ffc, p_ff_100m/30m/10m
#   bash scripts/download.sh base replay    # sai_ctx2 with its optimizer state, and the replay streams (for scripts/pretrain.sh)
#   bash scripts/download.sh runs           # every trained arm of the original grid (to rerun only the evals and monitors)
# Needs the GitHub CLI (gh auth login) while the repository is private; with a public one, plain curl works.
set -euo pipefail
REPO=MJ10/scientist-ai-math
TAG=v1
DIR=${SAI_DIR:-work}
WORLD="p_ff p_ffpad2 p_ffc p_ff_100m p_ff_30m p_ff_10m"
REPLAY="fineweb_edu_ctx:2 wikipedia_ctx:2 python_ctx:1 facts:1 fact_chains:1"  # <source>:<number of files>
RUNS="e_kfpad2_s0"
for s in 0 1; do
    for a in e r rsw rcl rcv rtrue rtb rtbsw rtbcl; do RUNS="$RUNS ${a}_kf_s$s"; done
    RUNS="$RUNS rcv_kfpad2_s$s"
done
get() {  # <asset> <destination>
    [ -f "$2" ] && { echo "have $2"; return; }
    mkdir -p "$(dirname "$2")"
    if command -v gh > /dev/null; then
        gh release download $TAG --repo $REPO --pattern "$1" --output "$2.tmp"
    else
        curl -fL -o "$2.tmp" "https://github.com/$REPO/releases/download/$TAG/$1"
    fi
    mv "$2.tmp" "$2"
    echo "got $2"
}
for what in ${@:-world}; do
    case $what in
        world)  for r in $WORLD; do get $r.npz $DIR/runs/$r/ckpt.npz; done ;;
        base)   get sai_ctx2.npz $DIR/runs/sai_ctx2/ckpt.npz ;;
        replay) for sn in $REPLAY; do
                    s=${sn%:*}
                    for ((i = 0; i < ${sn#*:}; i++)); do
                        f=$(printf %04d $i); get replay_${s}_$f.bin $DIR/tokens/v2/$s/train/$f.bin
                    done
                done ;;
        runs)   for r in $RUNS; do get $r.npz $DIR/runs/$r/ckpt.npz; done ;;
        *) echo "unknown: $what (world | base | replay | runs)"; exit 1 ;;
    esac
done
