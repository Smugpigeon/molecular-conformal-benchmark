#!/usr/bin/env bash
# Uni-Mol 3D sweep. JCIM main line: attack GSHt covalent reactivity.
# Per 项目调研汇总.md §5 方案3: target Pearson R > 0.80 (Zhang JCIM 2025 SOTA).
#
# Usage:
#   bash scripts/sweep_unimol.sh                                  # GSHt only
#   DATASETS="gsht esol lipophilicity" bash scripts/sweep_unimol.sh

set -euo pipefail

N_GPUS=${N_GPUS:-8}
DATASETS_ARG="${DATASETS:-gsht}"   # JCIM main line default
SEEDS_ARG="${SEEDS:-42 1337 2024}"

read -r -a DATASETS <<< "$DATASETS_ARG"
read -r -a SEEDS <<< "$SEEDS_ARG"

LOGDIR=runs/unimol_launches
mkdir -p "$LOGDIR"

n_jobs=$((${#DATASETS[@]} * ${#SEEDS[@]}))
echo "==> Launching $n_jobs Uni-Mol runs across $N_GPUS GPUs"
echo "    Datasets: ${DATASETS[*]}"
echo "    Seeds:    ${SEEDS[*]}"
echo

idx=0
for d in "${DATASETS[@]}"; do
    for s in "${SEEDS[@]}"; do
        while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$N_GPUS" ]; do
            sleep 10
        done

        gpu=$(( idx % N_GPUS ))
        log="$LOGDIR/${d}_seed${s}_gpu${gpu}.log"
        echo "[$(date +%H:%M:%S)] -> GPU $gpu : $d seed=$s  ($log)"
        CUDA_VISIBLE_DEVICES=$gpu \
            python -m src.train \
                dataset=$d \
                model=unimol \
                seed=$s \
            > "$log" 2>&1 &
        idx=$((idx + 1))
        sleep 3
    done
done

echo
echo "==> All $idx jobs launched. Waiting..."
wait
echo "==> $(date) ALL UNIMOL RUNS COMPLETE"
