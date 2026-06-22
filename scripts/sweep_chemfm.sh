#!/usr/bin/env bash
# ChemFM-3B LoRA fine-tune sweep across N_GPUS.
# Per 项目调研汇总.md §5 方案1: target SOTA on ESOL/FreeSolv/BBBP/BACE.
#
# Usage:
#   bash scripts/sweep_chemfm.sh               # full 6x3 sweep
#   N_GPUS=4 bash scripts/sweep_chemfm.sh      # share with other jobs
#   DRY=1 DATASETS=gsht bash scripts/sweep_chemfm.sh   # quick GSHt sanity

set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

N_GPUS=${N_GPUS:-8}
DRY=${DRY:-0}
DATASETS_ARG="${DATASETS:-esol freesolv lipophilicity bace bbbp gsht}"
SEEDS_ARG="${SEEDS:-42 1337 2024}"

read -r -a DATASETS <<< "$DATASETS_ARG"
read -r -a SEEDS <<< "$SEEDS_ARG"

EPOCHS_OVERRIDE=""
if [ "$DRY" = "1" ]; then
    echo "==> DRY mode: 1 epoch only"
    EPOCHS_OVERRIDE="model.epochs=1"
fi

LOGDIR=runs/chemfm_launches
mkdir -p "$LOGDIR"

n_jobs=$((${#DATASETS[@]} * ${#SEEDS[@]}))
echo "==> Launching $n_jobs ChemFM-3B runs across $N_GPUS GPUs"
echo "    Datasets: ${DATASETS[*]}"
echo "    Seeds:    ${SEEDS[*]}"
echo

idx=0
for d in "${DATASETS[@]}"; do
    for s in "${SEEDS[@]}"; do
        # Wait until fewer than N_GPUS jobs in flight.
        while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$N_GPUS" ]; do
            sleep 15
        done

        gpu=$(( idx % N_GPUS ))
        tag="${d}_seed${s}_gpu${gpu}"
        log="$LOGDIR/${tag}.log"

        echo "[$(date +%H:%M:%S)] -> GPU $gpu : $d seed=$s  ($log)"
        CUDA_VISIBLE_DEVICES=$gpu \
            python -m src.train \
                dataset=$d \
                model=chemfm \
                seed=$s \
                $EPOCHS_OVERRIDE \
            > "$log" 2>&1 &

        idx=$((idx + 1))
        sleep 5  # stagger model loading (12GB each)
    done
done

echo
echo "==> All $idx jobs launched. Waiting..."
wait
echo "==> $(date) ALL CHEMFM RUNS COMPLETE"
