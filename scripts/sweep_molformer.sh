#!/usr/bin/env bash
# Launch MolFormer-XL fine-tune across all 6 datasets x 3 seeds,
# distributed across N_GPUS in parallel.
# Per CLAUDE.md §12: must dry-run first (--max-steps small) before going wide.
#
# Usage:
#   bash scripts/sweep_molformer.sh              # full sweep (18 runs / 8 GPU)
#   N_GPUS=4 bash scripts/sweep_molformer.sh     # use only 4 GPUs
#   DRY=1 bash scripts/sweep_molformer.sh        # 1 epoch sanity run only

set -euo pipefail

# HF mirror for China (server has no direct HF access; cached lookups still OK)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Skip HF Hub health check on each run (cache already populated)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

N_GPUS=${N_GPUS:-8}
DRY=${DRY:-0}

DATASETS=(esol freesolv lipophilicity bace bbbp gsht)
SEEDS=(42 1337 2024)

EPOCHS_OVERRIDE=""
if [ "$DRY" = "1" ]; then
    echo "==> DRY mode: 1 epoch only"
    EPOCHS_OVERRIDE="model.epochs=1"
fi

LOGDIR=runs/molformer_launches
mkdir -p "$LOGDIR"

echo "==> Launching $((${#DATASETS[@]} * ${#SEEDS[@]})) MolFormer-XL runs across $N_GPUS GPUs"
echo "    Datasets: ${DATASETS[*]}"
echo "    Seeds:    ${SEEDS[*]}"
echo "    Log dir:  $LOGDIR/"
echo

idx=0
for d in "${DATASETS[@]}"; do
    for s in "${SEEDS[@]}"; do
        # Wait until fewer than N_GPUS jobs in flight.
        while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$N_GPUS" ]; do
            sleep 10
        done

        gpu=$(( idx % N_GPUS ))
        tag="${d}_seed${s}_gpu${gpu}"
        log="$LOGDIR/${tag}.log"

        echo "[$(date +%H:%M:%S)] -> GPU $gpu : $d seed=$s  ($log)"
        CUDA_VISIBLE_DEVICES=$gpu \
            python -m src.train \
                dataset=$d \
                model=molformer \
                seed=$s \
                $EPOCHS_OVERRIDE \
            > "$log" 2>&1 &

        idx=$((idx + 1))
        sleep 2  # stagger so HF cache loads don't collide
    done
done

echo
echo "==> All $idx jobs launched. Waiting for completion..."
wait
echo "==> $(date) ALL MOLFORMER RUNS COMPLETE"
