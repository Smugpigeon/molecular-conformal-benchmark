#!/usr/bin/env bash
# Generate val_predictions.csv for ROC/confusion figures.
# 4 backbones x 3 cls datasets, seed 42 only (predictions don't need 3 seeds).
# Requires the predictions patch in train.py + wrappers.

set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

N_GPUS=${N_GPUS:-8}
CLS=(esol_cls bace_cls bbbp_cls)
SEED=42
LOGDIR=runs/pred_launches
mkdir -p "$LOGDIR"

# RF (CPU)
for d in "${CLS[@]}"; do
    python -m src.train dataset="$d" model=rf seed=$SEED > "$LOGDIR/rf_${d}.log" 2>&1
    echo "  [done] RF $d"
done

# Neural (GPU)
idx=0
for model in molformer chemfm unimol; do
    for d in "${CLS[@]}"; do
        while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$N_GPUS" ]; do sleep 10; done
        gpu=$(( idx % N_GPUS ))
        echo "[$(date +%H:%M:%S)] GPU $gpu : $model $d"
        CUDA_VISIBLE_DEVICES=$gpu python -m src.train dataset="$d" model="$model" seed=$SEED \
            > "$LOGDIR/${model}_${d}.log" 2>&1 &
        idx=$((idx + 1)); sleep 3
    done
done
wait
echo "==> $(date) PREDICTIONS DONE"
