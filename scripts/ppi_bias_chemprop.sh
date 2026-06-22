#!/usr/bin/env bash
# ppi_bias: Chemprop D-MPNN (策略2 GNN) on the 6 datasets, official framing
# (5 regression + BBBP classification), 3 seeds. Adds the message-passing-GNN
# backbone to the comparison.

set -euo pipefail
N_GPUS=${N_GPUS:-8}
DATASETS=(esol freesolv lipophilicity bace gsht bbbp)
SEEDS=(42 1337 2024)
LOGDIR=runs/ppi_bias_chemprop
mkdir -p "$LOGDIR"

idx=0
for d in "${DATASETS[@]}"; do
    for s in "${SEEDS[@]}"; do
        while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$N_GPUS" ]; do sleep 10; done
        gpu=$(( idx % N_GPUS ))
        echo "[$(date +%H:%M:%S)] GPU $gpu : chemprop $d seed=$s"
        CUDA_VISIBLE_DEVICES=$gpu \
            python -m src.train dataset="$d" model=chemprop seed="$s" save_test_preds=true \
            > "$LOGDIR/${d}_seed${s}.log" 2>&1 &
        idx=$((idx + 1)); sleep 3
    done
done
wait
echo "==> $(date) ppi_bias chemprop sweep done ($idx runs)"
