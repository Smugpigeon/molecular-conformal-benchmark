#!/usr/bin/env bash
# ppi_bias evaluation sweep: 4 backbones x 5 regression datasets, with held-out
# test predictions saved (for the conformal-prediction study). seed 42.
# Logs + run dirs live under the ppi_bias project, consistent with the repo.

set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

N_GPUS=${N_GPUS:-8}
REG=(esol freesolv lipophilicity bace gsht)
SEED=42
LOGDIR=runs/ppi_bias_conformal
mkdir -p "$LOGDIR"

echo "==> ppi_bias eval sweep: RF + MolFormer + ChemFM + Uni-Mol x ${#REG[@]} datasets"

# RandomForest (CPU)
for d in "${REG[@]}"; do
    python -m src.train dataset="$d" model=rf seed=$SEED save_test_preds=true \
        > "$LOGDIR/rf_${d}.log" 2>&1
    echo "  [done] rf $d"
done

# Neural backbones (GPU, parallel)
idx=0
for model in molformer chemfm unimol; do
    for d in "${REG[@]}"; do
        while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$N_GPUS" ]; do sleep 10; done
        gpu=$(( idx % N_GPUS ))
        echo "[$(date +%H:%M:%S)] GPU $gpu : $model $d"
        CUDA_VISIBLE_DEVICES=$gpu \
            python -m src.train dataset="$d" model="$model" seed=$SEED save_test_preds=true \
            > "$LOGDIR/${model}_${d}.log" 2>&1 &
        idx=$((idx + 1)); sleep 3
    done
done
wait
echo "==> $(date) ppi_bias eval sweep done"
