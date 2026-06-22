#!/usr/bin/env bash
# Part A classification benchmark.
# 4 backbones (RF + MolFormer + ChemFM + Uni-Mol) x 3 cls datasets x 3 seeds.
# Datasets built by `python -m src.data.binarize` first.
# 项目调研汇总.md v2 §2-3.
#
# Usage:
#   bash scripts/sweep_partA_cls.sh
#   N_GPUS=4 bash scripts/sweep_partA_cls.sh

set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

N_GPUS=${N_GPUS:-8}
CLS_DATASETS=(esol_cls bace_cls bbbp_cls)
SEEDS=(42 1337 2024)
NEURAL_MODELS=(molformer chemfm unimol)

LOGDIR=runs/partA_launches
mkdir -p "$LOGDIR"

# ---- 1. RandomForest (CPU, fast, inline) ----
echo "=== RF (CPU) on 3 classification datasets ==="
for d in "${CLS_DATASETS[@]}"; do
    for s in "${SEEDS[@]}"; do
        python -m src.train dataset="$d" model=rf seed="$s" \
            > "$LOGDIR/rf_${d}_seed${s}.log" 2>&1
        echo "  [done] RF $d seed=$s"
    done
done

# ---- 2. Neural backbones (GPU, N_GPUS-way parallel) ----
echo "=== Neural backbones across $N_GPUS GPUs ==="
idx=0
for model in "${NEURAL_MODELS[@]}"; do
    for d in "${CLS_DATASETS[@]}"; do
        for s in "${SEEDS[@]}"; do
            while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$N_GPUS" ]; do
                sleep 10
            done
            gpu=$(( idx % N_GPUS ))
            log="$LOGDIR/${model}_${d}_seed${s}_gpu${gpu}.log"
            echo "[$(date +%H:%M:%S)] -> GPU $gpu : $model $d seed=$s"
            CUDA_VISIBLE_DEVICES=$gpu \
                python -m src.train dataset="$d" model="$model" seed="$s" \
                > "$log" 2>&1 &
            idx=$((idx + 1))
            sleep 3
        done
    done
done

echo "==> launched $idx neural jobs, waiting..."
wait
echo "==> $(date) PART A CLASSIFICATION COMPLETE"
