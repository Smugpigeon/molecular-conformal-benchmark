#!/usr/bin/env bash
# ppi_bias conformal MULTI-SEED: add seeds 1337 + 2024 to the conformal sweep
# (seed 42 already done). Fixes the single-seed violation (project §8) and
# enables error bars on interval width / coverage. Restricted to a GPU subset.
#
# Usage: GPU_LIST="4 5 6 7" bash scripts/ppi_bias_conformal_ms.sh

set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

read -r -a GPUS <<< "${GPU_LIST:-4 5 6 7}"
NG=${#GPUS[@]}
REG=(esol freesolv lipophilicity bace gsht)
SEEDS=(1337 2024)
NEURAL=(molformer chemfm unimol chemprop)
LOGDIR=runs/ppi_bias_conformal_ms
mkdir -p "$LOGDIR"

echo "==> multi-seed conformal on GPUs: ${GPUS[*]}  (seeds ${SEEDS[*]})"

# RF (CPU) first — fast, no GPU.
for d in "${REG[@]}"; do
  for s in "${SEEDS[@]}"; do
    python -m src.train dataset="$d" model=rf seed="$s" save_test_preds=true \
      > "$LOGDIR/rf_${d}_${s}.log" 2>&1
    echo "  [done] rf $d seed=$s"
  done
done

# Neural backbones on the GPU subset.
idx=0
for model in "${NEURAL[@]}"; do
  for d in "${REG[@]}"; do
    for s in "${SEEDS[@]}"; do
      while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$NG" ]; do sleep 10; done
      gpu=${GPUS[$(( idx % NG ))]}
      echo "[$(date +%H:%M:%S)] GPU $gpu : $model $d seed=$s"
      CUDA_VISIBLE_DEVICES=$gpu \
        python -m src.train dataset="$d" model="$model" seed="$s" save_test_preds=true \
        > "$LOGDIR/${model}_${d}_${s}.log" 2>&1 &
      idx=$((idx + 1)); sleep 3
    done
  done
done
wait
echo "==> $(date) multi-seed conformal done ($idx neural runs)"
