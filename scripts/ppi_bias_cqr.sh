#!/usr/bin/env bash
# ppi_bias CQR sweep: conformalized quantile regression across 3 representation
# classes (gbm / molformer / chemprop) x 5 regression datasets x 3 seeds.
# GBM on CPU; neural on a GPU subset.
#
# Usage: GPU_LIST="4 5 6 7" bash scripts/ppi_bias_cqr.sh

set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

read -r -a GPUS <<< "${GPU_LIST:-4 5 6 7}"
NG=${#GPUS[@]}
REG=(esol freesolv lipophilicity bace gsht)
SEEDS=(42 1337 2024)
LOGDIR=runs/ppi_bias_cqr
mkdir -p "$LOGDIR"

echo "==> CQR sweep on GPUs: ${GPUS[*]}"

# GBM (CPU)
for d in "${REG[@]}"; do
  for s in "${SEEDS[@]}"; do
    python scripts/run_cqr.py --model gbm --dataset "$d" --seed "$s" \
      > "$LOGDIR/gbm_${d}_${s}.log" 2>&1
    echo "  [done] gbm $d seed=$s"
  done
done

# Neural (GPU subset)
idx=0
for model in molformer chemprop; do
  for d in "${REG[@]}"; do
    for s in "${SEEDS[@]}"; do
      while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$NG" ]; do sleep 10; done
      gpu=${GPUS[$(( idx % NG ))]}
      echo "[$(date +%H:%M:%S)] GPU $gpu : $model $d seed=$s"
      CUDA_VISIBLE_DEVICES=$gpu \
        python scripts/run_cqr.py --model "$model" --dataset "$d" --seed "$s" \
        > "$LOGDIR/${model}_${d}_${s}.log" 2>&1 &
      idx=$((idx + 1)); sleep 3
    done
  done
done
wait
echo "==> $(date) CQR sweep done ($idx neural runs)"
