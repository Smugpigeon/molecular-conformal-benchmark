#!/usr/bin/env bash
# ppi_bias Uni-Mol CQR sweep: fine-tuned Uni-Mol (84M, 3D) quantile head ->
# conformalized quantile regression. Answers: does adaptive CQR fix Uni-Mol's
# "tightest but under-covers" split-conformal failure?
# 5 regression datasets x 3 seeds = 15 runs, fp32, one card each.
#
# Usage: GPU_LIST="4 5 6 7" bash scripts/ppi_bias_unimol_cqr.sh
#        EPOCHS=2 GPU_LIST="4" bash scripts/ppi_bias_unimol_cqr.sh   # dry-run

set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

read -r -a GPUS <<< "${GPU_LIST:-4 5 6 7}"
NG=${#GPUS[@]}
REG=(esol freesolv lipophilicity bace gsht)
SEEDS=(42 1337 2024)
EP_ARG=""
[ -n "${EPOCHS:-}" ] && EP_ARG="--epochs ${EPOCHS}"
LOGDIR=runs/ppi_bias_unimol_cqr
mkdir -p "$LOGDIR"

echo "==> Uni-Mol CQR sweep on GPUs: ${GPUS[*]}  ${EP_ARG:+(epochs override: $EPOCHS)}"
idx=0
for d in "${REG[@]}"; do
  for s in "${SEEDS[@]}"; do
    while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$NG" ]; do sleep 10; done
    gpu=${GPUS[$(( idx % NG ))]}
    echo "[$(date +%H:%M:%S)] GPU $gpu : unimol $d seed=$s"
    CUDA_VISIBLE_DEVICES=$gpu \
      python scripts/run_cqr.py --model unimol --dataset "$d" --seed "$s" $EP_ARG \
      > "$LOGDIR/unimol_${d}_${s}.log" 2>&1 &
    idx=$((idx + 1)); sleep 5
  done
done
wait
echo "==> $(date) Uni-Mol CQR sweep done ($idx runs)"
