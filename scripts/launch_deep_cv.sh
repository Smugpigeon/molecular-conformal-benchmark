#!/usr/bin/env bash
# B: CV-train the 3 deep models on the 50 frozen primary scaffold folds (3 seeds/fold)
# so they join the per-fold omnibus. 50 x 3 models x 3 seeds = 450 runs.
# Each run: dataset=cvfold (reads CVFOLD_CSV env, see loaders.py), save_test_preds.
#
# Concurrency is capped with an explicit counter + `wait -n` (NOT $(jobs -r), which
# runs in a subshell with an empty job table and never caps).
#
# Usage:
#   DRY=1 MODELS=molformer FOLDS=1 bash scripts/launch_deep_cv.sh   # 1-fold 1-epoch smoke
#   nohup bash scripts/launch_deep_cv.sh > runs/deepcv/launcher.log 2>&1 &   # full 450
set -uo pipefail
cd ~/ppi_bias
PY="${PY:-${CONDA_PREFIX:-$HOME/.conda/envs/drug}/bin/python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-$HOME/.conda/envs/drug}/lib:${LD_LIBRARY_PATH:-}"  # chemprop libstdc++

N_CONC=${N_CONC:-8}                          # max concurrent jobs
N_GPUS=${N_GPUS:-8}
DRY=${DRY:-0}
read -r -a SEEDS <<< "${SEEDS:-42 1337 2024}"
read -r -a MODELS <<< "${MODELS:-molformer chemfm chemprop}"
EPOCH_OVR=""; [ "$DRY" = "1" ] && EPOCH_OVR="model.epochs=1"
NFOLDS_LIMIT=${FOLDS:-9999}

LOG=runs/deepcv; mkdir -p "$LOG"
idx=0; running=0
for f in $(ls data/processed/cv/primary_r*_f*.csv | head -n "$NFOLDS_LIMIT"); do
  tag=$(basename "$f" .csv)                   # primary_rR_fF
  csv_rel="processed/cv/${tag}.csv"           # relative to data_root
  for m in "${MODELS[@]}"; do
    for s in "${SEEDS[@]}"; do
      # robust concurrency cap: block until a slot frees
      if [ "$running" -ge "$N_CONC" ]; then wait -n; running=$((running - 1)); fi
      gpu=$(( idx % N_GPUS ))
      out="runs/deepcv/${m}_${tag}_seed${s}"
      CVFOLD_CSV="$csv_rel" CUDA_VISIBLE_DEVICES=$gpu $PY -m src.train \
          dataset=cvfold model=$m seed=$s save_test_preds=true $EPOCH_OVR \
          hydra.run.dir="$out" > "$LOG/${m}_${tag}_seed${s}.log" 2>&1 &
      idx=$((idx + 1)); running=$((running + 1)); sleep 1
    done
  done
done
wait
echo "DEEP CV DONE: launched $idx jobs"
