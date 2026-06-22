#!/usr/bin/env bash
# GAP6: multi-seed conformal width/coverage error bars. The 5 existing datasets x backbones
# already have 3-seed preds; only QM7/QM8 need seeds 1337+2024 (seed42 from the conformal-paper
# sweep). Waits for that sweep + the deep-CV to release GPUs, then trains the QM backbones at the
# two extra seeds and runs run_conformal_multiseed.py (mean+/-std across seeds).
set -uo pipefail
cd ~/ppi_bias
PY="${PY:-${CONDA_PREFIX:-$HOME/.conda/envs/drug}/bin/python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-$HOME/.conda/envs/drug}/lib:${LD_LIBRARY_PATH:-}"
LOG=runs/gap6; mkdir -p "$LOG"

echo "[gap6] waiting for conformal-paper sweep + deep-CV to finish..."
while pgrep -f "[s]weep_conformal_paper" >/dev/null 2>&1; do sleep 120; done
while pgrep -f "[d]ataset=cvfold" >/dev/null 2>&1; do sleep 120; done
echo "[gap6] GPUs free at $(date +%H:%M:%S); training QM backbones at seeds 1337,2024."

N_CONC=6; N_GPUS=8; idx=0; running=0
for s in 1337 2024; do
  for d in qm7 qm8; do
    for m in molformer chemfm unimol; do
      if [ "$running" -ge "$N_CONC" ]; then wait -n; running=$((running - 1)); fi
      gpu=$(( idx % N_GPUS )); rd="runs/manual_${d}_${m}_seed${s}"; rm -rf "$rd"
      CUDA_VISIBLE_DEVICES=$gpu $PY -m src.train dataset=$d model=$m seed=$s save_test_preds=true \
          hydra.run.dir="$rd" > "$LOG/${d}_${m}_${s}.log" 2>&1 &
      idx=$((idx + 1)); running=$((running + 1)); sleep 3
    done
  done
done
wait

echo "[gap6] RF (CPU) at seeds 1337,2024 on QM..."
for s in 1337 2024; do
  for d in qm7 qm8; do
    CUDA_VISIBLE_DEVICES="" $PY -m src.train dataset=$d model=rf seed=$s save_test_preds=true \
        hydra.run.dir="runs/manual_${d}_rf_seed${s}" > "$LOG/${d}_rf_${s}.log" 2>&1
  done
done

echo "[gap6] multi-seed conformal aggregation..."
$PY scripts/run_conformal_multiseed.py > "$LOG/multiseed.log" 2>&1 && echo "    multiseed done" || echo "    multiseed FAIL"
echo "GAP6 DONE at $(date +%H:%M:%S)"
tail -30 "$LOG/multiseed.log"
