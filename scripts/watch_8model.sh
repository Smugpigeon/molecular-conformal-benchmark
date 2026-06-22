#!/usr/bin/env bash
# Decoupled watcher: build the 8-model per-fold omnibus (the B headline significance) the MOMENT
# deep-CV finishes -- it needs only the deep-CV folds + the CPU per-fold matrix, NOT the slow
# qm8/Uni-Mol conformal cascade. Runs independently of sweep_conformal_paper / gap6 / finalize.
set -uo pipefail
cd ~/ppi_bias
PY="${PY:-${CONDA_PREFIX:-$HOME/.conda/envs/drug}/bin/python}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-$HOME/.conda/envs/drug}/lib:${LD_LIBRARY_PATH:-}"
LOG=runs/finalize; mkdir -p "$LOG"

echo "[8model-watch] waiting for deep-CV (dataset=cvfold) to finish... ($(date +%H:%M:%S))"
while pgrep -f "[d]ataset=cvfold" >/dev/null 2>&1; do sleep 60; done
echo "[8model-watch] deep-CV done at $(date +%H:%M:%S); building 8-model per-fold omnibus."

$PY scripts/make_perfold_8model.py > "$LOG/perfold_8model_early.log" 2>&1 && echo "  8-model omnibus OK" || echo "  8-model FAIL (see $LOG/perfold_8model_early.log)"

{
  echo "===== 8-MODEL PER-FOLD OMNIBUS (decoupled-early) @ $(date +%H:%M:%S) ====="
  cat results/final/perfold_8model_omnibus.csv 2>/dev/null; echo
  cat results/final/perfold_8model_significance.csv 2>/dev/null
} > "$LOG/EIGHTMODEL.txt" 2>&1
echo "[8model-watch] wrote $LOG/EIGHTMODEL.txt"
tail -40 "$LOG/perfold_8model_early.log"
