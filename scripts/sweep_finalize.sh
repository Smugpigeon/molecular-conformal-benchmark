#!/usr/bin/env bash
# Final consolidation (last in the chain). Waits for GAP6 + deep-CV to finish, builds the 8-model
# per-fold omnibus from the deep-CV folds, and dumps every final table to one file.
set -uo pipefail
cd ~/ppi_bias
PY="${PY:-${CONDA_PREFIX:-$HOME/.conda/envs/drug}/bin/python}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-$HOME/.conda/envs/drug}/lib:${LD_LIBRARY_PATH:-}"
LOG=runs/finalize; mkdir -p "$LOG"

echo "[finalize] waiting for GAP6 + deep-CV..."
while pgrep -f "[s]weep_gap6" >/dev/null 2>&1; do sleep 120; done
while pgrep -f "[d]ataset=cvfold" >/dev/null 2>&1; do sleep 120; done

echo "[finalize] building 8-model per-fold omnibus..."
$PY scripts/make_perfold_8model.py > "$LOG/perfold_8model.log" 2>&1 && echo "  8-model omnibus done" || echo "  8-model FAIL (see $LOG/perfold_8model.log)"

{
  echo "===== 8-MODEL PER-FOLD OMNIBUS ====="; cat results/final/perfold_8model_omnibus.csv 2>/dev/null; echo
  cat results/final/perfold_8model_significance.csv 2>/dev/null
  echo; echo "===== 6-DATASET CONFORMAL (width + coverage) ====="; cat results/final/conformal_full.csv 2>/dev/null
  echo; echo "===== MULTI-SEED CONFORMAL ERROR BARS (GAP6) ====="; cat results/final/conformal_multiseed.csv 2>/dev/null
  echo; echo "===== UQ BASELINES (ensemble) ====="; cat results/final/uq_baselines.csv 2>/dev/null
  echo; echo "===== UQ EXTRA (Gaussian-NLL / MC-Dropout / TempScale, MLP) ====="; cat results/final/uq_extra.csv 2>/dev/null
  echo; echo "===== DEEP MC-DROPOUT (MolFormer real FM) ====="; cat results/final/mc_dropout_deep.csv 2>/dev/null
  echo; echo "===== GAP4 DUAL SPLIT (random vs scaffold) ====="; cat results/final/gap4_dual_split.csv 2>/dev/null
  echo; echo "===== CROSS-DATASET DEMSAR CD RANKS ====="; cat results/final/cd_crossdataset_ranks.csv 2>/dev/null
} > "$LOG/CONSOLIDATED.txt" 2>&1

echo "FINALIZE DONE at $(date +%H:%M:%S)"
echo "---- CONSOLIDATED.txt ----"; cat "$LOG/CONSOLIDATED.txt"
