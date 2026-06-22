#!/usr/bin/env bash
# Queue the 6-regression-dataset conformal sweep (PAPER track).
#   1. TabPFN on the 4 small datasets (CPU, now) -> runs/ format.
#   2. WAIT for the BACE deep-CV (B) to release the GPUs.
#   3. Train MolFormer / ChemFM / Uni-Mol on QM7 + QM8 (GPU, capped).
#   4. Run run_conformal_full.py over the 6 datasets x 5 models.
# nohup this so it idle-waits then runs: nohup bash scripts/sweep_conformal_paper.sh > runs/conformal_paper/launcher.log 2>&1 &
set -uo pipefail
cd ~/ppi_bias
PY="${PY:-${CONDA_PREFIX:-$HOME/.conda/envs/drug}/bin/python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-$HOME/.conda/envs/drug}/lib:${LD_LIBRARY_PATH:-}"
LOG=runs/conformal_paper; mkdir -p "$LOG"

echo "[1] TabPFN on small regression datasets (CPU; skip if already done)..."
for d in esol freesolv lipophilicity bace; do
  [ -f "runs/manual_${d}_tabpfn_seed42/test_predictions.csv" ] && { echo "    tabpfn $d exists, skip"; continue; }
  CUDA_VISIBLE_DEVICES="" SCIPY_ARRAY_API=1 $PY scripts/run_tabpfn_to_runs.py "$d" \
      > "$LOG/tabpfn_${d}.log" 2>&1 && echo "    tabpfn $d OK" || echo "    tabpfn $d FAIL (see $LOG/tabpfn_${d}.log)"
done

echo "[2] waiting for BACE deep-CV (dataset=cvfold) to finish before taking GPUs..."
while pgrep -f "[d]ataset=cvfold" >/dev/null 2>&1; do sleep 120; done
echo "    BACE deep-CV done at $(date +%H:%M:%S); training QM backbones."

echo "[3] GPU backbones on QM7+QM8 (molformer/chemfm/unimol)..."
N_CONC=6; N_GPUS=8; idx=0; running=0
for d in qm7 qm8; do
  for m in molformer chemfm unimol; do
    if [ "$running" -ge "$N_CONC" ]; then wait -n; running=$((running - 1)); fi
    gpu=$(( idx % N_GPUS )); rd="runs/manual_${d}_${m}_seed42"; rm -rf "$rd"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m src.train dataset=$d model=$m seed=42 save_test_preds=true \
        hydra.run.dir="$rd" > "$LOG/${d}_${m}.log" 2>&1 &
    idx=$((idx + 1)); running=$((running + 1)); sleep 3
  done
done
wait
echo "    QM backbone training done at $(date +%H:%M:%S)."

echo "[4] conformal sweep over 6 datasets x 5 models..."
$PY scripts/run_conformal_full.py > "$LOG/conformal.log" 2>&1 && echo "    conformal done" || echo "    conformal FAIL"

echo "[5] non-conformal UQ baselines (ensemble-Gaussian vs conformal)..."
$PY scripts/run_uq_baselines.py > "$LOG/uq_baselines.log" 2>&1 && echo "    UQ done" || echo "    UQ FAIL (see $LOG/uq_baselines.log)"

echo "[6] cross-dataset Demsar CD diagram..."
$PY scripts/make_cd_crossdataset.py > "$LOG/cd.log" 2>&1 && echo "    CD done" || echo "    CD FAIL (see $LOG/cd.log)"

echo "PAPER SWEEP (conformal + UQ + CD) DONE at $(date +%H:%M:%S)"
tail -20 "$LOG/conformal.log"; echo "---UQ---"; tail -10 "$LOG/uq_baselines.log"; echo "---CD---"; tail -6 "$LOG/cd.log"
