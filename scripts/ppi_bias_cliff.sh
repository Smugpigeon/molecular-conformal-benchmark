#!/usr/bin/env bash
# ppi_bias cliff sweep: neural backbones x 30 MoleculeACE datasets.
# Default molformer + unimol (the two SemiMol 2026 explicitly did NOT evaluate).
# Logs/run dirs under the ppi_bias project.

set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

N_GPUS=${N_GPUS:-8}
MODELS="${MODELS:-molformer unimol}"
LOGDIR=runs/ppi_bias_cliff
mkdir -p "$LOGDIR"

DATASETS=$(ls data/external/CHEMBL*.csv | xargs -n1 basename | sed 's/\.csv$//')
n=$(echo "$DATASETS" | wc -w)
echo "==> ppi_bias cliff sweep: [$MODELS] x $n MoleculeACE datasets across $N_GPUS GPUs"

idx=0
for model in $MODELS; do
    for d in $DATASETS; do
        while [ "$(jobs -r 2>/dev/null | wc -l)" -ge "$N_GPUS" ]; do sleep 10; done
        gpu=$(( idx % N_GPUS ))
        CUDA_VISIBLE_DEVICES=$gpu \
            python scripts/run_cliff_neural.py --model "$model" --dataset "$d" \
            > "$LOGDIR/${model}_${d}.log" 2>&1 &
        idx=$((idx + 1)); sleep 2
    done
done
wait
echo "==> $(date) ppi_bias cliff sweep done ($idx runs)"
