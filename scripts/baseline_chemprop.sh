#!/usr/bin/env bash
# Chemprop v2 baseline on all 6 datasets.
# Uses the chemprop CLI directly (no Python wrapper yet).
# Reference: https://github.com/chemprop/chemprop

set -euo pipefail

if ! command -v chemprop >/dev/null 2>&1; then
    echo "ERROR: chemprop CLI not found. Activate env first: conda activate drug"
    exit 1
fi

DATA_DIR="data"
OUT_ROOT="runs/chemprop"
mkdir -p "$OUT_ROOT"

declare -A DATASETS=(
    [esol]="ESOL.split.csv:regression"
    [freesolv]="FreeSol.split.csv:regression"
    [lipophilicity]="Lipophilicity.split.csv:regression"
    [bace]="BACE.split.csv:regression"
    [bbbp]="BBBP.split.csv:classification"
    [gsht]="GSHt-419.split.csv:regression"
)

for name in "${!DATASETS[@]}"; do
    IFS=':' read -r filename task <<< "${DATASETS[$name]}"
    save_dir="$OUT_ROOT/$name"
    echo "==> Training Chemprop on $name ($task)"
    chemprop train \
        --data-path "$DATA_DIR/$filename" \
        --task-type "$task" \
        --smiles-columns smiles \
        --target-columns label \
        --split-type predetermined \
        --split-key-molecule 0 \
        --save-dir "$save_dir" \
        --num-workers 8 \
        --epochs 50 \
        --batch-size 50 \
        || echo "WARN: chemprop failed on $name; continuing"
    echo
done

echo "==> Chemprop baselines complete."
