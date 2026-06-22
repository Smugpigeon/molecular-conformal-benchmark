#!/usr/bin/env bash
# Random Forest + Morgan FP baseline on all 6 datasets.
# Per CLAUDE.md §5: `make baseline-rf`
# Per CLAUDE.md §8.2: 3 seeds for mean ± std.

set -euo pipefail

DATASETS=(esol freesolv lipophilicity bace bbbp gsht)
SEEDS=(42 1337 2024)

echo "==> Running RF baseline on ${#DATASETS[@]} datasets x ${#SEEDS[@]} seeds"
echo "    Datasets: ${DATASETS[*]}"
echo "    Seeds:    ${SEEDS[*]}"
echo

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "==> [$dataset / seed=$seed]"
        python -m src.train dataset="$dataset" model=rf seed="$seed"
        echo
    done
done

echo "==> All RF baselines complete. Results under runs/."
echo "    Aggregate with: python -m scripts.aggregate_runs --model rf"
