#!/usr/bin/env bash
# Download foundation model weights from HuggingFace.
# Per CLAUDE.md §4 / 项目调研汇总.md §4.1.
# Set HF_HOME to control cache location; default ~/.cache/huggingface.

set -euo pipefail

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "ERROR: huggingface-cli not found. Install: pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

echo "==> Downloading foundation model weights..."
echo "    Cache dir: ${HF_HOME:-$HOME/.cache/huggingface}"
echo

# Tier 1: must-have for baseline benchmark
huggingface-cli download ibm-research/MoLFormer-XL-both-10pct
huggingface-cli download ibm-research/materials.smi-ted
huggingface-cli download DeepChem/ChemBERTa-77M-MLM

# Tier 2: ChemFM (large, slow download — 1B 4GB, 3B 12GB)
read -p "Download ChemFM-1B (~4GB)? [y/N] " yn
case "$yn" in [yY]*) huggingface-cli download ChemFM/ChemFM-1B ;; esac

read -p "Download ChemFM-3B (~12GB)? [y/N] " yn
case "$yn" in [yY]*) huggingface-cli download ChemFM/ChemFM-3B ;; esac

# Tier 3: LLM (optional, for LLM route in 项目调研汇总.md §5 方案2)
read -p "Download Qwen2.5-7B-Instruct (~15GB)? [y/N] " yn
case "$yn" in [yY]*) huggingface-cli download Qwen/Qwen2.5-7B-Instruct ;; esac

read -p "Download ChemLLM-7B-Chat (~14GB)? [y/N] " yn
case "$yn" in [yY]*) huggingface-cli download AI4Chem/ChemLLM-7B-Chat ;; esac

echo "==> Done. Models will be loaded from cache by transformers."
