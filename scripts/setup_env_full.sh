#!/usr/bin/env bash
# Install deep learning deps into existing 'drug' env.
# Run AFTER setup_env_minimal.sh has succeeded.
# unimol-tools is installed separately (has tricky native deps).
set -euo pipefail

source /nfs_share/miniconda3/etc/profile.d/conda.sh
conda activate drug

echo "[1/4] $(date) PyTorch 2.4 + CUDA 12.1 (pip wheel, fastest)..."
pip install --quiet \
    --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.*" torchvision torchaudio
echo "  -> $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())')"

echo "[2/4] $(date) Chemprop v2 + chemistry stack..."
pip install --quiet "chemprop>=2.0.4"
echo "  -> chemprop $(python -c 'import chemprop; print(chemprop.__version__)')"

echo "[3/4] $(date) Transformers stack (HF + LoRA + accelerate)..."
pip install --quiet \
    "transformers>=4.44" \
    "peft>=0.12" \
    "accelerate>=0.33" \
    "datasets>=2.20" \
    "pytorch-lightning>=2.4" \
    "wandb>=0.17"
echo "  -> transformers $(python -c 'import transformers; print(transformers.__version__)')"

echo "[4/4] $(date) Quick GPU sanity..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
n = torch.cuda.device_count()
print(f'CUDA OK: {n} GPU(s)')
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'  [{i}] {p.name} {p.total_memory/1e9:.1f}GB')
# Smoke test: tensor on GPU 0
x = torch.randn(100, 100, device='cuda:0')
y = x @ x.T
assert y.is_cuda
print('GPU matmul OK')
"

echo "$(date) FULL ENV READY"
