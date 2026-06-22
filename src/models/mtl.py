"""Masked multi-task learning across the physicochemical trio.

ESOL / FreeSolv / Lipophilicity have DISJOINT molecule sets (项目调研汇总.md
v2 §4.1). The ONLY valid multi-task scheme is:
  - shared encoder (Morgan FP -> latent)
  - one regression head per task
  - each batch comes from a single task -> only that head + the encoder update
  - round-robin over tasks
  - labels z-standardized per task using TRAIN stats only (CLAUDE.md §7.3)

We compare single-task (separate encoder+head per task) vs multi-task and
report per-task val RMSE. Per CLAUDE.md §16.1 we EXPECT and report negative
transfer where it happens — multi-task is not assumed to win.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from src.data.featurizers import batch_morgan
from src.data.loaders import load_dataset
from src.utils.metrics import regression_metrics

logger = logging.getLogger(__name__)

PHYSCHEM_TRIO = ["esol", "freesolv", "lipophilicity"]


@dataclass
class TaskData:
    x_train: torch.Tensor
    y_train: torch.Tensor  # standardized
    x_val: torch.Tensor
    y_val_raw: NDArray[np.float64]  # original units
    mean: float
    std: float


def _encoder(in_dim: int, latent: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(512, latent), nn.ReLU(),
    )


class MTLModel(nn.Module):
    """Shared encoder + per-task linear heads."""

    def __init__(self, tasks: list[str], in_dim: int = 2048, latent: int = 256, dropout: float = 0.2):
        super().__init__()
        self.encoder = _encoder(in_dim, latent, dropout)
        self.heads = nn.ModuleDict({t: nn.Linear(latent, 1) for t in tasks})

    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        return self.heads[task](self.encoder(x)).squeeze(-1)


class STLModel(nn.Module):
    """Single-task: own encoder + head (no sharing)."""

    def __init__(self, in_dim: int = 2048, latent: int = 256, dropout: float = 0.2):
        super().__init__()
        self.encoder = _encoder(in_dim, latent, dropout)
        self.head = nn.Linear(latent, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


def _prepare(data_root: str, device: torch.device) -> dict[str, TaskData]:
    """Featurize + z-standardize each task (train stats only)."""
    out: dict[str, TaskData] = {}
    for task in PHYSCHEM_TRIO:
        split = load_dataset(task, data_root=data_root)
        xtr, itr = batch_morgan(split.train["smiles"].tolist())
        xva, iva = batch_morgan(split.val["smiles"].tolist())
        ytr = split.train["label"].to_numpy()[itr].astype(np.float64)
        yva = split.val["label"].to_numpy()[iva].astype(np.float64)
        mean, std = float(ytr.mean()), float(ytr.std())
        out[task] = TaskData(
            x_train=torch.tensor(xtr, dtype=torch.float32, device=device),
            y_train=torch.tensor((ytr - mean) / std, dtype=torch.float32, device=device),
            x_val=torch.tensor(xva, dtype=torch.float32, device=device),
            y_val_raw=yva,
            mean=mean,
            std=std,
        )
    return out


def _eval(pred_std: torch.Tensor, td: TaskData) -> dict[str, float]:
    pred = pred_std.detach().cpu().numpy() * td.std + td.mean  # back to original units
    return regression_metrics(td.y_val_raw, pred)


def train_single_task(
    data: dict[str, TaskData], epochs: int, lr: float, bs: int, device: torch.device
) -> dict[str, dict[str, float]]:
    """Train a separate model per task."""
    results = {}
    for task, td in data.items():
        model = STLModel().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        loader = DataLoader(TensorDataset(td.x_train, td.y_train), batch_size=bs, shuffle=True)
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            results[task] = _eval(model(td.x_val), td)
    return results


def train_multi_task(
    data: dict[str, TaskData], epochs: int, lr: float, bs: int, device: torch.device
) -> dict[str, dict[str, float]]:
    """Shared encoder + per-task heads, round-robin batches, masked loss."""
    tasks = list(data)
    model = MTLModel(tasks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    loaders = {
        t: DataLoader(TensorDataset(d.x_train, d.y_train), batch_size=bs, shuffle=True)
        for t, d in data.items()
    }
    for _ in range(epochs):
        model.train()
        iters = {t: iter(ld) for t, ld in loaders.items()}
        # Round-robin until the largest loader is exhausted.
        n_steps = max(len(ld) for ld in loaders.values())
        for _step in range(n_steps):
            for t in tasks:
                try:
                    xb, yb = next(iters[t])
                except StopIteration:
                    iters[t] = iter(loaders[t])
                    xb, yb = next(iters[t])
                opt.zero_grad()
                # Only task t's head + the shared encoder receive gradient.
                loss_fn(model(xb, t), yb).backward()
                opt.step()

    model.eval()
    results = {}
    with torch.no_grad():
        for t, td in data.items():
            results[t] = _eval(model(td.x_val, t), td)
    return results


def run_mtl_experiment(
    data_root: str = "data",
    epochs: int = 60,
    lr: float = 1e-3,
    bs: int = 64,
    device: torch.device | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run single-task and multi-task, return {scheme: {task: metrics}}."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _prepare(data_root, device)
    logger.info(f"Single-task training (trio) on {device} ...")
    stl = train_single_task(data, epochs, lr, bs, device)
    logger.info(f"Multi-task training (shared encoder) on {device} ...")
    mtl = train_multi_task(data, epochs, lr, bs, device)
    return {"single_task": stl, "multi_task": mtl}
