"""MolFormer-XL fine-tune wrapper.

Per CLAUDE.md §4 / 项目调研汇总.md §4.1: MolFormer-XL (IBM, Nat Mach Intell 2022)
is the most reliable foundation model baseline. 47M params, fits easily in a
single GPU. Full fine-tune (no LoRA needed) is fine.

Per CLAUDE.md §8.8: bf16 OK for transformer; fp16 risky.
Per CLAUDE.md §8.1: caller of train_molformer() is responsible for set_all_seeds.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.data.loaders import Split
from src.utils.metrics import metrics_for_task

logger = logging.getLogger(__name__)


class SmilesDataset(Dataset):
    """SMILES + label dataset with on-the-fly tokenization."""

    def __init__(
        self,
        smiles: list[str],
        labels: list[float],
        tokenizer,
        max_len: int = 128,
    ):
        self.smiles = smiles
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        enc = self.tok(
            self.smiles[i],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(float(self.labels[i]), dtype=torch.float32),
        }


class MolFormerRegressor(nn.Module):
    """MolFormer-XL backbone + small MLP head."""

    def __init__(
        self,
        backbone_id: str = "ibm-research/MoLFormer-XL-both-10pct",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_id, trust_remote_code=True)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Masked-mean pool last hidden state.
        # Trigger: pooler_output may not exist depending on HF version.
        # Why:     mean pool is robust across MolFormer variants.
        # Outcome: deterministic [B, hidden] representation.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def train_molformer(
    cfg,
    split: Split,
    run_dir: Path,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Fine-tune MolFormer-XL on a split, return val metrics.

    Saves model.state_dict + training log to run_dir.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"MolFormer on device={device}, backbone={cfg.model.backbone_id}")

    # ---- 1. Tokenizer + model ----
    tok = AutoTokenizer.from_pretrained(cfg.model.backbone_id, trust_remote_code=True)
    model = MolFormerRegressor(backbone_id=cfg.model.backbone_id).to(device)
    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M"
    )

    # ---- 2. Dataloaders ----
    bs = cfg.model.batch_size
    max_len = cfg.model.max_seq_length

    def make_loader(df, shuffle: bool):
        ds = SmilesDataset(
            df["smiles"].tolist(),
            df["label"].astype(float).tolist(),
            tok,
            max_len=max_len,
        )
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=cfg.model.get("num_workers", 2),
            pin_memory=True,
        )

    train_loader = make_loader(split.train, shuffle=True)
    val_loader = make_loader(split.val, shuffle=False)

    # ---- 3. Optimizer + scheduler ----
    no_decay = {"bias", "LayerNorm.weight"}
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay_params if any(nd in n for nd in no_decay) else decay_params).append(p)

    optim = AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.model.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.model.lr,
    )
    total_steps = len(train_loader) * cfg.model.epochs
    # Trigger: very short runs (e.g., DRY=1 sanity, total_steps < 10).
    # Why:     OneCycleLR with pct_start=0.1 zero-divs when warmup < 1 step.
    # Outcome: fall back to constant LR — fine because short runs are just sanity checks.
    if total_steps >= 20:
        sched = OneCycleLR(
            optim,
            max_lr=cfg.model.lr,
            total_steps=total_steps,
            pct_start=cfg.model.warmup_ratio,
        )
    else:
        from torch.optim.lr_scheduler import ConstantLR  # noqa: PLC0415
        sched = ConstantLR(optim, factor=1.0, total_iters=total_steps)

    # ---- 4. Loss ----
    if split.task == "regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    # ---- 5. Train loop ----
    best_metric: float | None = None
    best_state: dict | None = None
    metric_key = split.metric_main
    higher_is_better = metric_key in {"PearsonR", "SpearmanRho", "ROC-AUC", "PR-AUC", "F1", "MCC"}

    for epoch in range(1, cfg.model.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optim.zero_grad()
            pred = model(input_ids, attention_mask)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            train_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
        train_loss /= n_seen

        # Val
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                pred = model(input_ids, attention_mask)
                if split.task == "classification":
                    pred = torch.sigmoid(pred)
                preds.append(pred.cpu().numpy())
                ys.append(batch["label"].numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(ys)
        val_metrics = metrics_for_task(split.task, y_true, y_pred)

        cur = val_metrics[metric_key]
        is_best = (
            best_metric is None
            or (higher_is_better and cur > best_metric)
            or (not higher_is_better and cur < best_metric)
        )
        if is_best:
            best_metric = cur
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics
            best_y_pred = y_pred.copy()
            best_y_true = y_true.copy()

        logger.info(
            f"epoch {epoch:3d}/{cfg.model.epochs} "
            f"train_loss={train_loss:.4f} val_{metric_key}={cur:.4f} "
            f"(best={best_metric:.4f}) [{time.time()-t0:.1f}s]"
        )

    # ---- 6. Save best ----
    assert best_state is not None
    torch.save(best_state, run_dir / "model.pt")
    # Per-molecule val predictions (val_loader is shuffle=False -> aligns with split.val).
    import pandas as pd  # noqa: PLC0415
    pd.DataFrame({
        "smiles": split.val["smiles"].tolist()[: len(best_y_pred)],
        "y_true": best_y_true,
        "y_score": best_y_pred,
    }).to_csv(run_dir / "val_predictions.csv", index=False)

    # §16.4 one-time test predictions for conformal/cliff studies.
    if cfg.get("save_test_preds", False) and len(split.test) > 0:
        model.load_state_dict(best_state)
        model.eval()
        test_loader = make_loader(split.test, shuffle=False)
        tpreds = []
        with torch.no_grad():
            for batch in test_loader:
                ii = batch["input_ids"].to(device, non_blocking=True)
                am = batch["attention_mask"].to(device, non_blocking=True)
                p = model(ii, am)
                if split.task == "classification":
                    p = torch.sigmoid(p)
                tpreds.append(p.cpu().numpy())
        t_pred = np.concatenate(tpreds)
        pd.DataFrame({
            "smiles": split.test["smiles"].tolist()[: len(t_pred)],
            "y_true": split.test["label"].to_numpy()[: len(t_pred)],
            "y_score": t_pred,
        }).to_csv(run_dir / "test_predictions.csv", index=False)

    logger.info(f"Best val_{metric_key}={best_metric:.4f}; saved state_dict + predictions")
    return best_metrics
