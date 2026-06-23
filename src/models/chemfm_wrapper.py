"""ChemFM-3B LoRA fine-tune wrapper.

Per CLAUDE.md §4 / 项目调研汇总.md §4.1: ChemFM-3B is current SMILES-class SOTA
on ESOL/FreeSolv/BBBP/BACE. 3B Llama-architecture model.

Per CLAUDE.md §8.9: LoRA mandatory (full fine-tune of 3B model = 24GB+ optim states).
Per CLAUDE.md §8.8: bf16 on transformers backbone; head stays fp32.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.loaders import Split
from src.utils.metrics import metrics_for_task

logger = logging.getLogger(__name__)


class CausalSmilesDataset(Dataset):
    """SMILES dataset for causal-LM-based models.

    Important: ChemFM is decoder-only, so we need to know where the "last real
    token" is per example to pool. Returns input_ids + attention_mask + label.
    """

    def __init__(
        self,
        smiles: list[str],
        labels: list[float],
        tokenizer,
        max_len: int = 256,
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
            add_special_tokens=True,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(float(self.labels[i]), dtype=torch.float32),
        }


class ChemFMRegressor(nn.Module):
    """ChemFM-3B backbone + LoRA + regression/classification head."""

    def __init__(
        self,
        backbone_id: str = "ChemFM/ChemFM-3B",
        peft_cfg=None,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Trigger: 3B-param model in fp32 = 12GB weights + 24GB optim states.
        # Why:     bf16 keeps weights in 6GB; LoRA keeps optim tiny.
        # Outcome: full fine-tune fits in <24GB on an 80GB A100 card.
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_id,
            torch_dtype=torch.bfloat16,
        )
        # Enable gradient checkpointing to save activation memory.
        self.backbone.gradient_checkpointing_enable()
        # Disable cache (incompatible with gradient checkpointing).
        self.backbone.config.use_cache = False

        # Apply LoRA.
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=peft_cfg.r,
            lora_alpha=peft_cfg.alpha,
            lora_dropout=peft_cfg.dropout,
            target_modules=list(peft_cfg.target_modules),
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        # Head in fp32 for numerical stability.
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last hidden state from the base model: [B, L, hidden].
        last_hidden = out.hidden_states[-1]

        # Pool by last non-pad token (causal LM convention).
        # attention_mask: [B, L], 1 = real, 0 = pad.
        seq_lens = attention_mask.sum(dim=1) - 1  # [B]
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = last_hidden[batch_idx, seq_lens]  # [B, hidden]

        return self.head(pooled.float()).squeeze(-1)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_chemfm(
    cfg,
    split: Split,
    run_dir: Path,
    device: torch.device | None = None,
) -> dict[str, float]:
    """LoRA fine-tune ChemFM-3B on a split, return val metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"ChemFM on device={device}, backbone={cfg.model.backbone_id}")

    # ---- Tokenizer + model ----
    tok = AutoTokenizer.from_pretrained(cfg.model.backbone_id)
    # Causal-LM tokenizers often lack a pad token; reuse EOS.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = ChemFMRegressor(
        backbone_id=cfg.model.backbone_id,
        peft_cfg=cfg.model.peft,
    ).to(device)
    n_trainable = model.trainable_param_count()
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {n_trainable/1e6:.2f}M / {n_total/1e9:.2f}B "
        f"({100*n_trainable/n_total:.2f}%)"
    )

    # ---- Dataloaders ----
    bs = cfg.model.batch_size
    grad_accum = cfg.model.get("gradient_accumulation_steps", 1)
    max_len = cfg.model.max_seq_length

    def make_loader(df, shuffle: bool):
        ds = CausalSmilesDataset(
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

    # ---- Optimizer + scheduler ----
    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
    )
    total_steps = (len(train_loader) // grad_accum) * cfg.model.epochs
    if total_steps >= 20:
        sched = OneCycleLR(
            optim,
            max_lr=cfg.model.lr,
            total_steps=total_steps,
            pct_start=cfg.model.get("warmup_ratio", 0.1),
        )
    else:
        sched = ConstantLR(optim, factor=1.0, total_iters=total_steps)

    # ---- Loss ----
    if split.task == "regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    # ---- Train ----
    best_metric: float | None = None
    best_state: dict | None = None
    best_metrics: dict[str, float] = {}
    metric_key = split.metric_main
    higher_is_better = metric_key in {
        "PearsonR", "SpearmanRho", "ROC-AUC", "PR-AUC", "F1", "MCC"
    }

    for epoch in range(1, cfg.model.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_seen = 0

        optim.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            pred = model(input_ids, attention_mask)
            loss = loss_fn(pred, y) / grad_accum
            loss.backward()
            train_loss += loss.item() * y.size(0) * grad_accum
            n_seen += y.size(0)

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optim.step()
                sched.step()
                optim.zero_grad()

        train_loss /= n_seen

        # ---- Eval ----
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                pred = model(input_ids, attention_mask)
                if split.task == "classification":
                    pred = torch.sigmoid(pred)
                preds.append(pred.float().cpu().numpy())
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
            # CLAUDE.md §8.5: save state_dict only. LoRA + head -- skip frozen base.
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if "lora_" in k or "head" in k
            }
            best_metrics = val_metrics
            best_y_pred = y_pred.copy()
            best_y_true = y_true.copy()

        logger.info(
            f"epoch {epoch:3d}/{cfg.model.epochs} "
            f"train_loss={train_loss:.4f} val_{metric_key}={cur:.4f} "
            f"(best={best_metric:.4f}) [{time.time()-t0:.1f}s]"
        )

    # ---- Save ----
    assert best_state is not None
    torch.save(best_state, run_dir / "model.pt")
    import pandas as pd  # noqa: PLC0415
    pd.DataFrame({
        "smiles": split.val["smiles"].tolist()[: len(best_y_pred)],
        "y_true": best_y_true,
        "y_score": best_y_pred,
    }).to_csv(run_dir / "val_predictions.csv", index=False)

    # §16.4 one-time test predictions for conformal/cliff studies.
    if cfg.get("save_test_preds", False) and len(split.test) > 0:
        model.load_state_dict(best_state, strict=False)  # best_state = LoRA+head only
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
                tpreds.append(p.float().cpu().numpy())
        t_pred = np.concatenate(tpreds)
        pd.DataFrame({
            "smiles": split.test["smiles"].tolist()[: len(t_pred)],
            "y_true": split.test["label"].to_numpy()[: len(t_pred)],
            "y_score": t_pred,
        }).to_csv(run_dir / "test_predictions.csv", index=False)

    logger.info(f"Best val_{metric_key}={best_metric:.4f}; saved LoRA+head + predictions")
    return best_metrics
