"""Temperature-scaling calibration on BBBP.

Per 项目调研汇总.md v2 §4.2: neural classifiers are typically over-confident.
We train a Morgan-FP MLP on BBBP, then fit a single temperature T on a
calibration split and measure Expected Calibration Error (ECE) before/after.
BBBP is the ideal showcase: train ~76% positive vs real-world ~98% negative.

Run: python scripts/calibrate_bbbp.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from src.data.featurizers import batch_morgan
from src.data.loaders import load_dataset
from src.utils.logging_setup import configure_logging
from src.utils.metrics import classification_metrics
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


class MorganMLP(nn.Module):
    def __init__(self, in_dim: int = 2048, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logit


def expected_calibration_error(
    probs: NDArray[np.float64], labels: NDArray[np.int_], n_bins: int = 10
) -> float:
    """Binary ECE: bin by predicted confidence, sum |acc - conf| weighted."""
    conf = np.maximum(probs, 1 - probs)        # confidence of predicted class
    pred = (probs >= 0.5).astype(int)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit scalar T minimizing BCE on a held-out calibration set (LBFGS)."""
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=100)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3).item())


def dump_calibration_csv(probs_before, probs_after, labels, T, ece_before, ece_after, roc, n_bins=10):
    """Persist reliability-diagram bins + summary so the figure can be drawn
    elsewhere in a unified (scienceplots) style. Writes:
      results/bbbp_calibration.csv       (kind, conf, acc, n)
      results/bbbp_calibration_meta.csv  (ece_before, ece_after, T, roc_auc, n_test)
    """
    def bins(probs):
        conf = np.maximum(probs, 1 - probs)
        correct = ((probs >= 0.5).astype(int) == labels).astype(float)
        edges = np.linspace(0, 1, n_bins + 1)
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() == 0:
                continue
            out.append({"conf": float(conf[m].mean()), "acc": float(correct[m].mean()), "n": int(m.sum())})
        return out

    rows = []
    for kind, probs in [("before", probs_before), ("after", probs_after)]:
        for r in bins(probs):
            r["kind"] = kind
            rows.append(r)
    pd.DataFrame(rows).to_csv("results/bbbp_calibration.csv", index=False)
    pd.DataFrame([{"ece_before": float(ece_before), "ece_after": float(ece_after),
                   "T": float(T), "roc_auc": float(roc), "n_test": int(len(labels))}]
                 ).to_csv("results/bbbp_calibration_meta.csv", index=False)
    logger.info("wrote results/bbbp_calibration.csv + _meta.csv")


def reliability_diagram(probs_before, probs_after, labels, out_path: Path, n_bins=10):
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    def bin_stats(probs):
        conf = np.maximum(probs, 1 - probs)
        pred = (probs >= 0.5).astype(int)
        correct = (pred == labels).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() == 0:
                continue
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
        return xs, ys

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0.5, 1], [0.5, 1], "k--", label="perfect calibration")
    for probs, name, mk in [(probs_before, "before T", "o"), (probs_after, "after T", "s")]:
        xs, ys = bin_stats(probs)
        ax.plot(xs, ys, marker=mk, label=name)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("BBBP reliability diagram (temperature scaling)")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info(f"saved {out_path}")


def main() -> int:
    configure_logging("INFO")
    set_all_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = load_dataset("bbbp_cls")
    xtr, itr = batch_morgan(split.train["smiles"].tolist())
    xva, iva = batch_morgan(split.val["smiles"].tolist())
    ytr = split.train["label"].to_numpy()[itr].astype(np.float32)
    yva = split.val["label"].to_numpy()[iva].astype(int)

    Xtr = torch.tensor(xtr, dtype=torch.float32, device=device)
    Ytr = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva = torch.tensor(xva, dtype=torch.float32, device=device)

    # ---- Train MLP ----
    model = MorganMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True)
    for _ in range(80):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits_va = model(Xva).cpu()

    # ---- Split val -> calibration / test halves ----
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(yva))
    half = len(perm) // 2
    cal_idx, test_idx = perm[:half], perm[half:]
    logits_cal = logits_va[cal_idx]
    y_cal = torch.tensor(yva[cal_idx], dtype=torch.float32)
    logits_test = logits_va[test_idx]
    y_test = yva[test_idx]

    # ---- Before calibration ----
    probs_before = torch.sigmoid(logits_test).numpy()
    ece_before = expected_calibration_error(probs_before, y_test)

    # ---- Fit temperature on calibration set, apply to test ----
    T = fit_temperature(logits_cal, y_cal)
    probs_after = torch.sigmoid(logits_test / T).numpy()
    ece_after = expected_calibration_error(probs_after, y_test)

    metrics = classification_metrics(y_test, probs_before)

    print("\n" + "=" * 50)
    print("BBBP TEMPERATURE-SCALING CALIBRATION")
    print("=" * 50)
    print(f"n_test            = {len(y_test)}")
    print(f"ROC-AUC           = {metrics['ROC-AUC']:.3f}  (unchanged by T)")
    print(f"fitted temperature= {T:.3f}  ({'over' if T > 1 else 'under'}-confident)")
    print(f"ECE before        = {ece_before:.4f}")
    print(f"ECE after         = {ece_after:.4f}")
    print(f"ECE reduction     = {100*(ece_before-ece_after)/max(ece_before,1e-9):+.1f}%")

    # CSV is the canonical output (plotted in unified style by make_benchmark_figures).
    dump_calibration_csv(probs_before, probs_after, y_test, T, ece_before, ece_after,
                         metrics["ROC-AUC"])
    # Legacy in-script PNG is best-effort: some servers have a broken matplotlib
    # (libstdc++ CXXABI mismatch). Never let a plotting failure fail the data run.
    try:
        reliability_diagram(
            probs_before, probs_after, y_test,
            Path("results/figures/bbbp_calibration.png"),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"skipped legacy reliability PNG (plot backend unavailable): {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
