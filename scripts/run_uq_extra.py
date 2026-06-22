"""GAP3-rest: non-conformal UQ baselines on a learned mean-variance MLP, each turned into a
90% prediction interval and compared to split-conformal on the SAME model:

  * Gaussian-NLL (heteroscedastic mean+variance head)  -> mean +/- 1.645*sigma
  * MC-Dropout (N stochastic forward passes, dropout on at inference)
  * Temperature scaling (post-hoc: scale sigma on val so val coverage = nominal, apply to test)
  * Split-conformal on the MLP mean (the principled reference; valid by construction)

Message (GAP3): native UQ (Gaussian-NLL, MC-Dropout) is typically MISCALIBRATED (empirical
coverage != 0.90); temperature scaling partially recovers it; split-conformal is valid by
construction -> conformal ADDS value. This MLP is the locally-testable proxy whose MC-Dropout +
temp-scaling + coverage/width logic the deep-model wrappers reuse on the server.

Run: /opt/anaconda3/bin/python3 scripts/run_uq_extra.py [dataset=bace_clean] [alpha=0.1]
Output: results/final/uq_extra.csv (one row per UQ method, appended per dataset).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import load_dataset  # noqa: E402
from src.eval.conformal import _conformal_quantile  # noqa: E402  (finite-sample, method="higher")
from src.utils.seed import set_all_seeds  # noqa: E402

FINAL = Path("results/final"); FINAL.mkdir(parents=True, exist_ok=True)


def morgan(smiles, n=2048):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator as rfg
    RDLogger.DisableLog("rdApp.*")
    gen = rfg.GetMorganGenerator(radius=2, fpSize=n)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


class MeanVarMLP(nn.Module):
    """Morgan -> hidden -> [mean, log_var]. Dropout kept so the same net supports MC-Dropout."""

    def __init__(self, d, h1=256, h2=128, p=0.2):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(p),
                                  nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p))
        self.head = nn.Linear(h2, 2)

    def forward(self, x):
        o = self.head(self.body(x))
        return o[:, 0], o[:, 1]  # mean, log_var


def nll(mean, log_var, y):
    return 0.5 * (log_var + (y - mean) ** 2 * torch.exp(-log_var)).mean()


def _cov_width(y, mean, half):
    return float((np.abs(y - mean) <= half).mean()), float(2 * np.mean(half))


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "bace_clean"
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    z = 1.6448536269514722  # 90% two-sided Gaussian (alpha=0.1)
    set_all_seeds(42)
    sp = load_dataset(ds)
    Xtr, Xva, Xte = (morgan(sp.train.smiles.tolist()), morgan(sp.val.smiles.tolist()),
                     morgan(sp.test.smiles.tolist()))
    ytr = sp.train.label.to_numpy(np.float64); yva = sp.val.label.to_numpy(np.float64)
    yte = sp.test.label.to_numpy(np.float64)
    ymu, ysd = ytr.mean(), ytr.std() + 1e-8

    dev = "cpu"
    tX = lambda A: torch.tensor(A, dtype=torch.float32, device=dev)
    tY = lambda v: torch.tensor((v - ymu) / ysd, dtype=torch.float32, device=dev)
    net = MeanVarMLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xtr_t, ytr_t, Xva_t = tX(Xtr), tY(ytr), tX(Xva)
    best, best_state, bad = 1e9, None, 0
    for ep in range(400):
        net.train()
        opt.zero_grad()
        m, lv = net(Xtr_t)
        loss = nll(m, lv, ytr_t)
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            mv, lvv = net(Xva_t)
            vloss = float(nll(mv, lvv, tY(yva)))
        if vloss < best - 1e-4:
            best, best_state, bad = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 40:
                break
    net.load_state_dict(best_state)

    def predict_meanvar(X, train_mode=False):
        net.train() if train_mode else net.eval()
        with torch.no_grad():
            m, lv = net(tX(X))
        return m.cpu().numpy(), np.exp(lv.cpu().numpy())  # mean(std-scaled), var(std-scaled)

    rows = []
    # --- Gaussian-NLL (eval) ---
    m_te, v_te = predict_meanvar(Xte)
    mean_te = m_te * ysd + ymu
    sd_te = np.sqrt(v_te) * ysd
    cov, w = _cov_width(yte, mean_te, z * sd_te)
    rows.append({"dataset": ds, "method": "Gaussian-NLL", "coverage": round(cov, 3), "width": round(w, 3)})

    # --- MC-Dropout (N stochastic passes, dropout on) ---
    N = 30
    means, vars = [], []
    for _ in range(N):
        mm, vv = predict_meanvar(Xte, train_mode=True)
        means.append(mm); vars.append(vv)
    means = np.stack(means); vars = np.stack(vars)
    pred_mean = means.mean(0) * ysd + ymu
    pred_sd = np.sqrt(vars.mean(0) + means.var(0)) * ysd  # aleatoric + epistemic
    cov, w = _cov_width(yte, pred_mean, z * pred_sd)
    rows.append({"dataset": ds, "method": "MC-Dropout", "coverage": round(cov, 3), "width": round(w, 3)})

    # --- Temperature scaling on the Gaussian-NLL sigma (fit T on val so val coverage = 1-alpha) ---
    m_va, v_va = predict_meanvar(Xva)
    mean_va = m_va * ysd + ymu; sd_va = np.sqrt(v_va) * ysd
    Ts = np.linspace(0.2, 8.0, 400)
    covs = [(np.abs(yva - mean_va) <= z * T * sd_va).mean() for T in Ts]
    T_opt = float(Ts[int(np.argmin(np.abs(np.array(covs) - (1 - alpha))))])
    cov, w = _cov_width(yte, mean_te, z * T_opt * sd_te)
    rows.append({"dataset": ds, "method": f"Gaussian-NLL+TempScale(T={T_opt:.2f})",
                 "coverage": round(cov, 3), "width": round(w, 3)})

    # --- Split-conformal on the MLP mean (principled reference) ---
    # Canonical finite-sample quantile (method="higher"), matching src.eval.conformal.
    q = _conformal_quantile(np.abs(yva - mean_va), alpha)
    cov, w = _cov_width(yte, mean_te, np.full_like(yte, q))
    rows.append({"dataset": ds, "method": "Split-Conformal(MLP)", "coverage": round(cov, 3), "width": round(w, 3)})

    out = pd.DataFrame(rows)
    print(f"\n=== UQ on {ds} (target coverage {1 - alpha:.2f}) ===")
    print(out.to_string(index=False))
    print("KEY: native (Gaussian-NLL / MC-Dropout) coverage != 0.90 = miscalibrated; "
          "TempScale recovers it; Split-Conformal valid by construction.")
    path = FINAL / "uq_extra.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    prev = prev[prev.dataset != ds] if len(prev) else prev
    pd.concat([prev, out], ignore_index=True).to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
