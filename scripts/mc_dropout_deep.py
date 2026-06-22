"""GAP3: MC-Dropout on the REAL deep models (not the MLP proxy). Reloads a trained model's
state_dict, enables every dropout module at inference, runs N stochastic forward passes ->
predictive (mean, std) -> 90% Gaussian interval -> coverage/width, alongside split-conformal
on the same mean. Confirms the MLP finding (native MC-Dropout is over/under-confident; conformal
is valid by construction) on the actual foundation-model backbones.

Supported (HF-transformer wrappers): molformer, chemfm. No re-training (uses run_dir/model.pt).
Run on the server: CUDA_VISIBLE_DEVICES=<free> python scripts/mc_dropout_deep.py <model> <dataset> [n=30]
Output: results/final/mc_dropout_deep.csv (appended).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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

Z = 1.6448536269514722  # 90% two-sided Gaussian
ALPHA = 0.1
RUNS = Path("runs")
FINAL = Path("results/final"); FINAL.mkdir(parents=True, exist_ok=True)


def enable_dropout(m: nn.Module) -> int:
    k = 0
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.train(); k += 1
    return k


def latest_run(dataset: str, model: str) -> Path | None:
    for d in reversed(sorted(RUNS.glob(f"*_{dataset}_{model}_seed42"))):
        if (d / "model.pt").exists():
            return d
    return None


def _hf_passes(model_obj, tok, smiles, dev, n, max_len=128):
    enc = tok(list(smiles), padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
    ii, am = enc.input_ids.to(dev), enc.attention_mask.to(dev)
    outs = []
    with torch.no_grad():
        for _ in range(n):
            outs.append(model_obj(ii, am).cpu().numpy())
    return np.stack(outs)  # (n, m)


def build_hf(model: str, run_dir: Path, dev: str):
    from transformers import AutoTokenizer
    if model == "molformer":
        from src.models.molformer_wrapper import MolFormerRegressor
        bid = "ibm-research/MoLFormer-XL-both-10pct"
        net = MolFormerRegressor(backbone_id=bid)
        tok = AutoTokenizer.from_pretrained(bid, trust_remote_code=True)
    elif model == "chemfm":
        from src.models.chemfm_wrapper import ChemFMRegressor  # class name per wrapper
        bid = "ChemFM/ChemFM-3B"
        net = ChemFMRegressor(backbone_id=bid)
        tok = AutoTokenizer.from_pretrained(bid, trust_remote_code=True)
    else:
        raise ValueError(model)
    net.load_state_dict(torch.load(run_dir / "model.pt", map_location=dev))
    net.to(dev).eval()
    return net, tok


def main():
    set_all_seeds(42)  # reproducible MC-Dropout sampling (§8.1)
    model, dataset = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    rd = latest_run(dataset, model)
    if rd is None:
        print(f"no trained {model}/{dataset} (need runs/*_{dataset}_{model}_seed42/model.pt)"); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, tok = build_hf(model, rd, dev)
    nd = enable_dropout(net)
    print(f"{model}/{dataset}: reloaded {rd.name}, {nd} dropout modules active, device={dev}")

    sp = load_dataset(dataset)
    yva, yte = sp.val.label.to_numpy(np.float64), sp.test.label.to_numpy(np.float64)
    Pva, Pte = _hf_passes(net, tok, sp.val.smiles, dev, n), _hf_passes(net, tok, sp.test.smiles, dev, n)
    mean_te, std_te = Pte.mean(0), Pte.std(0)
    cov = float((np.abs(yte - mean_te) <= Z * std_te).mean()); width = float(2 * Z * std_te.mean())
    mean_va = Pva.mean(0); res = np.abs(yva - mean_va)
    q = float(_conformal_quantile(res, ALPHA))  # canonical finite-sample quantile (method="higher")
    cov_c = float((np.abs(yte - mean_te) <= q).mean()); width_c = 2 * q

    print(f"  MC-Dropout cov={cov:.3f} width={width:.3f}  |  split-conformal cov={cov_c:.3f} width={width_c:.3f}")
    row = {"dataset": dataset, "model": model, "coverage_mcdropout": round(cov, 3),
           "width_mcdropout": round(width, 3), "coverage_conformal": round(cov_c, 3),
           "width_conformal": round(width_c, 3), "n_passes": n}
    p = FINAL / "mc_dropout_deep.csv"
    prev = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if len(prev):
        prev = prev[~((prev.dataset == dataset) & (prev.model == model))]
    pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(p, index=False)
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
