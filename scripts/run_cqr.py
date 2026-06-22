"""Conformalized Quantile Regression for one (dataset, backbone, seed).

Trains a quantile regressor (q05/q95), then CQR-conformalizes on val and
evaluates adaptive intervals on test. Three representation classes:
  gbm       — GradientBoosting quantile on Morgan FP (classical, CPU)
  molformer — MolFormer-XL backbone + 2-output quantile head, pinball loss (GPU)
  chemprop  — native QuantileFFN D-MPNN (GPU)

Saves CQR coverage+width to results/cqr/<model>_<dataset>_<seed>.csv.
Run: python scripts/run_cqr.py --model molformer --dataset esol --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.featurizers import batch_morgan
from src.data.loaders import load_dataset
from src.eval.cqr import evaluate_cqr
from src.utils.logging_setup import configure_logging
from src.utils.seed import set_all_seeds

from src.utils.rdkit_compat import patch_rdkit_pandas_compat

logger = logging.getLogger(__name__)
Q_LO, Q_HI, ALPHA = 0.05, 0.95, 0.1

# Must run before any `import unimol_tools` (transitively imports rdkit
# PandasTools, which is broken on pandas 2.2 + setuptools>=81). See §11.
patch_rdkit_pandas_compat()


# ---------- GBM (classical, CPU) ----------
def gbm_quantile(split, seed):
    from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415

    Xtr, itr = batch_morgan(split.train["smiles"].tolist())
    Xva, iva = batch_morgan(split.val["smiles"].tolist())
    Xte, ite = batch_morgan(split.test["smiles"].tolist())
    ytr = split.train["label"].to_numpy()[itr]
    common = dict(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed)
    g_lo = GradientBoostingRegressor(loss="quantile", alpha=Q_LO, **common).fit(Xtr, ytr)
    g_hi = GradientBoostingRegressor(loss="quantile", alpha=Q_HI, **common).fit(Xtr, ytr)
    return (
        split.val["label"].to_numpy()[iva], g_lo.predict(Xva), g_hi.predict(Xva),
        split.test["label"].to_numpy()[ite], g_lo.predict(Xte), g_hi.predict(Xte),
    )


# ---------- MolFormer quantile head (GPU) ----------
def molformer_quantile(split, seed, epochs=30, lr=1e-4, bs=32):
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from torch.utils.data import DataLoader, Dataset  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bid = "ibm-research/MoLFormer-XL-both-10pct"
    tok = AutoTokenizer.from_pretrained(bid, trust_remote_code=True)

    class DS(Dataset):
        def __init__(self, df):
            self.s = df["smiles"].tolist(); self.y = df["label"].astype(float).tolist()

        def __len__(self): return len(self.s)

        def __getitem__(self, i):
            e = tok(self.s[i], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            return e["input_ids"][0], e["attention_mask"][0], torch.tensor(self.y[i], dtype=torch.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = AutoModel.from_pretrained(bid, trust_remote_code=True)
            h = self.bb.config.hidden_size
            self.head = nn.Sequential(nn.Linear(h, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 2))

        def forward(self, ii, am):
            o = self.bb(input_ids=ii, attention_mask=am)
            m = am.unsqueeze(-1).float()
            pooled = (o.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1)
            return self.head(pooled)  # [B,2] = (q_lo, q_hi)

    # z-standardize targets on train.
    ytr = np.asarray(DS(split.train).y); mu, sd = ytr.mean(), ytr.std() + 1e-8

    def loader(df, sh): return DataLoader(DS(df), batch_size=bs, shuffle=sh, num_workers=2)
    tr = loader(split.train, True)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)

    def pinball(pred, target, q):
        d = target - pred
        return torch.maximum(q * d, (q - 1) * d).mean()

    for _ in range(epochs):
        net.train()
        for ii, am, y in tr:
            ii, am = ii.to(dev), am.to(dev)
            yz = ((y.to(dev) - mu) / sd)
            opt.zero_grad()
            out = net(ii, am)
            loss = pinball(out[:, 0], yz, Q_LO) + pinball(out[:, 1], yz, Q_HI)
            loss.backward(); opt.step()

    def predict(df):
        net.eval(); los, his = [], []
        with torch.no_grad():
            for ii, am, _ in loader(df, False):
                out = net(ii.to(dev), am.to(dev)).cpu().numpy()
                los.append(out[:, 0]); his.append(out[:, 1])
        lo = np.concatenate(los) * sd + mu; hi = np.concatenate(his) * sd + mu
        return np.minimum(lo, hi), np.maximum(lo, hi)  # enforce lo<=hi

    vlo, vhi = predict(split.val); tlo, thi = predict(split.test)
    return (split.val["label"].to_numpy(), vlo, vhi,
            split.test["label"].to_numpy(), tlo, thi)


# ---------- Chemprop native QuantileFFN (GPU) ----------
def chemprop_quantile(split, seed, epochs=50, bs=64):
    from lightning import pytorch as pl  # noqa: PLC0415
    from rdkit import Chem  # noqa: PLC0415

    from chemprop import data, featurizers, models, nn  # noqa: PLC0415

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def mk(df):
        dps, keep = [], []
        for s, y in zip(df["smiles"], df["label"]):
            if Chem.MolFromSmiles(s) is None:
                keep.append(False); continue
            dps.append(data.MoleculeDatapoint.from_smi(s, np.array([float(y)])))
            keep.append(True)
        return data.MoleculeDataset(dps, feat), np.array(keep, bool)

    tr, _ = mk(split.train); va, vk = mk(split.val); te, tk = mk(split.test)
    sc = tr.normalize_targets(); va.normalize_targets(sc)
    tl = data.build_dataloader(tr, batch_size=bs)
    mp = nn.BondMessagePassing(); agg = nn.MeanAggregation()
    # QuantileFFN with the two target quantiles; output_transform unscales.
    ot = nn.UnscaleTransform.from_standard_scaler(sc)
    ffn = nn.QuantileFFN(n_tasks=1, output_transform=ot)
    model = models.MPNN(mp, agg, ffn, batch_norm=True)
    tr_ = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1,
                     enable_checkpointing=False, enable_progress_bar=False,
                     logger=False, deterministic=False)
    tr_.fit(model, tl)

    def pred(dset, keep, df):
        dl = data.build_dataloader(dset, batch_size=bs, shuffle=False)
        out = np.concatenate([p.numpy() for p in tr_.predict(model, dl)])
        out = out.reshape(out.shape[0], -1)  # [N, n_quantiles]
        lo, hi = out[:, 0], out[:, -1]
        y = df["label"].to_numpy()[keep]
        return y, np.minimum(lo, hi), np.maximum(lo, hi)

    vy, vlo, vhi = pred(va, vk, split.val)
    ty, tlo, thi = pred(te, tk, split.test)
    return vy, vlo, vhi, ty, tlo, thi


# ---------- Uni-Mol 3D quantile head (GPU; fp32 ONLY per CLAUDE.md 8.8) ----------
def unimol_quantile(split, seed, epochs=40, lr=1e-4, bs=16):
    """Fine-tune Uni-Mol (84M, 3D) with a 2-output pinball head -> CQR.

    Unlike unimol_wrapper (MolTrain, no seed control), we drive UniMolModel
    directly, so this DOES vary across seeds. We reuse Uni-Mol's own conformer
    featurization (DataHub, is_train=False) + batch_collate_fn, and z-standardize
    targets ourselves. fp32 throughout (Uni-Mol 3D coords must not run fp16).
    """
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from src.models.unimol_wrapper import _canonicalize, _unimol_accepts  # noqa: PLC0415
    from unimol_tools.data import DataHub  # noqa: PLC0415
    from unimol_tools.models import UniMolModel  # noqa: PLC0415
    from unimol_tools.models.nnmodel import NNDataset  # noqa: PLC0415

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clean(df):
        """Strict pre-filter so featurized count == label count (no silent drop)."""
        s = df["smiles"].map(_canonicalize)
        df2 = df.loc[s.notna()].copy()
        df2["smiles"] = s[s.notna()].to_numpy()
        df2 = df2.loc[df2["smiles"].map(_unimol_accepts)].reset_index(drop=True)
        return df2

    def featurize(df):
        d = tempfile.mkdtemp()
        csv = os.path.join(d, "u.csv")
        pd.DataFrame({"SMILES": df["smiles"], "TARGET": df["label"].astype(float)}).to_csv(csv, index=False)
        dh = DataHub(data=csv, is_train=False, save_path=d, task="regression",
                     data_type="molecule", target_normalize="none",
                     model_name="unimolv1", model_size="84m")
        x = np.asarray(dh.data["unimol_input"], dtype=object)
        if len(x) != len(df):
            raise RuntimeError(f"Uni-Mol featurized {len(x)} != {len(df)} input molecules")
        return x

    tr_df, va_df, te_df = clean(split.train), clean(split.val), clean(split.test)
    Xtr, Xva, Xte = featurize(tr_df), featurize(va_df), featurize(te_df)
    ytr = tr_df["label"].to_numpy(dtype=np.float64)
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-8)
    ytr_z = ((ytr - mu) / sd).astype(np.float32)

    model = UniMolModel(output_dim=2, data_type="molecule", model_size="84m").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def pinball(out, tgt):
        tgt = tgt.view(-1).float()
        lo, hi = out[:, 0], out[:, 1]
        e_lo, e_hi = tgt - lo, tgt - hi
        return (torch.maximum(Q_LO * e_lo, (Q_LO - 1) * e_lo).mean()
                + torch.maximum(Q_HI * e_hi, (Q_HI - 1) * e_hi).mean())

    def loader(x, y, sh):
        return DataLoader(NNDataset(x, y), batch_size=bs, shuffle=sh,
                          collate_fn=model.batch_collate_fn)

    tl = loader(Xtr, ytr_z, True)
    for _ in range(epochs):
        model.train()
        for ni, nt in tl:
            ni = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ni.items()}
            loss = pinball(model(**ni), nt.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()

    def predict(x):
        model.eval(); outs = []
        with torch.no_grad():
            for ni, _ in loader(x, np.zeros(len(x), np.float32), False):
                ni = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ni.items()}
                outs.append(model(**ni).cpu().numpy())
        p = np.concatenate(outs)
        lo, hi = p[:, 0] * sd + mu, p[:, 1] * sd + mu
        return np.minimum(lo, hi), np.maximum(lo, hi)  # enforce lo<=hi

    vlo, vhi = predict(Xva); tlo, thi = predict(Xte)
    return (va_df["label"].to_numpy(), vlo, vhi,
            te_df["label"].to_numpy(), tlo, thi)


def main() -> int:
    configure_logging("INFO")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["gbm", "molformer", "chemprop", "unimol"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (neural models only)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/cqr"))
    args = ap.parse_args()

    set_all_seeds(args.seed)
    split = load_dataset(args.dataset)
    fn = {"gbm": gbm_quantile, "molformer": molformer_quantile,
          "chemprop": chemprop_quantile, "unimol": unimol_quantile}[args.model]
    # gbm has no epochs arg; neural models accept an optional override.
    kw = {} if (args.model == "gbm" or args.epochs is None) else {"epochs": args.epochs}
    cal_y, cal_lo, cal_hi, test_y, test_lo, test_hi = fn(split, args.seed, **kw)

    res = evaluate_cqr(cal_y, cal_lo, cal_hi, test_y, test_lo, test_hi, ALPHA)
    res.update({"dataset": args.dataset, "model": args.model, "seed": args.seed})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).to_csv(args.out_dir / f"{args.model}_{args.dataset}_{args.seed}.csv", index=False)
    logger.info(f"CQR {args.model}/{args.dataset}/s{args.seed}: "
                f"cov={res['coverage']:.3f} width={res['width']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
