"""Activity-cliff evaluation of a neural backbone on one MoleculeACE dataset.

Novelty thrust (项目调研汇总.md v2 §7.5.1): SemiMol (arXiv 2601.04507, 2026-01)
explicitly does NOT evaluate Uni-Mol or MolFormer on activity cliffs. This
script fills that gap: train a neural backbone on a MoleculeACE dataset,
predict the held-out test, and report cliff vs non-cliff RMSE.

One (model, dataset) per invocation -> a sweep distributes across GPUs.
Run: python scripts/run_cliff_neural.py --model molformer --dataset CHEMBL204_Ki
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from rdkit import Chem, RDLogger

from src.data.loaders import Split
from src.eval.cliff import cliff_stratified_rmse, load_moleculeace
from src.utils.logging_setup import configure_logging
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def build_split(name: str, seed: int):
    """MoleculeACE -> Split (carve 10% val from train) + canonical-SMILES->cliff map."""
    train, test, target = load_moleculeace(name)
    tr = train.rename(columns={target: "label"})[["smiles", "label"]].copy()
    te = test.rename(columns={target: "label"})[["smiles", "label"]].copy()
    val = tr.sample(frac=0.1, random_state=seed)
    tr2 = tr.drop(val.index)
    split = Split(
        train=tr2.reset_index(drop=True), val=val.reset_index(drop=True),
        test=te.reset_index(drop=True), task="regression", name=name, metric_main="RMSE",
    )
    cliff_map = {
        _canon(s): int(c)
        for s, c in zip(test["smiles"], test["cliff_mol"]) if _canon(s)
    }
    return split, cliff_map


def build_cfg(model: str, seed: int):
    mcfg = OmegaConf.load(f"configs/model/{model}.yaml")
    return OmegaConf.create({
        "model": mcfg, "seed": seed, "save_test_preds": True,
        "deterministic": True, "log_level": "INFO",
    })


def main() -> int:
    configure_logging("INFO")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["molformer", "chemfm", "unimol"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("results/cliff_neural"))
    args = ap.parse_args()

    set_all_seeds(args.seed)
    split, cliff_map = build_split(args.dataset, args.seed)
    cfg = build_cfg(args.model, args.seed)
    run_dir = Path("runs/cliff_neural") / f"{args.model}_{args.dataset}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "molformer":
        from src.models.molformer_wrapper import train_molformer as fn
    elif args.model == "chemfm":
        from src.models.chemfm_wrapper import train_chemfm as fn
    else:
        from src.models.unimol_wrapper import train_unimol as fn

    fn(cfg, split, run_dir)

    tp_path = run_dir / "test_predictions.csv"
    if not tp_path.exists():
        logger.error(f"no test predictions for {args.model}/{args.dataset}")
        return 1
    tp = pd.read_csv(tp_path)
    # Align cliff labels by canonical SMILES (robust to unimol's filtering/canon).
    cliff = np.array([cliff_map.get(_canon(s), 0) for s in tp["smiles"]], dtype=bool)
    res = cliff_stratified_rmse(tp["y_true"].to_numpy(), tp["y_score"].to_numpy(), cliff)
    res.update({"dataset": args.dataset, "model": args.model})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).to_csv(args.out_dir / f"{args.model}_{args.dataset}.csv", index=False)
    logger.info(
        f"{args.dataset} {args.model}: cliff={res['rmse_cliff']:.3f} "
        f"noncliff={res['rmse_noncliff']:.3f} ratio={res['cliff_ratio']:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
