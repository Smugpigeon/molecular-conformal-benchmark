"""Hyperparameter tuning for the four classic ML models (Ridge / SVR / GBM / RF) on BACE,
per BACE_项目设计方案.md §5.3.

Protocol (small-sample QSAR best practice):
  * search the hyperparameters with Optuna TPE (Bayesian), not a coarse grid;
  * the objective is 5-fold cross-validated out-of-fold MAE on the *training* split only;
  * every fold rebuilds its own StandardScaler inside an sklearn Pipeline, so no scaler is
    ever fit on held-out data (leak-safe, CLAUDE.md §7.3);
  * validation / test are never touched during the search -- they are scored once at the end
    (CLAUDE.md §8.4). MAE is the primary BACE regression metric; we report MAE+RMSE+R+rho (§9).

Outputs (results/tuning/):
  summary.json                       best params + CV/val/test metrics for every (model, rep)
  <model>_<rep>_{val,test}.csv       per-molecule predictions (for the figures)
  <model>_<rep>_trials.csv           the full Optuna trial table (for F9/F11)

Run a smoke check first, then the full search:
  /opt/anaconda3/bin/python3 scripts/tune_classic_ml.py --smoke
  /opt/anaconda3/bin/python3 scripts/tune_classic_ml.py --trials 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.metrics import regression_metrics  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)
import os  # noqa: E402
DATA_CSV = os.environ.get("BACE_DATA", "data/processed/bace_clean.csv")
OUT = Path(os.environ.get("TUNE_OUT", "results/tuning")); OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["ridge", "svr", "gbm", "rf"]


# --------------------------------------------------------------------------- features
def featurize(smiles: list[str], rep: str) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if rep == "morgan":
        # Trigger: rep == "morgan"  Why: canonical strategy-1 fingerprint (CLAUDE.md §9, r2/2048)
        # Outcome: 2048-bit dense 0/1 matrix
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        X = np.zeros((len(mols), 2048), dtype=np.float64)
        for i, m in enumerate(mols):
            X[i] = gen.GetFingerprint(m)
        return X
    if rep == "descriptors":
        # Trigger: rep == "descriptors"  Why: interpretable physchem vector (same 12 as the EDA)
        # Outcome: 12-D continuous matrix (scaled inside the Pipeline)
        rows = []
        for m in mols:
            rows.append([Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
                         rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
                         rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
                         rdMolDescriptors.CalcNumAromaticRings(m), m.GetNumHeavyAtoms(),
                         rdMolDescriptors.CalcFractionCSP3(m), QED.qed(m),
                         rdMolDescriptors.CalcNumAtomStereoCenters(m)])
        return np.asarray(rows, dtype=np.float64)
    raise ValueError(f"unknown rep {rep!r}")


# --------------------------------------------------------------------------- search space
def suggest(trial: optuna.Trial, name: str) -> dict:
    """Return RAW optuna-suggested params; build() does all type conversion so the objective
    and the best-params replay stay byte-identical."""
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    if name == "svr":
        return {"C": trial.suggest_float("C", 1e-1, 1e3, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1e0, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.01, 0.5)}
    if name == "rf":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
                "max_depth": trial.suggest_categorical("max_depth", ["none", "5", "10", "20", "30"]),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "0.3", "0.5"]),
                "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 5, 10])}
    if name == "gbm":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_iter": trial.suggest_int("max_iter", 200, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50, log=True),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1e1, log=True)}
    raise ValueError(name)


def build(name: str, p: dict):
    p = dict(p)
    if name == "ridge":
        return Ridge(random_state=SEED, **p)
    if name == "svr":
        return SVR(kernel="rbf", **p)
    if name == "rf":
        md = p.pop("max_depth"); mf = p.pop("max_features")
        return RandomForestRegressor(max_depth=None if md == "none" else int(md),
                                     max_features="sqrt" if mf == "sqrt" else float(mf),
                                     random_state=SEED, n_jobs=-1, **p)
    if name == "gbm":
        return HistGradientBoostingRegressor(random_state=SEED, early_stopping=True, **p)
    raise ValueError(name)


def make_pipe(name: str, p: dict) -> Pipeline:
    est = build(name, p)
    # scaling matters for Ridge/SVR; tree ensembles are scale-invariant so skip the cost
    if name in ("ridge", "svr"):
        return Pipeline([("scale", StandardScaler()), ("model", est)])
    return Pipeline([("model", est)])


# --------------------------------------------------------------------------- tuning
def objective(trial: optuna.Trial, name: str, X: np.ndarray, y: np.ndarray, cv: KFold) -> float:
    pipe = make_pipe(name, suggest(trial, name))
    oof = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
    return float(mean_absolute_error(y, oof))


def tune_one(name: str, rep: str, data: dict, n_trials: int) -> dict:
    Xtr, ytr = data["Xtr"], data["ytr"]
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, name, Xtr, ytr, cv), n_trials=n_trials,
                   show_progress_bar=False)

    best = study.best_params
    pipe = make_pipe(name, best).fit(Xtr, ytr)
    pv = pipe.predict(data["Xva"]); pt = pipe.predict(data["Xte"])
    mv = regression_metrics(data["yva"], pv); mt = regression_metrics(data["yte"], pt)

    pd.DataFrame({"smiles": data["sva"], "y_true": data["yva"], "y_pred": pv}).to_csv(
        OUT / f"{name}_{rep}_val.csv", index=False)
    pd.DataFrame({"smiles": data["ste"], "y_true": data["yte"], "y_pred": pt}).to_csv(
        OUT / f"{name}_{rep}_test.csv", index=False)
    study.trials_dataframe().to_csv(OUT / f"{name}_{rep}_trials.csv", index=False)

    print(f"  [{name:5s}|{rep:11s}] cvMAE={study.best_value:.3f}  "
          f"val MAE={mv['MAE']:.3f} R={mv['PearsonR']:.3f}  "
          f"test MAE={mt['MAE']:.3f} R={mt['PearsonR']:.3f}  best={best}")
    return {"model": name, "rep": rep, "n_trials": n_trials, "cv_mae": study.best_value,
            "best_params": best, "val": mv, "test": mt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="morgan", choices=["morgan", "descriptors"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--smoke", action="store_true", help="2 trials, ridge+gbm only (pipeline check)")
    args = ap.parse_args()
    if args.smoke:
        args.trials, args.models = 2, ["ridge", "gbm"]

    set_all_seeds(SEED)
    df = pd.read_csv(DATA_CSV)
    sp = {k: df[df.set == k] for k in ["train", "validation", "test"]}
    X = {k: featurize(v.smiles.tolist(), args.rep) for k, v in sp.items()}
    data = {"Xtr": X["train"], "ytr": sp["train"].label.to_numpy(float),
            "Xva": X["validation"], "yva": sp["validation"].label.to_numpy(float),
            "Xte": X["test"], "yte": sp["test"].label.to_numpy(float),
            "sva": sp["validation"].smiles.tolist(), "ste": sp["test"].smiles.tolist()}
    print(f"rep={args.rep}  dim={data['Xtr'].shape[1]}  "
          f"n_train={len(data['ytr'])} n_val={len(data['yva'])} n_test={len(data['yte'])}  "
          f"trials={args.trials}")

    results = [tune_one(m, args.rep, data, args.trials) for m in args.models]
    if not args.smoke:
        path = OUT / "summary.json"
        prev = json.loads(path.read_text()) if path.exists() else []
        prev = [r for r in prev if (r["model"], r["rep"]) not in {(x["model"], x["rep"]) for x in results}]
        path.write_text(json.dumps(prev + results, indent=2))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
