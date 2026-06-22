"""B: predefined split vs Murcko scaffold split (generalization to novel chemistry) for all models.
Reads predefined metrics from results/final/master_table.csv and scaffold test predictions from
results/bace_scaffold_final/. Reports MAE/Pearson drop under scaffold split + a comparison figure."""
from __future__ import annotations
import sys, warnings; from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, scienceplots  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.metrics import regression_metrics  # noqa: E402
plt.style.use(["science","no-latex"]); plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman"],"mathtext.fontset":"stix"})
warnings.filterwarnings("ignore")
FIG=Path("results/figures"); FINAL=Path("results/final"); SC=Path("results/bace_scaffold_final")
SEEDS=[42,1337,2024]
# scaffold prediction files per model
SPEC={"Ridge":[SC/"ridge_morgan_test.csv"],"SVR":[SC/"svr_morgan_test.csv"],
      "GBM":[SC/"gbm_morgan_test.csv"],"RF":[SC/"rf_morgan_test.csv"],
      "TabPFN":[SC/"tabpfn_desc_test.csv"],
      "MolFormer":[SC/f"molformer_seed{s}_test.csv" for s in SEEDS],
      "ChemFM":[SC/f"chemfm_seed{s}_test.csv" for s in SEEDS],
      "Chemprop":[SC/f"chemprop_seed{s}_test.csv" for s in SEEDS]}
# map to predefined master_table model names
PRE={"Ridge":"Ridge","SVR":"SVR","GBM":"GBM","RF":"RF","TabPFN":"TabPFN (desc)",
     "MolFormer":"MolFormer-XL","ChemFM":"ChemFM-3B","Chemprop":"Chemprop"}
def _pc(d): return d["y_pred"] if "y_pred" in d.columns else d["y_score"]
def scaf_metric(files):
    files=[f for f in files if f.exists()]
    if not files: return None
    mats=[pd.read_csv(f).sort_values("smiles").reset_index(drop=True) for f in files]
    pred=np.mean([_pc(m).to_numpy(float) for m in mats],axis=0)
    return regression_metrics(mats[0].y_true.to_numpy(float),pred)
def main():
    M=pd.read_csv(FINAL/"master_table.csv").set_index("model")
    rows=[]
    for name,files in SPEC.items():
        sm=scaf_metric(files)
        if sm is None: print(f"  {name}: scaffold preds missing"); continue
        pre=M.loc[PRE[name]]
        rows.append({"model":name,"pre_MAE":pre.test_MAE,"scaf_MAE":round(sm["MAE"],3),
                     "dMAE":round(sm["MAE"]-pre.test_MAE,3),"pre_R":pre.test_PearsonR,
                     "scaf_R":round(sm["PearsonR"],3),"dR":round(sm["PearsonR"]-pre.test_PearsonR,3)})
    T=pd.DataFrame(rows); T.to_csv(FINAL/"scaffold_comparison.csv",index=False)
    print(T.to_string(index=False))
    fig,(a,b)=plt.subplots(1,2,figsize=(12,4.4)); x=np.arange(len(T)); w=0.38
    a.bar(x-w/2,T.pre_MAE,w,label="predefined split",color="#4C72B0",edgecolor="k",linewidth=0.4)
    a.bar(x+w/2,T.scaf_MAE,w,label="scaffold split",color="#C44E52",edgecolor="k",linewidth=0.4)
    a.set_xticks(x); a.set_xticklabels(T.model,rotation=25,ha="right",fontsize=8); a.set_ylabel("test MAE"); a.legend(fontsize=8); a.set_title("(a) MAE: predefined vs scaffold",fontsize=10)
    b.bar(x-w/2,T.pre_R,w,label="predefined",color="#4C72B0",edgecolor="k",linewidth=0.4)
    b.bar(x+w/2,T.scaf_R,w,label="scaffold",color="#C44E52",edgecolor="k",linewidth=0.4)
    b.set_xticks(x); b.set_xticklabels(T.model,rotation=25,ha="right",fontsize=8); b.set_ylabel("test Pearson R"); b.legend(fontsize=8); b.set_ylim(0,1); b.set_title("(b) Pearson: predefined vs scaffold",fontsize=10)
    fig.suptitle("B  Generalization stress test: predefined split vs Murcko scaffold split (novel chemistry)",y=1.02,fontsize=12)
    fig.tight_layout(); fig.savefig(FIG/"fig_scaffold_comparison_j.png",dpi=300,bbox_inches="tight")
    print("wrote fig_scaffold_comparison_j.png + scaffold_comparison.csv")
if __name__=="__main__": main()
