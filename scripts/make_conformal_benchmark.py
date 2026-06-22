"""A: split-conformal benchmark across ALL models on BACE + AD-conditional coverage (the original
novelty thread). Calibrate on validation residuals -> 90% intervals on test. Report empirical
coverage + mean width per model (tightest VALID interval wins). Then stratify coverage by
applicability domain (nearest-train Tanimoto) to show undercoverage concentrates on novel mols."""
from __future__ import annotations
import sys, warnings; from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, scienceplots  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
plt.style.use(["science","no-latex"]); plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman"],"mathtext.fontset":"stix"})
warnings.filterwarnings("ignore")
FIG=Path("results/figures"); FINAL=Path("results/final"); TUN=Path("results/tuning"); BC=Path("results/bace_clean_final")
ALPHA=0.1; SEEDS=[42,1337,2024]
SPEC=[("Ridge",[("val",TUN/"ridge_morgan_val.csv"),("test",TUN/"ridge_morgan_test.csv")]),
      ("SVR",[("val",TUN/"svr_morgan_val.csv"),("test",TUN/"svr_morgan_test.csv")]),
      ("GBM",[("val",TUN/"gbm_morgan_val.csv"),("test",TUN/"gbm_morgan_test.csv")]),
      ("RF",[("val",TUN/"rf_morgan_val.csv"),("test",TUN/"rf_morgan_test.csv")]),
      ("TabPFN",[("val",BC/"tabpfn_desc_val.csv"),("test",BC/"tabpfn_desc_test.csv")]),
      ("MolFormer",[("val",[BC/f"molformer_seed{s}_val.csv" for s in SEEDS]),("test",[BC/f"molformer_seed{s}_test.csv" for s in SEEDS])]),
      ("ChemFM",[("val",[BC/f"chemfm_seed{s}_val.csv" for s in SEEDS]),("test",[BC/f"chemfm_seed{s}_test.csv" for s in SEEDS])]),
      ("Chemprop",[("val",[BC/f"chemprop_seed{s}_val.csv" for s in SEEDS]),("test",[BC/f"chemprop_seed{s}_test.csv" for s in SEEDS])])]

def _pc(df): return df["y_pred"] if "y_pred" in df.columns else df["y_score"]
def load(files):
    files=files if isinstance(files,list) else [files]
    files=[f for f in files if Path(f).exists()]
    mats=[pd.read_csv(f).sort_values("smiles").reset_index(drop=True) for f in files]
    pred=np.mean([_pc(m).to_numpy(float) for m in mats],axis=0)
    return pd.DataFrame({"smiles":mats[0].smiles,"y":mats[0].y_true.to_numpy(float),"p":pred})

def conf_q(scores,a):
    n=len(scores); k=min(int(np.ceil((n+1)*(1-a))),n); return float(np.sort(scores)[k-1])

def ad_scores(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog("rdApp.*")
    tr=pd.read_csv("data/processed/bace_clean.csv"); tr=tr[tr.set=="train"].smiles.tolist()
    gen=AllChem.GetMorganGenerator(radius=2,fpSize=2048)
    ftr=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)),ftr)) for s in smiles])

def main():
    rows=[]; per_model={}
    test_smiles=None
    for name,sp in SPEC:
        dd=dict(sp); v=load(dd["val"]); t=load(dd["test"])
        qhat=conf_q(np.abs(v.y-v.p),ALPHA)
        cov=float(np.mean(np.abs(t.y-t.p)<=qhat)); width=2*qhat
        rows.append({"model":name,"coverage":round(cov,3),"width":round(width,3),"qhat":round(qhat,3)})
        per_model[name]=(t.sort_values("smiles").reset_index(drop=True),qhat)
        if test_smiles is None: test_smiles=t.sort_values("smiles").smiles.tolist()
    R=pd.DataFrame(rows); R.to_csv(FINAL/"conformal_benchmark.csv",index=False)
    print(R.to_string(index=False))
    # AD-conditional coverage
    ad=ad_scores(test_smiles)
    bins=[("low\n(Tan<0.4)",ad<0.4),("med\n(0.4-0.6)",(ad>=0.4)&(ad<0.6)),("high\n(Tan>=0.6)",ad>=0.6)]
    print("\nAD-conditional coverage (nominal 0.90):")
    adcov={}
    for name,(t,qhat) in per_model.items():
        cov_arr=(np.abs(t.y-t.p)<=qhat).astype(float).to_numpy()
        adcov[name]=[float(cov_arr[m].mean()) if m.sum()>0 else np.nan for _,m in bins]
    adf=pd.DataFrame(adcov,index=[b[0].replace(chr(10),' ') for b in bins]).T
    adf.to_csv(FINAL/"conformal_ad_coverage.csv"); print(adf.round(3).to_string())
    print("bin sizes:",{b[0].replace(chr(10),' '):int(m.sum()) for b,m in [(b,b[1]) for b in bins]})

    fig,(a,b)=plt.subplots(1,2,figsize=(11.5,4.4))
    # (a) coverage vs width Pareto
    for _,r in R.iterrows():
        a.scatter(r.width,r.coverage,s=70,edgecolor="k",linewidth=0.5,zorder=3)
        a.annotate(r.model,(r.width,r.coverage),xytext=(4,3),textcoords="offset points",fontsize=7.5)
    a.axhline(0.9,ls="--",color="#C44E52",lw=1,label="nominal 0.90")
    a.set_xlabel("mean interval width (log units)"); a.set_ylabel("empirical coverage")
    a.set_title("(a) tightest VALID interval (left+on 0.90 line = best)",fontsize=9.5); a.legend(fontsize=8)
    # (b) AD-conditional coverage
    x=np.arange(len(bins)); w=0.1
    for k2,name in enumerate(adf.index):
        a_=adf.loc[name].to_numpy()
        b.plot(x,a_,"o-",ms=4,lw=1,label=name)
    b.axhline(0.9,ls="--",color="#C44E52",lw=1)
    b.set_xticks(x); b.set_xticklabels([bb[0] for bb in bins],fontsize=8)
    b.set_ylabel("conditional coverage"); b.set_title("(b) coverage drops on novel chemistry (low AD)",fontsize=9.5)
    b.legend(fontsize=6.5,ncol=2)
    fig.suptitle("A  Conformal-prediction benchmark across models + AD-conditional coverage (BACE, 90%)",y=1.02,fontsize=12)
    fig.tight_layout(); fig.savefig(FIG/"fig_conformal_benchmark_j.png",dpi=300,bbox_inches="tight")
    print("wrote fig_conformal_benchmark_j.png + conformal_benchmark.csv + conformal_ad_coverage.csv")

if __name__=="__main__": main()
