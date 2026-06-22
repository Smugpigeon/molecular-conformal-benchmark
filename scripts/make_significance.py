"""C: statistical significance of model differences on BACE test (n=303).
- Friedman test + Nemenyi critical-difference (CD) diagram on per-molecule abs-error ranks --
  DESCRIPTIVE ONLY: per-molecule errors are not independent replicates (pseudo-replication), so
  the CD diagram is a visualization, not a valid significance test.
- Pairwise paired bootstrap of dMAE (10k resamples) with 95% CI + Bonferroni -- this and the
  per-fold Nadeau-Bengio test (scripts/perfold_significance.py) are the significance evidence.
Makes the TabPFN point-estimate lead and selected pairwise differences defensible: the bootstrap
separates TabPFN from RF/GBM/Ridge/SVR/MolFormer but NOT from Chemprop/ChemFM (p>0.1), so the
calibrated claim is 'first tier, not a clean sweep' (CLAUDE.md §16.4)."""
from __future__ import annotations
import sys, warnings; from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, scienceplots  # noqa
from scipy.stats import friedmanchisquare, rankdata
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_error_analysis import load_wide, MODELS  # noqa: E402
plt.style.use(["science","no-latex"]); plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman"],"mathtext.fontset":"stix"})
warnings.filterwarnings("ignore")
FIG=Path("results/figures"); FINAL=Path("results/final")
Q05={2:1.960,3:2.344,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,10:3.164}  # Nemenyi q_0.05

def main():
    wide=load_wide(); labels=[m for m,_,_ in MODELS if m in wide.columns]
    y=wide.y_true.to_numpy(float)
    E=np.array([np.abs(wide[m].to_numpy(float)-y) for m in labels])  # (k, n)
    k,n=E.shape
    # Friedman + Nemenyi CD on per-molecule error ranks (1=best)
    ranks=np.apply_along_axis(rankdata,0,E); avg=ranks.mean(1)
    chi,p=friedmanchisquare(*E)
    CD=Q05[k]*np.sqrt(k*(k+1)/(6*n))
    order=np.argsort(avg)
    print(f"Friedman chi2={chi:.1f} p={p:.2e} | CD(0.05)={CD:.3f}")
    for i in order: print(f"  {labels[i]:12s} avg_rank={avg[i]:.2f} testMAE={E[i].mean():.3f}")
    # CD diagram
    fig,ax=plt.subplots(figsize=(9.5,3.7))
    lo,hi=1,k
    ax.set_xlim(lo-0.3,hi+0.3); ax.set_ylim(-len(labels)*0.5-1,1.75); ax.axis("off")
    ax.plot([lo,hi],[0,0],"k",lw=1)
    for r in range(int(lo),int(hi)+1): ax.plot([r,r],[0,0.12],"k",lw=1); ax.text(r,0.52,str(r),ha="center",fontsize=9.5)
    # CD bar
    ax.plot([lo,lo+CD],[1.25,1.25],"k",lw=2); ax.text(lo+CD/2,1.45,f"CD={CD:.2f}",ha="center",fontsize=9.5)
    so=order  # best->worst
    for rank_pos,i in enumerate(so):
        yy=-0.5*(rank_pos+1)
        side = -0.3 if rank_pos < k/2 else hi+0.3
        ax.plot([avg[i],avg[i]],[0,yy],color="0.5",lw=0.8)
        ax.plot([avg[i],side],[yy,yy],color="0.5",lw=0.8)
        ax.text(side+(-0.05 if side<0 else 0.05),yy,f"{labels[i]} ({avg[i]:.2f})",
                ha="right" if side<0 else "left",va="center",fontsize=8.5)
    # cliques: connect models within CD
    sa=avg[so]; ymark=0.18; grp=0
    j=0
    while j<k:
        m=j
        while m+1<k and sa[m+1]-sa[j]<=CD: m+=1
        if m>j:
            ax.plot([sa[j]-0.03,sa[m]+0.03],[ymark+grp*0.12,ymark+grp*0.12],"r",lw=2.5); grp+=1
        j=m+1 if m>j else j+1
    ax.set_title(f"C  Critical-difference diagram (descriptive; Friedman p={p:.1e}, per-molecule n={n} "
                 f"-- not independent replicates); lower rank = better. "
                 f"Significance: paired bootstrap + per-fold Nadeau-Bengio",fontsize=9)
    fig.tight_layout(); fig.savefig(FIG/"fig_cd_diagram_j.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # pairwise bootstrap dMAE vs TabPFN (and full matrix)
    rng=np.random.default_rng(42); B=10000
    idx=rng.integers(0,n,size=(B,n))
    mae_bs=np.array([E[:,idx[b]].mean(1) for b in range(B)])  # (B,k)
    rows=[]
    ti=labels.index("TabPFN") if "TabPFN" in labels else int(order[0])
    for i in range(k):
        if i==ti: continue
        d=mae_bs[:,i]-mae_bs[:,ti]  # >0 means model i worse than TabPFN
        ci=np.percentile(d,[2.5,97.5]); pv=2*min((d<0).mean(),(d>0).mean())
        rows.append({"model":labels[i],"dMAE_vs_TabPFN":round(float(d.mean()),3),
                     "CI_lo":round(ci[0],3),"CI_hi":round(ci[1],3),"p":round(float(pv),4),
                     "sig_bonf":bool(pv<0.05/(k-1))})
    T=pd.DataFrame(rows).sort_values("dMAE_vs_TabPFN"); T.to_csv(FINAL/"significance_vs_tabpfn.csv",index=False)
    print(T.to_string(index=False))
    print("wrote fig_cd_diagram_j.png + significance_vs_tabpfn.csv")

if __name__=="__main__": main()
