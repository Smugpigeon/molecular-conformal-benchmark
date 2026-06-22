"""EDA Step 1 for BACE-1 (design doc §3): preprocessing funnel + Murcko scaffold
grid + chemical-space t-SNE (with split overlay) + activity-cliff network, plus a
cleaning log. Real data only (1513 BACE molecules from data/BACE.split.csv).

Convention (per BACE_项目设计方案.md): statistical/data plots use scienceplots +
Times New Roman; molecule structures use RDKit (scienceplots can't draw molecules).

Run: /opt/anaconda3/bin/python3 scripts/eda_bace.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401
from matplotlib.collections import LineCollection  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, DataStructs, Draw  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

RDLogger.DisableLog("rdApp.*")
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix", "axes.unicode_minus": True,
})
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
OUT = Path("results/eda"); OUT.mkdir(parents=True, exist_ok=True)
PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv("data/BACE.split.csv")
    n0 = len(df)
    recs = [{"smiles": s, "label": float(l), "set": sp}
            for s, l, sp in zip(df.smiles, df.label, df.set)]
    log = []   # (step, n_in, n_out, n_affected, note)

    # ---------- L1: cleaning pipeline (funnel) ----------
    for r in recs:
        r["mol"] = Chem.MolFromSmiles(r["smiles"])
    recs = [r for r in recs if r["mol"] is not None]
    log.append(("① 解析 SMILES", n0, len(recs), n0 - len(recs), "解析失败被剔除"))

    lfc = rdMolStandardize.LargestFragmentChooser()
    n_multi = 0
    for r in recs:
        if len(Chem.GetMolFrags(r["mol"])) > 1:
            n_multi += 1
        r["mol"] = lfc.choose(r["mol"])
    log.append(("② 去盐/取最大片段", len(recs), len(recs), n_multi, "含多片段→取最大"))

    un = rdMolStandardize.Uncharger()
    n_charged = 0
    for r in recs:
        b = Chem.MolToSmiles(r["mol"])
        r["mol"] = un.uncharge(r["mol"])
        if Chem.MolToSmiles(r["mol"]) != b:
            n_charged += 1
    log.append(("③ 中和电荷", len(recs), len(recs), n_charged, "带形式电荷→中和"))

    te = rdMolStandardize.TautomerEnumerator()
    n_taut = 0
    for r in recs:
        b = Chem.MolToSmiles(r["mol"])
        try:
            r["mol"] = te.Canonicalize(r["mol"])
        except Exception:
            pass
        if Chem.MolToSmiles(r["mol"]) != b:
            n_taut += 1
    log.append(("④ 互变异构标准化", len(recs), len(recs), n_taut, "归一到 canonical 互变异构体"))

    for r in recs:
        r["can"] = Chem.MolToSmiles(r["mol"], canonical=True)
    seen, dedup, n_dup = {}, [], 0
    for r in recs:
        if r["can"] in seen:
            n_dup += 1
        else:
            seen[r["can"]] = 1
            dedup.append(r)
    before = len(recs); recs = dedup
    log.append(("⑤ canonical 去重", before, len(recs), n_dup, "标准化后重复→去除"))

    # ---------- L2: quality audit ----------
    ik14 = {}
    for r in recs:
        try:
            k = Chem.MolToInchiKey(r["mol"])[:14]
        except Exception:
            k = None
        r["ik14"] = k
        ik14.setdefault(k, []).append(r["label"])
    n_taut_dup = sum(len(v) - 1 for k, v in ik14.items() if k and len(v) > 1)
    conflicts = [(k, max(v) - min(v)) for k, v in ik14.items()
                 if k and len(v) > 1 and (max(v) - min(v)) > 0.5]
    n_nostereo = sum(("@" not in r["can"]) for r in recs)
    audit = {
        "n_clean": len(recs), "canonical_dups_removed": n_dup,
        "tautomer_InChIKey14_dups": n_taut_dup,
        "label_conflicts(>0.5 log)": len(conflicts),
        "stereo_undefined_pct": round(100 * n_nostereo / len(recs), 1),
    }

    # save cleaned data + logs
    pd.DataFrame([{"smiles": r["can"], "label": r["label"], "set": r["set"]} for r in recs]
                 ).to_csv(PROC / "bace_clean.csv", index=False)
    logdf = pd.DataFrame(log, columns=["步骤", "进入", "保留", "影响数", "说明"])
    logdf.to_csv(OUT / "preprocessing_log.csv", index=False)
    pd.DataFrame([audit]).to_csv(OUT / "quality_audit.csv", index=False)

    y = np.array([r["label"] for r in recs], dtype=float)
    split = np.array([r["set"] for r in recs])

    # ---------- features: Morgan FP ----------
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(r["mol"]) for r in recs]
    X = np.zeros((len(fps), 2048), dtype=np.int8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, X[i])

    # ===================== F1: preprocessing funnel =====================
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    stages_en = ["1. Parse SMILES", "2. Desalt (largest frag)", "3. Neutralize charges",
                 "4. Canonical tautomer", "5. Dedup (canonical)"]
    kept = [r[2] for r in log]; aff = [r[3] for r in log]
    ypos = np.arange(len(stages_en))[::-1]
    ax.barh(ypos, kept, color="#4C72B0", edgecolor="k", linewidth=0.4, height=0.6)
    for yp, k, a in zip(ypos, kept, aff):
        ax.text(k + 8, yp, f"{k}" + (f"  ($-$/mod {a})" if a else ""), va="center", fontsize=8)
    ax.axvline(n0, ls=":", color="0.5", lw=0.8)
    ax.set_yticks(ypos); ax.set_yticklabels(stages_en, fontsize=8.5)
    ax.set_xlabel("molecules retained"); ax.set_xlim(0, n0 * 1.20)
    ax.set_title(f"BACE preprocessing funnel: {n0} $\\to$ {len(recs)} clean molecules")
    fig.tight_layout(); fig.savefig(FIG / "eda_funnel_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote eda_funnel_j.png")

    # ===================== F2: Murcko scaffold grid (RDKit) =====================
    scaf = []
    for r in recs:
        try:
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=r["mol"]))
        except Exception:
            scaf.append("")
    sdf = pd.DataFrame({"scaf": scaf, "y": y})
    vc = pd.Series(scaf).value_counts()
    top = [(s, c) for s, c in vc.items() if s][:12]
    n_unique = int((vc.index != "").sum()); n_single = int((vc == 1).sum())
    gmols = [Chem.MolFromSmiles(s) for s, _ in top]
    legends = [f"#{i+1}  n={c}, pIC50={sdf[sdf.scaf == s].y.mean():.1f}"
               for i, (s, c) in enumerate(top)]
    img = Draw.MolsToGridImage(gmols, legends=legends, molsPerRow=4, subImgSize=(270, 200))
    img.save(str(FIG / "eda_scaffold_grid.png"))
    print(f"wrote eda_scaffold_grid.png  (unique scaffolds={n_unique}, singletons={n_single})")

    # ===================== t-SNE embedding (shared by F4, F6) =====================
    emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
               random_state=42).fit_transform(X.astype(float))

    # ===================== F4: chemical space (pIC50 + split overlay) =====================
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    sc = a1.scatter(emb[:, 0], emb[:, 1], c=y, cmap="viridis", s=11,
                    edgecolor="k", linewidth=0.15)
    a1.set_title(r"Chemical space colored by pIC$_{50}$"); a1.set_xlabel("t-SNE 1"); a1.set_ylabel("t-SNE 2")
    fig.colorbar(sc, ax=a1, fraction=0.046, pad=0.04).set_label(r"pIC$_{50}$")
    colmap = {"train": "#4C72B0", "validation": "#DD8452", "test": "#C44E52"}
    for sp in ["train", "validation", "test"]:
        m = split == sp
        a2.scatter(emb[m, 0], emb[m, 1], s=11, color=colmap[sp], alpha=0.65,
                   edgecolor="white", linewidth=0.2, label=f"{sp} (n={int(m.sum())})")
    a2.set_title("Train / val / test overlap (leakage view)")
    a2.set_xlabel("t-SNE 1"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "eda_tsne_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote eda_tsne_j.png")

    # ===================== F6: activity-cliff network =====================
    edges = []   # (i, j, sim, dy)
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for k, s in enumerate(sims):
            if s > 0.60:
                j = i + 1 + k
                edges.append((i, j, s, abs(y[i] - y[j])))
    cliffs = [(i, j, s, dy) for (i, j, s, dy) in edges if s > 0.70 and dy > 2.0]
    # SALI for top cliffs
    sali = sorted(((dy / (1 - s + 1e-6), i, j, s, dy) for (i, j, s, dy) in edges if s > 0.6 and dy > 1.0),
                  reverse=True)[:25]
    pd.DataFrame([{"SALI": round(v, 1), "smiles_A": recs[i]["can"], "smiles_B": recs[j]["can"],
                   "Tanimoto": round(s, 3), "delta_pIC50": round(dy, 2)}
                  for v, i, j, s, dy in sali]).to_csv(OUT / "activity_cliffs_top.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.scatter(emb[:, 0], emb[:, 1], s=5, color="0.8", zorder=1)
    segs = [[(emb[i, 0], emb[i, 1]), (emb[j, 0], emb[j, 1])] for (i, j, s, dy) in edges]
    lc = LineCollection(segs, array=np.array([dy for *_, dy in edges]), cmap="RdYlBu_r",
                        linewidths=0.5, alpha=0.55, zorder=2)
    ax.add_collection(lc)
    for (i, j, s, dy) in cliffs:
        ax.plot([emb[i, 0], emb[j, 0]], [emb[i, 1], emb[j, 1]], color="#9b2226",
                lw=1.3, zorder=3)
    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$|\Delta$pIC$_{50}|$ per edge")
    ax.set_title(f"BACE activity-cliff landscape "
                 f"(edges Tanimoto$>$0.6: {len(edges)}; red cliffs: {len(cliffs)})", fontsize=10)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    fig.tight_layout(); fig.savefig(FIG / "eda_cliff_network_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote eda_cliff_network_j.png")

    # ---------- summary ----------
    print("\n=== 预处理日志 ===")
    print(logdf.to_string(index=False))
    print("\n=== 质量审计 ===", audit)
    print(f"=== 活性悬崖：高相似边 {len(edges)}，悬崖对(sim>0.7 & Δ>2) {len(cliffs)} ===")
    print(f"=== 化学多样性：{n_unique} 独特骨架，{n_single} 单例 "
          f"({100*n_single/len(recs):.0f}%) ===")


if __name__ == "__main__":
    main()
