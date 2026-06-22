"""EDA Step 1 (part 2): quality dashboard (F5) + descriptor distributions (F7) +
label distribution & conflicts (F8). Reads the cleaned data (data/processed/bace_clean.csv,
1504 molecules) produced by eda_bace.py. Statistical plots use scienceplots + Times New Roman.

Run: /opt/anaconda3/bin/python3 scripts/eda_bace_part2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401
from scipy.stats import pearsonr  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, DataStructs, Crippen, Descriptors, rdMolDescriptors, QED  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": True})
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)

DESCS = ["MW", "cLogP", "TPSA", "HBD", "HBA", "RotB", "Rings", "AromRings",
         "HeavyAtoms", "Fsp3", "QED", "Stereo"]


def descriptors(m):
    return [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
            rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumAromaticRings(m), m.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcFractionCSP3(m), QED.qed(m),
            rdMolDescriptors.CalcNumAtomStereoCenters(m)]


def main():
    df = pd.read_csv("data/processed/bace_clean.csv")
    mols = [Chem.MolFromSmiles(s) for s in df.smiles]
    y = df.label.to_numpy(dtype=float); split = df.set.to_numpy()
    D = pd.DataFrame([descriptors(m) for m in mols], columns=DESCS)

    # ---- F7: descriptor distributions (3x4) ----
    fig, axs = plt.subplots(3, 4, figsize=(12, 7.4))
    for ax, name in zip(axs.ravel(), DESCS):
        v = D[name].to_numpy(dtype=float)
        ax.hist(v, bins=26, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(v), ls="--", color="#9b2226", lw=1.0)
        med = np.median(v)
        ax.set_title(f"{name}  (med {med:.0f})" if med > 20 else f"{name}  (med {med:.2f})",
                     fontsize=9)
        ax.set_ylabel("count", fontsize=7)
    fig.suptitle("BACE descriptor distributions (1504 molecules; red = median)",
                 fontsize=12, y=1.00)
    fig.tight_layout(); fig.savefig(FIG / "eda_descriptors_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote eda_descriptors_j.png")

    # ---- F8: label distribution + HONEST two-level noise analysis ----
    # canonical (same exact molecule incl. stereo) = measurement/curation noise (the floor)
    # InChIKey14 (skeleton, ignores stereo/tautomer) = noise + REAL stereo/tautomer SAR (upper bound)
    from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: PLC0415
    raw = pd.read_csv("data/BACE.split.csv")
    lfc = rdMolStandardize.LargestFragmentChooser(); unch = rdMolStandardize.Uncharger()
    taut = rdMolStandardize.TautomerEnumerator()
    can_g, ik_g = {}, {}
    for s, lab in zip(raw.smiles, raw.label):
        mm = Chem.MolFromSmiles(s)
        if mm is None:
            continue
        mm = unch.uncharge(lfc.choose(mm))
        try:
            mm = taut.Canonicalize(mm)
        except Exception:
            pass
        can_g.setdefault(Chem.MolToSmiles(mm), []).append(float(lab))
        try:
            kk = Chem.MolToInchiKey(mm)[:14]
        except Exception:
            kk = None
        if kk:
            ik_g.setdefault(kk, []).append(float(lab))
    can_d = [max(v) - min(v) for v in can_g.values() if len(v) > 1]
    n_can = sum(1 for d in can_d if d > 0)
    noise = float(np.median([d for d in can_d if d > 0])) if any(d > 0 for d in can_d) else 0.0
    ik_conf = [(min(v), max(v)) for v in ik_g.values() if len(v) > 1 and (max(v) - min(v)) > 0.5]
    n_conf = len(ik_conf)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(y.min(), y.max(), 30)
    for sp, c in [("train", "#4C72B0"), ("validation", "#DD8452"), ("test", "#55A868")]:
        v = y[split == sp]
        a1.hist(v, bins=bins, density=True, alpha=0.45, color=c,
                label=f"{sp} (n={len(v)})", edgecolor="white", linewidth=0.3)
    a1.axvline(y.mean(), ls="--", color="k", lw=0.9)
    a1.set_xlabel(r"pIC$_{50}$"); a1.set_ylabel("density")
    a1.set_title("Label distribution by split"); a1.legend(fontsize=8)
    if ik_conf:
        lo = np.array([c[0] for c in ik_conf]); hi = np.array([c[1] for c in ik_conf])
        a2.scatter(lo, hi, s=22, color="#9b2226", alpha=0.6, edgecolor="k", linewidth=0.3)
        lim = [y.min(), y.max()]; a2.plot(lim, lim, ls="--", color="0.5", lw=0.9)
    a2.set_xlabel(r"min pIC$_{50}$ in group"); a2.set_ylabel(r"max pIC$_{50}$ in group")
    a2.set_title(f"Skeleton-level (InChIKey$_{{14}}$): {n_conf} variant pairs\n"
                 f"$\\Leftarrow$ incl. real stereo/tautomer SAR (NOT pure noise)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "eda_label_dist_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote eda_label_dist_j.png | canonical-level (same molecule) conflicts={n_can}, "
          f"noise floor={noise:.2f} log | InChIKey14 skeleton variation={n_conf}")

    # ---- features for leakage / scaffold ----
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in mols]
    tr = split == "train"; te = split == "test"
    fp_tr = [f for f, t in zip(fps, tr) if t]; fp_te = [f for f, t in zip(fps, te) if t]
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(f, fp_tr)) for f in fp_te])
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    s_tr = set(s for s, t in zip(scaf, tr) if t)
    s_te = [s for s, t in zip(scaf, te) if t]
    overlap = 100 * np.mean([s in s_tr for s in s_te])

    # rule compliance
    lip = ((D.MW < 500) & (D.cLogP < 5) & (D.HBD <= 5) & (D.HBA <= 10)).mean() * 100
    veb = ((D.RotB <= 10) & (D.TPSA <= 140)).mean() * 100
    # descriptor-pIC50 abs correlation
    cors = {n: abs(pearsonr(D[n], y)[0]) for n in DESCS}
    cors = dict(sorted(cors.items(), key=lambda x: -x[1]))

    # ---- F5: quality dashboard (2x3) ----
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    a, b, c, d, e, f = axs.ravel()
    # (a) data-quality counts
    qk = ["parse\nfail", "canonical\ndup", "tautomer\ndup", "label\nnoise"]
    qv = [0, 9, 39, n_can]
    a.bar(range(4), qv, color=["#55A868", "#C44E52", "#C44E52", "#DD8452"], edgecolor="k", linewidth=0.4)
    a.set_xticks(range(4)); a.set_xticklabels(qk, fontsize=8)
    a.set_title("Data-quality issues (count)"); a.set_ylabel("molecules")
    for i, v in enumerate(qv): a.annotate(str(v), (i, v + 0.5), ha="center", fontsize=8)
    # (b) stereo: molecules with >=1 UNSPECIFIED stereocenter (same definition as the L2 audit,
    #     so the dashboard agrees with the reported ~75% undefined headline; NOT "contains @")
    n_undef = int(sum(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(m) > 0 for m in mols))
    n_def = len(mols) - n_undef
    b.pie([n_undef, n_def], labels=[f"undefined\n{n_undef}", f"fully\ndefined\n{n_def}"],
          colors=["#bbbbbb", "#4C72B0"], autopct="%1.0f%%", textprops={"fontsize": 8})
    b.set_title(f"Stereo: $\\geq$1 unspecified center ({100*n_undef/len(mols):.0f}%)", fontsize=9)
    # (c) nearest-train Tanimoto (leakage)
    c.hist(nn, bins=26, color="#8172B3", alpha=0.85, edgecolor="white", linewidth=0.3)
    c.axvline(0.85, ls="--", color="#9b2226", lw=1.1); c.axvspan(0.85, 1, color="#9b2226", alpha=0.1)
    c.set_title(f"Test nearest-train Tanimoto\n(>0.85: {100*(nn>0.85).mean():.0f}%)", fontsize=9)
    c.set_xlabel("Tanimoto")
    # (d) scaffold overlap
    d.bar([0, 1], [overlap, 100 - overlap], color=["#C44E52", "#4C72B0"], edgecolor="k", linewidth=0.4)
    d.set_xticks([0, 1]); d.set_xticklabels(["test scaffold\nin train", "novel\nscaffold"], fontsize=8)
    d.set_title(f"Predefined-split scaffold overlap\n({overlap:.0f}% test in train)", fontsize=9)
    d.set_ylabel("%")
    # (e) drug-likeness compliance
    e.bar([0, 1], [lip, veb], color="#55A868", edgecolor="k", linewidth=0.4)
    e.set_xticks([0, 1]); e.set_xticklabels(["Lipinski Ro5", "Veber"], fontsize=8.5)
    e.set_ylim(0, 100); e.set_title("Drug-likeness pass rate"); e.set_ylabel("%")
    for i, v in enumerate([lip, veb]): e.annotate(f"{v:.0f}%", (i, v + 1), ha="center", fontsize=8)
    # (f) descriptor-pIC50 correlation
    names = list(cors)[:8][::-1]; vals = [cors[n] for n in names]
    f.barh(range(len(names)), vals, color="#4C72B0", edgecolor="k", linewidth=0.4)
    f.set_yticks(range(len(names))); f.set_yticklabels(names, fontsize=8)
    f.set_xlabel(r"$|$Pearson$|$ with pIC$_{50}$"); f.set_title("Descriptor $\\leftrightarrow$ activity", fontsize=9)
    fig.suptitle("BACE data-quality dashboard (1504 molecules)", fontsize=12, y=1.00)
    fig.tight_layout(); fig.savefig(FIG / "eda_quality_dashboard_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote eda_quality_dashboard_j.png")

    print(f"\nsummary: Lipinski {lip:.0f}%, Veber {veb:.0f}%, scaffold overlap {overlap:.0f}%, "
          f"test NN>0.85 {100*(nn>0.85).mean():.0f}%, label-noise floor {noise:.2f}")


if __name__ == "__main__":
    main()
