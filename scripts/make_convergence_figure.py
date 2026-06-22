"""Training-convergence verification figure (scienceplots + Times New Roman).

Reads the per-epoch curves under results/convergence/ (3 seeds each, all from real
training logs -- neural models parsed from server logs, RF/Chemprop recomputed)
and renders a 2x3 panel proving each representation trained to a plateau:

  (a) RF + Morgan : val RMSE vs number of trees      (its convergence analog)
  (b) MolFormer-XL: val RMSE vs epoch + train loss   (full fine-tune, noisy)
  (c) ChemFM-3B   : val RMSE vs epoch + train loss   (LoRA, clean plateau)
  (d) Chemprop    : val loss  vs epoch + train loss   (D-MPNN, final = plateau)
  (e) Uni-Mol     : val loss  vs epoch + train loss   (internal normalized val)
  (f) normalized overlay: fraction of improvement achieved vs fraction of budget

Each panel: 3-seed mean (solid) +/- 1 s.d. band; train loss on a faded twin axis;
a star at the selected/best epoch. No fabricated numbers.

Run: python scripts/make_convergence_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": True,
})

CONV = Path("results/convergence")
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)

COLOR = {"rf": "#7f7f7f", "chemprop": "#CC79A7", "molformer": "#0072B2",
         "chemfm": "#009E73", "unimol": "#D55E00"}
SEEDS = [42, 1337, 2024]


def load_epoch(model: str, val_col: str) -> pd.DataFrame:
    """Stack the 3-seed per-epoch curves; return mean/std by epoch."""
    frames = []
    for s in SEEDS:
        d = pd.read_csv(CONV / f"{model}_bace_seed{s}.csv")
        frames.append(d.assign(seed=s))
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby("epoch").agg(
        val_m=(val_col, "mean"), val_s=(val_col, "std"),
        tr_m=("train_loss" if "train_loss" in df else "train_loss_epoch", "mean"),
    ).reset_index()
    return g


def panel_epoch(ax, model, g, ylabel, title, subtitle):
    c = COLOR[model]
    x = g["epoch"].to_numpy()
    v = g["val_m"].to_numpy(); s = g["val_s"].to_numpy()
    # val curve + 3-seed band
    ax.plot(x, v, "-", color=c, lw=1.6, marker="o", ms=2.6, zorder=4, label="val (3-seed mean)")
    ax.fill_between(x, v - s, v + s, color=c, alpha=0.16, lw=0, zorder=2)
    # best/selected epoch
    bi = int(np.nanargmin(v)); bx, by = x[bi], v[bi]
    ax.axvline(bx, ls=":", color="0.45", lw=0.8, zorder=1)
    ax.plot([bx], [by], marker="*", ms=11, color=c, mec="k", mew=0.4, zorder=5)
    ax.annotate(f"best {by:.3f}\n@ep{bx}", (bx, by), textcoords="offset points",
                xytext=(6, 10), fontsize=6.5, color="0.2")
    # train loss on faded twin axis (drop leading warmup spike for readability)
    ax2 = ax.twinx()
    tr = g["tr_m"].to_numpy()
    med = np.nanmedian(tr)
    keep = tr <= 5 * med
    ax2.plot(x[keep], tr[keep], ls="--", color=c, alpha=0.40, lw=1.0, zorder=1)
    ax2.set_ylabel("train loss", color="0.5", fontsize=7)
    ax2.tick_params(axis="y", labelcolor="0.5", labelsize=6)
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, pad=8)
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.6, color="0.35")
    return ax2


def main():
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.4))
    a, b, c, d, e, f = axs.ravel()

    # (a) RF learning curve: val RMSE vs n_trees
    rf = pd.read_csv(CONV / "rf_bace_learning_curve.csv")
    rg = rf.groupby("n_trees").agg(m=("val_rmse", "mean"), s=("val_rmse", "std")).reset_index()
    x = rg["n_trees"].to_numpy()
    a.plot(x, rg["m"], "-o", color=COLOR["rf"], lw=1.6, ms=3, zorder=4)
    a.fill_between(x, rg["m"] - rg["s"], rg["m"] + rg["s"], color=COLOR["rf"], alpha=0.16, lw=0)
    # "enough trees": first n where within 1% of the 500-tree value
    final = rg["m"].iloc[-1]
    enough = int(x[np.argmax(rg["m"].to_numpy() <= final * 1.01)])
    a.axvline(enough, ls=":", color="0.45", lw=0.8)
    a.annotate(f"within 1% of\nfinal by ~{enough} trees", (enough, final),
               textcoords="offset points", xytext=(10, 18), fontsize=6.5, color="0.2")
    a.set_xlabel("number of trees"); a.set_ylabel(r"val RMSE $\downarrow$")
    a.set_title("(a) RF + Morgan", fontsize=9, pad=8)
    a.text(0.5, 1.005, "from-scratch ensemble; plateaus early", transform=a.transAxes,
           ha="center", va="bottom", fontsize=6.6, color="0.35")

    # (b) MolFormer  (c) ChemFM  -> val RMSE (pIC50 scale)
    gm = load_epoch("molformer", "val_metric")
    panel_epoch(b, "molformer", gm, r"val RMSE $\downarrow$", "(b) MolFormer-XL",
                "base model, full fine-tune (44.6M); noisy val -> best-on-val")
    b.set_ylim(0.82, 1.9)  # plateau lives at 0.85-1.05; ep1-2 (~3.5) clipped for clarity
    b.annotate("ep1-2 off-scale (~3.5)", (0.97, 0.96), xycoords="axes fraction",
               ha="right", va="top", fontsize=6, color="0.45")
    gc = load_epoch("chemfm", "val_metric")
    panel_epoch(c, "chemfm", gc, r"val RMSE $\downarrow$", "(c) ChemFM-3B",
                "base model, LoRA (6.1M / 3.0B = 0.2%); clean plateau")

    # (d) Chemprop -> val loss
    gd = load_epoch("chemprop", "val_loss")
    panel_epoch(d, "chemprop", gd, r"val loss $\downarrow$", "(d) Chemprop (D-MPNN)",
                "from-scratch GNN; gap closed: ep50 sits on plateau")
    f.axis("off")  # Uni-Mol removed -> 6th cell blank

    # (e) normalized overlay: fraction of improvement achieved vs fraction of budget
    def norm_curve(g, xcol):
        x = g[xcol].to_numpy(dtype=float); v = g["val_m"].to_numpy(dtype=float)
        xf = (x - x.min()) / (x.max() - x.min())
        frac = (v[0] - v) / (v[0] - v.min())   # 0 -> 1 as it converges
        return xf, frac
    series = [("rf", rg.rename(columns={"m": "val_m"}), "n_trees"),
              ("molformer", gm, "epoch"), ("chemfm", gc, "epoch"),
              ("chemprop", gd, "epoch")]
    label = {"rf": "RF", "molformer": "MolFormer", "chemfm": "ChemFM",
             "chemprop": "Chemprop"}
    for name, g, xc in series:
        xf, frac = norm_curve(g, xc)
        e.plot(xf, frac, "-", color=COLOR[name], lw=1.5, label=label[name])
    e.axhline(1.0, ls=":", color="0.6", lw=0.8)
    e.set_xlabel("fraction of training budget")
    e.set_ylabel("fraction of improvement achieved")
    e.set_title("(e) convergence speed (normalized)", fontsize=9, pad=8)
    e.set_ylim(-0.03, 1.08)
    e.legend(fontsize=6.5, loc="lower right", ncol=1, framealpha=0.9)

    fig.suptitle("Training convergence verification on BACE "
                 "(4 representations, 3 seeds: mean $\\pm$ s.d.)",
                 fontsize=12, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = FIG / "fig_convergence_j.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
