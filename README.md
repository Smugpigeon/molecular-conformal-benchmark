# Fudan AIDD: BACE-1 Property Prediction + Conformal UQ Benchmark

复旦大学药学院《AI 赋能药物设计发现前沿》期末项目。本仓库有两条交付线：

- **课程交付（主线）**：BACE-1 pIC50 单数据集端到端 pipeline —— EDA、8 个模型基准（经典 QSAR / Chemprop / MolFormer-XL / ChemFM-3B / Uni-Mol / TabPFN）、split-conformal 可信区间、4D8C 结构对接、可解释性 / 分类 / 筛选 / 指纹消融。产物为课程报告 docx + 汇报 pptx（见 `report/`）。
- **论文交付（并行）**：跨分子基础模型的 **conformal 预测区间质量基准**（6 个回归数据集：ESOL / FreeSolv / Lipophilicity / BACE / QM7 / QM8），目标投稿 **J. Chem. Inf. Model.**。论文仓库的 README 见 [README_paper.md](README_paper.md)（公开发布时改名为 `README.md`）。

> 注：早期计划的「GSHt 共价反应性」方向已放弃，GSHt 数据集在论文流程中被隔离，不进入主文 / SI 主表（见 CLAUDE.md §1）。

## Quick Start

```bash
mamba env create -f env.yml -n drug
conda activate drug && pip install -e .
make baseline-rf                # RF + Morgan FP baseline
pytest -q tests/                # sanity check（可选课程分类数据缺失时会 skip）
```

## What's Here

| 路径 | 用途 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | **项目规矩**（Claude Code 自动加载，开发前必读） |
| [项目调研汇总.md](项目调研汇总.md) | SOTA 论文索引 + 方案对比 + 计划 |
| [data/](data/) | 老师给的 6 个数据集 CSV，只读 |
| [src/](src/) | 主代码（`pip install -e .` 后可 import）|
| [configs/](configs/) | Hydra YAML 配置 |
| [scripts/](scripts/) | 一键 shell / 分析脚本 |
| [results/final/](results/final/) | 论文真值表（CSV）—— 单一数据来源 |
| [paper/](paper/) | achemso LaTeX 手稿 + SI + refs.bib + Overleaf 包 |
| [report/](report/) | 课程报告 docx + 汇报 pptx |
| [tests/](tests/) | pytest 单元测试 |

## Datasets

| 数据集 | 任务 | 分子数 | 用途 |
|---|---|---|---|
| ESOL | 水溶性 logS（回归）| 1128 | 论文 conformal 基准 |
| FreeSolv | 水合自由能（回归）| 642 | 论文 conformal 基准 |
| Lipophilicity | logD@pH7.4（回归）| 4200 | 论文 conformal 基准 |
| BACE | β-secretase pIC50（回归）| 1513 | 课程主线 + 论文 |
| QM7 | 原子化能（回归）| 6832 | 论文 conformal 基准 |
| QM8 | 电子谱能 E1-CC2（回归）| 21766 | 论文 conformal 基准 |

完整 SOTA 数字和论文引用（含 split / metric / n）见 [项目调研汇总.md](项目调研汇总.md)。

## Repro Checklist (JCIM)

提交论文前 `bash scripts/check_repro.sh` 自动检查所有条目（见 CLAUDE.md §13）。论文的全部表 / 图由 `scripts/` 从 `results/final/` 重生成。

## License

MIT（见 [LICENSE](LICENSE)）。`data/` 内 MoleculeNet CSV 沿用各自上游许可。
