# BACE-1 课程项目设计方案（主文档）

> 复旦《AI 赋能药物发现前沿》(PHAR30050) 期末项目 · 汤雨凡 23307130372
> 本文件是项目的"主设计图"：对齐课程打分点、把数据分析/调参做到极致、并接入 PBCNet2 的 novelty。
> **画图约定**：所有统计/数据图一律用 **scienceplots**（`plt.style.use(["science","no-latex"])` + Times New Roman，与现有 `results/figures/*_j.png` 一致）；**分子结构图用 RDKit**（`MolsToGridImage`，scienceplots 画不了分子）。
> 配套文件：[CLAUDE.md](CLAUDE.md)（做事规矩）、[项目调研汇总.md](项目调研汇总.md)（调研）。

---

## 0. 一句话定位

在课程给的 **BACE 数据集（1513 个分子）** 上做 **BACE-1 抑制活性 pIC50 的回归预测**，把作业要求的 **数据整理 / 模型构建 / 训练 / 分析** 四项做到极致（对齐课程教学内容、不堆高大上），并在"分析"层接入 **PBCNet2（结构-based 相对亲和力）** 做一个别人做不了的 novelty。

---

## 1. 老师怎么打分（从大纲 + 第五讲提取）

**本项目 = 大纲里 40% 的「课题研讨 / 汇报」**（另有 40% 是读经典论文的读书报告，与本项目无关）。第五讲原文：

> "选择课程介绍的某项 AI 药物设计预测任务，完成 **①数据整理 ②模型构建 ③训练 ④分析**，期末口头汇报（每人都讲）+ 撰写书面实验报告。"

**BACE 任务定义**（第五讲官方表）：预测 BACE-1 的 IC50，**1513 分子，回归，官方指标 = MAE + Pearson R**。

**官方"两种策略"**（第五讲）：
- **策略 1**：SMILES → 编码器（指纹/描述符，RDKit/DeepChem） → ML（**随机森林 / 梯度提升 / 支持向量机**，或神经网络）
- **策略 2**：SMILES → 二维分子图 → **GNN**

### 1.1 打分点 × 课程周 × 三位老师 对照表（报告里贴这张）

| 完成步骤 | 课程依据（周/讲） | 对应老师 |
|---|---|---|
| 选题（BACE，锁钥/靶向） | 第1讲 导论、CADD | **王任小** |
| 数据整理（描述符 / RDKit / 降维） | 第2讲 描述符、第5讲 RDKit-DeepChem、第3讲 PCA-tSNE | 王任小 / **戚逸飞** |
| 模型构建（两种策略） | 第5讲 两策略、第3讲 ML+GNN | **戚逸飞** |
| 训练 + 调参 + 防过拟合 | 第3讲 train/val/test、调参、过拟合 | **戚逸飞** |
| 分析（评估指标） | 第3讲 评估、第5讲 官方指标 | **戚逸飞** |
| 结构对接 / 亲合性（加分） | 第4讲 对接+打分函数+PLANET、第1讲 CADD | **李嫣 / 王任小** |
| Novelty（PBCNet2 桥接） | 第4讲 亲合性预测（PLANET/PBCNet 本课题组） | **李嫣** |
| 汇报 + 报告 | 第16讲 | 三人 |

> **核心洞察**：本项目横跨三位老师的内容（王任小=选题/CADD/打分函数；戚逸飞=ML/调参/GNN/RDKit；李嫣=亲合性预测/对接/PBCNet2）。贴这张表 = 老师一眼看到"课程教的我全用上了"。

---

## 2. 总体架构（一条龙）

```
数据整理(EDA)─→ 表征/建模 ─→ 调参+训练 ─→ 评估分析 ─→ 结构对接 ─→ Novelty(PBCNet2) ─→ 报告/汇报
  第2/4/5讲       第2/3/5讲     第3讲        第4讲       第11/12讲     第4讲              第16讲
```

模块：① 数据整理（§3） ② 模型构建（§4） ③ 训练+调参（§5） ④ 评估分析（§6） ⑤ 结构对接（§7） ⑥ Novelty（§8）。

---

## 3. 数据整理（详细 EDA + 预处理）—— 打分点①

### 3.1 关键原则：分析与预处理的顺序

```
基础清洗(能解析的分子) ─→ EDA 分析(侦察) ─┐
        ↑                                    │ 发现问题
        └──── 精细预处理(施工,按侦察结果) ←─┘
                        ↓
              特征化 + 划分(fit 只在 train!)
```

- **基础清洗先行**（要有能被 RDKit 读出来的分子才能分析）；
- **EDA 是预处理的"导航"**：先分析才知道该怎么处理（71% 立体未定义 → 决定保留 `@/@@`；标签冲突 → 决定怎么合并；骨架泄漏 → 决定要不要补 scaffold split）；
- **唯一红线（第3讲 防过拟合）**：任何需要 fit 的变换（标准化/特征选择/PCA 做特征）**只在训练集上 fit**，绝不在划分前对全量 fit。

### 3.2 十层流水线

| 层 | 做什么 | 课件 |
|---|---|---|
| L0 原始契约 | 1513 / pIC50 / 预定义切分；列含义、来源（MoleculeNet, Subramanian 2016） | — |
| L1 解析+标准化清洗 | RDKit 解析（失败记日志）、canonical 化、**互变异构标准化**（MolStandardize）、**去盐/取最大片段+中和**、立体保留 `@/@@` | 第5讲 RDKit |
| L2 数据质量审计 | canonical 重复、**InChIKey₁₄ 互变异构重复**、**🌟标签冲突检测**（同分子不同 pIC50 → 量出内禀噪声） | 第3讲 |
| L3 标签画像 | pIC50 直方图/KDE、按 split、偏度/离群、噪声 σ → "RMSE/噪声下限" | — |
| L4 分子描述符画像 | 12 描述符（MW/cLogP/TPSA/HBD/HBA/可旋转键/环数/芳香环/重原子/Fsp³/QED/立体中心）+ 与 pIC50 相关 + 描述符相关矩阵 + Lipinski/Veber | **第2讲 描述符核心** |
| L5 化学空间 | Morgan FP → **PCA/t-SNE/UMAP** 按 pIC50 上色 + train/val/test 叠加（泄漏可视化） | 第3讲 降维 |
| L6 骨架与多样性 | Bemis-Murcko：独特数/频率/单例/Top-N、骨架级活性 | — |
| L7 泄漏+划分审计 | 训练-测试 canonical/tautomer 重复、Murcko 骨架重叠、最近邻 Tanimoto 分布 → 量化泄漏 + 另建 scaffold split | 第3讲 train/val/test |
| L8 SAR + 活性悬崖 | **活性悬崖**（SALI：高 Tanimoto + 大 ΔpIC50）、**🌟子结构富集**（卡方/优势比，哪些片段→活性） | 第1/4讲 SAR |
| L9 特征化(leak-safe) | Morgan / 描述符 / 二维图 / tokenize / 3D 构象；fit 只在 train | 第5讲 两策略 |
| L10 产出+可复现 | `data/processed/*`、`notebooks/01_EDA.ipynb`、`make data`、清洗日志 | 第5讲 |

### 3.3 "高大上但不浮夸"的图清单（数据整理交付）

| # | 图 | 内容 | 工具 |
|---|---|---|---|
| F1 | **预处理漏斗 / Sankey** | 1513 → 解析失败 → 去重 → 去盐 → 标准化 → N clean | **scienceplots**（瀑布）/ matplotlib-sankey |
| F2 | **🌟 骨架结构网格** | Top-12 Murcko 骨架的真实结构图 | **RDKit** `MolsToGridImage` |
| F3 | **🌟 活性悬崖对照** | 悬崖对"同骨架·一基团之差·活性差几 log" | **RDKit** 结构图 |
| F4 | **化学空间** | t-SNE/UMAP 按 pIC50 上色 + train/val/test 三色叠加 | **scienceplots** |
| F5 | **质量审计仪表盘** | 多面板：失败/重复/立体/标签冲突/泄漏 | **scienceplots** 多子图 |
| F6 | **活性悬崖网络** | 分子为点、相似连边、边按活性差上色（接 §8 novelty） | **scienceplots** + networkx |
| F7 | 描述符分布 + 相关热图 | 12 描述符分布 + 与 pIC50 相关 | **scienceplots** |
| F8 | 标签分布 | pIC50 直方图/KDE 按 split | **scienceplots** |
| — | **预处理日志表** | 每步（RDKit 函数 / 影响分子数 / 理由） | 表格 |

### 3.4 三个杀手锏（别人不做）

1. **标签冲突 → 内禀噪声**（L2）：让你能说"MAE 0.60 已逼近数据本身噪声下限"。
2. **化学空间画 train/val/test 重叠**（L5/F4）：把"泄漏"从数字变成一眼看懂的图。
3. **子结构富集 + 活性悬崖**（L8/F3/F6）：从"预测得准"升到"理解了构效关系"。

---

## 4. 模型构建（7–8 个，覆盖完整方法谱）—— 打分点②

### 4.1 模型清单（表征 × 学习器矩阵）

| 策略 | 模型 | 范式 | 训练成本 | 状态 |
|---|---|---|---|---|
| 策略1 | **Ridge 线性回归** + Morgan | 线性 | 秒级 sklearn | **拟加**（第3讲 baseline） |
| 策略1 | **SVR（支持向量机）** + Morgan | 核方法 | 秒级 sklearn | **拟加**（第5讲点名，同学在做） |
| 策略1 | **GBM（LightGBM/梯度提升）** + Morgan | 树集成 | 秒级 sklearn | **拟加**（第5讲点名） |
| 策略1 | RF + Morgan（r2/2048） | 树集成 | 有 | 已有 |
| 策略1 | MolFormer-XL（IBM, 11亿预训练） | 预训练序列 Transformer | GPU | 已有 |
| 策略1 | ChemFM-3B（Clemson, LoRA） | 预训练大模型 | GPU | 已有 |
| 策略2 | Chemprop D-MPNN（MIT） | 图网络 | GPU | 已有 |
| 策略2 | **DeepChem GraphConv** | 图网络 | GPU | 可选（踩"DeepChem"课程点） |

→ 7（+1 可选）覆盖：**线性 → 核 → 树集成×2 → 预训练序列 → 预训练大模型 → 图网络**。

**原则**：填满"特征 × 学习器"矩阵里不同的格子，**不堆冗余**（不加 10 个雷同深度模型）。
**红利**：范式越全 → §8 的"跨族残差 → 数据内禀 vs 模型特有误差" novelty 信号越干净。

### 4.2 当前真实结果（4 模型，验证集 3-seed，待补三个 sklearn）

| 模型 | Pearson R | MAE |
|---|---|---|
| Chemprop（图网络） | **0.835 ± 0.005** | **0.604** |
| RF + Morgan | 0.804 ± 0.002 | 0.643 |
| ChemFM-3B | 0.803 ± 0.004 | 0.651 |
| MolFormer-XL | 0.798 ± 0.037 | 0.657 |

> 头条结论：**从零训练的图网络（Chemprop）赢过更大的预训练模型** → "模型大 ≠ 效果好，合适的表征在小数据上更值钱"。

---

## 5. 训练 + 调参策略 —— 打分点③（第3讲主场）

### 5.1 ML 工作流（第3讲原图）

`研究问题 → 训练 → 评估 → 分析（误差）→ 迭代`。

### 5.2 调参协议（小样本 QSAR 的标准姿势，有文献依据）

**调研结论（2026-06，先查证再设计）**：1059 条训练分子属于"小样本 QSAR"。该体量下调参的两大风险与对策都有定论——

1. **乐观偏差（optimistic bias）**：在同一份 CV 上既调参又报分，会系统性高估泛化能力（Cawley & Talbot, JMLR 2010；Krstajic et al., *J Cheminform* 2014, 6:10）。对策 = **嵌套交叉验证**（外环估性能、内环调参），外环分数才是诚实的泛化估计。
2. **预处理泄漏**：scaler / 特征选择 / PCA 若在划分前 fit，测试信息泄进训练（CLAUDE §7.3）。对策 = **把所有预处理塞进 `sklearn.Pipeline`**，让它在每个 CV 折内部各自 fit。
3. **搜索效率**：小样本下单次 CV 很便宜、但 SVR 的 `C×gamma×epsilon` 是连续大空间，**网格搜索浪费预算**。对策 = **Optuna TPE 贝叶斯搜索**（按历史 trial 自适应采样），与 AstraZeneca 的 QSARtuna（*JCIM* 2024，专做 QSAR 自动调参，支持 RF/SVR/Ridge/Lasso）同思路。

**两类模型、两套策略**（不能一锅烩）：

```
经典 ML（Ridge/SVR/GBM/RF，秒级训练）          深度模型（Chemprop/MolFormer/ChemFM，分钟–小时级）
┌─────────────────────────────────────┐      ┌──────────────────────────────────────┐
│ 嵌套 CV：外 5 折(估性能) × 内 5 折(调参)│      │ 全 Optuna 不可行 → 只调最敏感的几个旋钮  │
│ 内环 = Optuna TPE 50–100 trial         │      │ lr × (LoRA rank) 小网格 6–12 配置       │
│ 预处理全进 Pipeline，折内各自 fit       │      │ best-on-val 早停 = epoch 也是被调的超参  │
│ 选 CV 均值最优 → train 上重训          │      │ 报 stop-epoch 分布（§16.1）            │
└─────────────────────────────────────┘      └──────────────────────────────────────┘
        │ 跨模型族选型只看验证集（151，一次）                                  │
        └────────────────────────┬──────────────────────────────────────────┘
                                 ▼
          最终模型在 train+val 重训 → 测试集（303）只评估一次 → 报告
```

**为什么这样设计能拿分**：嵌套 CV 的"外环诚实分 vs 单划分乐观分"差值，本身就是第3讲"评估"主题的最佳展示；而"测试集只碰一次"是王任小/戚逸飞最看重的纪律。

### 5.3 每模型调参空间（贝叶斯连续区间，非粗网格）

| 模型 | 超参（Optuna 分布） | 区间 / 候选 | 备注 |
|---|---|---|---|
| Ridge | `alpha` ~ logU | 1e-3 – 1e3 | 唯一旋钮，做基线下限 |
| SVR(RBF) | `C` ~ logU / `gamma` ~ logU / `epsilon` ~ U | 1e-1–1e3 / 1e-4–1e0 / 0.01–0.5 | 连续空间，TPE 收益最大 |
| GBM(HistGB) | `lr` ~ logU / `max_depth` / `max_iter`(早停) / `min_samples_leaf` / `subsample` | 0.01–0.3 / 2–8 / ≤1000 / 1–50 / 0.6–1.0 | 内置早停防过拟合 |
| RF | `n_estimators` / `max_depth` / `max_features` / `min_samples_leaf` | 200–1000 / {None,5,10,20,30} / {sqrt,0.3,0.5} / {1,2,5,10} | bagging 自带正则 |
| 表征（也是超参） | Morgan `radius×nBits` **vs** RDKit 描述符 | {2,3}×{1024,2048} / 12+ 描述符 | 让 Optuna 选表征 |
| Chemprop | `depth`/`hidden`/`dropout`/`ffn_layers` | {3,4,5}/{300,600}/{0,0.1,0.2}/{2,3} | 随机搜索 ~20 配置 |
| MolFormer/ChemFM | `lr`/`dropout`/`LoRA rank`/`weight_decay` | {1e-4,5e-5,1e-5}/{0.1,0.2}/{8,16,32}/{0,1e-2} | 小网格 + 早停 |

**三道诚实闸门（出表前必跑，§16）**：
1. **嵌套外环分** 与单划分分一起报 → 量化乐观偏差有多大；
2. **阴性对照**：打乱训练标签后用同一套调参流程重跑，若调出来的模型仍 |R|>0.15 → 有泄漏，**结果作废**（§16.8）；
3. **超参稳定性**：报 top-k 配置的 CV 分布，若最优只比第 50 名高 0.001，叙述**不写"最优超参是 X"**，只写"这一族超参表现相当"（防把噪声当信号）。

### 5.4 调参/训练图（全 scienceplots）

| # | 图 | 内容 |
|---|---|---|
| F9 | **验证曲线** | 扫单个超参（如 RF max_depth、Chemprop depth），train vs CV → 过拟合甜点 |
| F10 | **学习曲线** | 训练集大小 vs 分数 → 数据瓶颈 or 模型瓶颈 |
| F11 | **超参热图** | RF (n_estimators × max_depth) CV 分数 |
| F12 | **默认 vs 调参** | 每模型提升对比（诚实，有时只升一点） |
| F13 | **训练收敛曲线** | 神经网络逐 epoch val（3-seed 带），★ best-on-val（已有 `fig_convergence_j`） |

### 5.5 防过拟合（多层）

best-on-val 早停（神经网络）、LoRA（ChemFM 只训 0.2%）、dropout/weight decay/梯度裁剪、RF bagging、**阴性对照**（打乱标签 R≈0）、**scaffold split 压力测试**（R 0.843→0.778）、leak-safe 预处理。

---

## 6. 评估与分析 —— 打分点④（核心）

### 6.1 第 1 层 · 核心硬指标（作业要求，原封不动）

7 模型 → **绝对 pIC50** → 全局 **MAE + Pearson R**（+RMSE+Spearman），3-seed 均值±std。

**指标互相打架 = 诊断信号**：
- Pearson ≫ Spearman → 高杠杆点撑着/排序乱；
- Spearman ≫ Pearson → 排序好但标度/非线性偏（适合筛选）；
- RMSE ≫ MAE → 重尾，少数大错样本；
- R 高 + MAE 高 → 系统偏置（slope≠1）→ 校准问题。
- 解构：`MSE = bias² + var`；pred~true 拟合的 slope（压缩=回归均值）/intercept（偏移）。

### 6.2 第 2 层 · 把 MAE/Pearson "切开看"（还是同一指标，分层）

按 **骨架 / 活性区间 / on-cliff vs off-cliff** 分层重算 Pearson/MAE → 预期发现 **"全局 0.835 漂亮，但同骨架内 / 悬崖上崩"**。**没引入新指标，就是把 MAE/Pearson 做细 = "分析"打分点要的"详细"。核心论点在这层诞生：全局指标好 ≠ 局部可用。**

### 6.3 逐样本分析

逐分子误差排行（最难分子）、跨模型一致性（全对/全错/分歧）、**残差相关**（误差是数据内禀还是模型特有）、误差归因（标签噪声/AD-out/活性悬崖/模型特有）。

### 6.4 分析图（全 scienceplots）

| # | 图 | 内容 |
|---|---|---|
| F14 | 逐模型指标棒图 | MAE/RMSE/Pearson/Spearman，误差棒（已有 `fig_metrics_bars_j`） |
| F15 | 预测-实测散点阵 | 每模型 pred vs true（已有 `fig_pred_grid_j`，**验证集 n=151**，与表一致） |
| F16 | 残差分析 | 残差 vs 真值 + 直方图（已有 `fig_residuals_j`） |
| F17 | **分层 Pearson/MAE** | per-scaffold / on-cliff vs off-cliff / 按活性区间 |
| F18 | **跨模型残差相关热图** | 7 模型残差两两相关 → 数据内禀 vs 模型特有 |
| F19 | **分子难度指数** | 跨模型平均误差 vs AD/cliff/标签冲突 |

---

## 7. 结构对接模块（加分，第11/12讲）

受体 PDB **4D8C** → Meeko 准备 → 共晶配体 **redock 验证（已得 0.92 Å）** → 高活性配体对接 → 打分 vs 活性（**已得：Vina 打分不预测活性，r=−0.34 n.s.，配体效率归一后归零**）。
- 课件映射：第11讲 力场/分子模拟（Vina/MMFF）、第12讲 蛋白结构、第1讲 CADD。
- 图：F20 redock 叠合（PyMOL 渲染，已有）、F21 打分 vs pIC50（**scienceplots**，已有 `fig_docking_le_j`）。

---

## 8. Novelty —— PBCNet2 桥接（第4讲，李嫣主场；汤雨凡是 PBCNet2 作者）

### 8.1 三层结构（MAE/Pearson 是脊柱，novelty 从它长出来）

```
第1层 核心硬指标(绝对 MAE/Pearson, 7模型) ── 作业主线，原封不动
        │
第2层 分层 MAE/Pearson(揭示"全局好、局部崩") ── 还是同一指标，做细
        │
第3层 PBCNet2 补盲区 + Anchor+Refine 混合 ── novelty，且折回 MAE/Pearson 收口
```

### 8.2 为什么是真 novelty（调研后诚实校准）

- ❌ 已做烂（别当卖点）：BACE-1 ligand-based QSAR + Pearson；"结构/FEP > ligand on 活性悬崖"（Schrödinger JCTC 2019 已证）；PBCNet2 的 SAR-Diff 已是悬崖类基准。
- ✅ **空白（三者交集）**：把 **PBCNet2（快 AI 相对预测）桥接到 MoleculeNet BACE（纯配体-based ML 基准）** + **逐样本互补分解** + **绝对/相对混合**。全世界在该基准上只比配体-based 绝对预测、从不用蛋白结构。

### 8.3 三件可交付

1. **桥接 + 逐样本互补**：在 1513 里挖同系列+悬崖对 → 这些对用 4D8C + 位姿喂 PBCNet2 → 看 PBCNet2 是否在配体模型崩的地方判对悬崖（F22 互补热图，scienceplots）。
2. **CASF 四种能力框架量化**（第4讲、王任小）：Pearson=scoring power、Spearman=ranking power、+ top-k EF=screening power → "配体模型强 scoring 弱 ranking，PBCNet2 强同系列 ranking"（F23 三能力对比表 + top-k 富集曲线，scienceplots）。
3. **Anchor+Refine 混合**：配体模型给全局绝对锚点 → PBCNet2 的 ΔpAct 在同骨架内相对微调 → 折回 **绝对 MAE/Pearson** 比，看悬崖/同系列子集的提升。

### 8.4 各自怎么量（别混着比）

| 对象 | 预测 | 指标 |
|---|---|---|
| 7 配体模型 | 绝对 pIC50 | **MAE + Pearson R**（全局 & 分层） |
| PBCNet2 | 相对 ΔpAct（成对） | 排序正确率 / 同系列 Spearman / ΔpAct MAE |
| Anchor+Refine 混合 | 绝对 pIC50 | **MAE + Pearson R**（回到硬指标） |

### 8.5 诚实边界（§16 规矩）

- **能说**：首次把结构-based 相对预测（快 AI 替身 PBCNet2）桥接到 MoleculeNet BACE 这个纯配体 ML 基准，做逐样本互补 + 绝对/相对混合，定位结构在何处救活性悬崖。→ 新角度/新桥接（course→workshop 级）。
- **不能说**："PBCNet2 解决活性悬崖"（论文 SAR-Diff 已展示）、"结构>配体 on cliffs"（Schrödinger 2019 已证）、任何"首次/SOTA"。
- 课程红线仍在：只用老师的 BACE 数据 + 公开 4D8C，**不引入外部化合物**（PBCNet2 是模型不是化合物库，OK）。
- 混合主要赢在**局部**（同系列 Spearman、悬崖判对率）；全局 MAE/Pearson 可能只小幅动——而"全局好≠局部好"恰是核心论点。

---

## 9. 代码结构与可复现

```
src/data/      加载、canonical、描述符、featurize（leak-safe）
src/models/    ridge / svr / gbm / rf / molformer / chemfm / chemprop (+deepchem)
src/tuning/    grid/random search + 5-fold CV（新增）
src/evaluate/  指标、分层、残差相关、误差归因、CASF 三能力
src/structure/ 对接 + PBCNet2 桥接
notebooks/     01_EDA / 02_model / 03_tuning / 04_eval / 05_docking / 06_pbcnet
configs/       Hydra YAML（含调参搜索空间）
```
统一 `set_all_seeds`（42/1337/2024）、`make data` 一键重建、清洗日志写进 run metadata。所有图由 `scripts/make_*figure*.py`（scienceplots）/ notebook 一键重出。

---

## 10. 交付物 + 执行顺序

**交付物**：① 详细 EDA notebook + 8 张数据图（F1–F8）② 7–8 模型 + 绝对 MAE/Pearson 表 ③ 调参 notebook + F9–F13 ④ 评估 + 分层 + 逐样本 F14–F19 ⑤ 对接 F20–F21 ⑥ PBCNet2 novelty F22–F23 ⑦ 书面报告 + 汇报 PPT（含演讲者备注）。

**建议执行顺序**：
1. **数据整理（§3）**：先出 F1 漏斗 + F2 骨架网格 + F4 t-SNE 泄漏 + F6 悬崖网络（最出彩）+ 预处理日志；
2. **补 3 个 sklearn 模型（§4）**：Ridge/SVR/GBM → 7 模型绝对 MAE/Pearson 表；
3. **调参（§5）**：F9–F12；
4. **分层分析（§6.2）**：F17 分层 MAE/Pearson（核心论点）；
5. **Novelty（§8）**：PBCNet2 桥接（先验证"配体模型在悬崖上崩"，再上 PBCNet2 补刀）；
6. 报告 + PPT 重组，每页标注对应课程周。

---

## 附录 · 关键文献（按需引用，DOI 已查证）

- **BACE/数据**：Subramanian 2016（BACE 来源）；MoleculeNet（Wu et al. 2018, Chem Sci）。
- **第4讲方法谱**：CASF-2016（Su…Wang R.X. 2019 JCIM）、CASF-2013 protocol（Nat Protoc 2017）；打分函数分类（Liu & Wang R.X. 2015 JCIM）；PLANET（JCIM 2024）。
- **PBCNet**：PBCNet（Yu…Zheng, Nat Comput Sci 2023）；**PBCNet2.0（Yu…Zheng, Nat Chem Biol 2026, doi:10.1038/s41589-026-02241-x）**。
- **活性悬崖**：MoleculeACE（van Tilborg, JCIM 2022）；Predicting Activity Cliffs with FEP（Schrödinger, JCTC 2019）。
- **模型**：MolFormer-XL（IBM, Nat Mach Intell 2022）；ChemFM（Clemson, Comm Chem 2025）；Chemprop v2（MIT, JCIM 2024）。
- **方法论**：Gal/ensemble UQ（MPNN ensembles, 2021）；Demšar 2006（多模型比较）。

> 引用时必须标 split 类型（random/scaffold/predefined）与指标定义，禁止"new SOTA"措辞（§16）。
