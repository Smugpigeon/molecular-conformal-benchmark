// Build the course experiment report as a professional .docx (official framing).
// Run: node scripts/build_report_docx.js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, LevelFormat, TabStopType, TabStopPosition,
  TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber,
  PageBreak,
} = require("docx");

const FIG = "results/figures/";
const CW = 9360;               // content width (US Letter, 1" margins)
const HEAD_FILL = "1F3864";    // dark blue header row
const ALT_FILL = "EEF3FA";     // light row stripe

// ---------- helpers ----------
const P = (text, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 276 }, ...opts,
  children: typeof text === "string" ? [new TextRun(text)] : text,
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 140 }, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 100 }, children: [new TextRun(t)] });
const B = (label, rest) => new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: label, bold: true }), new TextRun(rest)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60 }, children: typeof t === "string" ? [new TextRun(t)] : t });

// Read true pixel dimensions from the PNG IHDR header (8-byte sig, then
// 4 len + 4 type "IHDR" + width@16 + height@20, both big-endian uint32).
function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function figure(file, caption, w = 560) {
  const img = fs.readFileSync(FIG + file);
  // Preserve the real aspect ratio so nothing is stretched, regardless of filename.
  const { w: rw, h: rh } = pngSize(img);
  const h = Math.round(w * rh / rw);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: img, transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: "555555" })] }),
  ];
}

function table(headers, rows, widths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mkCell = (txt, i, isHead, alt) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    shading: { fill: isHead ? HEAD_FILL : (alt ? ALT_FILL : "FFFFFF"), type: ShadingType.CLEAR },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: String(txt), bold: isHead, color: isHead ? "FFFFFF" : "000000", size: 18 })] })],
  });
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => mkCell(h, i, true, false)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => mkCell(c, i, false, ri % 2 === 1)) })),
    ] });
}

// ---------- document ----------
const BODY_FONT = "Songti SC";   // 宋体 — body
const HEAD_FONT = "STHeiti";     // 黑体 — headings
const styles = {
  default: { document: { run: { font: BODY_FONT, size: 21 } } },
  paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 30, bold: true, font: HEAD_FONT, color: "1F3864" },
      paragraph: { spacing: { before: 240, after: 140 }, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 24, bold: true, font: HEAD_FONT, color: "2E5496" },
      paragraph: { spacing: { before: 160, after: 100 }, outlineLevel: 1 } },
  ],
};

const numbering = { config: [{ reference: "b", levels: [
  { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] }] };

// Title page
const titlePage = [
  new Paragraph({ spacing: { before: 2400, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "基于五类分子表征的 ADMET 性质预测、", bold: true, size: 40, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "结构对接与可信度评估", bold: true, size: 40, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 },
    children: [new TextRun({ text: "复旦大学药学院《AI 赋能药物设计发现前沿》期末实验报告", size: 24, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "姓名/学号：____________________", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "授课教师：戚逸飞 / 王任小", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "算力：8 × NVIDIA RTX 5880 Ada (48GB)", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "日期：2026 年 6 月", size: 22 })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

// TOC
const toc = [
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "目录", bold: true, size: 28, color: "1F3864" })] }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------- body ----------
const body = [];
const push = (...xs) => xs.forEach((x) => body.push(x));

// Abstract
push(H1("摘要"));
push(P("成药性质（ADMET）的早期预测是现代药物发现的关键环节。本报告按课程期末作业要求，对教师提供的 6 个 MoleculeNet 数据集（5 个回归任务 + 1 个分类任务），系统比较了课程介绍的两种建模策略——策略一（SMILES → 编码器 → 机器学习）与策略二（SMILES → 二维分子图 → 图神经网络）——并补充了一个超纲的三维几何模型，共 5 类分子表征骨架：随机森林+Morgan 指纹、MolFormer-XL、ChemFM-3B、Chemprop D-MPNN（图网络）、Uni-Mol（3D）。"));
push(P("在回归任务上，图网络（Chemprop）与三维几何模型（Uni-Mol）总体领先于指纹/序列方法：二者包揽 Pearson R 前二，Chemprop 在 BACE 上最高（0.835）；BBBP 分类则以序列模型 MolFormer-XL 最强（ROC-AUC 0.977）。我们进一步将 BACE 高活性配体对接进 β-secretase 晶体结构，并贯彻一套科研级可复现性审计（数据泄漏检查、阴性对照、分子大小混杂控制），主动披露全部局限。方法学主线上，我们给出跨基础模型的 conformal 预测可信度评估。"));

// 1 引言
push(H1("1  引言"));
push(P("当代新药研发平均耗时 10 年以上、花费超 10 亿美元；每 1 万个进入研究的化合物中，最终仅约 1 个上市。在早期高效淘汰成药性质差的候选物，是降低研发成本的核心策略，而 ADMET（吸收、分布、代谢、排泄、毒性）预测正是这一环节的关键计算工具。"));
push(P("本课程作业要求：选择课程介绍的某项人工智能药物设计预测任务，完成数据整理、模型构建、训练与分析，并撰写书面报告。课程给出两种建模策略：策略一基于 SMILES 字符串编码 + 机器学习；策略二基于二维分子图的图神经网络。本报告不局限于单一任务，而是在全部 6 个数据集上系统比较两种策略的代表模型，并补充结构对接与方法学探索，力求覆盖课程全部知识点。"));

// 2 数据
push(H1("2  数据与任务设定"));
push(P("课程提供 6 个 MoleculeNet 数据集，任务类型与评价指标如下表（均为教师钦定）。"));
push(table(
  ["任务", "预测性质", "分子数", "类别", "指标"],
  [
    ["ESOL", "水溶性 logS", "1128", "回归", "MAE / Pearson R"],
    ["FreeSolv", "水合自由能 ΔG", "642", "回归", "MAE / Pearson R"],
    ["Lipophilicity", "脂溶性 logD@7.4", "4200", "回归", "MAE / Pearson R"],
    ["BACE", "β-secretase IC50", "1513", "回归", "MAE / Pearson R"],
    ["BBBP", "血脑屏障渗透", "2039", "分类", "ROC-AUC"],
    ["GSHt1/2", "共价反应性", "419", "回归", "MAE / Pearson R"],
  ],
  [1700, 3260, 1100, 1100, 2200],
));
push(B("数据纪律：", "所有 SMILES 经 RDKit canonical 化一次；解析失败的分子记录数量后剔除（不静默丢弃）；任何标准化仅在训练折上拟合，防止信息泄漏。"));

// 3 方法
push(H1("3  方法"));
push(H2("3.1  两种建模策略与五类表征"));
push(P("我们覆盖课程的两种策略，并各取代表模型；额外加入一个三维几何模型作为超纲对照："));
push(bullet([new TextRun({ text: "策略一（SMILES→编码器→ML）：", bold: true }), new TextRun("随机森林 + Morgan 指纹（radius 2, 2048 位）；MolFormer-XL（11 亿分子预训练的序列 Transformer，47M 参数）；ChemFM-3B（30 亿参数解码器大模型，LoRA 微调）。")]));
push(bullet([new TextRun({ text: "策略二（SMILES→二维图→GNN）：", bold: true }), new TextRun("Chemprop D-MPNN——有向键消息传递网络，在原子（节点）+ 键（边）特征上做图学习。")]));
push(bullet([new TextRun({ text: "超纲（三维几何）：", bold: true }), new TextRun("Uni-Mol，SE(3)-Transformer，显式利用 RDKit ETKDGv3 生成的 3D 构象。")]));
push(H2("3.2  评估协议"));
push(P("每个（数据集 × 模型）组合跑 3 个随机种子（42 / 1337 / 2024）取均值 ± 标准差。回归报告 MAE 与 Pearson R（课程指标），分类报告 ROC-AUC；并辅以 RMSE / Spearman ρ / F1 / MCC。回归目标在训练折上 z 标准化（防泄漏），评估时还原到原单位。"));
push(H2("3.3  结构对接模块"));
push(P("从 PDB 4D8C 提取 β-secretase A 链，用 Meeko 准备受体，以共晶配体 BXD 定义口袋盒子（中心 30.58, 6.24, 14.52；尺寸 22×22×26 Å）。BACE 配体经 RDKit 3D 嵌入 + Meeko 转 PDBQT，用 AutoDock Vina 对接，对比 docking 打分与实验 pIC50。"));

// 4 结果
push(H1("4  结果"));
push(H2("4.1  回归基准（Pearson R / MAE）"));
push(table(
  ["数据集", "RF", "Chemprop", "MolFormer", "ChemFM", "Uni-Mol"],
  [
    ["ESOL", "0.807", "0.953", "0.900", "0.769", "0.966"],
    ["FreeSolv", "0.896", "0.937", "0.833", "0.679", "0.967"],
    ["Lipophilicity", "0.695", "0.875", "0.860", "0.756", "0.875"],
    ["BACE", "0.804", "0.835", "0.798", "0.803", "0.800"],
    ["GSHt", "0.722", "0.648", "0.545", "0.327", "0.749"],
  ],
  [2160, 1440, 1440, 1440, 1440, 1440],
));
push(P("表中为 Pearson R（越高越好，3 seed 均值）。下图同时给出 MAE 与 Pearson R 两个面板。", { spacing: { after: 60 } }));
push(...figure("fig_benchmark_regression.png", "图 1  五类骨架在 5 个回归任务上的性能：上 MAE（越低越好）、下 Pearson R（越高越好）；误差棒为 3 seed 标准差（验证集）", 600));
push(H2("4.2  分类基准（BBBP）"));
push(table(
  ["RF", "Chemprop", "MolFormer", "ChemFM", "Uni-Mol"],
  [["0.926", "0.899", "0.977", "0.971", "0.891"]],
  [1872, 1872, 1872, 1872, 1872],
));
push(P("表中为 ROC-AUC（3 seed 均值）。下图同时报告四项分类指标——MCC 才暴露真实差距：ROC-AUC / PR-AUC / F1 都在 0.9 附近难分高下，但 MCC 上 MolFormer 与 ChemFM（约 0.80）明显领先 RF / Chemprop / Uni-Mol（约 0.65）。这正是单一指标会掩盖、必须四指标并报的原因。", { spacing: { after: 60 } }));
push(...figure("fig_benchmark_classification.png", "图 2  BBBP 分类四指标（ROC-AUC / PR-AUC / F1 / MCC）；误差棒为 3 seed 标准差（验证集）", 580));
push(H2("4.3  策略一 vs 策略二"));
push(P("核心结论是没有单一最优骨架，最佳表征取决于任务性质与数据规模。回归任务上，图网络（策略二，Chemprop）与三维几何模型（Uni-Mol）总体领先于指纹/序列方法（策略一）：二者包揽 Pearson R 前二，Chemprop 在 BACE 上达全场最高 0.835、MAE 一致优于随机森林。ChemFM-3B 因 LoRA 微调在小数据集失配而最弱（GSHt Pearson R 仅 0.327）。BBBP 分类则由序列 Transformer（MolFormer-XL）领先。"));
push(H2("4.4  BACE 结构对接"));
push(P("在重原子 [20,40] 的类药窗口内取 30 个跨活性范围的配体对接。原始 Pearson(pIC50, vina) = −0.34；由于 Vina 打分对大分子有偏，控制分子大小后的偏相关为 −0.29——物理对接打分与活性呈弱负相关（结合越强、打分越负，符合预期），但明显弱于机器学习对活性的排序能力。对接的价值更多在于提供可解释的结合模式假设。"));
push(...figure("fig_bace_docking.png", "图 3  BACE 配体 docking 打分 vs 实验 pIC50（颜色为重原子数）；n=30，Pearson r=−0.34 未达显著（p=0.069），控制分子大小后偏相关 −0.29", 430));
push(P("需强调：n=30 时 r=−0.34 并未达到统计显著（p=0.069）。我们如实标注这一点，不夸大物理打分的活性预测力——其价值在于提供可解释的结合模式假设，而非定量排序。"));

push(H2("4.5  稳健性补充：活性悬崖压力测试"));
push(P("作为对泛化能力的额外压力测试，我们在 MoleculeACE 基准（30 个 ChEMBL 活性靶；属外部数据，仅作确认性分析，不计入课程 6 数据集）上比较各骨架在“活性悬崖”（结构相似但活性悬殊的分子对）与非悬崖样本上的 RMSE。结论与近期文献（CheMeleon 2026；SemiMol）一致：所有骨架在悬崖上误差都更高（cliff ratio > 1），其中三维 Uni-Mol 受影响最重（中位 ratio 1.25，97% 的靶 > 1），序列 MolFormer 最轻（1.04）。即基础模型并未真正攻克活性悬崖——我们在此确认该现象，但不将其作为主要贡献。"));
push(...figure("fig_activity_cliff.png", "图 4  活性悬崖压力测试（MoleculeACE，30 靶）：左为各靶 off- vs on-cliff RMSE（对角线上方 = 悬崖更差），右为 cliff ratio 分布（> 1 = 受悬崖惩罚）", 560));

// 5 审计
push(H1("5  科研级可复现性审计"));
push(P("我们对自身结果执行主动审计并披露全部局限。"));
push(B("数据泄漏：", "6 个数据集的预定义划分均存在不同程度的训练-测试泄漏——GSHt 的 Murcko scaffold 重合高达 89%，BBBP 有 21 个完全重复的 SMILES，意味着部分报告性能来自训练-测试相似性而非真实泛化。"));
push(B("阴性对照：", "将训练标签随机置换后重训，5/6 数据集通过（回归 Pearson R ≈ 0）；但 BBBP 因类别不均衡，置换标签 ROC-AUC 仍达 0.558（而非 0.5），GSHt 的置换 Pearson R 范围 [−0.33, +0.20]——因此 ChemFM-3B 在 GSHt 上 0.327 的相关性落在噪声区间内，不构成真实信号。"));

// 6 conformal (expanded — methodological main line)
push(H1("6  方法学探索：基础模型的 Conformal 预测可信度"));
push(P("点预测基准回答了“哪个骨架更准”，却没回答“何时可以相信它”。Conformal prediction 提供分布无关、有限样本的覆盖率保证，使我们能问一个更实用的问题：哪个骨架在保持有效覆盖（经验覆盖率 ≥ 名义 90%）的前提下给出最紧的预测区间？这是本报告的方法学主线。"));
push(B("诚实定位：", "“conformal 覆盖随化学相似度下降而退化”、“神经模型过度自信”这两个现象，在 conformal-QSAR 与不确定性文献中均已充分报道（Norinder 2014；MUBen TMLR 2024；CoDrug NeurIPS 2023；Hosni ACS Omega 2026 等）。本节并非声称发现这些现象，而是在统一协议下首次对现代基础模型（MolFormer-XL / Uni-Mol-3D / ChemFM-3B / RF-Morgan / Chemprop）做横向 conformal 比较，并量化“最紧 ≠ 有效”的代价。"));

push(H2("6.1  覆盖率总览"));
push(P("图 5 是 5 骨架 × 5 回归任务在 90% 名义覆盖下的经验覆盖率热图。多数格点接近 0.90（标定良好），但 Uni-Mol 在 ESOL（0.83）与 Lipophilicity（0.87）两格明显欠覆盖（红色），GSHt 行整体过覆盖（蓝色，≈0.98–1.00）。"));
push(...figure("fig_coverage_heatmap.png", "图 5  各骨架 × 数据集的经验覆盖率（名义 0.90；红 = 欠覆盖、蓝 = 过覆盖）", 470));

push(H2("6.2  最紧 ≠ 有效"));
push(P("将区间宽度与覆盖率并置（图 6）可见：Uni-Mol 在多数任务上给出最窄区间，但其 ESOL 区间虽最紧、覆盖率却跌进阴影“欠覆盖”区——窄而不可信。图 7 的宽度排行榜对此以斜纹标注：覆盖率未达标的骨架，其“最紧”不计入有效排名。在覆盖有效的任务上，Uni-Mol 与 Chemprop 通常给出最紧的合法区间。"));
push(...figure("fig_tradeoff_width_coverage.png", "图 6  宽度 vs 覆盖率（每个分面一个数据集；阴影 = 欠覆盖 90% 名义）", 600));
push(...figure("fig_width_leaderboard.png", "图 7  区间宽度排行榜（斜纹 = 覆盖率未达标，宽度不可比）", 580));

push(H2("6.3  条件覆盖：边际达标掩盖了 OOD 失效"));
push(P("边际覆盖达标不代表处处可信。按 applicability domain（测试分子到最近训练分子的 Tanimoto 相似度）分层后（图 8），所有骨架在低相似度（新化学，T≈0.46）子群系统性欠覆盖、在高相似度子群过覆盖——二者相互抵消，使边际覆盖“看起来”达标。这对真实虚拟筛选（常需外推到新骨架）是直接的可信度警示，也是本节相对边际指标的增量。"));
push(...figure("fig_ad_conditional_coverage.png", "图 8  按 applicability domain 分层的条件覆盖率：低相似度子群全骨架欠覆盖", 460));

push(H2("6.4  自适应区间：CQR 能否补回欠覆盖？"));
push(P("固定宽度的 split conformal 无法刻画异方差误差。我们进一步为每个骨架训练分位回归头（pinball 损失预测 5%/95% 分位），再用 Conformalized Quantile Regression（CQR；Romano et al. 2019）做有限样本校正，得到自适应宽度区间。结果（图 9）呈骨架依赖："));
push(bullet("Uni-Mol：在其欠覆盖的 ESOL 上，CQR 把覆盖率从 0.83 修复到 0.90（自适应加宽）；在过覆盖的 GSHt 上反而收紧区间——CQR 对这个三维、异方差骨架最有价值。"));
push(bullet("经典 / 图网络（GBM、Chemprop）：小数据上分位头噪声大，CQR 区间普遍变宽、并不优于固定宽度——“自适应”在样本不足时未必划算。"));
push(P("即 CQR 的价值取决于骨架与误差结构，而非普适改进。这是一个诚实的负-正混合结论，而非“CQR 全面更优”的夸大。"));
push(...figure("fig_split_vs_cqr_scatter.png", "图 9  split conformal → CQR：左为覆盖率（对角线上方 = CQR 改善，Uni-Mol/ESOL 越过 0.90），右为宽度（对角线下方 = CQR 更紧）", 560));

push(H2("6.5  分类校准：温度缩放"));
push(P("对分类，可信度对应概率校准。BBBP 上的 Morgan-MLP 高度过度自信（约 93% 的预测把置信度堆在 ≈1.0、却仅 87% 正确）。单参数温度缩放（T≈5.0）将期望校准误差 ECE 从 0.130 降到 0.070（−46%），如图 10 中主簇（标记面积 ∝ 样本数）向对角线移动所示。"));
push(...figure("fig_bbbp_calibration.png", "图 10  BBBP 可靠性图（温度缩放前后；标记面积 ∝ 该置信度区间样本数；ECE 降低 46%）", 430));

// 7 讨论
push(H1("7  讨论"));
push(P("模型选型启示：(1) 对小型理化/活性数据集，图网络与三维表征优势明显，但传统 RF + 指纹仍是极强的性价比基线；(2) 大语言模型（ChemFM-3B）的默认 LoRA 配置在小数据集上失效，提示实用部署需谨慎调参；(3) 物理对接对活性的预测能力弱于纯数据驱动的机器学习排序。"));
push(P("可信度的实用启示：最准的骨架未必最可信——Uni-Mol 点预测最强、conformal 区间最紧，却在 ESOL / Lipophilicity 欠覆盖；自适应 CQR 能修复特定骨架的欠覆盖却非普适；所有骨架在低相似度子群条件欠覆盖。因此实际部署应同时报告区间宽度、条件覆盖与校准，而非仅看点预测精度——这是本报告方法学主线的核心结论。"));
push(P("严谨性的价值：本报告的阴性对照与泄漏检查“拆穿”了若干表面良好的数字（ChemFM-GSHt、BBBP 基线），避免把噪声当信号、把泄漏当泛化——这正是其核心价值。"));
push(B("局限：", "(1) 主基准（点预测）数字基于验证集，测试集仅在 conformal 一次性最终评估中使用；(2) 对接样本 n=30 统计力有限，r=−0.34 未达显著；(3) 点预测基准中的 Uni-Mol 经 unimol-tools 训练、库内随机种子固定，实为单次测量——但 §6.4 的 Uni-Mol CQR 由自建训练循环驱动，已实现真正的 3 seed 变异（覆盖率标准差 0.001–0.018）；(4) 活性悬崖分析基于外部 MoleculeACE 数据，仅作确认性结论。"));

// 8 结论
push(H1("8  结论"));
push(P("本报告构建了一个完整、严谨、可复现的 ADMET 预测与结构对接 pipeline，覆盖课程的两种建模策略与全部数据集。核心结论：没有单一最优骨架——图网络（策略二）与三维模型长于回归，序列 Transformer 稳于分类，传统 RF 是极强基线，大模型 LoRA 在小数据上需谨慎；物理对接弱于机器学习排序；现有 benchmark 划分普遍存在数据泄漏。在方法学上，我们给出跨基础模型的统一 conformal 可信度评估：量化“最紧 ≠ 有效”、揭示条件覆盖的 OOD 失效、并以 CQR 检验自适应区间能否补回欠覆盖（结论为骨架依赖）。贯穿全文的方法论基石，是对自身结果的彻底审计与诚实披露。"));

// References
push(H1("参考文献"));
const refs = [
  "Wu Z, et al. MoleculeNet: A Benchmark for Molecular Machine Learning. Chem Sci, 2018.",
  "Ross J, et al. Large-scale chemical language representations (MolFormer). Nat Mach Intell, 2022.",
  "Zhou G, et al. Uni-Mol: A Universal 3D Molecular Representation Learning Framework. ICLR, 2023.",
  "Heid E, et al. Chemprop: Machine Learning Package for Molecular Property Prediction (v2). JCIM, 2025.",
  "Cai F, et al. ChemFM: A Foundation Model for Chemistry. 2024.",
  "Daina A, et al. SwissADME. Sci Rep, 2017.",
  "Trott O, Olson AJ. AutoDock Vina. J Comput Chem, 2010.",
  "Vovk V, et al. Algorithmic Learning in a Random World (conformal prediction). Springer, 2005.",
  "Li Y, et al. MUBen: Benchmarking the Uncertainty of Molecular Representation Models. TMLR, 2024.",
  "Norinder U, et al. Conformal Prediction to Define Applicability Domain. JCIM, 2014.",
  "Romano Y, Patterson E, Candès EJ. Conformalized Quantile Regression. NeurIPS, 2019.",
  "Guo C, et al. On Calibration of Modern Neural Networks. ICML, 2017.",
  "Joeres R, et al. DataSAIL: data splitting against information leakage. Nat Commun, 2025.",
];
refs.forEach((r, i) => push(new Paragraph({ spacing: { after: 60 }, children: [new TextRun(`[${i + 1}]  ${r}`)] })));

// ---------- assemble ----------
const doc = new Document({
  styles, numbering,
  sections: [
    { properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      children: [...titlePage, ...toc] },
    { properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "第 ", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 }), new TextRun({ text: " 页", size: 18 })] })] }) },
      children: body },
  ],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("report/课程实验报告.docx", buf); console.log("wrote report/课程实验报告.docx"); });
