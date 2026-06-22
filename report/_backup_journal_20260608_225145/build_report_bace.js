// Build the BACE single-task course report as a professional .docx.
// Focused version: BACE-1 is the whole story (activity prediction -> docking ->
// conformal trustworthiness); the 6-dataset benchmark is compressed into one
// short "extended validation" section. Mirrors the BACE-focused deck.
// Run: node scripts/build_report_bace.js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, PageBreak, TableOfContents,
} = require("docx");

const FIG = "results/figures/";
const CW = 9360;
const HEAD_FILL = "1F3864";
const ALT_FILL = "EEF3FA";

// ---------- helpers ----------
const P = (text, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 276 }, ...opts,
  children: typeof text === "string" ? [new TextRun(text)] : text,
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 140 }, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 100 }, children: [new TextRun(t)] });
const B = (label, rest) => new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: label, bold: true }), new TextRun(rest)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60 }, children: typeof t === "string" ? [new TextRun(t)] : t });

function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
function figure(file, caption, w = 560) {
  const img = fs.readFileSync(FIG + file);
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
const BODY_FONT = "Songti SC", HEAD_FONT = "STHeiti";
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

const titlePage = [
  new Paragraph({ spacing: { before: 2400, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "BACE-1 抑制剂活性预测：", bold: true, size: 40, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "分子表征、结构对接与 Conformal 可信度评估", bold: true, size: 36, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 },
    children: [new TextRun({ text: "复旦大学药学院《AI 赋能药物设计发现前沿》期末实验报告", size: 24, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "姓名/学号：____________________", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "授课教师：戚逸飞 / 王任小", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "算力：8 × NVIDIA RTX 5880 Ada (48GB)", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "日期：2026 年 6 月", size: 22 })] }),
  new Paragraph({ children: [new PageBreak()] }),
];
const toc = [
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "目录", bold: true, size: 28, color: "1F3864" })] }),
  new TableOfContents("目录", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
];

const body = [];
const push = (...xs) => xs.forEach((x) => body.push(x));

// Abstract
push(H1("摘要"));
push(P("β-secretase 1（BACE-1）是阿尔茨海默病的关键药物靶点。本报告以 BACE 数据集（1513 个分子）的抑制活性 pIC50 预测为核心任务，把一个药物设计问题从数据、五类分子表征建模、三维结构对接，到“何时可以相信预测”的 conformal 可信度评估，做成一条完整闭环；并将同一套方法学扩展到全部 6 个 MoleculeNet 数据集做交叉验证。"));
push(P("核心结果：图网络（Chemprop D-MPNN）在 BACE 上 Pearson R = 0.835、MAE = 0.604 双双最佳，其余四类表征聚于 R ≈ 0.80；共晶配体重对接（redock）RMSD 0.92 Å 验证了对接位姿几何，但 Vina 打分经配体效率归一化后与活性相关性归零（Pearson +0.10），即位姿可信而打分不预测活性；conformal 在 BACE 有效覆盖下给出约 ±1.3 pIC50 的最紧合法区间，并量化“最紧 ≠ 有效”——低相似度新骨架子群系统性欠覆盖。全程以阴性对照与数据泄漏审计自检。"));

// 1 引言
push(H1("1  引言"));
push(P("β-secretase 1（BACE-1）负责切割淀粉样前体蛋白、生成 β-淀粉样肽（Aβ），是抑制 Aβ 斑块形成、治疗阿尔茨海默病的关键药物靶点；准确预测候选分子对 BACE-1 的抑制活性，可在早期高效筛除弱活性化合物、加速先导发现。"));
push(P("本课程作业要求：选择课程介绍的某项人工智能药物设计预测任务，完成数据整理、模型构建、训练与分析，并撰写书面报告与课堂汇报。我们选择 BACE-1 抑制活性预测作为核心任务——在 6 个候选任务中，它是唯一能把课程的两种建模策略、三维表征、结构对接与可信度评估串成完整流程的任务：既能比较“SMILES→编码器→机器学习”（策略一）与“SMILES→二维分子图→图神经网络”（策略二）两条路线，又能引入蛋白晶体结构做对接，正中结构计算的核心。为检验方法学的普适性，我们再将同一套流程扩展到全部 6 个 MoleculeNet 数据集做交叉验证（§4）。"));

// 2 数据与方法
push(H1("2  数据与方法"));
push(H2("2.1  BACE 数据集"));
push(P("BACE 数据集含 1513 个 β-secretase 抑制剂，预测连续 pIC50（回归任务；指标 MAE / Pearson R），采用数据集预定义的训练 / 验证 / 测试划分。所有 SMILES 经 RDKit canonical 化一次，解析失败者记录数量后剔除（不静默丢弃）；BACE 约 71% 分子立体化学未定义，我们保留 @/@@ 标记、不强制去立体，并在讨论中说明其对 stereo-aware 模型的潜在影响。任何标准化仅在训练折上拟合，防止信息泄漏。"));
push(H2("2.2  五类分子表征"));
push(P("我们覆盖课程的两种策略并各取代表模型，额外加入一个三维几何模型作为超纲对照："));
push(bullet([new TextRun({ text: "策略一（SMILES→编码器→ML）：", bold: true }), new TextRun("随机森林 + Morgan 指纹（radius 2, 2048 位）；MolFormer-XL（11 亿分子预训练的序列 Transformer，47M 参数）；ChemFM-3B（30 亿参数解码器大模型，LoRA 微调）。")]));
push(bullet([new TextRun({ text: "策略二（SMILES→二维图→GNN）：", bold: true }), new TextRun("Chemprop D-MPNN——有向键消息传递网络，在原子（节点）+ 键（边）特征上做图学习。")]));
push(bullet([new TextRun({ text: "超纲（三维几何）：", bold: true }), new TextRun("Uni-Mol，SE(3)-Transformer，显式利用 RDKit ETKDGv3 生成的 3D 构象。")]));
push(H2("2.3  评估协议"));
push(P("每个（数据集 × 模型）组合跑 3 个随机种子（42 / 1337 / 2024）取均值 ± 标准差。回归报告 MAE 与 Pearson R（课程指标），并辅以 RMSE / Spearman ρ。回归目标在训练折上 z 标准化（防泄漏），评估时还原到原单位。"));
push(H2("2.4  结构对接"));
push(P("从 PDB 4D8C 提取 β-secretase A 链，用 Meeko 准备受体，以共晶配体 BXD 定义口袋盒子（中心 30.58, 6.24, 14.52；尺寸 22×22×26 Å）。BACE 配体经 RDKit 3D 嵌入 + Meeko 转 PDBQT，用 AutoDock Vina 对接。对接前先以共晶配体 BXD 重对接，验证协议能否重现晶体结合模式（§3.2）。"));
push(H2("2.5  Conformal 可信度协议"));
push(P("用 split conformal 给出名义 90% 覆盖的预测区间，以经验覆盖率检验有效性；进一步用 Conformalized Quantile Regression（CQR；Romano et al. 2019）做自适应宽度区间，并按 applicability domain（测试分子到最近训练分子的 Tanimoto 相似度）分层检验条件覆盖。"));

// 3 BACE results
push(H1("3  BACE 结果"));
push(H2("3.1  活性预测基准"));
push(table(
  ["表征", "策略", "Pearson R ↑", "MAE ↓"],
  [
    ["Chemprop D-MPNN", "策略二 · 图网络", "0.835 ± 0.005", "0.604"],
    ["RF + Morgan", "策略一 · 基线", "0.804 ± 0.002", "0.643"],
    ["ChemFM-3B", "策略一 · 大模型", "0.803 ± 0.004", "0.651"],
    ["Uni-Mol (3D)", "超纲 · 三维", "0.800 ± 0.006", "0.650"],
    ["MolFormer-XL", "策略一 · 序列", "0.798 ± 0.037", "0.657"],
  ],
  [2760, 2600, 2000, 2000],
));
push(P("（3 seed 均值 ± 标准差；pIC50 单位）。Chemprop（图网络）在 Pearson R 与 MAE 上双双最佳（0.835 / 0.604），其余四类表征聚于 R ≈ 0.80——BACE 的构效信号各类表征都能抓到；MolFormer 跨种子方差较大（±0.037），如实呈现、不挑种子。阴性对照：打乱训练标签重训后 permuted Pearson R ≈ −0.06（≈ 0），确认 0.835 是真实信号而非记忆噪声。", { spacing: { after: 60 } }));
push(H2("3.2  结构对接：位姿可信，打分不预测活性"));
push(P("我们先验证对接协议的几何可信度：将共晶配体 BXD 重新对接进受体，顶位姿与晶体构象的原位 RMSD 为 0.92 Å（优于 2 Å 的“成功重现”阈值，图 1），表明口袋盒子与受体准备能可靠重现已知结合模式。"));
push(...figure("fig_redock_overlay.png", "图 1  redock 验证：共晶 BXD（橙）与重对接顶位姿（绿）在 β-secretase 口袋内高度重合，原位 RMSD 0.92 Å", 360));
push(P("但对接打分预测活性的能力很弱。在重原子 [20,40] 的类药窗口内取 30 个跨活性配体，原始 Pearson(pIC50, Vina) = −0.34（n=30，p=0.069，未达显著，图 2）。采用配体效率（LE = Vina 打分 / 重原子数）这一标准的尺寸归一化后，相关性进一步消失（Pearson +0.10，p=0.59）：原本的弱相关主要由分子大小驱动，而非与尺寸无关的结合强度信号。"));
push(...figure("fig_bace_docking.png", "图 2  BACE 配体 docking 打分 vs 实验 pIC50（颜色为重原子数）；n=30，Pearson r=−0.34 未达显著（p=0.069）", 430));
push(P("因此对接的价值在于可解释的结合模式假设，而非活性排序。图 3 中一个高活性抑制剂（pIC50 9.19）占据催化口袋并与天冬氨酸残基形成氢键；而打分最强的配体（Vina −10.5）实测活性却接近最低（pIC50 4.65），直观印证“打分 ≠ 活性”。需补充：即便最优数据驱动模型在 BACE 上的 MAE ≈ 0.60 pIC50（约 4 倍 IC50 误差），也仅够粗筛排序，不足以分辨先导优化所需的 2–3 倍活性差异。"));
push(...figure("fig_pose_potent.png", "图 3  高活性抑制剂（pIC50 9.19）的对接结合模式：配体（青）位于 β-secretase 催化口袋，与天冬氨酸残基（橙）形成氢键（红虚线）", 360));
push(H2("3.3  Conformal 可信度：何时可以相信预测"));
push(P("点预测回答了“哪个骨架更准”，conformal 回答“何时可以相信它”。下表为 BACE 上各骨架的 split conformal 区间宽度与经验覆盖率（名义 0.90）。"));
push(table(
  ["表征", "区间宽度 (pIC50) ↓", "经验覆盖率（名义 0.90）"],
  [
    ["ChemFM-3B", "2.63", "0.895（略欠）"],
    ["Chemprop", "2.66", "0.925"],
    ["RF + Morgan", "2.87", "0.934"],
    ["Uni-Mol (3D)", "2.88", "0.934"],
    ["MolFormer-XL", "2.93", "0.924"],
  ],
  [3000, 3180, 3180],
));
push(P("BACE 上各骨架均达有效覆盖（0.90–0.93），Chemprop 给出最紧的合法区间，约 ±1.33 pIC50。但区间宽度提醒我们“有效不等于可用”：±1.3 pIC50 ≈ 20 倍 IC50，对前瞻性优先级排序的价值有限。条件覆盖审计更进一步——按到训练集的 Tanimoto 分层，低相似度（新骨架）子群系统性欠覆盖（RF 0.87、ChemFM 0.82），高相似度子群过覆盖（0.94–0.97）；二者在边际上相互抵消，使总体覆盖“看起来”达标。这对需外推到新化学的虚拟筛选是直接的可信度警示。CQR 自适应区间在 BACE 上并不划算（Chemprop 的 CQR 区间反而增宽至 6.74），此处 split conformal 已足够。", { spacing: { after: 60 } }));

// 4 extended validation (compressed)
push(H1("4  扩展验证：6 个数据集"));
push(P("为检验上述方法学是否只在 BACE 成立，我们将同一流程扩展到全部 6 个 MoleculeNet 数据集（5 回归 + 1 分类）。这部分超出“单一任务”的课程要求，作为稳健性验证。"));
push(P("点预测上没有单一最优骨架：回归任务中图网络（Chemprop）与三维模型（Uni-Mol）总体领先（二者包揽 Pearson R 前二，ESOL / FreeSolv 上 Uni-Mol 达 0.97），BBBP 分类由序列 Transformer（MolFormer-XL，ROC-AUC 0.977）领先，传统 RF + 指纹始终是极强的性价比基线，而 ChemFM-3B 的默认 LoRA 在小数据集失配（GSHt Pearson R 仅 0.327）。图 4 给出 5 类骨架在 5 个回归任务上的 MAE 与 Pearson R。", { spacing: { after: 60 } }));
push(...figure("fig_benchmark_regression.png", "图 4  五类骨架在 5 个回归任务上的性能：上 MAE（越低越好）、下 Pearson R（越高越好）；误差棒为 3 seed 标准差（验证集）", 580));
push(P("可信度上，跨基础模型的 conformal 横评量化了“最紧 ≠ 有效”：Uni-Mol 在多数任务区间最紧，却在 ESOL 欠覆盖至 0.83（图 5）——窄而不可信；§3.3 在 BACE 上观察到的“低相似度子群欠覆盖”在 6 数据集上普遍成立。诚实定位：“神经模型过度自信”“覆盖随相似度退化”这两个现象在 MUBen（TMLR 2024）、Norinder（2014）等文献中已充分报道；本工作的增量是在统一协议下对现代基础模型做横向 conformal 比较并量化代价，而非声称发现这些现象。", { spacing: { after: 60 } }));
push(...figure("fig_tradeoff_width_coverage.png", "图 5  区间宽度 vs 覆盖率（每个分面一个数据集；阴影 = 欠覆盖 90% 名义）", 560));
push(P("作为额外的泛化压力测试，我们在外部 MoleculeACE 基准（30 个 ChEMBL 活性靶；属外部数据，仅作确认性分析、不计入课程 6 数据集）上确认：所有骨架在“活性悬崖”上误差都更高，与近期文献（CheMeleon 2026；SemiMol）一致——即基础模型并未真正攻克活性悬崖。"));

// 5 audit
push(H1("5  科研级可复现性审计"));
push(P("我们对自身结果执行主动审计并披露全部局限。"));
push(B("数据泄漏：", "数据集的预定义划分存在不同程度的训练-测试支架重叠。以核心任务 BACE 为例：RF + Morgan 基线在预定义切分（测试-训练 Murcko 骨架重叠 64%）的 Pearson R 为 0.843，改用骨架严格分离的 scaffold split（重叠 0%）后降至 0.778（MAE 0.55 → 0.65，3 seed），即约 0.065 R 的虚高来自训练-测试相似性——退化温和但确实存在，应在比较中披露。其余数据集中，GSHt 为共价同系列化合物、天然共享骨架（重叠 89%，属同系列设计而非可避免缺陷），BBBP 有 21 个完全重复的 SMILES。"));
push(B("阴性对照：", "将训练标签随机置换后重训，BACE 等回归任务 permuted Pearson R ≈ 0；GSHt 的置换 Pearson R 落在 [−0.33, +0.20]，因此 ChemFM-3B 在 GSHt 上 0.327 的相关性落在噪声区间内、不构成真实信号。"));

// 6 discussion + conclusion
push(H1("6  讨论与结论"));
push(P("本报告以 BACE-1 抑制活性预测为核心，完成了一条从数据、五类分子表征建模、三维结构对接到 conformal 可信度评估的完整药物设计流程。主要结论："));
push(bullet("机器学习能可靠预测 BACE 活性——Chemprop（图网络）R = 0.835、MAE = 0.604 最佳；但 MAE ≈ 0.60 pIC50（约 4 倍 IC50）仅够粗筛，不足以分辨先导优化的 2–3 倍活性差异。"));
push(bullet("结构对接位姿几何可信（redock 0.92 Å），但 Vina 打分不预测活性（配体效率归一化后相关性归零），价值在可解释的结合模式假设，不夸大物理打分。"));
push(bullet("Conformal 回答“何时可信”——BACE 有效覆盖下最紧区间约 ±1.3 pIC50；但新骨架（低相似度）系统性欠覆盖，虚拟筛选需警惕。"));
push(bullet("上述结论在全部 6 个数据集的交叉验证下稳健；贯穿全文的方法论基石，是对自身结果的彻底审计与诚实披露——阴性对照与泄漏检查“拆穿”了若干表面良好的数字。"));
push(B("局限：", "(1) 主基准（点预测）数字基于验证集，测试集仅在 conformal 一次性最终评估中使用；(2) 对接样本 n=30 统计力有限，r=−0.34 未达显著；(3) 点预测中的 Uni-Mol 经 unimol-tools 训练、库内随机种子固定，实为单次测量；(4) MolFormer-XL / ChemFM-3B / Uni-Mol 为外部大规模语料预训练模型，其表征隐含课程数据集以外的化学知识——此属课程“策略一：编码器”的既定用法，并未向任务数据引入额外化合物。"));

// Data availability
push(H1("数据与代码可用性"));
push(P("数据：本研究核心任务使用课程提供的 BACE 数据集，扩展验证使用其余 5 个 MoleculeNet 数据集（ESOL / FreeSolv / Lipophilicity / BBBP / GSHt）；β-secretase 晶体结构 PDB 4D8C 取自公开数据库 RCSB。代码与结果：五类表征建模、conformal / CQR、结构对接（含 redock 位姿验证与渲染）与可复现性审计的脚本及结果表存于项目仓库，并经统一入口一键复跑。面向期刊正式投稿时，应按 ACS 数据政策将代码与可公开数据存入带 DOI 的公共仓库（如 Zenodo / Hugging Face）。"));

// References
push(H1("参考文献"));
const refs = [
  "Wu Z, et al. MoleculeNet: A Benchmark for Molecular Machine Learning. Chem Sci, 2018.",
  "Subramanian G, et al. Computational Modeling of β-secretase 1 (BACE-1) Inhibitors. JCIM, 2016.",
  "Ross J, et al. Large-scale chemical language representations (MolFormer). Nat Mach Intell, 2022.",
  "Zhou G, et al. Uni-Mol: A Universal 3D Molecular Representation Learning Framework. ICLR, 2023.",
  "Heid E, et al. Chemprop: Machine Learning Package for Molecular Property Prediction (v2). JCIM, 2025.",
  "Cai F, et al. ChemFM: A Foundation Model for Chemistry. 2024.",
  "Trott O, Olson AJ. AutoDock Vina. J Comput Chem, 2010.",
  "Eberhardt J, et al. AutoDock Vina 1.2.0 / Meeko ligand preparation. JCIM, 2021.",
  "Li Y, et al. MUBen: Benchmarking the Uncertainty of Molecular Representation Models. TMLR, 2024.",
  "Norinder U, et al. Conformal Prediction to Define Applicability Domain. JCIM, 2014.",
  "Romano Y, Patterson E, Candès EJ. Conformalized Quantile Regression. NeurIPS, 2019.",
  "van Tilborg D, et al. Exposing the Limitations of Molecular ML with Activity Cliffs (MoleculeACE). JCIM, 2022.",
];
refs.forEach((r, i) => push(new Paragraph({ spacing: { after: 60 }, children: [new TextRun(`[${i + 1}]  ${r}`)] })));

// ---------- assemble ----------
const doc = new Document({
  features: { updateFields: true },
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

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("report/课程实验报告_BACE.docx", buf); console.log("wrote report/课程实验报告_BACE.docx"); });
