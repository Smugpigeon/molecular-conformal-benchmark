// Build the BACE course experiment report (实验报告 format) matching the provided template:
// cover (校名 banner + 校徽 + title + info table) -> experiment title -> 实验目的/原理/材料与工具/
// 步骤/结果/讨论与思考/结论/参考文献/附录. Noto Serif CJK SC + Times New Roman.
// Calibrated claims kept consistent with the oral deck. Run: node scripts/build_report_bace.js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, PageBreak,
  ImageRun, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, VerticalAlign,
} = require("docx");

const FIG = "results/figures/";
const ASSET = "assets/fudan/";
const CW = 9360;
const HEAD_FILL = "1F3864", ALT_FILL = "EEF3FA";
const SONG = "Noto Serif CJK SC", HEI = "Noto Serif CJK SC", KAI = "Noto Serif CJK SC", LATIN = "Times New Roman";
const F = (ea) => ({ ascii: LATIN, eastAsia: ea, hAnsi: LATIN, cs: LATIN });

const P = (text, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 300 }, alignment: AlignmentType.JUSTIFIED, ...opts,
  children: typeof text === "string" ? [new TextRun(text)] : text,
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 260, after: 130 }, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 170, after: 90 }, children: [new TextRun(t)] });
const B = (label, rest) => new Paragraph({ spacing: { after: 100, line: 300 }, alignment: AlignmentType.JUSTIFIED, children: [new TextRun({ text: label, bold: true }), new TextRun(rest)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60, line: 300 }, children: typeof t === "string" ? [new TextRun(t)] : t });

function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
let FIGNO = 0;
function figure(file, caption, w = 520) {
  const img = fs.readFileSync(FIG + file);
  const { w: rw, h: rh } = pngSize(img);
  const h = Math.round(w * rh / rw);
  const cap = `图 ${++FIGNO}　${caption}`;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: img, transformation: { width: w, height: h }, altText: { title: cap, description: cap, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new TextRun({ text: cap, size: 18, color: "333333", font: F(HEI) })] }),
  ];
}
let TABNO = 0;
function table(headers, rows, widths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mkCell = (txt, i, isHead, alt) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA }, margins: { top: 60, bottom: 60, left: 90, right: 90 },
    shading: { fill: isHead ? HEAD_FILL : (alt ? ALT_FILL : "FFFFFF"), type: ShadingType.CLEAR },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: String(txt), bold: isHead, color: isHead ? "FFFFFF" : "000000", size: 18, font: F(SONG) })] })],
  });
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => mkCell(h, i, true, false)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => mkCell(c, i, false, ri % 2 === 1)) }))] });
}
function tcap(t) { return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, children: [new TextRun({ text: `表 ${++TABNO}　${t}`, size: 18, color: "333333", font: F(HEI) })] }); }
const MONO = "Courier New";
const CODE = (code) => code.split("\n").map((ln) => new Paragraph({
  spacing: { after: 0, line: 240 }, indent: { left: 180 }, shading: { type: ShadingType.CLEAR, fill: "F4F4F2" },
  children: [new TextRun({ text: ln.length ? ln : " ", size: 16, color: "1A1A1A", font: { ascii: MONO, eastAsia: SONG, hAnsi: MONO, cs: MONO } })] }));

const styles = {
  default: { document: { run: { font: F(SONG), size: 21 } } },
  paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 28, bold: true, font: F(HEI), color: "1F3864" }, paragraph: { spacing: { before: 260, after: 130 }, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 24, bold: true, font: F(HEI), color: "2E5496" }, paragraph: { spacing: { before: 170, after: 90 }, outlineLevel: 1 } },
  ],
};
const numbering = { config: [{ reference: "b", levels: [
  { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] }] };

// ---------- cover ----------
function coverImg(file, w) { const img = fs.readFileSync(ASSET + file); const { w: rw, h: rh } = pngSize(img); return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new ImageRun({ type: "png", data: img, transformation: { width: w, height: Math.round(w * rh / rw) } })] }); }
function infoRow(cells) {
  const td = (txt, bold, w, fill) => new TableCell({ width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 120, right: 120 }, shading: { fill: fill || "FFFFFF", type: ShadingType.CLEAR },
    borders: { top: { style: BorderStyle.SINGLE, size: 4, color: "888888" }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "888888" }, left: { style: BorderStyle.SINGLE, size: 4, color: "888888" }, right: { style: BorderStyle.SINGLE, size: 4, color: "888888" } },
    children: [new Paragraph({ children: [new TextRun({ text: txt, bold, size: 22, font: F(SONG) })] })] });
  return new TableRow({ children: cells.map((c) => td(c.t, c.b, c.w, c.fill)) });
}
const cover = [
  coverImg("fudan_banner.png", 360),
  coverImg("fudan_crest.png", 84),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 360, after: 520 }, children: [new TextRun({ text: "AI 赋能药物发现前沿　课程实验报告", bold: true, size: 40, font: F(HEI), color: "1F3864" })] }),
  new Table({ width: { size: 8200, type: WidthType.DXA }, alignment: AlignmentType.CENTER, columnWidths: [1900, 2300, 1700, 2300],
    rows: [
      infoRow([{ t: "课程名称：", b: true, w: 1900, fill: ALT_FILL }, { t: "AI 赋能药物发现前沿", b: false, w: 6300 }].map((c, i) => i === 0 ? c : { ...c, w: 6300 })),
      infoRow([{ t: "姓　　名：", b: true, w: 1900, fill: ALT_FILL }, { t: "汤雨凡", b: false, w: 2300 }, { t: "学　　号：", b: true, w: 1700, fill: ALT_FILL }, { t: "23307130372", b: false, w: 2300 }]),
      infoRow([{ t: "学　　院：", b: true, w: 1900, fill: ALT_FILL }, { t: "药学院", b: false, w: 2300 }, { t: "专　　业：", b: true, w: 1700, fill: ALT_FILL }, { t: "药学", b: false, w: 2300 }]),
    ] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600 }, children: [new TextRun({ text: "2026 年 6 月", size: 22, font: F(SONG) })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------- body ----------
const body = [];
const push = (...xs) => xs.forEach((x) => body.push(x));

push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 80 }, children: [new TextRun({ text: "基于机器学习与结构信息的 BACE-1 抑制剂活性预测", bold: true, size: 32, font: F(HEI), color: "1F3864" })] }));
push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 }, children: [new TextRun({ text: "—— 小样本 QSAR 下表格基础模型、传统模型与分子基础模型的系统评估", size: 20, font: F(KAI), color: "555555" })] }));

// 实验目的
push(H1("实验目的"));
push(P("β-secretase 1（BACE-1）是阿尔茨海默病的关键药物靶点。本实验以 BACE 数据集（1513 个分子）的抑制活性 pIC50 预测为核心任务，围绕一个科学问题展开："));
push(P("在小样本 BACE 这类 QSAR 任务中，新兴的“表格基础模型”能否在【点预测精度】【对新化学的外推泛化】【不确定性可信度】三条轴上，达到或超过传统 QSAR 与分子基础模型？"));
push(P("围绕该问题，本实验设定以下目标：(1) 在同一数据与同一划分上，公平比较覆盖课程两种建模策略的 8 个配体模型，并做统计显著性检验；(2) 不止比“谁更准”，而是解剖误差落在哪类分子上、换成新骨架是否仍成立、何时可以相信预测；(3) 以 β-secretase 晶体结构（PDB 4D8C）的分子对接与课题组的结构-成对模型 PBCNet2，检验结构信息能否补上配体模型的短板；(4) 全程以阴性对照、泄漏审计与显著性检验自检，诚实报告包括阴性结果在内的全部发现。"));
push(B("任务口径说明：", "BACE 在 MoleculeNet 中既可作回归（pIC50）也可作分类（官方 Class 标签）。本实验聚焦 pIC50 回归 + 结构对接模块，不以官方 Class 为训练目标；但在结果中由 pIC50 阈值化派生一条分类评估轴（见结果 §8），作为对回归模型实用能力的附加考察。"));
push(B("课程内容对应：", "本项目遵循课程第五讲 MoleculeNet/DeepChem 的作业口径（BACE / SMILES / 1513 / 回归 / MAE + Pearson R），完成“数据整理→模型构建→训练→分析”的完整链路；实现上用 RDKit + scikit-learn / Chemprop / HuggingFace Transformers 在同一数据任务上复现。各课程环节与本项目的对应关系如下表。"));
push(table(["课程环节", "本项目对应内容"],
  [["数据整理", "清洗漏斗 / 去重 / Murcko 骨架与泄漏审计（图 1–2）"],
   ["分子表征", "Morgan 指纹 + RDKit 物化描述符 + 多指纹消融（结果 §6）"],
   ["模型构建", "策略一：传统 ML（RF/GBM/SVR/Ridge）+ 基础模型（MolFormer/ChemFM/TabPFN）；策略二：Chemprop D-MPNN 图网络"],
   ["训练与评估", "Optuna 调参 + 5 折 CV + 3 seed；MAE / Pearson R（回归）、ROC-AUC / MCC（分类）、早期富集 EF"],
   ["药物发现应用", "β-secretase 4D8C 分子对接（Vina）/ PBCNet2 结构-成对模型 / 活性悬崖分析"],
   ["可复现性", "scaffold split + 泄漏审计 + 阴性对照 + 显著性检验 + conformal/AD 分层"]],
  [2200, 7160]));

// 实验原理
push(H1("实验原理"));
push(H2("1. 任务与分子表征"));
push(P("活性预测属回归任务，课程指标为平均绝对误差 MAE 与皮尔逊相关系数 Pearson R（辅以 RMSE、Spearman ρ）。机器学习不能直接读分子，须先把 SMILES 翻译成数值表征，课程给出两种策略：策略一“SMILES→编码器→机器学习”，策略二“SMILES→二维分子图→图神经网络”。"));
push(H2("2. 八个模型覆盖的方法谱"));
push(bullet([new TextRun({ text: "策略一：", bold: true }), new TextRun("Morgan 指纹（r2, 2048 位）+ 随机森林 / 梯度提升 / 支持向量回归 / 岭回归；分子基础模型 MolFormer-XL（11 亿分子预训练 Transformer，47M 参数，全参微调）、ChemFM-3B（30 亿参解码器大模型，LoRA 微调）；表格基础模型 TabPFN（前向一次、零调参，输入 RDKit 描述符）。")]));
push(bullet([new TextRun({ text: "策略二：", bold: true }), new TextRun("Chemprop D-MPNN——原子为节点、化学键为边的有向消息传递图神经网络。")]));
push(H2("3. 可信度、对接与结构-成对模型"));
push(P("Conformal 预测在验证集上校准，给出名义覆盖（如 90%）的预测区间，并可按适用域（applicability domain，测试分子到最近训练分子的 Tanimoto 相似度）分层检验条件覆盖。分子对接（AutoDock Vina）把配体放进 β-secretase 口袋打分；PBCNet2（Yu & Sheng，郑明月课题组）是基于笛卡尔张量的结构-成对模型，预测一对配体在同一口袋中的相对结合活性 ΔpAct，专为先导优化 / 活性悬崖式场景设计。"));

// 实验材料与工具
push(H1("实验材料与工具"));
push(B("数据：", "课程提供的 BACE 数据集（1513 个 β-secretase 抑制剂，含 pIC50 与预定义训练/验证/测试划分）；公开蛋白结构 PDB 4D8C（β-secretase 与共晶配体 BXD）用于对接，未引入任何外部化合物库。"));
push(B("软件：", "RDKit 2024.03（SMILES 标准化、Morgan 指纹、描述符、Murcko 骨架、活性悬崖）；scikit-learn（经典模型）、Optuna（贝叶斯调参）、TabPFN 2.0（表格基础模型）、Chemprop v2、HuggingFace Transformers + PEFT（MolFormer/ChemFM）、AutoDock Vina + Meeko（对接）、PBCNet2.0（结构-成对，作者公开 MIT 仓库）；统计用 scipy。"));
push(B("硬件：", "8 × NVIDIA RTX 5880 Ada（48 GB）服务器，conda 环境统一管理；随机性由 set_all_seeds 锁定。"));

// 实验步骤
push(H1("实验步骤"));
push(H2("1. 数据清洗与探索性分析"));
push(P("在入口处一次性清洗：RDKit 解析（0 失败）、去最大片段、电荷中和（838 个）、互变异构规范化（253 个）、去隐藏重复（9 个），得到 1504 个 canonical 分子；解析失败者记录数量而非静默丢弃（图 1）。质量审计发现：约 75% 分子立体化学未定义、预定义划分测试-训练 Murcko 骨架重叠约 64%（63.7%）、约 29%（29.4%）测试分子到最近训练分子 Tanimoto>0.85；同一 canonical 分子的重复标注仅 3 对、中位差 0.16 log，说明数据在分子级别相当干净（图 2）。数据中有 242 对活性悬崖（Tanimoto>0.7 且 ΔpIC50>2，图 3），是后续误差分析与结构模块的动机。"));
push(...figure("eda_funnel_j.png", "BACE 预处理漏斗：1513 → 1504（电荷中和 838、互变异构规范 253、去隐藏重复 9）", 430));
push(...figure("eda_quality_dashboard_j.png", "数据质量仪表盘（1504 分子）：质量计数、立体化学、测试-训练 Tanimoto、骨架重叠、类药通过率、描述符-活性相关", 520));
push(...figure("eda_cliff_pairs_j.png", "活性悬崖结构对照（RDKit）：近乎相同的结构、活性相差 2–3 个对数单位；红色为差异原子，第一行为纯立体翻转", 430));
push(H2("2. 训练、调参与防过拟合"));
push(P("以全参微调的 MolFormer-XL 为例：SMILES 经分词器编码为定长 128 token，送入主干得到逐 token 隐状态，经 attention_mask 加权 masked-mean 池化压成 768 维分子向量，再过两层 MLP 回归头输出 pIC50；训练主循环为标准五步（清梯度→前向→MSE→反传→裁剪更新，配 AdamW + OneCycleLR），每个 epoch 末在验证集挑最优权重（best-on-val）："));
push(...CODE(`enc = tok(smiles, padding="max_length", max_length=128, return_tensors="pt")
out = backbone(enc.input_ids, enc.attention_mask)
pooled = masked_mean(out.last_hidden_state, enc.attention_mask)   # (B, 768)
pred = head(pooled)                                               # predicted pIC50
# loop: zero_grad -> forward -> MSE -> backward -> clip_grad -> step (AdamW+OneCycleLR)
# best-on-val early stopping; ChemFM uses LoRA (rank 16, ~0.2% params)`));
push(P("经典模型用 Optuna TPE 贝叶斯调参，目标为 5 折交叉验证折外 MAE；所有标准化塞进 sklearn Pipeline 在折内各自拟合（leak-safe），避免乐观偏差（Cawley & Talbot 2010）。防过拟合分多层：深度模型 best-on-val 早停、ChemFM LoRA 仅训 0.2% 参数、dropout 与梯度裁剪、经典模型 5 折 CV——验证与测试 MAE 未见明显落差，未观察到明显过拟合迹象。深度模型各跑 3 个种子（42/1337/2024）报均值±标准差，测试集仅最终评估一次。"));
push(H2("3. 评估、结构模块与审计"));
push(P("在统一测试集（n=303）上报点预测四指标 + Friedman/配对 bootstrap 显著性 + 逐分子误差解剖 + conformal 可信度；按 Murcko 骨架严格重划做泛化压力测试；将测试集配体对接进 4D8C 并比较 docking/ML/实验活性；在活性悬崖对上检验 PBCNet2；以阴性对照（置换标签）与泄漏审计自检。此外补充三项课程方法学深化：表征消融（固定随机森林、更换第五讲指纹清单）、特征与子结构可解释性（置换重要性 + Morgan 位渲染）、以及由 pIC50 派生的分类（ROC-AUC/PR-AUC/F1/MCC）与早期富集（EF）评估轴。"));

// 实验结果
push(H1("实验结果"));
push(H2("1. 活性预测基准与统计显著性"));
push(tcap("8 个配体模型在 BACE 测试集上的活性预测（pIC50；深度模型 3 seed 均值±标准差，n=303）"));
push(table(["模型", "类别", "MAE ↓", "Pearson R ↑"],
  [["TabPFN", "表格基础模型 · 零调参", "0.498", "0.873"],
   ["Chemprop D-MPNN", "策略二 · 图网络", "0.551 ± 0.008", "0.842 ± 0.007"],
   ["RF + Morgan", "策略一 · 树集成", "0.554", "0.841"],
   ["ChemFM-3B", "策略一 · 大模型 LoRA", "0.559 ± 0.022", "0.831 ± 0.009"],
   ["GBM", "策略一 · 树集成", "0.569", "0.835"],
   ["Ridge", "策略一 · 线性", "0.596", "0.814"],
   ["SVR (RBF)", "策略一 · 核方法", "0.599", "0.815"],
   ["MolFormer-XL", "策略一 · 全参微调", "0.674 ± 0.054", "0.775 ± 0.035"]],
  [2300, 2660, 2200, 2200]));
push(P("TabPFN 在 MAE 与 Pearson R 上均为最佳点估计。关于显著性需谨慎区分两类检验：基于 n=303 个测试分子的 Friedman + Nemenyi 临界差异图（图 5）把逐分子误差当作配对块，分子并非独立重复（伪重复），因此本报告只把它当作**描述性可视化**，不作为严格显著性判据。严格的显著性来自两条互补证据：(i) 逐分子配对 bootstrap（10k 重采样 + Bonferroni）给出 ΔMAE 的置信区间，进一步将 RF、GBM 与 TabPFN 分开；(ii) 论文级的折级 Nadeau–Bengio 校正重采样检验（以折而非分子为重复单元，避免伪重复）。两类检验灵敏度不同。综合的严谨表述为：TabPFN 与图网络、化学大模型同属第一梯队，而非全面碾压；MolFormer 全参微调在小数据上方差最大、表现最弱。"));
push(...figure("fig_all_models_mae_j.png", "全模型测试 MAE（深度模型 3 seed 误差棒；TabPFN 最低）", 520));
push(...figure("fig_cd_diagram_j.png", "临界差异图（Friedman p=1.3e-5，n=303，仅作描述性可视化——逐分子非独立重复）：横轴为逐分子误差平均秩；严格显著性见配对 bootstrap 与折级 Nadeau–Bengio 检验", 540));
push(H2("2. 调参的真实增量与“是否还需要调参”"));
push(P("调参的诚实结论（图 6）：对欠正则模型（岭回归默认在 2048 维上几乎失效，R 0.65→0.81）调参大幅受益，对随机森林/梯度提升等“默认即好”的树集成仅边际改善、个别略降——可见调参不必然带来稳定增益。进一步，在本实验预算内，任何 Optuna 预算（梯度提升 80 trial、约 986 秒）都未追上零调参的 TabPFN（约 5 秒、MAE 0.498，图 7）；在该预算的耗时-误差权衡上，零调参的 TabPFN 最优，与小样本 BACE 场景契合。"));
push(...figure("fig_tune_default_vs_tuned_j.png", "默认 vs Optuna 调参（测试集）：欠正则模型大幅受益，树集成仅边际", 500));
push(...figure("fig_exp_tabpfn_vs_optuna_j.png", "TabPFN（零调参）vs Optuna 调参的经典模型：精度 vs 调参耗时（红星 = TabPFN）", 520));
push(H2("3. 逐分子误差解剖"));
push(P("误差并非随机：所有模型都“回归到均值”（低估高活性、高估低活性，TabPFN 收缩最小）；误差最大的十分位分子中，19% 落在活性悬崖（其余仅 1%）、16% 为标签冲突骨架（其余 2%），且更偏新骨架与极端活性（图 8）；跨 8 模型共识最难的分子多为大环、碎片样小分子或极端活性（图 9）。调参只缩小误差幅度、不改变“谁难”（大误差集合重叠 76%）。TabPFN 的预测区间宽度可提前标出难分子——58% 的共识难例落在其 90% 区间之外（易预测分子仅 6%，图 10）。"));
push(...figure("fig_err_hard_vs_easy_j.png", "大误差（共识 top-decile）分子 vs 其余：前者更新（低 Tanimoto）、活性更极端、更常落在悬崖与标签冲突上", 520));
push(...figure("fig_err_hard_structures.png", "跨 8 模型共识最难的 12 个分子（结构 + 真实 pIC50 + 平均误差）", 520));
push(...figure("fig_err_uncertainty_j.png", "TabPFN 不确定性提前标出难分子：拒识最不确定者后保留集 MAE 明显下降；58% 难例落在 90% 区间外", 520));
push(H2("4. 外推泛化与可信度"));
push(P("预定义划分有约 64% 骨架重叠、会高估泛化。按 Murcko 骨架严格重划（训练/测试无共享骨架）重训全部 8 模型后（图 11）：TabPFN 几乎不退化（MAE 0.498→0.508，仍最低、差距最小），全参微调的 MolFormer 泛化性能明显退化（0.674→1.134）——TabPFN 对新化学的外推在本实验中表现最稳。可信度方面（图 12，名义 90%，仅用验证集校准、全模型同一协议、测试集不参与调 coverage）：各模型边际覆盖率均达标（0.90–0.95），Chemprop 区间最紧；但按相似度分层，8 个模型中有 6 个在低相似度（新骨架）子群覆盖率降至 0.2–0.5（低 AD 子群 n=10，为探索性结果，需更大外部集验证），仅 TabPFN 与 MolFormer 在低 AD 仍接近 0.9（MolFormer 的代价是区间显著更宽，TabPFN 则同时保持紧致）。"));
push(...figure("fig_scaffold_comparison_j.png", "泛化压力测试：预定义 vs Murcko 骨架严格划分的测试 MAE 与 Pearson；TabPFN 两 split 都最低、差距最小", 540));
push(...figure("fig_conformal_benchmark_j.png", "Conformal 全模型基准（名义 90%）：(a) 覆盖率 vs 区间宽度；(b) 按适用域分层的条件覆盖——低 AD 普遍欠覆盖", 540));
push(H2("5. 结构对接与 PBCNet2"));
push(P("将测试集配体用 Vina 对接进 4D8C（图 13）：对接打分与实验活性的相关较弱（Pearson −0.33），且控制分子尺寸后接近 0（偏相关 −0.17）——因为活性配体偏大、Vina 又偏好大配体，是尺寸混淆；监督 QSAR 在同一测试集上相关性更高（R 0.84–0.87），但两者用途不同（对接给的是结合位姿，不作同类性能比较）。共晶配体 BXD 重对接 RMSD 仅 1.13 Å，支持基础对接设置合理（图 14）。在 161 个活性悬崖对上检验 PBCNet2（指标为相对 ΔpIC50 排序 |Spearman ρ|，已按 ΔΔG 符号对齐）：其排序（0.375，约束位姿复测 0.174）低于配体模型（0.78）——这是一个诚实的阴性结果，可能主要受 cross-docking 位姿质量限制（而非 PBCNet2 本身——它在 FEP 基准上表现优异），界定了结构-成对模型的适用范围（图 15）。需要说明的是，悬崖对之间存在共享配体，pair-level 相关性不应解释为 161 个完全独立样本。"));
push(...figure("fig_docking_analysis_j.png", "对接 vs 活性 vs ML：(a) Vina 打分 vs 实验 pIC50（按重原子数着色，尺寸混淆）；(b) |Pearson| 对比——监督 QSAR 相关性更高", 520));
push(...figure("fig_pbcnet_faircompare_j.png", "PBCNet2（两种位姿协议）vs 配体模型在活性悬崖上的相对活性排序 |Spearman|", 460));

// 实验结果 (continued): representation ablation / interpretability / extra evaluation axes
push(H2("6. 表征消融：模型排名是否依赖指纹选择"));
push(P("为排除“基准排名只是 Morgan 指纹的产物”这一可能，固定模型为随机森林、只更换分子表征——覆盖课程第五讲给出的指纹清单（Morgan、Atom Pair、Topological Torsion、MACCS、Avalon）与 RDKit 217 维物化描述符，在同一划分上各跑 3 个种子（表 2、图 15）。Morgan（MAE 0.559）与 Topological Torsion 并列最优，Atom Pair 的 Pearson R 最高（0.847），三者同属第一梯队；低容量的 MACCS（167 位）明显最弱（MAE 0.609）。两点结论：其一，主基准选用 Morgan 合理，它处在表征第一梯队；其二，固定模型下不同表征的 MAE 仅相差约 0.05，远小于模型之间的差距，模型排名并非表征选择的假象。此处未调参的 RF+Morgan（0.559）与基准表中经 Optuna 调参的 RF（0.554）几乎一致，互为印证。"));
push(tcap("固定随机森林、仅更换分子表征的活性预测（测试集，3 seed 均值，按 MAE 升序）"));
push(table(["分子表征", "维度", "MAE ↓", "Pearson R ↑"],
  [["Morgan (ECFP4)", "2048 位", "0.559", "0.837"],
   ["Topological Torsion", "2048 位", "0.559", "0.833"],
   ["Atom Pair", "2048 位", "0.566", "0.847"],
   ["Avalon", "2048 位", "0.574", "0.824"],
   ["RDKit 描述符", "217 维", "0.593", "0.840"],
   ["MACCS keys", "167 位", "0.609", "0.807"]],
  [2760, 2400, 2100, 2100]));
push(...figure("fp_ablation.png", "表征消融（模型固定为随机森林）：(a) 测试 MAE；(b) 测试 Pearson R——Morgan / Atom Pair / Topological Torsion 同属第一梯队，MACCS 最弱", 520));

push(P("进一步固定 Morgan、做配置与特征选择的稳健性检查：radius 2 vs 3 × 1024/2048/4096 的扫描中，r2/2048 与最优配置 r2/4096 仅相差 0.006 MAE、加大半径（radius 3）不带来增益；对 Morgan-2048 做特征选择（方差阈值保留 783 位、互信息 top-512、Lasso 选 53 位，选择器均仅在训练集拟合以防泄漏）对随机森林无改善（树集成自带隐式特征选择），而激进的 Lasso 53 位选择反而明显变差（RF MAE 0.559→0.592、Ridge 0.596→0.741）。因此选用全 2048 位 Morgan（r2）合理，额外的指纹半径/位数调参或特征选择在本任务上无实质收益。"));
push(H2("7. 特征与子结构可解释性"));
push(P("为回答“模型究竟学到了什么化学”，对随机森林做两层可解释性分析（均不依赖额外黑箱工具）。其一，在 RDKit 描述符上用置换重要性（permutation importance，测试集 R² 的下降量）排序（图 16）：分子大小 / 连接度类描述符主导（Chi0 居首，与 pIC50 的 Spearman ρ=+0.47；重原子量、分子复杂度 BertzCT 同为正相关），辅以 E-state 电子拓扑指数——表明在该 BACE 同系物系列中，更大、更复杂的分子普遍更强（描述符彼此高度相关，应按“主题”而非单个特征解读）。其二，把随机森林重要性最高的 Morgan 位映射回原子环境并渲染为子结构（图 17）：可定位若干增效 / 致弱基团（如某含氟芳环环境一旦出现，平均 pIC50 下降约 1.73）。该视角呼应课程对“可解释性 + 适用域”的强调，也为先导优化指明取舍方向。"));
push(...figure("interpretability_descriptors.png", "描述符置换重要性（随机森林）：分子大小 / 复杂度（Chi0、BertzCT）与 E-state 指数主导；蓝 = 与 pIC50 正相关，红 = 负相关", 460));
push(...figure("interpretability_substructures.png", "随机森林重要性最高的 6 个 Morgan 子结构及其对平均 pIC50 的方向（绿 = 增效，红 = 致弱）", 520));

push(H2("8. 分类与早期富集：回归之外的两条评估轴"));
push(P("BACE 在作业中是回归任务（主指标 MAE + Pearson R）；为完整演练课程第五讲的评估指标、并考察模型的实用能力，补充两条由回归预测派生的评估轴（同一测试集，复用预测、不重新训练）。"));
push(P("分类轴（表 3、图 18）：以 pIC50≥7（IC50≤100 nM；测试集 43.9% 为活性、类别均衡）二分，用连续预测 pIC50 作打分计算 ROC-AUC / PR-AUC，在阈值 7 处二值化算 F1 / MCC。Chemprop 图网络给出最高 ROC-AUC（0.915）与 MCC（0.650），TabPFN 次之（0.900 / 0.631）。一个值得注意的细节：回归点估计最优的 TabPFN 在分类轴上略逊于 Chemprop——没有单一模型在所有评估轴上通吃，评估口径会改变“谁最好”的结论。"));
push(P("早期富集轴（图 19）：模拟先导优化中“能否把最强的分子排到最前”。以最高活性 10%（pIC50≥8.04，n=31）为高活性命中、按预测活性降序计算富集因子 EF。最稳的 EF@10%（前 30 个）以 ChemFM（6.84）、随机森林（5.86）领先，各模型在前 5–10% 富集约 4–7 倍于随机；MolFormer 最弱，与其回归表现一致（EF@1% 仅含 3 个分子、噪声大，仅作参考）。"));
push(tcap("回归模型的分类视角（active = pIC50 ≥ 7，测试集 n=303，按 ROC-AUC 降序）"));
push(table(["模型", "ROC-AUC ↑", "PR-AUC ↑", "F1 ↑", "MCC ↑"],
  [["Chemprop", "0.915", "0.888", "0.797", "0.650"],
   ["TabPFN", "0.900", "0.858", "0.792", "0.631"],
   ["GBM", "0.900", "0.865", "0.761", "0.603"],
   ["ChemFM-3B", "0.899", "0.855", "0.753", "0.576"],
   ["RF + Morgan", "0.893", "0.854", "0.751", "0.576"],
   ["Ridge", "0.885", "0.840", "0.725", "0.557"],
   ["SVR (RBF)", "0.879", "0.840", "0.741", "0.562"],
   ["MolFormer-XL", "0.865", "0.798", "0.758", "0.571"]],
  [2160, 1800, 1800, 1800, 1800]));
push(...figure("classification_view.png", "分类视角：(a) 8 个模型的 ROC 曲线（active = pIC50 ≥ 7）；(b) MCC——Chemprop 最高，回归最优的 TabPFN 在分类轴略逊", 540));
push(...figure("screening_power.png", "早期富集（高活性 = 最高 10%，pIC50 ≥ 8.04）：(a) 富集曲线；(b) EF@5%——各模型在排序表头富集约 4–7 倍于随机", 540));

// 讨论与思考
push(H1("讨论与思考"));
push(P("本实验围绕一个科学问题给出三条相互印证的证据，统一指向“对新化学的泛化”这一主线："));
push(bullet("零调参的表格基础模型 TabPFN 在点预测上为最佳点估计，在 scaffold 外推与低 AD 条件覆盖上表现最稳，整体属于第一梯队；它与 Chemprop、RF、ChemFM 同属第一梯队。显著性依检验口径而定：在逐分子配对 bootstrap 与主分析 ρ=1/9 的 Nadeau–Bengio 检验下，TabPFN 显著优于线性/核模型与 RF/GBM；但在保守 ρ=0.25 NB-Holm 校正下，仅全参微调的 MolFormer 仍显著更差，其余差异不显著——故宜表述为第一梯队，而非全面碾压。"));
push(bullet("误差不是随机的：高度集中在活性悬崖、标签冲突、新骨架与极端活性分子；调参只缩小幅度、不改变“谁难”；不确定性可提前标出这些难分子，使“何时降低预测可信度”变得可操作。"));
push(bullet("结构方法（对接 / PBCNet2）在本设置下未能弥补配体模型短板：Vina 打分弱且受尺寸混淆，PBCNet2 在对接位姿上未超过配体模型——这是一个界定适用范围的阴性结果，瓶颈可能主要在位姿质量而非模型类别。"));
push(bullet("表征与评估口径的稳健性检查进一步支撑主结论：固定模型只换指纹，模型排名不变（Morgan 处第一梯队）；换成分类（ROC-AUC / MCC）与早期富集（EF）两条轴后，各模型整体仍可用，但“谁第一”随口径轻微变化（回归看 TabPFN、分类看 Chemprop）——评估应多轴并看，不可只报单一指标。"));
push(bullet("可解释性显示模型抓到的是真实构效信号：分子大小 / 复杂度与若干增效 / 致弱子结构主导预测，而非记忆噪声（与附录阴性对照相互印证）。"));
push(B("方法论思考：", "本实验最看重的不是把某个数字刷高，而是让每个结论都能被质疑也能被辩护——用阴性对照排除随机标签记忆、用 scaffold split 量化并披露相似性泄漏、用统计检验约束“显著”一词、对结构方法的阴性结果不作过度归因。这是把“看起来很准”升级为“知道它何时可信”的关键。"));
push(B("局限：", "(1) 单数据集（BACE），结论外推到其他靶点需验证；(2) 预定义划分存在骨架泄漏，已用 scaffold split 缓解并披露；(3) PBCNet2 的检验受限于 cross-docking 位姿质量、缺少共晶位姿；(4) 低相似度子群样本量小（n=10），相关结论为探索性；(5) 未做前瞻性实验验证。"));
push(B("研究范围界定：", "本实验聚焦先导优化阶段的“活性预测 + 结构复核”，未涉及分子生成、逆合成或新靶标发现——课程允许选择其中一项预测任务，本项目选定并完整实现了 BACE pIC50 活性预测这一项。"));

// 结论
push(H1("结论"));
push(P("在小样本 BACE-1 抑制活性预测上，零调参的表格基础模型 TabPFN 给出最优点估计，并在统计显著性、骨架外推鲁棒性、conformal 条件覆盖三条轴上对新化学的泛化均达第一梯队或最稳；模型误差系统性集中在活性悬崖、标签冲突与新骨架，且可被不确定性提前标出；结构方法（对接、PBCNet2）在 cross-docking 位姿域未能超过配体模型，构成一个界定适用范围的阴性结果。全程以阴性对照、泄漏审计与显著性检验自检，结论既可被质疑也可被辩护。"));

// 参考文献
push(H1("参考文献"));
const refs = [
  "VASSAR R, BENNETT B D, BABU-KHAN S, et al. β-secretase cleavage of Alzheimer's amyloid precursor protein by the transmembrane aspartic protease BACE[J]. Science, 1999, 286(5440): 735-741. DOI: 10.1126/science.286.5440.735.",
  "WU Z, RAMSUNDAR B, FEINBERG E N, et al. MoleculeNet: a benchmark for molecular machine learning[J]. Chemical Science, 2018, 9(2): 513-530. DOI: 10.1039/C7SC02664A.",
  "SUBRAMANIAN G, RAMSUNDAR B, PANDE V, et al. Computational modeling of β-secretase 1 (BACE-1) inhibitors using ligand based approaches[J]. Journal of Chemical Information and Modeling, 2016, 56(10): 1936-1949. DOI: 10.1021/acs.jcim.6b00290.",
  "ROGERS D, HAHN M. Extended-connectivity fingerprints[J]. Journal of Chemical Information and Modeling, 2010, 50(5): 742-754. DOI: 10.1021/ci100050t.",
  "HOLLMANN N, MÜLLER S, PURUCKER L, et al. Accurate predictions on small data with a tabular foundation model[J]. Nature, 2025, 637(8045): 319-326. DOI: 10.1038/s41586-024-08328-6.",
  "ROSS J, BELGODERE B, CHENTHAMARAKSHAN V, et al. Large-scale chemical language representations capture molecular structure and properties[J]. Nature Machine Intelligence, 2022, 4(12): 1256-1264. DOI: 10.1038/s42256-022-00580-7.",
  "HEID E, GREENMAN K P, CHUNG Y, et al. Chemprop: a machine learning package for chemical property prediction[J]. Journal of Chemical Information and Modeling, 2024, 64(1): 9-17. DOI: 10.1021/acs.jcim.3c01250.",
  "CAWLEY G C, TALBOT N L C. On over-fitting in model selection and subsequent selection bias in performance evaluation[J]. Journal of Machine Learning Research, 2010, 11: 2079-2107.",
  "AKIBA T, SANO S, YANASE T, et al. Optuna: a next-generation hyperparameter optimization framework[C]//Proceedings of the 25th ACM SIGKDD. 2019: 2623-2631. DOI: 10.1145/3292500.3330701.",
  "DEMŠAR J. Statistical comparisons of classifiers over multiple data sets[J]. Journal of Machine Learning Research, 2006, 7: 1-30.",
  "MAGGIORA G M. On outliers and activity cliffs — why QSAR often disappoints[J]. Journal of Chemical Information and Modeling, 2006, 46(4): 1535. DOI: 10.1021/ci060117s.",
  "EBERHARDT J, SANTOS-MARTINS D, TILLACK A F, et al. AutoDock Vina 1.2.0[J]. Journal of Chemical Information and Modeling, 2021, 61(8): 3891-3898. DOI: 10.1021/acs.jcim.1c00203.",
  "YU J, SHENG X, et al. Atomic-level protein-ligand recognition with PBCNet2.0 for probe discovery[J]. Nature Chemical Biology, 2026. DOI: 10.1038/s41589-026-02241-x.",
  "ANGELOPOULOS A N, BATES S. Conformal prediction: a gentle introduction[J]. Foundations and Trends in Machine Learning, 2023, 16(4): 494-591. DOI: 10.1561/2200000101.",
  "ROMANO Y, PATTERSON E, CANDÈS E J. Conformalized quantile regression[C]//NeurIPS 32. 2019.",
];
refs.forEach((r, i) => push(new Paragraph({ spacing: { after: 60 }, alignment: AlignmentType.JUSTIFIED, children: [new TextRun({ text: `[${i + 1}]　${r}`, size: 18, font: F(SONG) })] })));

// 附录
push(H1("附录"));
push(B("A. 可复现性审计：", "阴性对照——置换训练标签后用同一流程重训，RF / GBM / TabPFN 测试 Pearson R 分别降至 −0.11 / −0.04 / −0.05，支持 R≈0.84–0.87 来自可学习的结构-活性信号、而非对随机标签的记忆；泄漏审计——预定义划分骨架重叠约 64%、约 29% 测试分子 Tanimoto>0.85，scaffold split 已将相似性虚高剥离（RF MAE 0.554→0.684）。统一 set_all_seeds、3 seed、调参只用验证 / 折内 CV、测试集只看一次、所有图由 scripts/ 脚本一键重出。"));
push(B("B. 完整指标（深度模型 3 seed 均值±标准差，n=303）：", ""));
push(table(["模型", "MAE", "RMSE", "Pearson R", "Spearman ρ"],
  [["TabPFN", "0.498", "0.668", "0.873", "0.834"],
   ["Chemprop", "0.551±0.008", "0.737±0.015", "0.842±0.007", "0.827±0.009"],
   ["RF", "0.554", "0.747", "0.841", "0.822"],
   ["ChemFM-3B", "0.559±0.022", "0.762±0.021", "0.831±0.009", "0.816±0.007"],
   ["GBM", "0.569", "0.753", "0.835", "0.813"],
   ["Ridge", "0.596", "0.796", "0.814", "0.802"],
   ["SVR", "0.599", "0.793", "0.815", "0.797"],
   ["MolFormer-XL", "0.674±0.054", "0.885±0.057", "0.775±0.035", "0.733±0.034"]],
  [2160, 1800, 1800, 1800, 1800]));
push(B("C. 数据与代码可用性：", "本研究使用课程提供的 BACE 数据集与公开蛋白结构 PDB 4D8C；建模、调参、误差解剖、对接、PBCNet2、conformal 与审计脚本及结果表存于项目仓库，经统一入口一键复跑；PBCNet2 代码与权重来自作者公开 MIT 仓库。"));

const doc = new Document({
  styles, numbering,
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, font: F(SONG) })] })] }) },
    children: [...cover, ...body],
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("report/课程实验报告_BACE.docx", buf); console.log("wrote report/课程实验报告_BACE.docx (实验报告 format, " + FIGNO + " figs, " + TABNO + " tables)"); });
