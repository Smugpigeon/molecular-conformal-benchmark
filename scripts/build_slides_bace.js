// BACE oral deck — scientific-talk structure on the Fudan template (JinnyWong/FudanPPT Series2-4).
// One science question -> 3 evidence axes; conclusion-style titles; one claim per slide;
// detail moved to a Backup section. Run: NODE_PATH=$(npm root -g) node scripts/build_slides_bace.js
const pptxgen = require("pptxgenjs");
const fs = require("fs");

const FIG = "results/figures/";
const W = 13.333, H = 7.5;
const PAPER = "FCFBF8", INK = "1A1A1A", NAVY = "1F3864", SUB = "2E5496",
      MUTE = "47433B", HAIR = "D9D2C6", WHITE = "FFFFFF", TINT = "EEF1F6", ACCENT = "9B2226";
const SERIF = "Noto Serif CJK SC", LATIN = "Times New Roman", MONO = "Courier New";
const FN = "3E4E63", FSL = "8FA0B3";
const LOGO = "assets/fudan/fudan_logo.png", TOWERS = "assets/fudan/fudan_towers.png", GATE = "assets/fudan/fudan_gate.png";

function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function fit(file, bx, by, bw, bh) {
  const { w, h } = pngSize(FIG + file); const a = w / h;
  let iw = bw, ih = bw / a; if (ih > bh) { ih = bh; iw = bh * a; }
  return { path: FIG + file, x: bx + (bw - iw) / 2, y: by + (bh - ih) / 2, w: iw, h: ih };
}

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: W, height: H }); pres.layout = "WIDE";
pres.author = "汤雨凡"; pres.title = "BACE-1 抑制剂活性预测";

let pageNo = 0;
function runhead(s, section) {
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: H, fill: { color: FN }, line: { color: FN } });
  s.addImage({ path: LOGO, x: W - 2.02, y: 0.26, w: 1.62, h: 0.55 });
  s.addText("AI 赋能药物设计发现前沿" + (section ? "　·　" + section : ""), { x: 0.5, y: 0.34, w: 8.3, h: 0.3, fontFace: SERIF, fontSize: 11.5, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 0.5, y: 0.66, w: W - 2.75, h: 0, line: { color: HAIR, width: 1 } });
}
function foot(s) { pageNo++; s.addText(String(pageNo), { x: 0, y: H - 0.46, w: W, h: 0.3, fontFace: LATIN, fontSize: 11.5, color: MUTE, align: "center", margin: 0 }); }
function content(section, title) {
  const s = pres.addSlide(); s.background = { color: PAPER };
  runhead(s, section);
  s.addText(title, { x: 0.6, y: 0.84, w: W - 2.7, h: 0.66, fontFace: SERIF, fontSize: 23, bold: true, color: NAVY, margin: 0 });
  foot(s); return s;
}
const b = (t, opts = {}) => ({ text: t, options: { bullet: { code: "2022", indent: 16 }, breakLine: true, color: INK, fontFace: SERIF, fontSize: 15, paraSpaceAfter: 8, ...opts } });
function card(s, x, y, w, h, heading, runs) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: WHITE }, line: { color: HAIR, width: 1 } });
  if (heading) {
    s.addText(heading, { x: x + 0.26, y: y + 0.18, w: w - 0.5, h: 0.34, fontFace: SERIF, fontSize: 15, bold: true, color: NAVY, margin: 0 });
    s.addShape(pres.shapes.LINE, { x: x + 0.26, y: y + 0.56, w: w - 0.52, h: 0, line: { color: HAIR, width: 0.75 } });
  }
  s.addText(runs, { x: x + 0.26, y: y + (heading ? 0.64 : 0.2), w: w - 0.5, h: h - (heading ? 0.84 : 0.36), fontFace: SERIF, fontSize: 14, color: INK, margin: 0, paraSpaceAfter: 7, lineSpacingMultiple: 1.12, valign: "top" });
}
function jtable(s, x, y, w, colW, rows, hi) {
  const tbl = rows.map((r, ri) => r.map((c, ci) => {
    const head = ri === 0; const isHi = hi != null && ri === hi;
    return { text: String(c), options: { fill: { color: head ? FN : (isHi ? TINT : WHITE) },
      color: head ? WHITE : (isHi ? FN : INK), bold: head || isHi, align: ci === 0 ? "left" : "center",
      valign: "middle", fontFace: SERIF, fontSize: head ? 12 : 12.5 } };
  }));
  s.addTable(tbl, { x, y, w, colW, rowH: 0.4, border: { type: "solid", color: HAIR, pt: 0.5 }, margin: 4, valign: "middle" });
}
function codebox(s, x, y, w, h, code) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: "F4F4F2" }, line: { color: HAIR, width: 1 } });
  s.addText(code, { x: x + 0.22, y: y + 0.16, w: w - 0.44, h: h - 0.32, fontFace: MONO, fontSize: 12.5, color: INK, align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.06 });
}
function caption(s, t, x, y, w) { s.addText(t, { x, y, w, h: 0.3, fontFace: SERIF, fontSize: 11.5, italic: true, color: MUTE, align: "center", margin: 0 }); }
function figslide(section, title, file, capt, notes, fw) {
  const s = content(section, title);
  const im = fit(file, 0.55, 1.55, fw || 8.3, 5.05);
  s.addImage(im); if (capt) caption(s, capt, 0.55, 6.7, fw || 8.3);
  s.addNotes(notes); return s;
}
function box(s, x, y, w, h, title, sub, fill, tcol) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: FN, width: 1 } });
  s.addText([{ text: title, options: { fontSize: 14.5, bold: true, color: tcol, breakLine: true } },
             ...(sub ? [{ text: sub, options: { fontSize: 11.5, color: tcol } }] : [])],
    { x: x + 0.08, y, w: w - 0.16, h, fontFace: SERIF, align: "center", valign: "middle", margin: 0, lineSpacingMultiple: 1.02 });
}
function arrow(s, x, y, w) { s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: FSL, width: 2, endArrowType: "triangle" } }); }

// ===================================================== 1 TITLE
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 1.4, h: H, fill: { color: FN }, line: { color: FN } });
  s.addImage({ path: LOGO, x: W - 3.05, y: 0.5, w: 2.55, h: 0.87 });
  s.addImage({ path: TOWERS, x: W - 3.55, y: H - 2.95, w: 3.1, h: 3.1 * 627 / 845 });
  s.addText("BACE-1 抑制剂活性预测", { x: 1.4, y: 2.15, w: W - 1.8, h: 0.9, fontFace: SERIF, fontSize: 40, bold: true, color: FN, align: "center", margin: 0 });
  s.addText("小样本 QSAR 下表格基础模型、传统模型与分子基础模型的系统评估", { x: 1.4, y: 3.1, w: W - 1.8, h: 0.5, fontFace: SERIF, fontSize: 19, bold: true, color: FSL, align: "center", margin: 0 });
  s.addText("Tabular Foundation Models vs Classical QSAR and Molecular Foundation Models on Small-Sample BACE-1",
    { x: 1.4, y: 3.72, w: W - 1.8, h: 0.4, fontFace: LATIN, fontSize: 13, italic: true, color: MUTE, align: "center", margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 4.2, y: 4.4, w: W - 8.4, h: 0, line: { color: HAIR, width: 1 } });
  s.addText("复旦大学药学院《AI 赋能药物设计发现前沿》期末实验报告 · 课堂汇报", { x: 1.4, y: 4.6, w: W - 4.2, h: 0.34, fontFace: SERIF, fontSize: 14, color: INK, align: "center", margin: 0 });
  s.addText("汇报人：汤雨凡（学号 23307130372）", { x: 1.4, y: 5.05, w: W - 4.2, h: 0.34, fontFace: SERIF, fontSize: 15.5, bold: true, color: FN, align: "center", margin: 0 });
  s.addText("授课教师：戚逸飞 · 王任小 · 李嫣        2026 年 6 月", { x: 1.4, y: 5.55, w: W - 4.2, h: 0.34, fontFace: SERIF, fontSize: 12.5, color: MUTE, align: "center", margin: 0 });
  s.addNotes(`【开场 ~30s】各位老师好，我是汤雨凡。本次汇报围绕一个科学问题：在小样本 BACE 这类 QSAR 任务上，新兴的"表格基础模型"能否在点预测、外推泛化、不确定性可信度这三条轴上，达到或超过传统 QSAR 与分子基础模型。下面用约十五分钟给出三条主证据，再补充表征、可解释性与多评估轴三项课程方法学深化，并诚实交代限制。`);
}

// ===================================================== 2 ONE-SLIDE TAKEAWAY
{
  const s = content("核心结论", "表格基础模型在三条轴上均达第一梯队，并在外推与条件覆盖上更稳");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.62, w: W - 1.25, h: 1.0, rectRadius: 0.05, fill: { color: TINT }, line: { color: FN, width: 1 } });
  s.addText([{ text: "科学问题：", options: { bold: true, color: FN } },
             { text: "在小样本 BACE QSAR 中，表格基础模型 TabPFN 能否在【点预测】【外推泛化】【不确定性】上达到或超过传统 QSAR 与分子基础模型？", options: { color: INK } }],
    { x: 0.85, y: 1.7, w: W - 1.7, h: 0.84, fontFace: SERIF, fontSize: 16.5, valign: "middle", margin: 0, lineSpacingMultiple: 1.1 });
  const ev = [
    ["① 点预测", "MAE 0.498  最低", "与 Chemprop / RF / ChemFM 同属第一梯队；显著性依口径：bootstrap / ρ=1/9 下优于经典基线，保守 NB-Holm 下仅 MolFormer 仍显著更差"],
    ["② 外推泛化", "ΔMAE +0.01  最稳", "scaffold split 下几乎不退化；MolFormer 明显退化"],
    ["③ 可信度", "low-AD ~0.9", "提示在新化学上更稳；low-AD n=10，需更大外部集验证"],
  ];
  ev.forEach((e, i) => {
    const x = 0.6 + i * 4.18;
    card(s, x, 2.85, 3.95, 2.45, e[0], [
      { text: e[1], options: { fontSize: 21, bold: true, color: ACCENT, breakLine: true, paraSpaceAfter: 8 } },
      { text: e[2], options: { fontSize: 13.5, color: INK } }]);
  });
  s.addText([{ text: "结论（本数据集与评估协议下）：", options: { bold: true, color: FN } },
             { text: "TabPFN 在三轴上达到第一梯队，并在外推 / 低 AD 覆盖上表现最稳；结构方法（对接 / PBCNet2）未能弥补配体模型短板（可能受位姿质量影响）——一并如实报告。", options: { color: MUTE } }],
    { x: 0.6, y: 5.55, w: W - 1.25, h: 0.8, fontFace: SERIF, fontSize: 14.5, valign: "top", margin: 0, lineSpacingMultiple: 1.12 });
  s.addNotes(`【~70s】先给结论，后面逐条证明。科学问题如上方框。在本数据集与评估协议下，零调参的 TabPFN 在三条轴上达到第一梯队——点预测误差最低 0.498、且与图网络 Chemprop、化学大模型 ChemFM 统计打平；在骨架严格划分的外推下几乎不退化、最鲁棒；不确定性区间在新化学上也最稳。同时我诚实交代一个否定结果：结构方法因对接位姿质量受限、没能补上配体模型的短板。这页是全场的地图。`);
}

// ===================================================== 3 COURSE MAPPING
{
  const s = content("课程对应", "完整回应第五讲作业：数据整理 → 模型构建 → 训练评估 → 药物发现应用");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY } } });
  const rows = [
    [hdr("课程环节"), hdr("本项目对应内容")],
    ["数据整理", "清洗漏斗 / 去重 / Murcko 骨架与泄漏审计"],
    ["分子表征", "Morgan 指纹 + RDKit 物化描述符 + 多指纹消融"],
    ["模型构建", "策略一 传统 ML（RF / GBM / SVR / Ridge）+ 基础模型（MolFormer / ChemFM / TabPFN）；策略二 Chemprop D-MPNN 图网络"],
    ["训练与评估", "Optuna + 5 折 CV + 3 seed；MAE / Pearson R · ROC-AUC / MCC · 早期富集 EF"],
    ["药物发现应用", "β-secretase 4D8C 分子对接（Vina）/ PBCNet2 / 活性悬崖分析"],
    ["可复现性", "scaffold split · 泄漏审计 · 阴性对照 · 显著性检验 · conformal / AD 分层"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.7, w: W - 1.25, colW: [2.4, W - 1.25 - 2.4],
    fontFace: SERIF, fontSize: 13.5, color: INK, valign: "middle",
    border: { type: "solid", color: HAIR, pt: 0.5 }, rowH: 0.6 });
  s.addText("口径：MoleculeNet / DeepChem 作业（BACE / SMILES / 1513 / 回归 / MAE + Pearson R）；实现以 RDKit + scikit-learn / Chemprop / HuggingFace Transformers 复现同一数据任务。",
    { x: 0.6, y: 6.42, w: W - 1.25, h: 0.5, fontFace: SERIF, fontSize: 12.5, color: MUTE, margin: 0 });
  s.addNotes("【~40s】先给老师一张课程对应表：这门课要求选一个 AI 药物设计预测任务，做数据整理、模型构建、训练和分析——这张表逐条对上。数据整理是清洗加泄漏审计；表征是 Morgan 加 RDKit 描述符并做了多指纹消融；模型覆盖课程两种策略（指纹→ML、图→GNN）再加表格基础模型；评估以 MAE 和 Pearson R 为主，辅以分类和早期富集；最后落到 BACE 对接的药物发现语境。口径上跟随第五讲的 MoleculeNet / DeepChem 作业。");
}

// ===================================================== 4 STUDY DESIGN
{
  const s = content("研究设计", "研究设计：一个任务、八个模型、三轴评估 + 结构与审计");
  box(s, 0.6, 1.85, 1.95, 1.0, "BACE", "1513 → 清洗 1504", WHITE, FN);
  arrow(s, 2.6, 2.35, 0.5);
  box(s, 3.15, 1.85, 2.3, 1.0, "两种数据划分", "预定义 + 骨架严格", WHITE, FN);
  arrow(s, 5.5, 2.35, 0.5);
  box(s, 6.05, 1.85, 2.5, 1.0, "8 个模型", "传统 / 分子基础 / 表格基础 / 图", WHITE, FN);
  arrow(s, 8.6, 2.35, 0.5);
  box(s, 9.15, 1.55, 3.5, 1.6, "三轴评估", "① 点预测  ② scaffold 外推\n③ conformal / AD 可信度", TINT, FN);
  // lower row: structure module + audit
  box(s, 0.6, 3.95, 5.7, 1.5, "结构模块（对接 4D8C + PBCNet2）", "解释结合模式、检验结构-成对模型；\n与配体模型对照，非同类性能比较", WHITE, SUB);
  box(s, 6.65, 3.95, 6.0, 1.5, "可复现性审计", "阴性对照（置换标签）· 泄漏审计（骨架重叠）\n· 统一随机化 · 测试集只用一次", WHITE, SUB);
  s.addText("任务：BACE pIC50 回归 + 结构对接模块（官方 Class 标签为分类口径，本汇报聚焦回归）。指标：MAE / Pearson R 为主，辅以 RMSE / Spearman ρ；深度模型 3 seed mean±sd；测试集 n=303。", { x: 0.6, y: 5.7, w: W - 1.25, h: 0.5, fontFace: SERIF, fontSize: 13, color: MUTE, margin: 0 });
  s.addNotes(`【~65s】这是整个研究的设计图。一个真实任务 BACE、1513 个分子清洗到 1504；两种数据划分——预定义划分和骨架严格划分，后者专门测对新化学的外推；八个模型覆盖传统 QSAR、分子基础模型、表格基础模型和图网络；在三条轴上评估——点预测、外推、可信度。下面一行是两个支柱：结构模块用对接和 PBCNet2 引入蛋白信息、和配体模型对照，属对照而非同类性能比较；以及贯穿全程的可复现性审计。指标以平均绝对误差和相关系数为主，深度模型三个种子，测试集 303 个分子。`);
}

// ===================================================== 4 MODELS + FAIRNESS
{
  const s = content("模型与公平性", "八个模型覆盖两种策略，并施加统一的公平性控制");
  card(s, 0.6, 1.6, 6.0, 3.0, "策略一 · SMILES → 编码器 → ML", [
    b("传统 QSAR：随机森林 / 梯度提升 / SVR / 岭回归（Morgan 2048）"),
    b("分子基础模型：MolFormer-XL（47M，全参微调）、ChemFM-3B（30 亿参，LoRA 0.2%）"),
    b("表格基础模型：TabPFN（前向一次、零调参，输入 RDKit 描述符）", { color: ACCENT }),
  ]);
  card(s, 6.75, 1.6, 5.9, 3.0, "策略二 · SMILES → 二维图 → GNN", [
    b("Chemprop D-MPNN：原子-键有向消息传递图网络"),
    b("覆盖“表征 × 学习器”谱系，回答：表征、规模、还是不调参的基础模型更重要？"),
  ]);
  card(s, 0.6, 4.8, 12.05, 1.6, "公平性控制（避免偏置比较）", [
    b("同一份清洗后划分（bace_clean）输入全部 8 个模型；经典模型 Optuna 嵌套交叉验证、折内拟合标准化（防泄漏）；测试集只在最终评估一次。", { fontSize: 13.5 }),
    b("重复性策略不同（非完全统一）：深度模型各 3 seed（报 mean±sd），经典模型与 TabPFN 为确定性单次。", { fontSize: 12.5, color: MUTE }),
  ]);
  s.addNotes(`【~60s】八个模型分两类。策略一把分子当序列读：传统 QSAR、两个分子基础模型、以及表格基础模型 TabPFN；策略二把分子当图读，Chemprop。这么铺是为了回答——到底是表征更聪明重要、模型更大重要、还是干脆不调参的基础模型就够。关键是公平性：所有模型用同一份划分、同样三个种子、经典模型用嵌套交叉验证防止调参泄漏、测试集只看一次。先把比较的地基打平，后面的结论才站得住。`);
}

// ===================================================== 5 MAIN BENCHMARK
{
  const s = content("结果", "TabPFN 点预测 MAE 最低，与 Chemprop / RF / ChemFM 同属统计第一梯队");
  s.addImage(fit("fig_d_benchmark.png", 0.5, 1.55, 8.2, 5.0));
  card(s, 8.95, 1.7, 3.85, 4.7, "证据①：点预测", [
    b("TabPFN MAE 0.498 / R 0.873，最优。", { color: ACCENT }),
    b("Chemprop、RF、ChemFM 紧随（~0.55）——参数规模不决定性能。"),
    b("逐分子 CD 图（n=303）仅作描述性可视化（分子非独立重复）。", { fontSize: 12.5, color: MUTE }),
    b("严格显著性：配对 bootstrap（10k+Bonferroni）与折级 Nadeau–Bengio 检验。", { color: ACCENT }),
  ]);
  s.addNotes(`【~70s】第一条证据，点预测。TabPFN 误差最低、相关最高。但我没停在点估计。这里要区分两类检验：基于 n=303 个测试分子的临界差异图把逐分子误差当配对块，分子并非独立重复，所以我只把它当描述性可视化，不当严格显著性。严格的显著性来自两条：逐分子配对 bootstrap（10k 重采样 + Bonferroni）给 ΔMAE 置信区间，能进一步分开 RF、GBM 与 TabPFN；以及论文级的折级 Nadeau–Bengio 校正检验，以折为重复单元、避免伪重复。综合的严谨说法是 TabPFN 与图网络、化学大模型同属第一梯队，而不是碾压。`);
}

// ===================================================== 6 TABPFN vs BUDGET
{
  const s = figslide("调参", "在本实验预算内，TabPFN 优于 Optuna-tuned baselines", "fig_d_budget.png",
    "纵轴 test MAE，横轴调参耗时（对数）；红星 = 零调参 TabPFN",
    `【~55s】第二点关于调参。把传统模型在 1 到 80 次 Optuna 搜索下的表现，和零调参的 TabPFN 放在"精度对耗时"的图上。结论限定在本实验预算内：TabPFN 单次前向、约五秒、误差 0.498，落在左上角最优；而梯度提升调到八十次、耗时近千秒，仍没追上。我不说"任何预算都追不上"那种绝对话，只说在这个预算范围内、对路的小样本任务上，不调参的基础模型已经更划算。这一结论限定在本实验预算的耗时-误差权衡内。`, 8.3);
  card(s, 8.95, 1.7, 3.85, 4.0, "成本（效率分析）", [
    b("TabPFN：~5 s、0 trial、MAE 0.498。", { color: ACCENT }),
    b("GBM 最优需 80 trial / 986 s，仍逊。"),
    b("计时口径：同机 CPU、固定 exhaustiveness / trial 预算、单进程。", { fontSize: 12, color: MUTE }),
    b("限定“本实验预算内”，不作绝对结论。", { color: FN }),
  ]);
}

// ===================================================== 7 SCAFFOLD GENERALIZATION
{
  const s = figslide("外推泛化", "TabPFN 对新骨架最鲁棒，全参微调的 MolFormer 泛化性能明显退化", "fig_d_scaffold.png",
    "预定义 vs 骨架严格划分的绝对 MAE；TabPFN 两 split 都最低、差距最小",
    `【~60s】第二条证据，也是最重要的——外推泛化。预定义划分有约 64% 骨架重叠，会高估泛化。我按 Murcko 骨架严格重划、训练和测试无共享骨架，全部重训。鲁棒性差异很大：TabPFN 几乎不退化，误差只升 0.01、相关甚至上升，是对新化学最鲁棒的；而全参微调的 MolFormer 泛化性能明显退化、误差升 0.46。这和点预测、误差解剖指向同一条主线：TabPFN 在点预测上最佳、在 scaffold 外推与低 AD 覆盖上最稳，整体属第一梯队。这对真实虚拟筛选最关键，因为筛的就是没见过的新分子。`, 8.3);
  card(s, 8.95, 1.7, 3.85, 4.0, "证据②（外推）", [
    b("scaffold MAE：TabPFN 0.51 仍最低、差距最小（Δ+0.01）。", { color: ACCENT }),
    b("MolFormer 0.67→1.13（Δ+0.46），退化最明显。"),
    b("scaffold split = 更严格的新骨架压力测试。"),
  ]);
}

// ===================================================== 8 ERROR ANATOMY
{
  const s = content("误差解剖", "误差不是随机的：集中在悬崖、标签冲突与新骨架");
  s.addImage(fit("fig_d_hard3.png", 0.5, 1.7, 8.3, 4.2));
  caption(s, "跨 8 模型共识最难的三类代表分子", 0.5, 6.0, 8.3);
  card(s, 8.95, 1.7, 3.85, 4.7, "系统性，不随机", [
    b("回归到均值：低估高活性、高估低活性（TabPFN 收缩最小）。"),
    b("大误差分子：19% 在活性悬崖（其余仅 1%）、16% 标签冲突（其余 2%）。", { color: ACCENT }),
    b("更偏新骨架、极端活性。"),
    b("调参只缩小幅度、不改变“谁难”（大误差集合重叠 76%）。", { color: FN }),
  ]);
  s.addText("定义：活性悬崖 Tanimoto>0.7 且 ΔpIC50>2；标签冲突 = 同 InChIKey14 骨架多标签；“调参不改难” = 默认/调参 top-decile 误差集合 Jaccard=0.76。", { x: 0.5, y: 6.38, w: 8.3, h: 0.5, fontFace: SERIF, fontSize: 10, color: MUTE, margin: 0 });
  s.addNotes(`【~60s】这一页说清误差落在哪。两个系统性规律：一是所有模型都"回归到均值"、低估最强的高估最弱的；二是误差高度集中——误差最大的那批分子里，19% 落在活性悬崖、其余只有 1%，16% 是标签冲突、其余只有 2%，而且更偏新骨架和极端活性。左边是跨八个模型一致最难的三类代表：活性悬崖、标签冲突、新骨架。还有一点很关键：调参只是缩小误差幅度、并不改变哪些分子难。所以瓶颈在数据里的悬崖和噪声，不是调参不够。`);
}

// ===================================================== 9 UNCERTAINTY
{
  const s = figslide("不确定性", "TabPFN 不确定性能提前标出难分子", "fig_d_uncertainty.png",
    "按不确定性拒识 → 保留集 MAE 下降（vs 随机拒识）",
    `【~50s】既然误差集中在难分子上，能不能提前标出来？能。TabPFN 给的预测区间宽度和实际误差正相关；更直接的——58% 的共识难例落在它的 90% 区间之外，而容易预测的分子只有 6%。所以在虚拟筛选里，把最不确定的那批先拒掉交人工或实验复核，保留集误差会明显下降——红线明显低于随机拒识。这把"模型什么时候别信它"变成可操作的。`, 8.3);
  card(s, 8.95, 1.7, 3.85, 4.0, "可操作的预警", [
    b("主证据：拒识最不确定者 → 保留集 MAE 明显下降（vs 随机）。", { color: ACCENT }),
    b("区间宽度 vs 误差 Spearman ρ=0.32。"),
    b("解释性：58% 难例（按 test error 定义，后验）落在 90% 区间外，易预测仅 6%。", { fontSize: 12.5 }),
  ]);
}

// ===================================================== 10 CONFORMAL — MARGINAL
{
  const s = figslide("可信度 (1/2)", "边际 90% 覆盖对所有模型都达标", "fig_d_cov.png",
    "经验覆盖率 vs 区间宽度（红虚线 = 名义 0.90）",
    `【~45s】第三条证据，可信度，分两页。先看边际覆盖：给所有模型做 conformal 预测区间、名义 90%，每个模型的经验覆盖率都达标、在 0.90 到 0.95 之间，其中 Chemprop 的区间最紧、TabPFN 次之。单看这页会以为各模型区间都可信——但下一页会推翻这个表面结论。`, 8.3);
  card(s, 8.95, 1.7, 3.85, 4.0, "边际有效", [
    b("8 个模型边际覆盖率 0.90–0.95（达标）。"),
    b("Chemprop 区间最紧（2.26）、TabPFN 次之。"),
    b("协议：split-conformal，仅用 validation 校准，全模型同一协议，test 不参与调 coverage。", { fontSize: 11.5, color: MUTE }),
    b("但“边际有效 ≠ 处处有效” → 下页。", { color: FN }),
  ]);
}

// ===================================================== 11 CONFORMAL — AD CONDITIONAL
{
  const s = figslide("可信度 (2/2)", "但低相似度（新化学）分子明显欠覆盖", "fig_d_adcov.png",
    "按到训练集 Tanimoto 分层的条件覆盖（红虚线 = 0.90）",
    `【~55s】把覆盖率按到训练集的相似度分层，隐患就暴露了：8 个模型里有 6 个在低相似度、也就是新骨架子群上系统性欠覆盖、有的低到 0.2 到 0.5，而高相似度子群过覆盖——两者在总体上相互抵消，让边际覆盖"看起来"达标。也就是边际有效不等于处处有效。在低相似度上仍接近 0.9 的有两个：TabPFN 和 MolFormer——但 MolFormer 是靠区间显著更宽换来的，TabPFN 则同时保持紧致，是唯一又紧又稳的。这对外推到新化学的虚拟筛选是直接的可信度警示。需要说明的是，低相似度子群样本量较小（n=10），但欠覆盖的方向在八个模型上完全一致。`, 8.3);
  card(s, 8.95, 1.7, 3.85, 4.3, "条件覆盖失效", [
    b("8 中 6 个模型低 AD 覆盖率降至 0.2–0.5；TabPFN 与 MolFormer 仍接近 0.9（后者区间更宽）。", { color: ACCENT }),
    b("高 AD 过覆盖 → 边际相互抵消。"),
    b("→ 边际有效 ≠ 处处有效。", { color: FN }),
    b("局限：低 AD 子群 n=10（小），但欠覆盖方向跨 8 模型一致。", { fontSize: 12.5, color: MUTE }),
  ]);
}

// ===================================================== 12 DOCKING
{
  const s = figslide("结构对接", "对接打分与活性的相关主要由分子尺寸混淆；docking 价值在结合位姿", "fig_d_docking.png",
    "Vina 打分 vs 实验 pIC50（按重原子数着色，n=301）",
    `【~60s】配体模型只看分子，对接引入蛋白口袋。把测试集配体对接进 β-分泌酶晶体 4D8C，比较打分、机器学习和实验活性。结论：Vina 打分和活性只弱相关 −0.33，控制分子尺寸后接近 0（偏相关 −0.17）——因为活性分子普遍偏大、Vina 偏爱大分子，是尺寸混淆。监督 QSAR 在同一测试集上相关性更高，但两者用途不同：对接给的是结合位姿、不是排序工具。这里只对接了 301 个测试分子、不是全部 1504，因为对接是为下一步 PBCNet2 提供位姿。共晶配体重对接 RMSD 仅 1.13 埃，支持基础对接设置合理，在 backup。`, 8.3);
  card(s, 8.95, 1.7, 3.85, 4.3, "用途不同，非同类性能比较", [
    b("Vina vs 活性：raw r=−0.33，控尺寸 −0.17。", { color: ACCENT }),
    b("监督 QSAR 在同一测试集上相关性更高（R 0.84–0.87）。"),
    b("对接价值 = 结合位姿（作为 PBCNet2 输入）。"),
    b("基础对接设置合理：re-dock 1.13 Å（backup）。", { fontSize: 12.5, color: MUTE }),
  ]);
}

// ===================================================== 13 PBCNet2 (negative)
{
  const s = figslide("结构-成对模型", "结构-成对模型在对接位姿上未超过配体模型（可能受位姿质量影响）", "fig_d_pbcnet.png",
    "活性悬崖对上的相对活性排序 |Spearman|",
    `【~65s】配体模型在悬崖上集体失手，而 PBCNet2 是郑明月老师课题组的结构-成对模型、专门预测一对配体的相对活性、为悬崖场景而生——我也是这个工作的第二作者。我在 161 个活性悬崖对上检验它，用相对活性排序的 |Spearman ρ|（已按 ΔΔG 符号对齐）衡量。结果是一个阴性结果：它在悬崖对上只有 0.375、明显低于配体模型的 0.78；我还做了约束对接的公平复测、把位姿对齐到一致参考系，仍未改变结论。可能的主因是 cross-docking 位姿质量、而非 PBCNet2 本身——它在自己的 FEP 基准上很强，但那需要晶体级共晶位姿，而 MoleculeNet 这种多样配体只能批量对接、位姿不够可靠。这个否定结果界定了结构-成对模型该在什么条件下用。`, 8.3);
  card(s, 8.95, 1.7, 3.85, 4.3, "阴性结果（界定适用范围）", [
    b("悬崖对 n=161；指标 = |Spearman ρ|（已按 ΔΔG 符号对齐）。"),
    b("PBCNet2 0.38（约束 0.17）< RF/TabPFN 0.78。", { color: ACCENT }),
    b("可能主要受 cross-docking 位姿质量限制（非模型本身）。", { color: FN }),
    b("界定适用范围：需晶体/FEP 级位姿。"),
  ]);
}

// ===================================================== 14 ROBUSTNESS + INTERPRETABILITY
{
  const s = content("稳健性与可解释性", "换指纹不改模型排名；模型学到真实构效信号");
  s.addImage(fit("fig_d_ablation.png", 0.5, 1.6, 8.2, 4.9));
  caption(s, "固定随机森林、仅更换分子表征（第五讲指纹清单）", 0.5, 6.55, 8.2);
  card(s, 8.95, 1.7, 3.85, 4.8, "稳健性 + 可解释性", [
    b("表征消融：固定 RF 只换指纹，Morgan / Atom Pair / Topological Torsion 同属第一梯队，MACCS 最弱——模型排名非表征假象。", { color: ACCENT }),
    b("RF+Morgan 0.559 ≈ 调参后 0.554，互为印证。", { fontSize: 12.5, color: MUTE }),
    b("描述符置换重要性：分子大小 / 复杂度主导（Chi0 ρ=+0.47、BertzCT），即更大更复杂更强。"),
    b("Morgan 子结构可定位增效 / 致弱基团（backup）。"),
    b("呼应课程“可解释性 + 适用域”。", { color: FN, fontSize: 12.5 }),
  ]);
  s.addNotes(`【~50s】主结论之外，补两项稳健性与可解释性检查，呼应课程第四、五讲。其一表征消融：固定随机森林、只换分子指纹——Morgan、Atom Pair、Topological Torsion 几乎并列，MACCS 最弱；说明前面的模型排名不是某种指纹的产物，而且未调参的 RF+Morgan 0.559 和调参后的 0.554 几乎一样，互相印证。其二可解释性：描述符层面分子大小和复杂度主导，Chi0 与活性相关 +0.47，也就是这个系列里更大更复杂的分子更强；再把最重要的 Morgan 位画回子结构，能定位增效和致弱基团，backup 有图。这让模型不再是黑箱。`);
}

// ===================================================== 15 EVALUATION AXES
{
  const s = content("更多评估轴", "回归之外：分类与早期富集——谁第一随口径而变");
  s.addImage(fit("fig_d_classaxes.png", 0.5, 1.6, 8.2, 4.9));
  caption(s, "8 个模型的 ROC 曲线（active = pIC50 ≥ 7，类别均衡 43.9%）", 0.5, 6.55, 8.2);
  card(s, 8.95, 1.7, 3.85, 4.8, "分类 + 早期富集", [
    b("分类（active = pIC50≥7）：Chemprop ROC-AUC 0.915 / MCC 0.650 最高，TabPFN 次之。", { color: ACCENT }),
    b("回归最优（TabPFN）≠ 分类最优（Chemprop）——口径改变排名，故多轴并看。", { color: FN }),
    b("早期富集（高活性 = 最高 10%）：前 5–10% 富集约 4–7 倍于随机；EF@10% ChemFM / RF 领先（backup）。"),
    b("完整演练第五讲二分类指标（ROC / PR / F1 / MCC）。", { fontSize: 12.5, color: MUTE }),
  ]);
  s.addNotes(`【~50s】BACE 作业是回归，但我把回归预测派生出两条额外评估轴，顺带演练第五讲的全套指标。分类轴：在 100 纳摩阈值二分，Chemprop 图网络 ROC-AUC 0.915、MCC 0.650 最高，TabPFN 次之——一个有意思的细节是，回归点估计最好的 TabPFN 在分类上略输给 Chemprop，说明没有单一模型通吃、评估口径会改变谁第一，所以我多轴并看。早期富集轴模拟先导优化里“能否把最强的分子排到最前”：以最高活性 10% 为命中，各模型在表头富集 4 到 7 倍于随机，ChemFM 和随机森林在 EF@10% 上领先。富集曲线放在 backup。`);
}

// ===================================================== 16 REPRODUCIBILITY
{
  const s = content("可复现性", "泄漏、随机化与可复现性审计");
  card(s, 0.6, 1.65, 6.0, 4.8, "阴性对照（排除随机标签记忆）", [
    b("置换训练标签后用同一流程重训：RF / GBM / TabPFN 测试 R 分别降至 −0.11 / −0.04 / −0.05。"),
    b("→ 支持 R≈0.84–0.87 来自可学习的结构-活性信号、而非对随机标签的记忆。", { color: ACCENT }),
    b("8 个模型共享同一划分，减少划分差异带来的比较偏置。"),
    b("置换覆盖经典+表格代表（RF/GBM/TabPFN）；深度模型未跑置换（成本），但共享同一干净划分。", { fontSize: 12, color: MUTE }),
  ]);
  card(s, 6.75, 1.65, 5.9, 4.8, "泄漏审计 + 纪律", [
    b("预定义划分骨架重叠约 64%、约 29% 测试分子 Tanimoto>0.85。"),
    b("scaffold split 已把相似性虚高剥离（RF MAE 0.554→0.684）。", { color: ACCENT }),
    b("统一 set_all_seeds；3 seed；调参只用验证 / 折内 CV；测试集只看一次；所有图由 scripts/ 脚本一键重出（仓库可查）。"),
  ]);
  s.addNotes(`【~45s】方法论的地基是对自己结果的审计。阴性对照：把训练标签打乱重训，三个模型的测试相关都降至零附近——支持我报的 0.84 到 0.87 来自可学习的结构-活性信号、不是对随机标签的记忆。泄漏审计：预定义划分骨架重叠约 64%，我用骨架严格划分把这部分虚高定量剥离、如实报告。加上统一随机化、三个种子、调参只用验证集、测试集只看一次、所有图一键重出。这些让我敢说每个数字都站得住、也都可被质疑。`);
}

// ===================================================== 17 TAKE-HOME + LIMITATIONS
{
  const s = content("结论", "Implications and limitations");
  card(s, 0.6, 1.65, 6.0, 4.8, "Take-home", [
    b("小样本分子回归：零调参 TabPFN 在三轴（点预测 / scaffold 外推 / conformal 可信度）均达第一梯队，并在外推与低 AD 覆盖上、于本实验中表现最稳。", { color: ACCENT }),
    b("误差系统性集中在悬崖 / 标签冲突 / 新骨架；不确定性可提前标出难分子。"),
    b("结构方法（对接 / PBCNet2）未超过配体模型（可能受位姿质量影响）——一个界定适用范围的阴性结果。"),
  ]);
  card(s, 6.75, 1.65, 5.9, 4.8, "Limitations", [
    b("单数据集（BACE），结论外推到其他靶点需验证。"),
    b("预定义划分存在骨架泄漏（已用 scaffold split 缓解并披露）。"),
    b("PBCNet2 检验受限于 cross-docking 位姿质量，缺共晶位姿。"),
    b("未做前瞻性实验验证（prospective validation）。"),
  ]);
  s.addText("谢谢！恳请老师批评指正。", { x: 0.6, y: 6.55, w: 12, h: 0.45, fontFace: SERIF, fontSize: 17, bold: true, color: FN, align: "center", margin: 0 });
  s.addNotes(`【收尾 ~45s】三个 take-home：小样本分子回归上，零调参的 TabPFN 在点预测、外推、可信度三轴均达第一梯队、并在外推与可信度上最稳；误差系统性集中在悬崖和新骨架、不确定性能提前标出；结构方法未超过配体模型（可能受位姿质量影响）、我诚实报告了否定结果。限制也讲清楚：单数据集、预定义划分有泄漏（已用 scaffold split 缓解）、PBCNet2 检验可能受位姿质量影响、且没做前瞻性实验。最大的收获是把一个问题做到能 defend、并诚实面对否定结果。谢谢老师，请批评指正。`);
}

// ===================================================== BACKUP DIVIDER
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 1.4, h: H, fill: { color: FN }, line: { color: FN } });
  s.addImage({ path: LOGO, x: W - 2.02, y: 0.32, w: 1.62, h: 0.55 });
  s.addShape(pres.shapes.RECTANGLE, { x: 4.9, y: 2.55, w: 2.0, h: 2.0, fill: { color: FN }, line: { color: FN } });
  s.addText("B", { x: 4.9, y: 2.55, w: 2.0, h: 2.0, fontFace: LATIN, fontSize: 60, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText("Backup　补充材料", { x: 1.4, y: 4.8, w: W - 1.8, h: 0.6, fontFace: SERIF, fontSize: 24, bold: true, color: FN, align: "center", margin: 0 });
  s.addImage({ path: GATE, x: W / 2 - 1.1, y: H - 1.6, w: 2.2, h: 2.2 * 708 / 1000 });
}

// ===== B1 full table =====
{
  const s = content("Backup · 完整指标", "全模型 × 四指标（深度模型 3 seed mean ± sd；测试集 n=303）");
  jtable(s, 0.55, 1.65, 12.2, [2.5, 2.5, 2.5, 2.5, 2.5],
    [["模型 (split: predefined)", "MAE↓", "RMSE↓", "Pearson R↑", "Spearman ρ↑"],
     ["TabPFN (desc)", "0.498", "0.668", "0.873", "0.834"],
     ["Chemprop (3 seed)", "0.551±0.008", "0.737±0.015", "0.842±0.007", "0.827±0.009"],
     ["RF + Morgan", "0.554", "0.747", "0.841", "0.822"],
     ["ChemFM-3B (3 seed)", "0.559±0.022", "0.762±0.021", "0.831±0.009", "0.816±0.007"],
     ["GBM", "0.569", "0.753", "0.835", "0.813"],
     ["Ridge", "0.596", "0.796", "0.814", "0.802"],
     ["SVR (RBF)", "0.599", "0.793", "0.815", "0.797"],
     ["MolFormer-XL (3 seed)", "0.674±0.054", "0.885±0.057", "0.775±0.035", "0.733±0.034"]], 1);
  s.addText("说明：经典模型与 TabPFN 为确定性单次（重复性策略不同，见 p4）；深度模型报 3 seed（42/1337/2024）mean ± sd。划分 = bace_clean 预定义。", { x: 0.55, y: 5.8, w: 12.2, h: 0.5, fontFace: SERIF, fontSize: 12, color: MUTE, margin: 0 });
  s.addNotes("Backup：完整四指标 + 标准差，回答“方差在哪”。");
}

// ===== B2 training + nested CV =====
{
  const s = content("Backup · 训练流程", "训练 / 验证 / 测试 + 嵌套交叉验证");
  codebox(s, 0.6, 1.65, 6.1, 3.0,
`# MolFormer-XL: SMILES -> molecular vector -> pIC50
enc = tok(smiles, max_length=128, ...)
out = backbone(enc.input_ids, enc.attention_mask)
pooled = masked_mean(out.last_hidden_state, mask)
pred = head(pooled)
# 5 steps: zero_grad -> forward -> MSE
#          -> backward -> clip + step (AdamW + OneCycleLR)
# best-on-val early stopping; ChemFM LoRA trains 0.2%`);
  card(s, 6.85, 1.65, 5.8, 3.0, "嵌套交叉验证（经典模型）", [
    b("外层：留出测试只用一次。"),
    b("内层：5 折 CV + Optuna TPE 调参，标准化折内拟合（leak-safe）。"),
    b("避免乐观偏差（Cawley-Talbot 2010）。"),
  ]);
  card(s, 0.6, 4.85, 12.05, 1.55, "防过拟合", [
    b("best-on-val 早停；LoRA 0.2%；dropout + 梯度裁剪；经典 5 折 CV。未观察到明显验证-测试落差，未见明显过拟合迹象。", { fontSize: 13.5 }),
  ]);
  s.addNotes("Backup：训练实现 + 嵌套 CV 细节 + 防过拟合。");
}

// ===== B3 default vs tuned =====
{
  figslide("Backup · 调参", "默认 vs 调参：欠正则模型大幅受益、树集成边际", "fig_tune_default_vs_tuned_j.png",
    "调参对岭回归/SVR 大幅提升，对 RF/GBM 仅边际（个别略降）",
    "Backup：调参的真实增量——调参不必然带来稳定增益。", 9.0);
}

// ===== B4 CD diagram =====
{
  figslide("Backup · 显著性", "临界差异图（描述性；Friedman p=1.3e-5，n=303）", "fig_cd_diagram_j.png",
    "逐分子误差平均秩——逐分子非独立重复，仅作描述性可视化，非严格显著性检验",
    "Backup：CD 图仅作描述性可视化（逐分子非独立重复=伪重复）。严格显著性来自两条：(i) 逐分子配对 bootstrap（10k+Bonferroni）显示 TabPFN 显著优于 RF / GBM（p<0.01）；(ii) 折级 Nadeau–Bengio：主分析 ρ=1/9 下其余模型均显著更差，但保守 ρ=0.25 NB-Holm 下仅 MolFormer 仍显著更差——两种检验灵敏度不同。", 10.0);
}

// ===== B5 data dashboard =====
{
  figslide("Backup · 数据审计", "数据质量仪表盘揭示评估风险", "eda_quality_dashboard_j.png",
    "立体未定义 75% · 骨架重叠约 64% · 测试-训练 Tanimoto · 类药通过率 · 描述符-活性相关",
    "Backup：完整数据审计——立体未定义、骨架重叠、相似性泄漏。", 9.5);
}

// ===== B6 re-dock =====
{
  const s = figslide("Backup · 对接验证", "共晶配体 re-dock 支持基础对接设置合理（RMSD 1.13 Å）", "fig_redock_overlay.png",
    "对接位姿（彩色）与晶体位姿叠合",
    "Backup：re-dock 1.13 Å 支持基础对接设置合理；弱相关更可能源于打分函数局限。", 7.0);
  card(s, 8.0, 1.9, 4.7, 3.6, "为什么 n=301 而非 1504", [
    b("对接为 PBCNet2 提供位姿，对测试集 303 配体对接（2 个超大柔性失败）。"),
    b("Vina 负分 = 更强结合 → 与 pIC50 负相关方向正确。"),
    b("尺寸混淆：活性配体偏大、Vina 偏好大配体。"),
  ]);
}

// ===== B7 PBCNet2 method =====
{
  figslide("Backup · PBCNet2 方法", "PBCNet2 接入：成对图 + 两种位姿协议", "fig_pbcnet_cliffs_j.png",
    "悬崖 vs 系列对照；cross-dock 与 constrained 两种位姿",
    "Backup：PBCNet2 输入 = 蛋白口袋 + 配体位姿 → 图 → 成对 ΔpAct；972 对中 161 悬崖。悬崖对间共享配体（非独立），pair-level Spearman 有效样本量被高估、严格 CI 需按配体 bootstrap；主表用 |ρ|（PBCNet2 输出 ΔΔG 与 ΔpIC50 反号），带符号版见仓库。", 9.0);
}

// ===== B8 interpretability detail =====
{
  figslide("Backup · 可解释性", "Morgan 子结构：增效 / 致弱基团定位", "interpretability_substructures.png",
    "随机森林重要性最高的 6 个 Morgan 子结构（绿 = 增效，红 = 致弱）",
    "Backup：把 RF 重要性最高的 Morgan 位映射回原子环境；某含氟芳环环境一旦出现，平均 pIC50 下降约 1.73。描述符层面 Chi0（分子大小）居首、ρ=+0.47，主题是“更大更复杂的分子更强”。", 8.0);
}

// ===== B9 enrichment detail =====
{
  figslide("Backup · 早期富集", "富集曲线 + EF@5%（高活性 = 最高 10%）", "screening_power.png",
    "潜力命中 = 最高 10% pIC50（n=31，pIC50≥8.04）；EF 天花板 9.8",
    "Backup：模拟先导优化的早期识别能力；EF@1% 仅 k=3、噪声大，仅作参考；主看 EF@5/10% 与富集曲线。各模型在排序表头富集 4–7 倍于随机。", 9.5);
}

pres.writeFile({ fileName: "report/课程汇报_BACE.pptx" }).then((f) => console.log("wrote " + f + " (main + backup; pageNo=" + pageNo + ")"));
