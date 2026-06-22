---
name: journal-polish
description: Polish academic/scientific prose to journal-submission quality (Chinese or English). Use when the user asks to 润色/polish a paper, report, abstract, or manuscript section, or to tighten academic writing before submission. Applies the Schimel OCAR structure, sentence-level concision rules, and calibrated-claim discipline. Not for casual copy or marketing text.
---

# Journal-Polish

Polish scholarly prose to journal quality **without changing meaning, data, numbers, or
citations**. Tighten language, enforce point-first structure, and calibrate every claim to
its evidence.

## Sources (sourced from high-star GitHub, do not re-derive)

- **`ahmetbersoz/chatgpt-prompts-for-academic-writing`** (4.7k★) — academic polish /
  proofread prompt patterns.
- **`khufkens/paper_writing_checklist`** — Joshua Schimel, *Writing Science* (OCAR story
  arc + sentence rules). This is the substantive backbone below.
- **Project `AGENTS.md` §16** — anti-BS / claim-calibration rules. These OVERRIDE any
  generic prompt when they conflict (a journal reviewer's bar beats a chatbot prompt).

## Hard rules (never violate)

1. **Preserve facts.** Do not alter numbers, units, dataset names, citations, or the
   direction of any finding. Polishing is wording only.
2. **Calibrate claims to evidence (§16).** No "SOTA / outperforms / 显著优于 / 首次发现"
   unless backed by a stated, significant test. If p > 0.05 → "comparable / 相当". Report
   non-significant and negative results plainly. Keep existing honest hedges (e.g.
   "n=30 未达显著", the MUBen distinction) — strengthen them, never delete them.
3. **No new claims.** Never add a result, comparison, or citation that wasn't there.

## OCAR structure (Schimel) — check each section serves its arc role

- **Opening** (Abstract/Intro start): broad context + why it matters.
- **Challenge** (Intro/Methods): the specific gap/problem + research question.
- **Action** (Methods/Results): what was done + what was found.
- **Resolution** (Discussion/Conclusion): how understanding shifted + take-aways + next.

Every paragraph: **one idea, stated point-first** — the first sentence is the paragraph's
conclusion; the rest support it. If a paragraph has two ideas, split it.

## Sentence-level checklist (apply per paragraph)

- [ ] **Point-first**: topic sentence states the claim, not background.
- [ ] **Active voice**, subject and verb close together.
- [ ] **Short**: aim 12–20 words, **hard cap ≤ 30** / 一句一义; split any sentence you must re-read.
- [ ] **Plain words**: no thesaurus inflation; reuse a term rather than vary it.
- [ ] **Cut**: hedges ("it is worth noting"), redundancy, empty intensifiers, and
      superlatives without evidence. Limit connectives (however/moreover/此外/值得注意).
- [ ] **Precise**: replace vague quantifiers with the actual number; consistent terminology.
- [ ] **Tense**: present for established facts; past for your own results/procedures.
- [ ] **Numbers**: EN — spell out < 10, numerals for ≥ 10; ZH — keep one consistent style.

## Chinese academic prose (when polishing 中文稿)

- 主题句前置:每段第一句给结论,后文支撑。
- 一段一义;一句一义,长句拆短。
- 去口语化、去最高级;"显著"必须有统计支撑,否则改"较 / 相对"。
- 术语统一(同一概念全程一个词);数字与单位精确。
- 少用连接词堆叠(然而/此外/值得注意);主动语态优先。

## Ordered pass (operational — adapted from `Yuan1z0825/nature-skills`, 15.4k★ MIT)

Run polishing as an ordered pipeline, not ad-hoc. Borrowed from the nature-polishing
12-step workflow but **stripped of its Nature house style** — do NOT import British English
or broad-significance framing; for an ACS/JCIM target use US English and technical depth.

1. **Sentence split** — break compound sentences; one claim each.
2. **Section ID** — confirm each paragraph's role in the OCAR arc.
3. **Hourglass check** — broad → specific → broad across the section.
4. **Tense audit** — present for facts, past for your results/procedures (dedicated pass).
5. **Sentence edit** — active voice, ≤ 30 words, subject and verb close.
6. **Vocabulary** — precise, consistent terms; no thesaurus inflation.
7. **Template check** — section makes its expected moves (Results: finding → evidence).
8. **Citation audit** — every claim needing a source has one; references resolve.
9. **House style** — target venue spelling/format (US English + ACS style for JCIM).
10. **Overclaim pass** — every claim calibrated to evidence (§16); soften unsupported "显著/SOTA".
11. **Proofread** — spelling, punctuation, numerals.
12. **Final read** — plain output, meaning preserved.

## Workflow

1. Read the source of truth (often a build script that generates the doc, not the doc
   itself — editing the rendered `.docx` would be overwritten on rebuild).
2. Polish section by section. For each paragraph: apply the checklist; keep meaning.
3. Show the user **before → after** for the substantive changes so they can defend each.
4. Rebuild the document; do not hand-edit the rendered output.
5. Leave one line per section noting what changed (concision / point-first / calibration).
