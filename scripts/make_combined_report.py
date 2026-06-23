#!/usr/bin/env python3
"""Concatenate the Chinese course report with the JCIM manuscript + SI as an appendix.

Output: report/课程实验报告_BACE_含附录.pdf

Each source PDF keeps its own native typesetting and pagination; the parts are
joined with a navigable top-level outline (bookmark) per part. Regenerate after
rebuilding either the report or the manuscript PDFs.
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter

ROOT = Path(__file__).resolve().parent.parent
PARTS: list[tuple[Path, str]] = [
    (ROOT / "report" / "课程实验报告_BACE.pdf", "课程实验报告（中文）"),
    (ROOT / "paper" / "manuscript_jcim.pdf", "附录 D · JCIM 论文正文（英文）"),
    (ROOT / "paper" / "manuscript_jcim_SI.pdf", "附录 D · JCIM 补充材料 SI（英文）"),
]
OUT = ROOT / "report" / "课程实验报告_BACE_含附录.pdf"


def main() -> None:
    writer = PdfWriter()
    for path, title in PARTS:
        if not path.exists():
            raise SystemExit(f"missing source PDF: {path}")
        before = len(writer.pages)
        writer.append(str(path), outline_item=title)
        print(f"  + {len(writer.pages) - before:>2d} pages  {title}  <- {path.name}")
    with open(OUT, "wb") as fh:
        writer.write(fh)
    print(f"wrote {OUT}  ({len(writer.pages)} pages total)")


if __name__ == "__main__":
    main()
