#!/usr/bin/env bash
# One-time host setup so LibreOffice (used to export the report .docx and slides
# .pptx to PDF) embeds the open-source Noto CJK fonts instead of substituting
# macOS Apple fonts (Hiragino / Songti / PingFang), whose licenses restrict PDF
# embedding and which are absent on other machines.
#
# Why this is needed on macOS 14+: LibreOffice resolves fonts via CoreText, but
# fonts merely dropped into ~/Library/Fonts (e.g. by the brew cask below) are not
# always registered with CoreText, and `atsutil` is a no-op on macOS 14+.
# LibreOffice DOES scan its own bundled font directory, so we copy the Noto CJK
# OTFs there. On Linux this script is unnecessary: fontconfig finds the fonts and
# soffice uses them directly.
#
# Prereq (macOS): brew install --cask font-noto-sans-cjk-sc font-noto-serif-cjk-sc
set -euo pipefail

LOFONTS="/Applications/LibreOffice.app/Contents/Resources/fonts/truetype"
SRC="${HOME}/Library/Fonts"

[ -d "${LOFONTS}" ] || { echo "LibreOffice font dir not found: ${LOFONTS}"; exit 1; }

copied=0
for f in NotoSansCJKsc-Regular NotoSansCJKsc-Bold NotoSansCJKsc-Medium \
         NotoSerifCJKsc-Regular NotoSerifCJKsc-Bold NotoSerifCJKsc-Medium; do
  if [ -f "${SRC}/${f}.otf" ]; then
    cp -f "${SRC}/${f}.otf" "${LOFONTS}/" && echo "copied ${f}.otf" && copied=$((copied + 1))
  else
    echo "warning: missing ${SRC}/${f}.otf (run the brew cask first)"
  fi
done

echo "Done (${copied} fonts). Re-export the report/slides PDFs with soffice --convert-to pdf."
