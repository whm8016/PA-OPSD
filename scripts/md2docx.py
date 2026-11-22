"""把中文 Markdown 译文批量转成 Word（.docx）。

用法：
    python md2docx.py <输入目录或 md 文件> [输出目录]
"""

import os
import re
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

CJK_FONT = "宋体"
CODE_FONT = "Consolas"


def _set_font(run, font_name=CJK_FONT, size=None, bold=None, italic=None):
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def _add_inline(paragraph, text):
    """解析行内 **粗体**、*斜体*、`代码`，其余按普通文本处理。"""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)  # 链接只留文字
    tokens = re.split(r"(\*\*.+?\*\*|`.+?`|\*[^*]+?\*)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = paragraph.add_run(tok[2:-2])
            _set_font(r, bold=True)
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = paragraph.add_run(tok[1:-1])
            _set_font(r, font_name=CODE_FONT, size=10)
            r.font.color.rgb = RGBColor(0x80, 0x00, 0x00)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = paragraph.add_run(tok[1:-1])
            _set_font(r, italic=True)
        else:
            r = paragraph.add_run(tok)
            _set_font(r)


def _add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(x in ("", None) or set(x) <= set("-: ") for x in c)]
    if not cells:
        return
    ncol = max(len(c) for c in cells)
    table = doc.add_table(rows=len(cells), cols=ncol)
    table.style = "Table Grid"
    for i, row in enumerate(cells):
        for j in range(ncol):
            txt = row[j] if j < len(row) else ""
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            _add_inline(p, txt)
            if i == 0:
                for r in p.runs:
                    r.bold = True
    doc.add_paragraph()


def md_to_docx(md_path, out_path):
    with open(md_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = CJK_FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CJK_FONT)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            p = doc.add_paragraph()
            r = p.add_run("\n".join(code))
            _set_font(r, font_name=CODE_FONT, size=9.5)
            p.paragraph_format.left_indent = Pt(12)
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"\s*\|[-: |]+\|\s*$", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            _add_table(doc, rows)
            continue

        m = re.match(r"(#{1,6})\s+", line)
        if m:
            level = len(m.group(1))
            text = line[m.end():].strip()
            h = doc.add_heading("", level=min(level, 4))
            _add_inline(h, text)
            for r in h.runs:
                r.font.color.rgb = RGBColor(0, 0, 0)
            i += 1
            continue

        if re.match(r"^---+$|^\*\*\*+$", stripped):
            i += 1
            continue

        m = re.match(r"^(\s*)([-*+])\s+", line)
        if m:
            text = line[m.end():]
            p = doc.add_paragraph(style="List Bullet")
            _add_inline(p, text.strip())
            i += 1
            continue

        m = re.match(r"^(\s*)(\d+)[.、)]\s+", line)
        if m:
            text = line[m.end():]
            p = doc.add_paragraph(style="List Number")
            _add_inline(p, text.strip())
            i += 1
            continue

        if stripped.startswith(">"):
            p = doc.add_paragraph()
            _add_inline(p, stripped.lstrip("> ").strip())
            for r in p.runs:
                r.italic = True
            p.paragraph_format.left_indent = Pt(20)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        _add_inline(p, stripped)
        i += 1

    for section in doc.sections:
        section.left_margin = Pt(64)
        section.right_margin = Pt(64)

    doc.save(out_path)
    return out_path


def main():
    if not sys.argv[1:]:
        print("usage: python md2docx.py <dir|file> [outdir]")
        return
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else (src if os.path.isdir(src) else os.path.dirname(src))

    if os.path.isdir(src):
        files = [os.path.join(src, f) for f in sorted(os.listdir(src)) if f.lower().endswith(".md")]
    else:
        files = [src]

    os.makedirs(outdir, exist_ok=True)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0] + ".docx"
        target = os.path.join(outdir, name)
        md_to_docx(f, target)
        print("OK", target)


if __name__ == "__main__":
    main()
