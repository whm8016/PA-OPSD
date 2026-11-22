# -*- coding: utf-8 -*-
"""Fill the BUPT master's proposal template with proposal_v3_content. Writes a new docx."""
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm
from docx.text.paragraph import Paragraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proposal_v3_content as C  # noqa: E402

TEMPLATE = Path(r"C:\Users\whm\Desktop\论文\02-教育-RAG与Agent直接相关工作\开题报告-唐祎迪.docx")
OUT = Path(r"D:\OPSD\开题报告_OPSD自进化_v3.docx")

MARK = re.compile(r"(\*\*.+?\*\*|_\{.+?\}|\^\{.+?\})")


class Protos:
    def __init__(self, doc):
        t1 = doc.tables[1].cell(0, 0).paragraphs
        t2 = doc.tables[2].cell(0, 0).paragraphs
        self.h1 = t1[1]._p
        self.body = t1[2]._p
        self.ref = t1[23]._p
        self.caption = t2[4]._p
        self.figure = t2[3]._p
        self.symtab = deepcopy(doc.tables[4].cell(1, 0)._tc.xpath("./w:tbl")[0])


def ppr_of(p_elm):
    ppr = p_elm.find(qn("w:pPr"))
    return deepcopy(ppr) if ppr is not None else None


def rpr_of(p_elm):
    r = p_elm.find(qn("w:r"))
    if r is None:
        return None
    rpr = r.find(qn("w:rPr"))
    return deepcopy(rpr) if rpr is not None else None


def clean_rpr(rpr):
    if rpr is None:
        rpr = OxmlElement("w:rPr")
    for tag in ("w:b", "w:bCs", "w:vertAlign", "w:i", "w:color", "w:highlight", "w:u"):
        for el in rpr.findall(qn(tag)):
            rpr.remove(el)
    return rpr


MATH_CHARS = re.compile(r"([\u0370-\u03ff·‖∈∪Σ−≥≤∼~]+)")


def add_run(p_elm, text, base_rpr, bold=False, vert=None):
    for i, piece in enumerate(MATH_CHARS.split(text)):
        if piece:
            _add_run(p_elm, piece, base_rpr, bold, vert, latin=bool(i % 2))


def _add_run(p_elm, text, base_rpr, bold, vert, latin):
    r = OxmlElement("w:r")
    rpr = clean_rpr(deepcopy(base_rpr))
    if latin:
        for el in rpr.findall(qn("w:rFonts")):
            rpr.remove(el)
        fonts = OxmlElement("w:rFonts")
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            fonts.set(qn(attr), "Times New Roman")
        rpr.insert(0, fonts)
    if bold:
        rpr.append(OxmlElement("w:b"))
    if vert:
        v = OxmlElement("w:vertAlign")
        v.set(qn("w:val"), vert)
        rpr.append(v)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    p_elm.append(r)


def new_par(proto_elm, text="", bold_all=False, align=None, drop_indent=False):
    p = OxmlElement("w:p")
    ppr = ppr_of(proto_elm)
    if ppr is not None:
        if drop_indent:
            for ind in ppr.findall(qn("w:ind")):
                ppr.remove(ind)
        if align:
            for jc in ppr.findall(qn("w:jc")):
                ppr.remove(jc)
            jc = OxmlElement("w:jc")
            jc.set(qn("w:val"), align)
            ppr.append(jc)
        p.append(ppr)
    add_marked(p, text, rpr_of(proto_elm), bold_all)
    return p


def add_marked(p, text, base, bold_all=False):
    for seg in MARK.split(text):
        if not seg:
            continue
        if seg.startswith("**"):
            add_run(p, seg[2:-2], base, bold=True)
        elif seg.startswith("_{"):
            add_run(p, seg[2:-1], base, bold=bold_all, vert="subscript")
        elif seg.startswith("^{"):
            add_run(p, seg[2:-1], base, bold=bold_all, vert="superscript")
        else:
            add_run(p, seg, base, bold=bold_all)


def resolve_citations(sections):
    order = {}
    pat = re.compile(r"\{([A-Za-z0-9]+)\}")

    def sub(m):
        key = m.group(1)
        if key not in C.REFS:
            return m.group(0)
        if key not in order:
            order[key] = len(order) + 1
        return f"[{order[key]}]"

    out = []
    for blocks in sections:
        nb = []
        for b in blocks:
            if b[0] in ("p", "h2") and isinstance(b[1], str):
                nb.append((b[0], pat.sub(sub, b[1])) + tuple(b[2:]))
            else:
                nb.append(b)
        out.append(nb)
    return out, order


def fill_symtab(tbl, rows, base):
    trs = tbl.findall(qn("w:tr"))
    proto_row = trs[1]
    for tr in trs[1:]:
        tbl.remove(tr)
    for sym, meaning in rows:
        tr = deepcopy(proto_row)
        for tc, text in zip(tr.findall(qn("w:tc")), (sym, meaning)):
            ps = tc.findall(qn("w:p"))
            for extra in ps[1:]:
                tc.remove(extra)
            p = ps[0]
            for child in list(p):
                if child.tag != qn("w:pPr"):
                    p.remove(child)
            add_marked(p, text, base)
        tbl.append(tr)
    return tbl


def fill_cell(doc, cell, blocks, protos, order):
    tc = cell._tc
    keep = tc.findall(qn("w:p"))[0]
    for child in list(tc):
        if child.tag in (qn("w:p"), qn("w:tbl")) and child is not keep:
            tc.remove(child)
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            tc.append(new_par(protos.h1, b[1], bold_all=True))
        elif kind == "h2":
            tc.append(new_par(protos.h1, b[1], bold_all=True))
        elif kind == "p":
            tc.append(new_par(protos.body, b[1]))
        elif kind == "eq":
            tc.append(new_par(protos.caption, f"{b[1]}        （{b[2]}）", align="center", drop_indent=True))
        elif kind == "cap":
            tc.append(new_par(protos.caption, b[1]))
        elif kind == "fig":
            p = new_par(protos.figure, "", align="center", drop_indent=True)
            tc.append(p)
            Paragraph(p, cell).add_run().add_picture(str(b[1]), width=Cm(15.0))
            tc.append(new_par(protos.caption, b[2]))
        elif kind == "symtab":
            tc.append(fill_symtab(deepcopy(protos.symtab), b[1], rpr_of(protos.body)))
        elif kind == "refs":
            tc.append(new_par(protos.h1, "主要参考文献目录", bold_all=True))
            for key, n in sorted(order.items(), key=lambda kv: kv[1]):
                tc.append(new_par(protos.ref, f"[{n}] {C.REFS[key]}"))
    tc.append(OxmlElement("w:p"))


def text_runs(p):
    return [r for r in p.runs if not r._r.xpath(".//w:drawing")]


def blank_after_colon(p):
    seen = False
    for r in text_runs(p):
        if seen:
            r.text = ""
        elif ":" in r.text:
            r.text = r.text[: r.text.index(":") + 1]
            seen = True


def set_cell_text(cell, text):
    p = cell.paragraphs[0]
    runs = text_runs(p)
    if runs:
        runs[0].text = text
        for r in runs[1:]:
            r.text = ""
    else:
        r = p.add_run(text)
        r.font.name = "宋体"
        r._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")
    for extra in cell.paragraphs[1:]:
        for r in text_runs(extra):
            r.text = ""


def fill_cover(doc):
    for r in doc.paragraphs[0]._p.findall(qn("w:r")):
        if r.xpath(".//w:drawing"):
            doc.paragraphs[0]._p.remove(r)
    for p in doc.paragraphs[11:19]:
        blank_after_colon(p)
    for r in text_runs(doc.paragraphs[21]):
        r.text = ""
    text_runs(doc.paragraphs[21])[0].text = "      年    月    日"


def fill_basic_table(doc):
    t = doc.tables[0]
    set_cell_text(t.cell(0, 1), C.TITLE)
    set_cell_text(t.cell(1, 1), "")
    set_cell_text(t.cell(1, 4), C.PAPER_TYPE)
    for ri in range(4, 16):
        for c in t.rows[ri].cells:
            for p in c.paragraphs:
                for r in text_runs(p):
                    r.text = ""
    last = t.rows[16].cells[0]
    set_cell_text_lines(last, ["已修课程是否满足培养方案要求：", "修课计划："])


def set_cell_text_lines(cell, lines):
    for p, line in zip(cell.paragraphs, lines):
        runs = text_runs(p)
        if runs:
            runs[0].text = line
            for r in runs[1:]:
                r.text = ""


def fill_plan(doc):
    t = doc.tables[5]
    for i, (when, what, expect) in enumerate(C.PLAN):
        row = t.rows[2 + i]
        for ci, text in enumerate((when, what, expect)):
            set_cell_text(row.cells[ci], text)
        for p in row.cells[1].paragraphs + row.cells[2].paragraphs:
            p.alignment = 0


def count_chars(blocks):
    n = 0
    for b in blocks:
        if b[0] in ("p", "h1", "h2") and isinstance(b[1], str):
            s = b[1]
            n += len(re.findall(r"[\u4e00-\u9fff]", s)) + len(re.findall(r"[A-Za-z0-9]+", s))
    return n


def main():
    shutil.copyfile(TEMPLATE, OUT)
    doc = docx.Document(OUT)
    protos = Protos(doc)
    sections, order = resolve_citations([C.SEC1, C.SEC2, C.SEC3, C.SEC4, C.SEC5])
    fill_cover(doc)
    fill_basic_table(doc)
    fill_cell(doc, doc.tables[1].cell(0, 0), sections[0], protos, order)
    fill_cell(doc, doc.tables[2].cell(0, 0), sections[1], protos, order)
    fill_cell(doc, doc.tables[3].cell(0, 0), sections[2], protos, order)
    fill_cell(doc, doc.tables[4].cell(0, 0), sections[3], protos, order)
    fill_cell(doc, doc.tables[4].cell(1, 0), sections[4], protos, order)
    fill_plan(doc)
    doc.save(OUT)
    unused = sorted(set(C.REFS) - set(order))
    print("saved", OUT)
    print("refs cited:", len(order), "unused:", unused)
    for name, blocks in zip(["一", "二", "三", "四", "五"], sections):
        print(name, count_chars(blocks))


if __name__ == "__main__":
    main()
