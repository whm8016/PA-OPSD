from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "开题报告_OPSD自进化_v2.md"
OUTPUT = ROOT / "开题报告_OPSD自进化_v2.docx"
FIGURE_DIR = ROOT / "docs" / "figures"
LANDSCAPE_FIGURE = FIGURE_DIR / "opsd_evolution_landscape_v3.png"
LOOP_FIGURE = FIGURE_DIR / "opsd_skill_loop_v3.png"
AUDIT = ROOT / "开题报告_OPSD自进化_v2_验收.json"


def set_cell_text(cell, text: str, bold: bool = False, size: float = 10.5) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "宋体"
    run.font.size = Pt(size)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, end])


def add_toc(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-3" \\h \\z \\u '
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "请在 Word 中右键更新目录"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, placeholder, end])


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.5)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "宋体"
    normal.font.size = Pt(12)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.first_line_indent = Cm(0.74)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(0)

    for style_name, chinese_font, size, bold in (
        ("Title", "黑体", 22, True),
        ("Heading 1", "黑体", 16, True),
        ("Heading 2", "黑体", 14, True),
        ("Heading 3", "黑体", 12, True),
    ):
        style = styles[style_name]
        style.font.name = chinese_font
        style.font.size = Pt(size)
        style.font.bold = bold
        style._element.rPr.rFonts.set(qn("w:eastAsia"), chinese_font)
        style.paragraph_format.first_line_indent = Cm(0)
        style.paragraph_format.space_before = Pt(8)
        style.paragraph_format.space_after = Pt(4)



def configure_numbered_section(section, start: int = 1) -> None:
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.5)
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    header = section.header.paragraphs[0]
    header.text = ""
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = header.add_run("北京邮电大学硕士研究生学位论文开题报告")
    run.font.name = "宋体"
    run.font.size = Pt(9)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    footer = section.footer.paragraphs[0]
    footer.text = ""
    add_page_number(footer)
    page_number = OxmlElement("w:pgNumType")
    page_number.set(qn("w:start"), str(start))
    section._sectPr.append(page_number)


def use_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def add_box(ax, xy, width, height, text, fontsize=12) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.4,
        edgecolor="#315A7D",
        facecolor="#EAF2F8",
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#17324D",
    )


def make_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    use_chinese_font()

    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    columns = [
        (0.04, 0.73, "教育导师", "LongTutor / ScaffoldLM / DICE"),
        (0.04, 0.53, "特权蒸馏", "OPSD / PIDistill / PreSO"),
        (0.04, 0.33, "经验与技能", "ReasoningBank / EvolveR / Trace2Skill / REAT"),
        (0.04, 0.13, "参数化进化", "OPD-Evolver / EvoRouter / HeirCommit"),
    ]
    for x, y, title, body in columns:
        add_box(ax, (x, y), 0.48, 0.14, f"{title}\n{body}", 10.5)
    add_box(
        ax,
        (0.66, 0.23),
        0.3,
        0.56,
        "本课题切入点\n\n学生状态条件化\n教学动作技能\n\n独立验证准入\n\nOPSD 权重沉降\n\n去检索保持率",
        11,
    )
    for start_y in (0.8, 0.6, 0.4, 0.2):
        ax.annotate(
            "",
            xy=(0.66, 0.51),
            xytext=(0.52, start_y),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#315A7D"),
        )
    ax.text(
        0.5,
        0.04,
        "研究空白：外部教学经验是否能经可靠验证后，沉降为无需检索仍可保持的单模型教学能力？",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        color="#8A3B12",
    )
    fig.tight_layout()
    fig.savefig(LANDSCAPE_FIGURE, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6.8), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        (0.04, 0.64, "单模型导师\n基于 x 生成轨迹"),
        (0.25, 0.64, "独立评价\n成功/失败归因"),
        (0.46, 0.64, "技能候选\n聚类与冲突检测"),
        (0.67, 0.64, "互斥验证集\n准入/拒绝"),
        (0.67, 0.25, "特权教师\nx + z_skill"),
        (0.46, 0.25, "动作 + token\nOPSD"),
        (0.25, 0.25, "候选模型\n确认集继承/回滚"),
        (0.04, 0.25, "新一代模型\n关闭检索复测"),
    ]
    for x, y, text in nodes:
        add_box(ax, (x, y), 0.16, 0.17, text, 11)
    arrows = [
        ((0.20, 0.725), (0.25, 0.725)),
        ((0.41, 0.725), (0.46, 0.725)),
        ((0.62, 0.725), (0.67, 0.725)),
        ((0.75, 0.64), (0.75, 0.42)),
        ((0.67, 0.335), (0.62, 0.335)),
        ((0.46, 0.335), (0.41, 0.335)),
        ((0.25, 0.335), (0.20, 0.335)),
        ((0.12, 0.42), (0.12, 0.64)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#315A7D"),
        )
    add_box(ax, (0.84, 0.43), 0.13, 0.12, "拒绝技能\n归档证据", 9.5)
    ax.annotate(
        "未通过",
        xy=(0.84, 0.49),
        xytext=(0.83, 0.66),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#8A3B12"),
        ha="center",
        fontsize=9,
        color="#8A3B12",
    )
    add_box(ax, (0.04, 0.02), 0.16, 0.12, "候选未通过\n回滚旧模型\n可靠技能保留", 9)
    ax.annotate(
        "拒绝继承",
        xy=(0.12, 0.14),
        xytext=(0.30, 0.25),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#8A3B12"),
        fontsize=9,
        color="#8A3B12",
    )
    ax.text(
        0.5,
        0.93,
        "快循环：经验提炼与验证    |    慢循环：OPSD 权重沉降与跨代继承",
        ha="center",
        fontsize=15,
        weight="bold",
        color="#17324D",
    )
    ax.text(
        0.5,
        0.17,
        "关键判据：若关闭技能检索后收益消失，只能证明外部记忆有效，不能证明参数化自进化。",
        ha="center",
        fontsize=12,
        color="#8A3B12",
    )
    fig.tight_layout()
    fig.savefig(LOOP_FIGURE, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_caption(document: Document, text: str) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run(text)
    run.font.name = "宋体"
    run.font.size = Pt(10.5)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def add_cover(document: Document) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(30)
    run = p.add_run("北京邮电大学")
    run.bold = True
    run.font.name = "黑体"
    run.font.size = Pt(24)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(28)
    run = p.add_run("硕士研究生学位论文开题报告")
    run.bold = True
    run.font.name = "黑体"
    run.font.size = Pt(22)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(48)
    p.paragraph_format.space_after = Pt(28)
    run = p.add_run("面向长期个性化辅导的教育特权在策略自蒸馏\n与教学技能自进化研究")
    run.bold = True
    run.font.name = "黑体"
    run.font.size = Pt(18)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")

    info = [
        ("学　　号", "【待填写】"),
        ("姓　　名", "【待填写】"),
        ("学生类别", "【待填写】"),
        ("学　　院", "【待填写】"),
        ("学科/专业", "【待填写】"),
        ("研究方向", "人工智能教育应用与大语言模型智能体"),
        ("导师姓名", "【待填写】"),
        ("攻读学位", "【待填写】"),
    ]
    table = document.add_table(rows=len(info), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row, (label, value) in zip(table.rows, info):
        row.cells[0].width = Cm(4)
        row.cells[1].width = Cm(8.5)
        set_cell_text(row.cells[0], label, bold=True, size=12)
        set_cell_text(row.cells[1], value, size=12)
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(30)
    run = p.add_run("2026 年 11 月")
    run.font.name = "宋体"
    run.font.size = Pt(14)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    front_section = document.add_section(WD_SECTION.NEW_PAGE)
    configure_numbered_section(front_section, start=1)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.first_line_indent = Cm(0)
    run = title.add_run("开题基本信息")
    run.bold = True
    run.font.name = "黑体"
    run.font.size = Pt(16)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    meta = [
        ("论文题目", "面向长期个性化辅导的教育特权在策略自蒸馏与教学技能自进化研究"),
        ("英文题目", "Pedagogical Privileged On-Policy Self-Distillation with Validated Skill Evolution for Long-Term Personalized Tutoring"),
        ("选题来源", "【导师科研项目与研究生自选研究方向相结合，待导师确认】"),
        ("论文类型", "【基础研究/应用基础研究，待确认】"),
        ("开题日期", "【待填写】"),
        ("开题地点", "【待填写】"),
    ]
    table = document.add_table(rows=len(meta), cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row, (label, value) in zip(table.rows, meta):
        row.cells[0].width = Cm(4)
        row.cells[1].width = Cm(12)
        set_cell_text(row.cells[0], label, bold=True)
        set_cell_text(row.cells[1], value)

    document.add_paragraph()
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run("已修课程")
    run.bold = True
    run.font.name = "黑体"
    run.font.size = Pt(14)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    courses = document.add_table(rows=5, cols=4)
    courses.style = "Table Grid"
    courses.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["课程名称", "学分", "成绩", "班级排名"]
    for i, header in enumerate(headers):
        set_cell_text(courses.rows[0].cells[i], header, bold=True)
    for row in courses.rows[1:]:
        for cell in row.cells:
            set_cell_text(cell, "【待填写】")
    set_repeat_table_header(courses.rows[0])
    p = document.add_paragraph("已修课程是否满足培养方案要求：【是 / 否，待填写】")
    p.paragraph_format.first_line_indent = Cm(0)
    document.add_page_break()

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run("目　录")
    run.bold = True
    run.font.name = "黑体"
    run.font.size = Pt(18)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    toc = document.add_paragraph()
    toc.paragraph_format.first_line_indent = Cm(0)
    add_toc(toc)
    document.add_page_break()


def add_markdown_table(document: Document, lines: list[str]) -> None:
    rows = []
    for line in lines:
        values = [value.strip() for value in line.strip().strip("|").split("|")]
        rows.append(values)
    if len(rows) >= 2 and all(set(value) <= {"-", ":"} for value in rows[1]):
        rows.pop(1)
    if not rows:
        return
    width = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=width)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(rows):
        for c_idx in range(width):
            value = row[c_idx] if c_idx < len(row) else ""
            set_cell_text(table.rows[r_idx].cells[c_idx], value, bold=r_idx == 0, size=9.5)
    set_repeat_table_header(table.rows[0])


def add_body_from_markdown(document: Document, text: str) -> None:
    start = text.index("# 一、立题依据")
    body = text[start:]
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        line = raw.strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            add_markdown_table(document, table_lines)
            continue
        if line.startswith("# "):
            document.add_heading(line[2:].strip(), level=1)
        elif line.startswith("## "):
            heading = line[3:].strip()
            document.add_heading(heading, level=2)
            if heading == "1.2 国内外研究现状与发展趋势":
                p = document.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.first_line_indent = Cm(0)
                p.add_run().add_picture(str(LANDSCAPE_FIGURE), width=Cm(15.2))
                add_caption(document, "图 1-1　研究现状、直接前驱与本课题切入点")
            if heading == "3.1 研究方法与技术路线":
                p = document.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.first_line_indent = Cm(0)
                p.add_run().add_picture(str(LOOP_FIGURE), width=Cm(15.2))
                add_caption(document, "图 3-1　教学技能快循环与 OPSD 权重沉降慢循环")
        elif line.startswith("### "):
            document.add_heading(line[4:].strip(), level=3)
        elif line.startswith("- "):
            p = document.add_paragraph(style="List Bullet")
            p.paragraph_format.first_line_indent = Cm(0)
            p.add_run(line[2:].strip())
        elif re.match(r"^L_total\s*=", line):
            p = document.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            run = p.add_run(line)
            run.font.name = "Cambria Math"
            run.font.size = Pt(11)
        else:
            p = document.add_paragraph()
            if line.startswith("[") and re.match(r"^\[\d+\]", line):
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.left_indent = Cm(0.74)
                p.paragraph_format.hanging_indent = Cm(0.74)
                p.paragraph_format.line_spacing = 1.25
                p.style = document.styles["Normal"]
            p.add_run(line)
        i += 1


def count_chinese_and_latin(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))


def section_counts(text: str) -> dict[str, int]:
    names = [
        "一、立题依据",
        "二、研究内容和目标",
        "三、研究方案设计及可行性分析",
        "四、本研究课题可能的创新之处",
        "五、研究基础与工作条件",
    ]
    result = {}
    for index, name in enumerate(names):
        start = text.index(f"# {name}")
        end = text.find(f"# {names[index + 1]}", start) if index + 1 < len(names) else text.find("# 学位论文工作计划", start)
        section_text = text[start:end]
        result[name] = count_chinese_and_latin(section_text)
    return result


def build() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    make_figures()
    document = Document()
    configure_document(document)
    add_cover(document)
    add_body_from_markdown(document, text)
    properties = document.core_properties
    properties.title = "面向长期个性化辅导的教育特权在策略自蒸馏与教学技能自进化研究"
    properties.subject = "硕士研究生学位论文开题报告"
    properties.author = "【待填写】"
    properties.keywords = "OPSD; 自进化; 智能导师; 教学技能; 长期个性化辅导"
    document.save(OUTPUT)

    with zipfile.ZipFile(OUTPUT) as archive:
        bad = archive.testzip()
        root = ET.fromstring(archive.read("word/document.xml"))
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraph_text = "".join(node.text or "" for node in root.findall(".//w:t", namespace))
    counts = section_counts(text)
    references = len(re.findall(r"^\[\d+\]", text, flags=re.MULTILINE))
    audit = {
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "figures": [str(LANDSCAPE_FIGURE), str(LOOP_FIGURE)],
        "zip_integrity_error": bad,
        "document_text_units": count_chinese_and_latin(paragraph_text),
        "section_text_units": counts,
        "minimum_checks": {
            "第一章不少于800": counts["一、立题依据"] >= 800,
            "第二章不少于2500": counts["二、研究内容和目标"] >= 2500,
            "第三章不少于800": counts["三、研究方案设计及可行性分析"] >= 800,
            "第四章不少于500": counts["四、本研究课题可能的创新之处"] >= 500,
            "第五章不少于500": counts["五、研究基础与工作条件"] >= 500,
        },
        "reference_count": references,
        "reference_target_met": 20 <= references <= 30,
        "placeholder_count": text.count("【待填写】"),
        "placeholders_preserved": text.count("【待填写】") >= 6,
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
