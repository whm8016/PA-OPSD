# -*- coding: utf-8 -*-
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

OUT = r"C:\Users\whm\Desktop\PA-OPSD_科研进展汇报_20261026.pptx"
FONT = "微软雅黑"
NAVY = RGBColor(0x1B, 0x2A, 0x4A)
INK = RGBColor(0x30, 0x36, 0x42)
ACCENT = RGBColor(0xE0, 0x76, 0x2B)
LIGHT = RGBColor(0xF2, 0xF5, 0xFA)
LINEC = RGBColor(0xD5, 0xDC, 0xE8)
GRAY = RGBColor(0x6A, 0x72, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SKY = RGBColor(0xC9, 0xD4, 0xE8)
STEEL = RGBColor(0x8F, 0xA1, 0xC4)

PAGE_W = Inches(13.333)
PAGE_H = Inches(7.5)


def set_run(r, text, size=15, bold=False, color=NAVY, name=FONT):
    r.text = text
    f = r.font
    f.size = Pt(size)
    f.bold = bold
    f.name = name
    f.color.rgb = color
    rPr = r._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        e = rPr.find(qn(tag))
        if e is None:
            e = rPr.makeelement(qn(tag), {})
            rPr.append(e)
        e.set("typeface", name)


def txbox(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tb.text_frame.word_wrap = True
    return tb.text_frame


def para(tf, segs, size=15, bold=False, color=INK, first=False, align=None,
         space_after=6, space_before=0, line_spacing=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after = Pt(space_after)
    if space_before:
        p.space_before = Pt(space_before)
    if align is not None:
        p.alignment = align
    if line_spacing:
        p.line_spacing = line_spacing
    if isinstance(segs, str):
        segs = [(segs, {})]
    for text, kw in segs:
        r = p.add_run()
        set_run(r, text, size=kw.get("size", size), bold=kw.get("bold", bold),
                color=kw.get("color", color))
    return p


def rect(slide, l, t, w, h, fill, shape=MSO_SHAPE.RECTANGLE, line_color=None):
    sh = slide.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line_color is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line_color
        sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh


def header(slide, title, idx, kicker=None):
    rect(slide, 0.55, 0.42, 0.085, 0.52, ACCENT)
    tf = txbox(slide, 0.80, 0.28, 12.0, 0.8)
    if kicker:
        para(tf, kicker, size=11, bold=True, color=ACCENT, first=True, space_after=1)
        para(tf, title, size=22, bold=True, color=NAVY)
    else:
        para(tf, title, size=24, bold=True, color=NAVY, first=True)
    rect(slide, 0.55, 1.14, 12.25, 0.018, LINEC)
    tf = txbox(slide, 0.55, 7.12, 9.0, 0.3)
    para(tf, "PA-OPSD 科研进展汇报 · 2026-10-26", size=9, color=GRAY, first=True, space_after=0)
    tf = txbox(slide, 11.9, 7.12, 0.9, 0.3)
    para(tf, "%02d" % idx, size=9, color=GRAY, first=True, align=PP_ALIGN.RIGHT, space_after=0)


def body_bullets(slide, items, top=1.38, left=0.55, width=12.25, height=5.55, size=14, gap=9):
    tf = txbox(slide, left, top, width, height)
    first = True
    for it in items:
        kind = it[0]
        if kind == "h":
            para(tf, [("▍", {"color": ACCENT, "bold": True}), (it[1], {"bold": True, "color": NAVY})],
                 size=size + 1, first=first, space_after=5, space_before=7)
        else:
            segs = [("▪ ", {"color": ACCENT, "bold": True})]
            if len(it) > 1 and it[1]:
                segs.append((it[1], {"bold": True, "color": NAVY}))
            if len(it) > 2 and it[2]:
                segs.append((it[2], {}))
            para(tf, segs, size=size, first=first, space_after=gap, line_spacing=1.04)
        first = False


def add_table(slide, rows, top, col_w, left=0.55, width=12.25, row_h=0.5,
              header_h=0.42, size=11, header_size=11.5):
    n_r = len(rows)
    n_c = len(rows[0])
    gt = slide.shapes.add_table(n_r, n_c, Inches(left), Inches(top),
                                Inches(width), Inches(row_h * n_r)).table
    gt.first_row = False
    gt.horz_banding = False
    tot = float(sum(col_w))
    for i, cw in enumerate(col_w):
        gt.columns[i].width = Emu(int(Inches(width) * cw / tot))
    for ri, row in enumerate(rows):
        gt.rows[ri].height = Emu(int(Inches(header_h if ri == 0 else row_h)))
        for ci, val in enumerate(row):
            cell = gt.cell(ri, ci)
            cell.margin_left = Inches(0.07)
            cell.margin_right = Inches(0.07)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if ri == 0:
                cell.fill.fore_color.rgb = NAVY
            else:
                cell.fill.fore_color.rgb = LIGHT if ri % 2 == 0 else WHITE
            ctf = cell.text_frame
            ctf.word_wrap = True
            for li, ln in enumerate(val.split("\n")):
                p = ctf.paragraphs[0] if li == 0 else ctf.add_paragraph()
                p.alignment = PP_ALIGN.LEFT
                r = p.add_run()
                set_run(r, ln, size=(header_size if ri == 0 else size),
                        bold=(ri == 0), color=(WHITE if ri == 0 else INK))
    return gt


def note(slide, segs, top, left=0.55, width=12.25, size=11.5):
    tf = txbox(slide, left, top, width, 0.9)
    para(tf, segs, size=size, color=GRAY, first=True, space_after=2, line_spacing=1.05)


def flow_box(slide, l, t, w, h, lines, fill=NAVY, tcolor=WHITE, size=12, line_color=None):
    sh = rect(slide, l, t, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_color=line_color)
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate(lines):
        para(tf, ln, size=(size + 0.5 if i == 0 else size), bold=(i == 0),
             color=tcolor, first=(i == 0), align=PP_ALIGN.CENTER, space_after=2,
             line_spacing=1.0)
    return sh


def arrow(slide, l, t, w=0.36, h=0.3, shape=MSO_SHAPE.RIGHT_ARROW):
    rect(slide, l, t, w, h, ACCENT, shape=shape)


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])



OUT = r"C:\Users\whm\Desktop\PA-OPSD_科研进展汇报_20261026_v4.pptx"


def main():
    prs = Presentation()
    prs.slide_width = PAGE_W
    prs.slide_height = PAGE_H

    # ---- S1 封面 ----
    s = blank(prs)
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    rect(s, 0, 7.28, 13.333, 0.22, ACCENT)
    tf = txbox(s, 0.9, 1.45, 11.6, 0.5)
    para(tf, "研究生科研进展汇报", size=16, bold=True, color=ACCENT, first=True)
    rect(s, 0.92, 2.10, 1.6, 0.06, ACCENT)
    tf = txbox(s, 0.9, 2.42, 11.8, 2.0)
    para(tf, "教育特权信息驱动的教学动作级蒸馏研究", size=33, bold=True, color=WHITE, first=True, space_after=6)
    para(tf, "PA-OPSD：从多智能体教师团队到轻量导师", size=19, color=SKY)
    tf = txbox(s, 0.9, 4.55, 11.8, 0.6)
    para(tf, "Pedagogical Action-Level Distillation with Educational Privileged Information",
         size=12.5, color=STEEL, first=True)
    tf = txbox(s, 0.9, 6.15, 11.6, 0.5)
    para(tf, "汇报人：XXX      指导教师：XXX      2026 年 10 月 26 日", size=14, color=WHITE, first=True)

    # ---- S2 汇报框架 ----
    s = blank(prs)
    header(s, "汇报框架", 2)
    items = [
        ("", "01  阶段工作回顾", "—— 精读了哪些论文、调研了哪些数据集、方向如何收敛"),
        ("", "02  研究问题与创新点", "—— RQ1-5 与问题形式化；开题聚焦的两个创新"),
        ("", "03  我的理解", "—— 数据集各起什么作用；每篇论文拿什么、不拿什么"),
        ("", "04  整体方案串讲", "—— 训练期到评测期的端到端流程"),
        ("", "05  执行顺序与基线", "—— 现在训练什么、什么时候评测、和谁对比"),
        ("", "06  差距与困惑", "—— 还差什么；需要讨论确认的四个问题"),
        ("", "07  下一步计划与风险", "—— 11-12 月任务清单；五条风险与应对"),
    ]
    body_bullets(s, items, top=1.6, size=15.5, gap=13)

    # ---- S3 阶段工作回顾 ----
    s = blank(prs)
    header(s, "阶段工作回顾：读了什么、收敛了什么", 3)
    items = [
        ("h", "论文：精读 4 篇 + 泛读 2 篇"),
        ("", "OPSD（ICML'26）", "：全文 + 本地代码库 D:\\OPSD 通读，掌握 on-policy 蒸馏训练循环"),
        ("", "LongTutor", "：问题定义 + 三级评测节 + Gold 1000 条数据格式（主 benchmark）"),
        ("", "AgentArk / ScaffoldLM", "：多 Agent 压缩进单模型的叙事；教学动作分类体系的来源"),
        ("", "泛读 LUPI / Hinton KD", "：特权信息的理论依据；动作级硬标签蒸馏的出处"),
        ("h", "数据集：确定 3 个，排除 4 个"),
        ("", "确定", "：LongTutor Gold 1000（评测）、Eedi 对话（训练）、Eedi 误概念图谱（特权 m 来源）"),
        ("", "排除", "：EduAgentBench / MMTutorBench / ASTRA 非开题必做；MathDial 验证为可选、不下载"),
        ("h", "方向收敛"),
        ("", "开题边界", "：只做 PA-OPSD 主线（B1-B5 对比）；自进化、预算路由明确放到学位论文后半段"),
        ("", "产出", "：《执行总纲》单一真相表——符号、数据集、基线、阶段任务全部对齐成文"),
    ]
    body_bullets(s, items, size=13, gap=5)


    # ---- 新增 S4 研究问题与形式化 ----
    s = blank(prs)
    header(s, "研究问题与问题形式化", 4, kicker="RESEARCH QUESTIONS")
    rect(s, 0.55, 1.45, 4.6, 5.2, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_color=LINEC)
    tf = txbox(s, 0.85, 1.7, 4.05, 4.8)
    para(tf, [("问题形式化", {"bold": True, "color": ACCENT, "size": 15})], first=True, space_after=10)
    para(tf, [("学生可见  x = (q, h, r)", {"bold": True, "color": NAVY})], size=13.5, space_after=3)
    para(tf, "题目 q ；允许使用的历史 h ；当前对话 r", size=11.5, color=GRAY, space_after=10)
    para(tf, [("特权信息  z = (k, m, y*, rho)", {"bold": True, "color": NAVY})], size=13.5, space_after=3)
    para(tf, "知识状态 k ；误概念 m ；标答 y* ；量规 rho", size=11.5, color=GRAY, space_after=10)
    para(tf, [("非对称约束（LUPI）", {"bold": True, "color": NAVY})], size=13.5, space_after=3)
    para(tf, "训练期 Teacher Team 可用 z 做监督；部署期 Student 的输出不得依赖 z —— 训练用 z 不算作弊", size=12, space_after=0, line_spacing=1.15)
    tf = txbox(s, 5.5, 1.5, 7.3, 5.1)
    para(tf, [("开题聚焦（RQ1-3）", {"bold": True, "color": ACCENT, "size": 14})], first=True, space_after=5)
    para(tf, [("RQ1  ", {"bold": True, "color": NAVY}), ("教育特权信息（k / m / 量规）能否通过 on-policy 蒸馏，变成部署时可用的教学能力？", {})], size=12.5, space_after=5, line_spacing=1.12)
    para(tf, [("RQ2  ", {"bold": True, "color": NAVY}), ("蒸馏“教学动作”相比直接蒸文本，能否在保住教学质量的同时显著省 token？", {})], size=12.5, space_after=5, line_spacing=1.12)
    para(tf, [("RQ3  ", {"bold": True, "color": NAVY}), ("多 Agent 教师团队的能力，能否压缩进单个轻量 Tutor？", {})], size=12.5, space_after=10, line_spacing=1.12)
    para(tf, [("全文延伸（RQ4-5，开题后接）", {"bold": True, "color": GRAY, "size": 14})], space_after=5)
    para(tf, [("RQ4  ", {"bold": True, "color": GRAY}), ("按学生状态动态分配推理预算，能否优化 学习增益 / token 的 Pareto？", {"color": GRAY})], size=12.5, space_after=5, line_spacing=1.12)
    para(tf, [("RQ5  ", {"bold": True, "color": GRAY}), ("教学技能与 Agent 协作结构，能否随长期交互持续进化？", {"color": GRAY})], size=12.5, space_after=0, line_spacing=1.12)

    # ---- S4 数据集理解 ----
    s = blank(prs)
    header(s, "我的理解 1：两个数据集 = 训练场 + 考场", 5, kicker="DATASETS")
    flow_box(s, 0.55, 1.42, 5.95, 0.55, ["Eedi = 训练场（在这里学会辅导）"], fill=ACCENT, size=13)
    tf = txbox(s, 0.75, 2.12, 5.7, 3.3)
    para(tf, [("真人师生数学对话", {"bold": True, "color": NAVY}), ("，每轮 tutor 话语带", {}), ("TalkMovePrediction", {"bold": True, "color": NAVY}), (" 教学动作标注", {})], size=13, first=True, space_after=7, line_spacing=1.06)
    para(tf, [("误概念图谱", {"bold": True, "color": NAVY}), ("：misconception 与 construct 关联，是特权信息 m 的来源", {})], size=13, space_after=7, line_spacing=1.06)
    para(tf, [("为我提供", {"bold": True, "color": NAVY}), ("：训练语料 + 动作标签（映射到六类教学动作 A_ped）", {})], size=13, space_after=7, line_spacing=1.06)
    flow_box(s, 6.85, 1.42, 5.95, 0.55, ["LongTutor = 考场（在这里打分）"], fill=NAVY, size=13)
    tf = txbox(s, 7.05, 2.12, 5.7, 3.3)
    para(tf, [("Gold 1000 条", {"bold": True, "color": NAVY}), (" 长期辅导记录（学生历史行为 + 多轮对话）", {})], size=13, first=True, space_after=7, line_spacing=1.06)
    para(tf, [("三级评测，正好考我最关心的三种能力：", {"bold": True, "color": NAVY})], size=13, space_after=4, line_spacing=1.06)
    para(tf, [("   1) Evidence", {"bold": True, "color": ACCENT}), ("：从长期历史里提取证据", {})], size=12.5, space_after=3)
    para(tf, [("   2) Diagnosis", {"bold": True, "color": ACCENT}), ("：诊断学生知识状态（4 类）", {})], size=12.5, space_after=3)
    para(tf, [("   3) Teaching", {"bold": True, "color": ACCENT}), ("：生成教学策略（Judge + ROUGE）", {})], size=12.5, space_after=6)
    para(tf, [("定位：评测主表；不改它的方法，只拿它当尺子", {})], size=12.5, color=GRAY)
    rect(s, 0.55, 5.62, 12.25, 1.05, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_color=LINEC)
    tf = txbox(s, 0.85, 5.78, 11.7, 0.8)
    para(tf, [("关键设计：训练用 Eedi、评测用 LongTutor，两场分离", {"bold": True, "color": NAVY}),
              ("——防止“自产自销”，测的是模型跨场景的真实泛化能力", {})], size=13.5, first=True, line_spacing=1.15)

    # ---- S5 OPSD 理解 ----
    s = blank(prs)
    header(s, "我的理解 2：OPSD 论文学到了什么", 6, kicker="METHOD BASE")
    items = [
        ("h", "它解决什么问题"),
        ("", "SFT 的 off-policy / exposure bias", "：训练时看的全是标准答案轨迹；推理时从模型自己（可能跑偏的）轨迹接着走，一步偏、步步偏"),
        ("h", "它怎么做：同一个模型，扮两个角色"),
        ("", "学生", "：只看题目，自己生成轨迹（on-policy）"),
        ("", "教师", "：带着标准答案，但评估的是学生走出来的那条轨迹（prefill 同一后缀，不重新生成）"),
        ("", "对齐", "：逐 token 用 JSD 把教师分布“教”给学生；教师冻结（fixed_teacher + LoRA）"),
        ("h", "我的改造点（= PA-OPSD）"),
        ("", "特权信息升级", "：只有标答 y*  →  知识状态 k + 误概念 m + 标答 y* + 量规 rho"),
        ("", "蒸馏对象升级", "：只蒸文本 token  →  先蒸“选什么教学动作”(L_action)，再蒸“该动作下怎么说”(L_token)"),
    ]
    body_bullets(s, items, size=13.5, gap=7)

    # ---- S6 其他论文 ----
    s = blank(prs)
    header(s, "我的理解 3：其余论文，拿什么、不拿什么", 7, kicker="RELATED WORK")
    rows = [
        ["论文", "它做什么", "我拿", "我不拿"],
        ["AgentArk", "多 Agent 讨论压缩进单模型，省 token", "“MAS→Single”的叙事与划界；B5 质量上限的定位", "辩论数据 + GRPO 管线（我走 OPSD 系）"],
        ["ScaffoldLM", "由学生状态选择教学动作", "动作分类体系 A_ped：追问/提示/纠错/举例/评估/停止", "它的训练流程"],
        ["LUPI", "训练见特权 z、部署不见 z", "理论依据：“训练用 z 不算作弊”", "经典 SVM+ 细节"],
        ["Hinton KD", "硬标签知识蒸馏", "L_action 用交叉熵的出处", "—"],
    ]
    add_table(s, rows, 1.55, [1.5, 2.6, 3.4, 2.6], row_h=0.78, size=11.5)
    note(s, [("读法结论：精读 OPSD / LongTutor / ScaffoldLM / Eedi 官方材料；LUPI 与 KD 只记结论；自进化与预算路由类文献（EvolveR、Mem2Evolve、BudgetMem）开题阶段不读。", {})], 5.75, size=12)

    # ---- S7 总体思路（大白话流程） ----
    s = blank(prs)
    header(s, "总体思路：名师带实习老师，毕业独自上岗", 8, kicker="OVERALL PIPELINE")
    tf = txbox(s, 0.55, 1.28, 12.25, 0.35)
    para(tf, [("【训练期 · 名师带教】", {"bold": True, "color": ACCENT, "size": 14}),
              ("    名师什么都能看，实习老师只看题目和对话，逐句带教", {"color": GRAY, "size": 12})], first=True, space_after=0)
    b, aw = 2.10, 0.34
    xs = [0.55, 3.03, 5.51, 7.99, 10.47]
    axs = [2.69, 5.17, 7.65, 10.13]
    y1, h1 = 1.70, 1.34
    row1 = [
        (["素材：真人辅导记录", "Eedi 师生对话 6.5 万条", "（在这里学怎么教）"], NAVY),
        (["实习老师先试讲", "学生模型只看题目+对话", "自己生成每轮话术"], NAVY),
        (["名师团队逐句点评", "教师团队看得见档案 z：", "知识状态 / 错因 / 标答 / 评分标准"], NAVY),
        (["两级批改，学进脑子", "1 这轮该做什么动作", "2 这个动作下话怎么说", "（L_action + L_token）"], ACCENT),
        (["毕业上岗", "= 轻量导师 checkpoint", "部署只看题目和对话"], STEEL),
    ]
    for x, (lines, fill) in zip(xs, row1):
        flow_box(s, x, y1, b, h1, lines, fill=fill, size=9.5)
    for ax in axs:
        arrow(s, ax, y1 + h1 / 2 - 0.14, aw, 0.28)
    arrow(s, 11.30, y1 + h1 + 0.06, 0.34, 0.34, shape=MSO_SHAPE.DOWN_ARROW)
    tf = txbox(s, 0.55, 3.48, 12.25, 0.35)
    para(tf, [("【评测期 · 同考一张卷】", {"bold": True, "color": ACCENT, "size": 14}),
              ("    所有方法同一底座、同一流程，在 LongTutor 考场上比", {"color": GRAY, "size": 12})], first=True, space_after=0)
    b2 = 2.75
    xs2 = [0.55, 3.72, 6.89, 10.06]
    axs2 = [3.34, 6.51, 9.68]
    y2, h2 = 3.86, 1.34
    row2 = [
        (["考卷：LongTutor", "Gold 1000 条", "长期辅导场景"], NAVY),
        (["三场考试", "Evidence 会不会找证据", "Diagnosis 会不会诊断", "Teaching 教得好不好"], NAVY),
        (["同考一张卷", "B1-B5 + 我的方法", "同一底座、同一流程"], NAVY),
        (["比两件事", "教得好不好（分数）", "省不省 token（LGPT）"], ACCENT),
    ]
    for x, (lines, fill) in zip(xs2, row2):
        flow_box(s, x, y2, b2, h2, lines, fill=fill, size=9.5)
    for ax in axs2:
        arrow(s, ax, y2 + h2 / 2 - 0.14, aw, 0.28)
    rect(s, 0.55, 5.55, 12.25, 1.05, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_color=LINEC)
    tf = txbox(s, 0.85, 5.74, 11.7, 0.8)
    para(tf, [("一句话：", {"bold": True, "color": ACCENT}),
              ("训练期名师什么都能看、逐句带教（贵）；评测期学生独自上场、只看题目和对话（省）。", {"bold": True, "color": NAVY})], size=13.5, first=True, space_after=4, line_spacing=1.12)
    para(tf, "开题后再接：从带教记录里总结“教学心得”更新名师（自进化）；按学生状态决定本轮叫几个名师帮忙（预算路由）。", size=11.5, color=GRAY, line_spacing=1.1)

    # ---- 新增 S9 核心创新点 ----
    s = blank(prs)
    header(s, "核心创新点：开题聚焦两个，全文共六个", 9, kicker="CONTRIBUTIONS")
    flow_box(s, 0.55, 1.5, 6.0, 1.95, ["创新 1  教育特权 OPSD", "特权信息从“标答 y*”扩展为", "{ 知识状态 k, 误概念 m, 标答 y*, 量规 rho, 学习结果 }", "—— 把 OPSD 从数学推理引入教育场景"], fill=ACCENT, size=11.5)
    flow_box(s, 6.8, 1.5, 6.0, 1.95, ["创新 2  教学动作级蒸馏", "Long Reasoning → 教学动作 → 自然语言", "替代 “Long Reasoning → Long Response”", "—— 先教选动作，再教怎么说，省 token"], fill=ACCENT, size=11.5)
    tf = txbox(s, 0.55, 3.7, 12.25, 0.4)
    para(tf, [("全文创新（开题后逐步实现）", {"bold": True, "color": GRAY, "size": 13})], first=True, space_after=0)
    labels = [
        ["创新 3  状态感知动态路由", "学生状态 → 预算 → Agent 数量"],
        ["创新 4  自进化教学技能库", "轨迹 → 原则 → 特化 / 淘汰"],
        ["创新 5  MAS→单 Tutor 压缩", "训练用团队，部署用单模型"],
        ["创新 6  LGPT 指标", "学习增益 / 每千 token"],
    ]
    xs = [0.55, 3.68, 6.81, 9.94]
    for x, lb in zip(xs, labels):
        flow_box(s, x, 4.12, 2.85, 1.15, lb, fill=NAVY, size=10.5)
    rect(s, 0.55, 5.55, 12.25, 1.0, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_color=LINEC)
    tf = txbox(s, 0.85, 5.7, 11.7, 0.75)
    para(tf, [("定位一句话：", {"bold": True, "color": ACCENT}),
              ("研究重点不是“如何让更多 Agent 协作”，而是“如何把 Agent 协作产生的能力压缩进一个便宜、不泄密的部署模型”。", {"bold": True, "color": NAVY})], size=13, first=True, line_spacing=1.15)

    # ---- S8 现在做什么 ----
    s = blank(prs)
    header(s, "串起来：现在训练什么？什么时候用 LongTutor 评测？", 10, kicker="EXECUTION ORDER")
    items = [
        ("", "1) 先评测、后训练：", "第一周先在 LongTutor Gold 上跑 B1 零样本（Qwen2.5-7B + 固定 tutor prompt，不训练），拿三级 baseline 数字——这是所有后续方法的“锚点”，也是开题报告第一张表"),
        ("", "2) 备训练料：", "Eedi 对话下载归档；50 条真人话语→六类动作映射表 v1；单模型三 prompt 产 Team 离线轨迹 >= 5k 轮（x, z, a_T, 话术）"),
        ("", "3) 训 B2 / B4：", "B2 = SFT（只用 x + 话术做交叉熵）；B4 = 原版 OPSD（基于 D:\\OPSD，冒烟 50-100 step）"),
        ("", "4) 训 PA-OPSD：", "加动作头 + 特权 z prompt + 条件 L_token——我的主方法"),
        ("", "5) 统一评测：", "同一套 LongTutor Gold 流程给所有 checkpoint 打分；Eedi 动作准确率作辅表；B5（在线 Team）只评质量上限"),
    ]
    body_bullets(s, items, top=1.5, size=13.5, gap=10)
    rect(s, 0.55, 6.0, 12.25, 0.85, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_color=LINEC)
    tf = txbox(s, 0.85, 6.14, 11.7, 0.6)
    para(tf, [("一句话：", {"bold": True, "color": ACCENT}),
              ("LongTutor 从第一天就要用（B1 数字）；大规模对比评测等 checkpoint 齐了再统一跑。", {"bold": True, "color": NAVY})], size=13.5, first=True, line_spacing=1.1)

    # ---- S9 基线表（大白话人设版） ----
    s = blank(prs)
    header(s, "和谁比：5 个对照组 + 我的方法", 11, kicker="BASELINES")
    rows = [
        ["编号", "人设（通俗理解）", "做法（技术）", "它回答什么问题"],
        ["B1", "裸考的新手", "不训练：Qwen2.5-7B + 固定提示词直接上岗", "不训练行不行？（下限）"],
        ["B2", "背教案的新手", "SFT：死记硬背 Eedi 真人老师的话术", "只会学舌够不够？"],
        ["B3", "看名师录像自学", "离线蒸馏：模仿教师团队输出，但学的不是自己走出来的路", "学别人轨迹 vs 学自己轨迹"],
        ["B4", "名师逐句点评（只看标答）", "原版 OPSD：特权只有标准答案，逐词对齐", "原版方法搬到教育场景行不行？"],
        ["B5", "名师全程陪堂", "不蒸馏：评测时每一步都现场调用教师团队", "不计成本，质量上限有多高？"],
        ["PA-OPSD", "名师带完整档案带教（我的）", "特权=知识状态+错因+标答+评分标准；先教动作、再教措辞", "我的两个加法值不值？（主方法）"],
    ]
    add_table(s, rows, 1.5, [1.3, 2.4, 4.6, 3.0], row_h=0.68, size=11.5)
    note(s, [("全部方法从同一底座 Qwen2.5-7B + LoRA 出发，唯一变量是训练方法。", {"bold": True, "color": NAVY}),
             ("   期待结果：超过 B2 / B4、逼近 B5，同时 token 成本远低于 B5。", {})], 6.1, size=12)

    # ---- 新增 S12 还差什么 ----
    s = blank(prs)
    header(s, "还差什么：已完成 vs 缺口", 12, kicker="GAP ANALYSIS")
    flow_box(s, 0.55, 1.45, 5.6, 0.55, ["已完成（到 2026.10）"], fill=NAVY, size=13)
    tf = txbox(s, 0.8, 2.15, 5.3, 4.3)
    for t in ["论文精读与方法理解（OPSD / LongTutor / AgentArk / ScaffoldLM）",
              "数据集确定与排除（3 定 4 排）",
              "B1-B5 对比方案与 LGPT 主指标设计",
              "开题边界收敛：只做 PA-OPSD 主线",
              "《执行总纲》单一真相表成文"]:
        para(tf, [("[done]  ", {"bold": True, "color": ACCENT}), (t, {})], size=12.5, first=(t.startswith("论文")), space_after=9, line_spacing=1.1)
    flow_box(s, 6.85, 1.45, 5.95, 0.55, ["还缺（按依赖顺序）"], fill=ACCENT, size=13)
    tf = txbox(s, 7.1, 2.15, 5.6, 4.3)
    for i, t in enumerate(["代码：fork D:\\OPSD 建 pa-opsd 分支；动作头 + z prompt 改造",
              "数据：Eedi 下载归档；Team 离线轨迹 v0（>= 5k 轮）",
              "标签：50 条真人话语 → 六类动作映射表 v1",
              "数字：LongTutor B1 零样本三级 baseline",
              "实验：B2-B5 全跑 + 消融 + LGPT 主表",
              "写作：开题报告定稿 → 论文草稿"]):
        para(tf, [("(%d)  " % (i + 1), {"bold": True, "color": NAVY}), (t, {})], size=12.5, first=(i == 0), space_after=8, line_spacing=1.1)

    # ---- S10 困惑 ----
    s = blank(prs)
    header(s, "当前困惑：需要讨论确认的四个问题", 13, kicker="OPEN QUESTIONS")
    items = [
        ("", "1) 动作映射口径：", "Eedi 真人话语 → 六类教学动作，边界 case 怎么判？计划先做 50 条映射表 v1，人工抽验 10%"),
        ("", "2) 特权 z 的质量：", "k / m 由 Teacher Team 离线诊断生成，准确性如何验证？部署侧的信息泄漏率如何测量？"),
        ("", "3) 跨数据集泛化：", "Eedi 训练、LongTutor 评测，分布差异会不会压低指标？是否需要混入少量 LongTutor 训练子集？"),
        ("", "4) 评测成本：", "LongTutor 三级全量评测（Judge + ROUGE）的算力 / API 预算如何安排？"),
    ]
    body_bullets(s, items, top=1.7, size=14, gap=16)

    # ---- S11 下一步计划 ----
    s = blank(prs)
    header(s, "下一步计划：2026.11-12（开题与数据 v1）", 14, kicker="NEXT STEPS")
    items = [
        ("h", "工程"),
        ("", "Fork D:\\OPSD 建 pa-opsd 分支", "；克隆 liano3/LongTutor，跑通 run_longtutor_gold.sh"),
        ("h", "数据"),
        ("", "Eedi 对话 + 误概念图谱下载", "，许可与伦理截图归档；动作映射表 v1（50 条，抽验 10%）"),
        ("", "Team 离线轨迹 v0", "：单模型三 prompt，>= 5k 轮 JSONL，格式 (x, z, a_T, 话术)"),
        ("h", "实验"),
        ("", "LongTutor B1 零样本三级 baseline 数字", "——第一张真实实验表"),
        ("h", "文献"),
        ("", "精读 OPSD 全文 + LongTutor 评测节", "；开题报告定稿"),
        ("h", "阶段交付物"),
        ("", "trajectories_v1.jsonl；B1 数字；开题 PPT / 报告", ""),
    ]
    body_bullets(s, items, size=13, gap=5)
    note(s, [("其后（2027.01-02 核心期）：B4 冒烟 → PA-OPSD v1 → B2/B3/B4/B5/PA 全跑 + 消融（-k, -m, -rho, -y*；去 L_action；单 Teacher vs Team）→ LGPT 主表。", {})], 6.15, size=11.5)


    # ---- 新增 S15 风险与应对 ----
    s = blank(prs)
    header(s, "风险与应对（来自 1016 研究计划 §十七）", 15, kicker="RISKS")
    rows = [
        ["风险", "应对", "阶段"],
        ["公开数据缺少真实学习增益", "不把 LLM-Judge 分等同于学习效果；结合 knowledge tracing + 下一题表现 + 学生模拟器 + 小规模人工评价", "开题"],
        ["Eedi 涉及未成年学习者", "严格遵守非商业研究条款；不做个人识别、学生画像或高风险自动决策；许可截图归档", "开题"],
        ["自进化可能积累错误原则", "utility 打分 + 语义去重 + held-out 验证 + 回滚机制", "全文"],
        ["多 Agent 通信抬高 token 成本", "预算路由从系统设计之初就是核心模块，而非事后优化", "全文"],
        ["LLM Judge 本身有偏差", "自动评价 + 人工子集复核的双重评价方案", "开题"],
    ]
    add_table(s, rows, 1.5, [2.6, 6.6, 1.1], row_h=0.86, size=11.5)

    # ---- S12 结尾 ----
    s = blank(prs)
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    rect(s, 0, 7.28, 13.333, 0.22, ACCENT)
    tf = txbox(s, 1.0, 1.9, 11.3, 2.6)
    para(tf, "主线一句话", size=15, bold=True, color=ACCENT, first=True, space_after=10)
    para(tf, "训练期：用特权信息 + 教师团队，在学生自己的轨迹上做“动作 + 话术”两级蒸馏；", size=19, color=WHITE, space_after=6, line_spacing=1.2)
    para(tf, "部署期：只留一个只看题目和对话的轻量导师。", size=19, color=WHITE, line_spacing=1.2)
    tf = txbox(s, 1.0, 4.7, 11.3, 1.2)
    para(tf, "开题后接：自进化（轨迹→原则→更新 Team→再蒸馏） + 预算路由（按学生状态决定调几个 Agent）",
         size=13, color=SKY, first=True, line_spacing=1.3)
    tf = txbox(s, 1.0, 6.1, 11.3, 0.6)
    para(tf, "请各位老师批评指正", size=16, bold=True, color=ACCENT, first=True)

    prs.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
