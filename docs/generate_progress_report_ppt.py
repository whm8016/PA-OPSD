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


def main():
    prs = Presentation()
    prs.slide_width = PAGE_W
    prs.slide_height = PAGE_H

    # S1 cover
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
    para(tf, "汇报人：×××      指导教师：×××      2026 年 10 月 26 日", size=14, color=WHITE, first=True)

    # S2 agenda
    s = blank(prs)
    header(s, "汇报框架", 2)
    items = [
        ("1", "研究问题与定位", "为什么做：三个未被填补的缺口"),
        ("2", "总体思路与方法", "怎么做：训练贵、部署省；PA-OPSD 两级蒸馏"),
        ("3", "数据集与文献", "用什么、借什么：三数据集 + 各论文的角色"),
        ("4", "实验方案", "怎么验证：B1–B5 基线、三组消融、LGPT 主指标"),
        ("5", "当前进展与待办", "做到哪、下一步：文档/文献/工程就绪 → 阶段一清单"),
        ("6", "风险与对策", "预案：代理指标、数据合规、负结果、算力"),
    ]
    y = 1.55
    for num, t, d in items:
        tf = txbox(s, 0.9, y, 11.8, 0.75)
        para(tf, [(num + "  ", {"color": ACCENT, "bold": True, "size": 20}),
                  (t, {"bold": True, "color": NAVY, "size": 17}),
                  ("    " + d, {"color": GRAY, "size": 12.5})],
             first=True, space_after=0)
        y += 0.88

    # S3 research question
    s = blank(prs)
    header(s, "研究问题：把「看过小抄的名师团队」压进轻量导师", 3, kicker="01 研究问题与定位")
    body_bullets(s, [
        ("b", "训练阶段｜看过小抄的名师团队：", "Diagnoser / Pedagogue / Verifier 可见特权 z（知识状态 k、误概念 m、标答 y*、量规 ρ），先决定「怎么教」"),
        ("b", "蒸馏阶段｜压的是动作而不只是文字：", "在学生自己的 on-policy 轨迹上，把「何时追问、提示、纠错、举例、停止」先压成教学动作层，再压进轻量模型"),
        ("b", "部署阶段｜轻量导师只看 x：", "x =（当前题 q，长期历史 h，当前对话 r）；无 z、无团队、无额外模型调用"),
        ("b", "核心问题：", "只看 x 的 Student，能否接近看过 z 的教师团队的教学质量，同时显著降低 token 成本？"),
        ("b", "统一尺子 LGPT：", "每千 token 学习增益 = ΔG / (N_token / 1000)，质量与成本在同一坐标下比较"),
    ], top=1.45, size=15, gap=14)

    # S4 gap table
    s = blank(prs)
    header(s, "研究缺口：四条线索都没解决这个问题", 4, kicker="01 研究问题与定位")
    add_table(s, [
        ["研究线索", "代表工作", "已解决", "未解决（本课题切入点）"],
        ["LLM 智能导师", "LongTutor、ScaffoldLM", "长期辅导的任务定义与评价", "教学能力成本高：依赖大模型或多智能体"],
        ["特权信息蒸馏", "OPSD（ICML 2026）", "训练/部署信息不对称的蒸馏范式", "特权未含学生建模；监督对象是整段文本"],
        ["多智能体压缩", "AgentArk（NeurIPS 2026）", "多智能体到单模型的通用压缩", "未利用教育特权信息；未区分教学动作层"],
        ["教学动作建模", "Eedi 对话与误概念图谱", "真实辅导动作数据", "动作仅作评价维度，未作蒸馏中间表示"],
    ], top=1.5, col_w=[1.5, 2.1, 3.2, 5.45], row_h=0.72)
    note(s, [("定位：", {"bold": True, "color": NAVY}),
             ("不是再造一个「更多智能体」的教育系统，而是蒸馏范式的教育化改造——蒸馏对象从整段推理文本换成教学动作，特权信息从标准答案扩展为 k、m、y*、ρ。", {})],
         top=5.6, size=12.5)

    # S5 framework diagram
    s = blank(prs)
    header(s, "总体思路：训练贵、部署省的非对称结构", 5, kicker="02 总体思路与方法")
    tf = txbox(s, 0.55, 1.32, 4.0, 0.35)
    para(tf, "训练期（离线 · 一次性）", size=13, bold=True, color=ACCENT, first=True, space_after=0)
    flow_box(s, 0.55, 1.72, 2.72, 1.05, ["公开数据 + 特权 z", "LongTutor / Eedi", "k · m · y* · ρ"])
    arrow(s, 3.31, 2.10)
    flow_box(s, 3.71, 1.72, 2.72, 1.05, ["多智能体教师团队", "Diagnoser / Pedagogue", "/ Verifier（三 prompt）"], size=10.5)
    arrow(s, 6.47, 2.10)
    flow_box(s, 6.87, 1.72, 2.72, 1.05, ["动作标注轨迹", "(x, z, a_T, 话语)", "≥ 5,000 轮 JSONL"])
    arrow(s, 9.63, 2.10)
    flow_box(s, 10.03, 1.72, 2.75, 1.05, ["PA-OPSD 蒸馏", "L_action + L_token", "on-policy 学生轨迹"], fill=ACCENT)
    arrow(s, 11.35, 2.83, w=0.28, h=0.5, shape=MSO_SHAPE.DOWN_ARROW)
    tf = txbox(s, 0.55, 3.22, 4.0, 0.35)
    para(tf, "部署期（在线）", size=13, bold=True, color=ACCENT, first=True, space_after=0)
    flow_box(s, 0.55, 3.62, 3.5, 1.0, ["轻量 Student 导师", "仅输入 x = (q, h, r)", "7B + LoRA，单模型"], fill=WHITE, tcolor=INK, line_color=LINEC)
    arrow(s, 4.15, 3.97)
    flow_box(s, 4.61, 3.62, 3.5, 1.0, ["教学动作 + 辅导话术", "无 z · 无团队", "无额外模型调用"], fill=WHITE, tcolor=INK, line_color=LINEC)
    flow_box(s, 8.35, 3.62, 4.43, 1.0, ["统一尺子 LGPT", "每千 token 学习增益", "ΔG / (N_token / 1000)"], fill=LIGHT, tcolor=NAVY, line_color=LINEC)
    sh = rect(s, 0.55, 4.92, 12.23, 0.62, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf, [("后期（学位论文 2027H2）：", {"bold": True, "color": NAVY}),
              ("自进化教学技能库（EvolveR 式）+ 学生状态预算路由（BudgetMem 式）→ 教师进化后触发再蒸馏", {})],
         size=12, first=True, align=PP_ALIGN.CENTER, space_after=0)
    tf = txbox(s, 0.55, 5.75, 12.0, 0.35)
    para(tf, "三个关键科学问题（RQ）", size=13, bold=True, color=NAVY, first=True, space_after=0)
    rq = [
        "RQ1：教育特权 z（k、m、y*、ρ）能否被无特权 Student 有效吸收？",
        "RQ2：动作级蒸馏能否比整段文本蒸馏更省 token 且保质量？",
        "RQ3：教师团队分工能力能否压进单模型而不显著损失动作恰当性？",
    ]
    x = 0.55
    for t in rq:
        flow_box(s, x, 6.12, 3.95, 0.85, [t], fill=WHITE, tcolor=INK, size=10.5, line_color=LINEC)
        x += 4.14

    # S6 method
    s = blank(prs)
    header(s, "方法核心：PA-OPSD 两级蒸馏", 6, kicker="02 总体思路与方法")
    body_bullets(s, [
        ("b", "教学动作集 A_ped = {追问, 提示, 纠错, 举例, 部分解答, 停止}：", "六类，由 Eedi 真人导师话语映射而来"),
        ("b", "第一级 L_action = CE(a_T, a_S(·|x))：", "把教师动作当分类标签，学生先学「这一步该做什么」"),
        ("b", "第二级 L_token = Σ KL[ π_T(·|x, z, a_T) ‖ π_S(·|x, a_S) ]：", "以教师动作条件下的话语做 token 级蒸馏，再学「按动作怎么说」"),
        ("b", "总损失 L = λ_a·L_action + λ_t·L_token：", "按事后效用加权样本，只更新 Student"),
        ("b", "一步训练：", "Student(x) 自采轨迹 → Teacher(x+z) 在同一轨迹 prefill → 逐位置 JSD + 动作 CE"),
        ("b", "核心假设：", "教学决策比整段推理更可复用；动作标签远短于文本，监督与推理成本显著下降"),
    ], top=1.45, size=14, gap=12)

    # S7 boundary
    s = blank(prs)
    header(s, "关键设计边界：谁能看什么", 7, kicker="02 总体思路与方法")
    body_bullets(s, [
        ("b", "可见状态 x = (q, h, r)：", "当前题目、长期历史摘要、当前对话——Student 的全部输入"),
        ("b", "特权信息 z = (k, m, y*, ρ)：", "知识状态、误概念、标准解、评分量规——仅训练期 Teacher 可见"),
        ("b", "关键边界：", "后续学习结果 o_{t+1} 不进 Teacher 输入，仅做事后效用加权与样本筛选，避免信息泄漏"),
        ("b", "Student 不输出 k/m 诊断表：", "学的是 π_S(a, 话术 | x) ≈ π_T(· | x, z)；用 ΔG、动作一致率、LongTutor 三级验证，而非 k 复现率"),
        ("b", "教师团队降本：", "同一模型三种 prompt 分饰三角，监督一次性离线构建"),
        ("b", "工程基座：", "OPSD 官方代码已在本地（D:\\OPSD），B4 基线可直接复用"),
    ], top=1.45, size=14, gap=12)

    # S8 datasets
    s = blank(prs)
    header(s, "已确定数据集：开题三件套", 8, kicker="03 数据集与文献")
    add_table(s, [
        ["数据集", "规模", "在本研究中的作用"],
        ["LongTutor Gold\n（主 Benchmark）", "1,000 条人工标注\n（XES3G5M 日志 + 对齐 history_features）", "评部署 Tutor 仅看 x 时的长期辅导三级能力；B1–B5 主表之一"],
        ["Eedi 问题锚定辅导对话", "1,971 干预 / 68,717 消息\n1,073 学生 / 25 名真人导师", "动作监督源：映射六类动作、训练 L_action；提供真实导师动作分布"],
        ["Eedi 误概念图谱", "8,117 误概念 / 2,258 知识构件\n19,294 条边", "教师侧特权 m 与知识状态先验；Diagnoser 输入"],
    ], top=1.45, col_w=[2.6, 3.6, 6.05], row_h=0.85, size=11.5)
    note(s, [("口径说明：", {"bold": True, "color": NAVY}),
             ("LongTutor 已含 XES3G5M 专家标注，无需另下整库当第二个主 benchmark。", {})], top=4.55, size=11.5)
    note(s, [("后期可选（不纳入开题完成标准）：", {"bold": True, "color": NAVY}),
             ("XES3G5M 全量（KT 状态模型）、EdNet（学习增益代理与 Pareto）、EduAgentBench（路由泛化）、MMTutorBench（多模态）、ASTRA（编程域 MAS）。", {})], top=5.05, size=11.5)

    # S9 LongTutor detail
    s = blank(prs)
    header(s, "主考卷 LongTutor：三级递进评测", 9, kicker="03 数据集与文献")
    add_table(s, [
        ["层级", "模型要产出", "与 Gold 比什么", "指标"],
        ["① Evidence 证据获取", "3 道固定题：信息抽取 / 跨会话推理 / 幻觉检查", "精确匹配 + LLM 语义等价", "Memory SA（含分类型 Acc）"],
        ["② Diagnosis 状态诊断", "四类标签：Recall Failure / Conceptual Gap / Procedural Error / Transfer Deficit", "标签一致性", "Acc + Macro-F1"],
        ["③ Teaching 教学动作", "个性化辅导话术（利用历史 + 符合诊断策略）", "LLM Judge 四维 1–5 分 + ROUGE-L", "历史利用 / 策略对齐 / 连贯 / 可懂"],
    ], top=1.45, col_w=[2.1, 4.1, 3.0, 3.05], row_h=0.8, size=11)
    body_bullets(s, [
        ("b", "是考卷不是方法：", "LongTutor 测的是蒸馏后的 Student 只剩 x 时还保留多少教学能力"),
        ("b", "三级递进设计：", "证据找不准 → 诊断难对 → 教学策略分随之下降"),
        ("b", "Gold 结构：", "每条样本 memory 固定 3 题（Q1 信息抽取 / Q2 跨会话推理 / Q3 幻觉检查），1,000 条 → 3,000 小题级计数"),
    ], top=4.6, size=12.5, gap=8)

    # S10 papers map
    s = blank(prs)
    header(s, "文献地图：每篇论文拿什么、不拿什么", 10, kicker="03 数据集与文献")
    add_table(s, [
        ["论文", "在本研究中的作用"],
        ["LongTutor（ACL 2026）", "主考卷：长期辅导三级任务与指标；拿问题定义与尺子，不拿方法"],
        ["OPSD（ICML 2026）", "蒸馏主干：同模型分饰师生、特权 + on-policy token 级监督；特权从 y* 扩展为 k、m、ρ、y*"],
        ["ScaffoldLM（ACL 2026）", "教学规划 + 评估驱动记忆：支撑动作中间层设计；教育向 baseline"],
        ["AgentArk（NeurIPS 2026）", "多智能体→单模型压缩对标；本课题压缩的是教学策略（含动作监督）而非通用辩论轨迹"],
        ["Tutor CoPilot（2024）", "动机证据：专家策略有效但难规模化（RCT：900 导师 / 1,800 学生，掌握度 +4pp，约 $20/人/年）"],
        ["EvolveR / Mem²Evolve（2026）", "后期自进化：轨迹→原则→技能库→角色专化（学位论文阶段）"],
        ["BudgetMem（ICML 2026）", "后期路由：分档预算 + Pareto；改造为学生状态驱动的 L/M/H 升档"],
        ["SciEducator（CVPR 2026）", "划界：不竞争「多智能体+教育」系统贡献，聚焦蒸馏方法贡献"],
    ], top=1.42, col_w=[2.7, 9.55], row_h=0.54, header_h=0.36, size=10.5, header_size=11)
    note(s, [("原则：", {"bold": True, "color": NAVY}),
             ("不是复现五篇论文，而是在一个教育闭环里把各类机制改成 pedagogy-aware 版本。", {})], top=6.35, size=11.5)

    # S11 baselines
    s = blank(prs)
    header(s, "实验设计：B1–B5 基线与消融", 11, kicker="04 实验方案")
    add_table(s, [
        ["编号", "基线", "实现要点"],
        ["B1", "单模型直接提示", "Qwen2.5-7B + 固定 tutor prompt，无训练"],
        ["B2", "普通监督微调 SFT", "仅 x + 辅导文本 CE（Eedi 对话为主），无特权、无动作层"],
        ["B3", "文本级蒸馏", "教师团队整段话语作 target，无 on-policy、无动作"],
        ["B4", "原版 OPSD", "特权仅 y*，仅 token JSD（复用官方代码）"],
        ["B5", "完整教师团队部署", "评测时每轮调用 Team + z（token × 调用次数），质量上限与成本对照"],
        ["PA-OPSD", "本文方法", "教育特权 z + 动作级两级蒸馏"],
    ], top=1.42, col_w=[1.0, 2.2, 9.05], row_h=0.5, header_h=0.4, size=11)
    body_bullets(s, [
        ("b", "归因逻辑：", "vs B4 恰好隔离两个变量（教育 z、动作层）；vs B3 隔离蒸馏对象；vs B5 回答压缩损失"),
        ("b", "三组消融：", "特权构成（−k / −m / −ρ / −y*）；蒸馏粒度（token / 动作 / 两级）；教师结构（单教师 vs 团队）"),
    ], top=5.35, size=12.5, gap=8)

    # S12 LGPT
    s = blank(prs)
    header(s, "主指标 LGPT：每千 token 学习增益", 12, kicker="04 实验方案")
    body_bullets(s, [
        ("b", "定义：", "LGPT = ΔG / (N_token / 1000)，用一把尺子统一质量与成本"),
        ("b", "ΔG（代理学习增益）：", "Eedi——干预后下一答是否正确 / 误概念是否消除；LongTutor——官方子任务 success 或同知识点后续题（论文中写死一种）"),
        ("b", "N_token：", "部署 Student 单轮会话总 token；B5 另计教师团队调用"),
        ("b", "辅助指标：", "动作一致率、rubric Judge 分、人工抽检分、答案泄漏率；平均每轮 / 每会话 token"),
        ("b", "意义：", "让教育智能体研究与高效推理研究在同一坐标下比较"),
    ], top=1.45, size=14.5, gap=13)

    # S13 progress
    s = blank(prs)
    header(s, "当前进展：方案与工程准备已就绪", 13, kicker="05 当前进展与待办")
    body_bullets(s, [
        ("b", "选题与文档：", "开题报告初稿 + 全文愿景文档（1016）+ 执行总纲定稿，划清「开题 4 个月主体实验」与「学位论文扩展」边界"),
        ("b", "文献地图：", "9 篇核心论文逐篇明确「拿什么 / 不拿什么」，绪论五段素材完成"),
        ("b", "数据集口径：", "开题三件套确定（LongTutor-Gold + Eedi 对话 + 误概念图谱），后期 5 个扩展划清"),
        ("b", "方法细节定稿：", "六类动作集、两级损失、z 边界（o_{t+1} 仅事后加权）、LGPT 定义"),
        ("b", "工程准备：", "OPSD 官方代码本地就绪（B4 基座）；LongTutor 复现路径摸清（history_features 构建 + 三级评测流水线 + 一键脚本）"),
        ("b", "理论梳理：", "SFT / RL(GRPO) / OPD / OPSD 监督信号对比；AgentArk PAD（PRM + GRPO）精读"),
        ("b", "踩坑记录：", "XES 须按知识点切分生成 history_features（_key 才对齐 Gold）；评测 workers≥2、metrics 需 --limit 1000、全量前清空 preds/metrics"),
    ], top=1.42, size=12.5, gap=9)

    # S14 todo phase1
    s = blank(prs)
    header(s, "待办：本周 + 阶段一（2026.11–12）", 14, kicker="05 当前进展与待办")
    body_bullets(s, [
        ("h", "本周（开题前）"),
        ("b", "开题报告与汇报 PPT 定稿；", "明确边界：自进化与预算路由不纳入开题主体实验"),
        ("h", "阶段一 · 2026.11–12（数据与监督 v1）"),
        ("b", "Fork 整理 pa-opsd 分支", "（基于 OPSD 官方代码，复用 opsd_trainer / data_collator）"),
        ("b", "跑通 LongTutor Gold 一键评测：", "记录 B1 零样本三级 baseline（Memory SA / Diagnosis Macro-F1 / Teaching 四维 + ROUGE-L）"),
        ("b", "Eedi 数据落地：", "下载辅导对话 + 误概念图谱，伦理与许可截图归档"),
        ("b", "动作映射表 v1：", "50 条 Eedi 导师话语 → 六类动作，人工抽验 10%"),
        ("b", "教师团队离线 v0：", "单模型三 prompt 产出 (x, z, a_T, 话语) JSONL ≥ 5,000 轮"),
        ("b", "文献精读：", "OPSD 全文 + LongTutor 评测节"),
        ("h", "阶段交付物"),
        ("b", "", "trajectories_v1.jsonl；LongTutor B1 baseline 数字；开题报告与 PPT"),
    ], top=1.38, size=12.5, gap=7)

    # S15 todo phase2 + timeline
    s = blank(prs)
    header(s, "待办：阶段二（2027.01–02，核心实验）与全程时间线", 15, kicker="05 当前进展与待办")
    body_bullets(s, [
        ("b", "B4 冒烟：", "opsd_train.py 混合数据 50–100 step；PA-OPSD v1 = L_action 头 + z prompt + 条件 L_token"),
        ("b", "B2–B5 + PA 全跑：", "Eedi 子集 + LongTutor 评测各至少 1 次；三组消融（特权构成 / 蒸馏粒度 / 教师结构）"),
        ("b", "主表与投稿：", "LGPT + LongTutor 三级 + Eedi 动作准确率；目标 ACL 2027 滚动审稿"),
    ], top=1.38, size=12.5, gap=7, height=2.0)
    tf = txbox(s, 0.55, 3.55, 8.0, 0.35)
    para(tf, [("▍", {"color": ACCENT, "bold": True}), ("全程时间线", {"bold": True, "color": NAVY})], size=13.5, first=True, space_after=0)
    add_table(s, [
        ["时间", "工作内容", "阶段目标"],
        ["2026.11–12", "开题 + LongTutor/Eedi 数据预处理 + 教师团队监督管线", "动作标注轨迹数据集 v1"],
        ["2027.01–02", "PA-OPSD 两级蒸馏训练 + B1–B5 基线 + 主体消融", "主表结果；投 ACL 2027 滚动审稿"],
        ["2027.03–05", "补充实验 + 人工抽检 + 论文修改再投", "论文定稿或进入下一轮审稿"],
        ["2027.06–08", "后续方向探索（技能库、动态预算）", "学位论文展望章节初步实验"],
        ["2027.09–10", "学位论文撰写、盲审与答辩准备", "学位论文送审"],
    ], top=3.95, col_w=[1.6, 6.3, 4.35], row_h=0.5, header_h=0.38, size=11)

    # S16 risks
    s = blank(prs)
    header(s, "风险与对策", 16, kicker="06 风险与对策")
    add_table(s, [
        ["风险", "对策"],
        ["代理指标风险：公开数据上的学习增益只能用「后续题目对错」代理，与真实学习存在差距", "固定代理公式；LongTutor 留一评估；安排 50 条人工抽检校准结论"],
        ["数据合规风险：Eedi 限非商业研究使用，原始交互涉及未成年人", "仅用官方脱敏版本；不尝试还原个体信息；按学校要求完成伦理审查备案"],
        ["负结果风险：动作级蒸馏可能不优于文本级蒸馏", "对「动作粒度是否有独立价值」的否定性回答本身即为可发表的消融结论，重心转向分析"],
        ["算力风险：蒸馏训练可能超预算", "7B + LoRA；教师团队离线分批生成；先子集再全量，优先保主消融完整"],
    ], top=1.5, col_w=[5.3, 6.95], row_h=0.95, size=11.5)

    # S17 closing
    s = blank(prs)
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    rect(s, 0, 7.28, 13.333, 0.22, ACCENT)
    tf = txbox(s, 0.9, 2.0, 11.6, 0.9)
    para(tf, "开题只做 PA-OPSD", size=32, bold=True, color=WHITE, first=True)
    tf = txbox(s, 0.9, 3.05, 11.6, 1.4)
    para(tf, "在 LongTutor + Eedi 上，用教育特权 z + 六类教学动作扩展 OPSD，对比 B1–B5，主指标 LGPT",
         size=16, color=SKY, first=True, space_after=10)
    para(tf, "下一步：开题定稿 → 跑通 LongTutor B1 → Eedi 动作映射 → 教师团队离线轨迹 v1",
         size=13.5, color=STEEL)
    rect(s, 0.92, 5.05, 1.6, 0.06, ACCENT)
    tf = txbox(s, 0.9, 5.3, 11.6, 0.7)
    para(tf, "请各位老师同学批评指正", size=20, bold=True, color=WHITE, first=True)

    prs.core_properties.title = "PA-OPSD 科研进展汇报"
    prs.core_properties.author = ""
    prs.save(OUT)
    print("saved:", OUT, "slides:", len(prs.slides.__iter__.__self__._sldIdLst))


if __name__ == "__main__":
    main()
