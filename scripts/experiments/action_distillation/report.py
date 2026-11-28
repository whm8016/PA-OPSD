"""Produce the experiment report and insert verified findings in a separate copy of proposal V3."""

import argparse
import copy
import json
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree

NAMES = {
    "gold_sft": "原 Gold SFT",
    "v3_pa": "PA 联合",
    "v3_teacher_sft": "教师文本 SFT",
    "v3_content": "仅内容及格式",
    "v3_action_kl": "仅动作及格式",
    "offpolicy": "固定教师离策略",
    "v3_wrong": "错配特权",
    "v3_neutral": "等长无关文本",
    "v3_no_eedi": "移除 Eedi",
    "v3_solution_only": "仅标准解 OPSD",
    "base": "原始基座",
}


def get(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percent(x):
    return f"{100 * x:.1f}%"


def setup():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(1.9)
    sec.bottom_margin = Cm(1.8)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)
    for name, size, face, color in [
        ("Normal", 11, "宋体", "182532"),
        ("Title", 22, "微软雅黑", "000000"),
        ("Heading 1", 18, "微软雅黑", "000000"),
        ("Heading 2", 12, "微软雅黑", "175664"),
    ]:
        st = doc.styles[name]
        st.font.name = face
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor.from_string(color)
        st._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), face)
        st.paragraph_format.space_before = Pt(4)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing = 1.32
        p = st._element.get_or_add_pPr()
        for el in p.findall(qn("w:pBdr")):
            p.remove(el)
    foot = sec.footer.paragraphs[0]
    foot.alignment = 2
    rr = foot.add_run("OPSD 开题实验     /     2026.11.27     /     ")
    rr.font.size = Pt(8)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    foot._p.append(fld)
    return doc


def table(doc, heads, rows, widths):
    t = doc.add_table(rows=0, cols=len(heads))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for col, w in zip(t.columns, widths, strict=True):
        col.width = Cm(w)
    for i, row in enumerate([heads, *rows]):
        tr = t.add_row()
        tr._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for j, (cell, text, w) in enumerate(zip(tr.cells, row, widths, strict=True)):
            cell.width = Cm(w)
            sh = OxmlElement("w:shd")
            sh.set(qn("w:fill"), "D7E6E7" if not i else "F0F5F5" if i % 2 else "FFFFFF")
            cell._tc.get_or_add_tcPr().append(sh)
            p = cell.paragraphs[0]
            p.alignment = 0 if j == 0 else 1
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.space_before = Pt(3.5)
            p.paragraph_format.space_after = Pt(3.5)
            r = p.add_run(str(text))
            r.bold = not i
            r.font.name = "宋体"
            r.font.size = Pt(9.5)
            r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")


def integrated(source, dest, paragraphs):
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    w = lambda t: "{" + ns + "}" + t
    with ZipFile(source) as zin, ZipFile(dest, "w") as zout:
        original = zin.read("word/document.xml")
        root = etree.fromstring(original)
        nodes = [
            p
            for p in root.iter(w("p"))
            if "".join(t.text or "" for t in p.iter(w("t"))).strip() == "2. 实验条件"
        ]
        if len(nodes) != 1:
            raise ValueError("Cannot locate the proposal's research-foundation section")
        marker = nodes[0]
        body = marker.getprevious()
        for i, txt in enumerate(paragraphs):
            template = marker if i == 0 else body
            p = etree.Element(w("p"))
            if template.find(w("pPr")) is not None:
                p.append(copy.deepcopy(template.find(w("pPr"))))
            run = etree.SubElement(p, w("r"))
            attr = template.find(".//" + w("rPr"))
            if attr is not None:
                run.append(copy.deepcopy(attr))
            etree.SubElement(run, w("t")).text = txt
            marker.addprevious(p)
        for info in zin.infolist():
            data = (
                etree.tostring(
                    root, encoding="UTF-8", xml_declaration=True, standalone=True
                )
                if info.filename == "word/document.xml"
                else zin.read(info.filename)
            )
            zout.writestr(info, data)
    with ZipFile(source) as a, ZipFile(dest) as b:
        assert a.namelist() == b.namelist() and all(
            a.read(n) == b.read(n) for n in a.namelist() if n != "word/document.xml"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("docs", type=Path)
    args = ap.parse_args()
    d = args.bundle
    docs = args.docs
    s = get(d / "analysis.json")
    big = get(d / "scale/analysis.json")
    first = s["structured"]
    scale = big["structured"]
    original = s["official_prediction_audit"]
    biga = big["official_prediction_audit"]
    blind = get(d / "blind_confirmation/summary.json")
    c = s["cycles"]
    doc = setup()
    p = doc.add_paragraph

    def h(t):
        doc.add_heading(t, 2)

    def page(t):
        doc.add_page_break()
        doc.add_heading(t, 1)

    pa = "v3_pa"
    base = "gold_sft"
    ts = "v3_teacher_sft"
    tok = "v3_content"

    def ft(o):
        if o and o.get("teaching_mean") is not None:
            return f"{o['teaching_mean']:.3f}（{o['teaching_n']}）"
        return "—"

    def scored(db, k):
        o = db["official"].get(k, {})
        return o.get("n", o.get("incomplete", 0))

    anchor = original[base]["diagnosis"]
    small = original[pa]["diagnosis"]
    large = biga[pa]["diagnosis"]
    para1 = f"固定公开历史与官方生成设置后，原 Gold SFT、72 题 PA 和 373 题 PA 在相同 140 道开发题上，诊断分别答对 {anchor}、{small}、{large} 道。"
    para2 = f"固定结构化协议下，72 题 PA 将诊断与动作同时命中的题数由 {first[base]['joint']} 提升到 {first[pa]['joint']}。在同一开发集上，使用错配特权也达到 {first['v3_wrong']['joint']} 道。动作分布和语言形式会随训练改变，正确特权的独立贡献仍需从对照中识别。"
    cmp = s["paired"]["official_pa_vs_gold_sft"]["teaching"]
    teaching = f"72 题联合蒸馏的教学四维均分为 {s['official'][pa]['teaching_mean']:.3f}，原 Gold SFT 为 {s['official'][base]['teaching_mean']:.3f}（各 140 题）。同题均分差 {cmp['diff']:.3f}，按题号聚类的 95% 区间为 {cmp['ci95'][0]:.3f} 至 {cmp['ci95'][1]:.3f}。"
    semantic = get(d / "规模配对/教学同题差异.json")
    gain = semantic["gold_sft"]["dimensions"]["four_mean"]
    history = semantic["gold_sft"]["dimensions"]["history_score"]
    scale_finding = f"373 题 PA 的教学四维均分为 {gain['pa373_mean']:.3f}，原 Gold SFT 为 {gain['baseline_mean']:.3f}。同题改善 {gain['wins']} 道、退步 {gain['losses']} 道，均分差 {gain['diff']:+.3f}，按题号聚类的 95% 区间为 {gain['ci95'][0]:+.3f} 至 {gain['ci95'][1]:+.3f}。历史运用一项提高 {history['diff']:.3f} 分。"
    control = semantic["teacher373"]
    sub = control["dimensions"]["four_mean"]
    control_finding = f"同为 373 题训练的教师文本 SFT 已完成 {control['n']}/140 道裁判评分。这 {control['n']} 道上，教师文本 SFT 为 {sub['baseline_mean']:.3f}，PA 为 {sub['pa373_mean']:.3f}。其余题仅有预测。该子集按运行顺序形成，尚无法估计全组优势。"
    para3 = "三轮技能候选均未通过独立准入。为检验强行入库的后果，三轮分别对候选做了参数更新，并在关闭检索后复评。三轮均保留原权重。Codex 对 45 道确认题的匿名话术复核中，原模型安全可用 21 次，候选模型 16 次。"
    p("教学动作蒸馏与技能自进化实验", "Title")
    p("开题 V3 实证报告　　2026 年 11 月 27 日")
    p(
        "同一批学生记录先用于检验模型能否读到历史证据，再用于检验如何教、能否从经验中持续改进。前者已经出现可解释的增益；后两项研究已运行到权重更新、独立确认与外部复测。"
    )
    h("正式任务和诊断动作")
    p(scale_finding)
    p(para1)
    p(para2)
    h("技能是否值得继承")
    p(para3)
    h("实验范围")
    table(
        doc,
        ["实验", "样本、模型与产出"],
        [
            [
                "主消融",
                "同一起点，8 组工程对照和 8 组原公式与教师对照，各 72 题、36 次更新",
            ],
            ["规模扩展", "全部 373 道符合条件的纯文字训练题，3 组各 187 次更新"],
            [
                "开发与外部",
                "官方提示 9 组各 140 题；MathTutorBench 4 个模型各 4,800 条回答",
            ],
            ["跨轮确认", "三轮分别 16、14、15 条；300 条 LongTutor 封存测试保留"],
        ],
        [4.1, 12.4],
    )
    page("同一开发集上的正式生成与评分")
    p(
        "各组使用同一时序历史、公开问题和 LongTutor 固定提交，温度 0，输出上限 3,000 token。诊断与策略来自未改动的官方预测文件。教学用官方裁判给出的历史、策略、连贯、理解四维逐题评分取平均，满分 5。"
    )
    order = [base, pa, ts, "v3_action_kl", tok, "offpolicy"]
    tbl = []
    for sz, analysis, au, arms in [
        ("起点或72题", s, original, order),
        ("373题", big, biga, [pa, ts, tok]),
    ]:
        for k in arms:
            o = analysis["official"].get(k, {})
            mf = o.get("upstream", {}).get("diagnosis_macro_f1")
            tbl.append(
                [
                    NAMES[k],
                    sz if k != base else "起点",
                    au.get(k, {}).get("diagnosis", "—"),
                    "—" if mf is None else f"{mf:.3f}",
                    ft(o),
                    scored(analysis, k),
                ]
            )
    table(
        doc,
        ["训练方式", "规模", "诊断正确", "官方宏 F1", "教学均分（n）", "DK 已评分"],
        tbl,
        [3.65, 2.2, 2.5, 2.5, 3.6, 2.1],
    )
    p(
        teaching
        + " 表中教学均分是四项平均；官方另给独立总体评分。横线表示该组未取得全量 DK 汇总。"
    )
    p(control_finding)
    m0 = s["official"][base]["counts"]
    m1 = s["official"][pa]["counts"]
    p(
        "历史记忆三个子问题各 140 题。Gold SFT 的 Q1 / Q2 / Q3 命中为 "
        + " / ".join(str(m0[k]) for k in ("Q1", "Q2", "Q3"))
        + "，72 题 PA 为 "
        + " / ".join(str(m1[k]) for k in ("Q1", "Q2", "Q3"))
        + "。两组在 Q2 上都只答对约四分之一。"
    )
    h("与先前检索实验的联系")
    p(
        "同一基座、相近提示长度，普通检索和时序检索在历史事实提取 Q1 上分别答对 78 和 115 道；38 道改善、1 道退步。按 92 个当前题号整簇重抽样，增幅为 26.4 个百分点，95% 区间为 18.1 至 36.2。新增对照固定时序输入，检验的是权重变化。"
    )
    page("动作和语言分别贡献了什么")
    p(
        "每题先在公开上下文选择历史依据、诊断与 A–H 动作，再生成第一步话术。动作相容指所选动作属于参考教学策略的预设允许集合；联合命中还要求诊断正确。同行均使用相同的 140 道开发题。"
    )
    rows = []
    for k in [
        base,
        ts,
        pa,
        "v3_action_kl",
        tok,
        "v3_solution_only",
        "v3_wrong",
        "v3_neutral",
        "v3_no_eedi",
        "offpolicy",
    ]:
        r = first[k]
        rows.append(
            [
                NAMES[k],
                r["diagnosis"],
                r["compatible"],
                r["joint"],
                f"{r['macro_f1']:.3f}",
            ]
        )
    table(
        doc,
        ["72 题条件", "诊断正确", "动作相容", "联合命中", "四类宏 F1"],
        rows,
        [5.1, 2.8, 2.9, 2.8, 2.8],
    )
    a = s["paired"]["structured_pa_vs_gold_sft"]["joint"]
    b = s["paired"]["structured_pa_vs_v3_wrong"]["joint"]
    p(
        f"联合命中相对 SFT 起点同题改善 {a['wins']} 道、退步 {a['losses']} 道。按当前题号聚类，百分点差的 95% 区间为 {100 * a['ci95'][0]:.1f} 至 {100 * a['ci95'][1]:.1f}。正确与错配特权的同题结果为 {b['wins']} 胜、{b['losses']} 负。教师文本 SFT 在诊断指标上另有优势。"
    )
    p(
        "起点对 140 道题全部选择 H（部分解答）；联合模型有 118 道选择 C（追问），错配特权模型有 131 道选择 C。联合命中上升的同时，单独衡量的动作相容数从 71 降至 47，动作分布仍集中。"
    )
    h("扩大训练量")
    p(
        "373 题时，PA、教师文本 SFT、内容蒸馏的联合命中分别为 "
        + "、".join(str(scale[k]["joint"]) for k in [pa, ts, tok])
        + " 道。小规模每类取 18 题，大规模使用全部合格题，类别比例随规模一起改变。"
    )
    page("跨数据集泛化")
    p(
        "采用 MathTutorBench 固定提交的四项任务。普通教学提问 1,150 条、困难教学提问 327 条、错步定位 2,004 条、苏格拉底式提问 1,319 条。四种权重均使用官方提示和相同解码设置，共返回 19,200 条逐项回答。"
    )
    ext = s["external"]
    table(
        doc,
        ["权重", "普通提问", "困难提问", "错步宏 F1", "问题 BLEU"],
        [
            [
                NAMES[k],
                percent(ext[k]["pedagogy_following"]["match"]),
                percent(ext[k]["pedagogy_following_hard"]["match"]),
                f"{ext[k]['mistake_location']['f1_macro']:.3f}",
                f"{ext[k]['socratic_questioning']['bleu']:.3f}",
            ]
            for k in ["base", base, ts, pa]
        ],
        [4.1, 3, 3, 3.1, 3.2],
    )
    p(
        "普通和困难提问的官方指标统计输出中是否出现问号或提问词。错步任务检验模型给出的首个错误步骤编号。BLEU 衡量生成问题与参考问题的文字相似性。"
    )
    h("与训练起点比较")
    p(
        "PA 的困难提问命中率由 84.1% 变为 84.7%，错步宏 F1 由 0.199 变为 0.195。四项结果没有呈现一致的外部优势；较小变动应结合任务指标的分辨率解释。"
    )
    h("教师信号的来源")
    p(
        "同源模型分三次调用完成诊断、教学策略和审核。Eedi 对照对命中的题重跑全部三角色。140 道开发题中，完整教师最终诊断正确 33 道，去 Eedi 为 30 道；合格 JSON 均为 138 道。窄知识点映射覆盖主训练 33/72、开发 17/140。备选误概念的存在不代表学生实际持有该误概念。"
    )
    page("三轮技能提炼与继承")
    p(
        "从来源轨迹中提取按可见状态触发的模板技能，记录正反例。准入对照在另一批学生和题目上比较现有技能库与加入候选后的技能库。所有候选还各生成一份实验权重，以便观察未通过准入时强行更新的后果。"
    )
    tbl = []
    for k, r in c.items():
        a, b = r["admission"], r["confirmation"]
        tbl.append(
            [
                k,
                f"{a['before']['joint']}→{a['after']['joint']} / {a['before']['n']}",
                r["updates"],
                f"{b['before']['joint']}→{b['after']['joint']} / {b['before']['n']}",
                f"{blind[k]['incumbent']['U']}→{blind[k]['candidate']['U']}",
                "保留起点",
            ]
        )
    table(
        doc,
        ["轮次", "准入联合命中", "候选更新", "关检索联合命中", "首步盲评可用", "继承"],
        tbl,
        [1.3, 3.3, 2.25, 3.85, 3.35, 2.45],
    )
    p(
        "每轮的学生、当前题号和规范化题干与其他池隔离。准入后才能进入活动技能库。三轮均未入库，因此没有得到连续三代性能提升。原 Gold SFT 曾见过 560 条训练池；三轮新鲜确认池来自开发集，均缺少迁移不足类。"
    )
    h("复看具体话术")
    p(
        "三轮共 45 题、90 份条件输出；同题相同文本合并后，Codex 对 87 个匿名选项逐条复核。第 2 轮候选的可用首步从 8/14 降到 3/14。把要求从结论倒推的运煤题具体化为“从最后剩下的300吨开始倒推”，学生可以动手；把四个 3 填运算符误导到“一笔画规则”，则偏离了当前任务。"
    )
    p(
        "盲评是事后单评审，不参与候选训练或准入。第 1 轮探索性候选的去检索保持率为 −0.50；后两轮参考检索增益未达预设阈值，比例不报告。未通过准入的候选不作为已验证技能内化的正证据。"
    )
    probe = get(d / "history_probe/analysis.json")
    page("相同题目下，模型会依据学生历史换一种教法吗")
    p(
        "先固定每题原 Gold SFT 已生成的证据序号和诊断，再把可见历史记录里的“正确”和“错误”逐条互换。题目、知识点、时间、当前作答结果均保持原值。比较八种动作的概率分布。另从证据和诊断重新选择一次，观察整条决策过程会怎样变化。"
    )
    tab = []
    for label, key in [
        ("原 Gold SFT", "gold_sft"),
        ("联合蒸馏 · 72", "pa72"),
        ("错配特权 · 72", "wrong72"),
        ("联合蒸馏 · 373", "pa373"),
    ]:
        m = probe["models"][key]
        tab.append(
            [
                label,
                f"{m['fixed_action_probs']['tv_mean']:.3f}",
                m["fixed_action_probs"]["changed_argmax"],
                m["free_action_probs"]["changed_argmax"],
                f"{m['natural_direction']['mean_per_question']:+.3f}",
            ]
        )
    table(
        doc,
        ["模型", "固定前缀 TV", "固定首选改变", "自由首选改变", "自然配对方向"],
        tab,
        [3.9, 3.1, 3.1, 3.1, 3.2],
    )
    p(
        "TV 为总变差距离，0 表示两个八动作分布完全相同，1 表示没有重合。首选改变的分母各为 140。翻转历史后的理想教法未知，这项干预只检验模型的敏感度。"
    )
    contrast = probe["paired"]["pa72_minus_wrong72"]["fixed_action_probs"]
    mechanism = f"固定前缀时，72 题联合蒸馏与错配特权的平均 TV 差为 {contrast['diff']:+.3f}，按当前题号聚类的 95% 区间为 {contrast['ci95'][0]:+.3f} 至 {contrast['ci95'][1]:+.3f}。"
    p(mechanism)
    p(
        "同一真实历史、同一前缀下，从 Gold SFT 换成 72 题 PA 的平均 TV 为 0.961，而只翻转该模型输入里的历史正误为 0.110。原推理与本轮概率读取对真实输入的动作首选一致数为 133–140/140。表内的干预比较均使用同一实现。"
    )
    h("用真实的同题异生记录验证方向")
    p(
        f"开发集中有 {probe['natural_pairs']} 对同题异生、专家诊断对应动作集合互不重合的记录，仅覆盖 {probe['natural_questions']} 种当前题目。表中最后一列比较双方给各自参考动作的概率优势，再对题目等权平均；正值表示平均偏向各自参考。不同学生也有其他差异，这一比较用于提出假设。"
    )
    p(
        "373 道训练题中可组成 248 对，覆盖 31 种题目。后续用不同学生的真实题史构成难负例，先让独立学科教师判断两份下一步提示是否需要不同。每道题内的配对共享来源，单独的一对不能作为独立样本量。"
    )
    page("实验边界与两周安排")
    h("对照与复现")
    p(
        "训练采用 Qwen2.5-7B-Instruct 的 BF16 基座与同一 Gold SFT LoRA 起点。正式 PA 在学生采样的续写前缀上匹配教师到学生的八动作 KL 和话术 token JSD，并加格式交叉熵，系数为 0.5、0.5、0.05。温度 0.7 生成训练轨迹；推理评测温度为 0。早期八组动作 JSD 工程对照单独归档。"
    )
    p(
        "LongTutor 上游提交 ba40af3b9e976f5960c42a15a645c0e9c3a7d718。MathTutorBench 上游提交 6faed173ec2bef55cb899b2a3e0f93982f9cb176。训练日志含每题续写、损失、更新梯度与适配器摘要，正式预测保留格式失败。"
    )
    h("开题时应保留的研究问题")
    p(
        "拟把论文主线收紧到同题异生的教学决策。让教师先指出哪一条历史事实要求改变哪一步教学，再用同题另一名学生的真实记录构造对照。比较等量普通教师文本 SFT、正确特权及错配特权，并以盲评首步是否适合该学生为主终点。技能继承留作第二阶段。"
    )
    h("提交前两周")
    p(
        "第 1–3 天逐题复核现有配对与失分案例，学习同题比较及聚类区间。第 4–7 天请两名学科教师独立复核训练池内同题异生的教学首步并记录一致率。第 8–10 天冻结数据、判据、强基线和训练预算；只有教师能反复区分且同意理由时才启动配对蒸馏。第 11–14 天完成开题与答辩。投稿阶段需补独立数据、多个种子、第二基座和外部人工确认。"
    )
    doc.core_properties.title = "教学动作蒸馏与技能自进化实验"
    doc.save(docs / "OPSD_V3完整实验报告.docx")
    integrated(
        docs.parent / "开题报告_OPSD自进化_v3.docx",
        docs / "开题报告_OPSD自进化_v3_实证补充版.docx",
        [
            "前期实验与验证进展",
            "固定开发集 140 题，基座采用普通和时序检索时，历史提取 Q1 分别答对 78 和 115 道，同题改善 38 道、退步 1 道。"
            + para1,
            scale_finding + control_finding,
            para2 + teaching,
            mechanism
            + "后续拟在同题异生的真实记录上，用独立教师复核教学首步，并与同等数据量的普通教师文本 SFT 比较。",
            para3,
            "扩展实验在全部 373 道含标准解的纯文字训练题上比较了联合模型、教师文本监督微调及内容蒸馏，各更新 187 次。四项 MathTutorBench 外部任务使用原评测代码完成 19,200 条预测，未观察到跨指标一致优势。上述结果将作为后续独立教师验证和参数内化研究的比较基础。",
        ],
    )


if __name__ == "__main__":
    main()
