# One-off script to generate class presentation PPT
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

OUT = r"D:\OPSD\docs\PA-OPSD_上课思路梳理_20260324.pptx"


def add_title_slide(prs, title, subtitle=""):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    if subtitle and len(slide.placeholders) > 1:
        slide.placeholders[1].text = subtitle


def add_bullets(prs, title, bullets, note=None):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    tf = slide.placeholders[1].text_frame
    tf.clear()
    for i, line in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(20)
    if note:
        slide.notes_slide.notes_text_frame.text = note


def add_two_col(prs, title, left_title, left_items, right_title, right_items):
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # blank
    slide.shapes.title.text = title
    # text boxes
    left = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(4.5), Inches(5.5))
    right = slide.shapes.add_textbox(Inches(5.2), Inches(1.3), Inches(4.5), Inches(5.5))
    for box, t, items in [(left, left_title, left_items), (right, right_title, right_items)]:
        tf = box.text_frame
        tf.text = t
        tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.size = Pt(22)
        for it in items:
            p = tf.add_paragraph()
            p.text = it
            p.font.size = Pt(18)
            p.level = 0


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    add_title_slide(
        prs,
        "PA-OPSD 研究思路梳理",
        "数据集 · Benchmark · 与 OPSD 的关系\n（课堂汇报 · 2026-03-24）",
    )

    add_bullets(
        prs,
        "1. 一句话方向",
        [
            "训练期：多角色教师 + 特权信息 z，在学生 on-policy 轨迹上教「动作 + 怎么说」",
            "部署期：单一轻量 Student Tutor，只看 x = (q, h, r)，不泄漏 z、不跑多 Agent",
            "全文闭环：PA-OPSD（核心）→ 自进化技能库 → learner-state 预算路由（后期）",
            "主指标：LGPT = ΔG / (N_token/1000)（学习增益 / 千 token 成本）",
        ],
    )

    add_bullets(
        prs,
        "2. 要回答的核心问题",
        [
            "能否在「部署只看题目与对话」下，接近「训练时能看 k、m、标答、量规 + 多 Agent」的辅导质量？",
            "能否明显降低 token 与 Agent 调用成本？",
            "策略能否随交互持续改进（后期：EvolveR 式技能库）？",
            "注意：不是 Student 推理时恢复完整 z，而是模仿「已知 z 的 Teacher」的决策",
        ],
    )

    add_bullets(
        prs,
        "3. 三个现实矛盾（绪论动机）",
        [
            "辅导需要状态 k、误概念 m、教学动作 a —— 但产品不能把标答/后台标签塞给用户端模型",
            "多 Agent 教师质量好，但推理阶段 token × 调用次数贵（对照 AgentArk、Tutor CoPilot）",
            "静态 SFT/文本蒸馏：多在专家轨迹上，难覆盖学生 rollout 错轨（exposure bias）",
            "LongTutor：LLM 会找历史证据 h，但长期诊断 k + 自适应教学仍弱",
        ],
    )

    add_two_col(
        prs,
        "4. 符号：学生可见 x vs 特权 z",
        "x（部署 Student 输入）",
        [
            "q：当前题目 / 任务",
            "h：长期学习史摘要（证据，LongTutor 任务1）",
            "r：当前 session 多轮对话",
        ],
        "z（仅训练 Teacher）",
        [
            "k：知识状态诊断（LongTutor 任务2）",
            "m：误概念（Eedi Graph）",
            "y*：标答 / 参考解（OPSD 已有）",
            "ρ：量规（勿泄漏答案、先追问等）",
        ],
    )

    add_bullets(
        prs,
        "5. PA-OPSD 方法（相对原版 OPSD 两个旋钮）",
        [
            "旋钮 A：特权 z 从「仅 y*」扩展为 k, m, ρ, …（教育 LUPI）",
            "旋钮 B：蒸馏对象 = 教学动作 a（L_action, CE）+ token（L_token, JSD）",
            "轨迹：与 OPSD 相同 —— 仅 Student generate 后缀 S；Teacher prefill 同一 S",
            "L = λ_a L_action + λ_t L_token；梯度只更新 Student",
            "代码参照：D:\\OPSD（ICML OPSD ≈ 实验基线 B4）",
        ],
    )

    add_bullets(
        prs,
        "6. 训练一步（对齐 OPSD 代码直觉）",
        [
            "① Student 仅用 student prompt（x）→ generate → 后缀 S",
            "② Teacher prompt = x 的文本 + z 文本 + 过渡语；拼接同一 S（不 generate）",
            "③ Student forward → P_S；Teacher no_grad forward → P_T",
            "④ 每 token / 每步动作：JSD 或 CE；Verifier 防标答泄漏进 Student 可见文本",
            "部署：只保留 Student；Team 与 z 全部去掉",
        ],
    )

    add_two_col(
        prs,
        "7. 实验基线 B1–B5",
        "基线",
        [
            "B1：大模型 + 固定 prompt",
            "B2：普通 SFT（仅 x，文本）",
            "B3：蒸馏 Teacher 整段文本",
            "B4：原版 OPSD（y* + token JSD）",
            "B5：部署时真跑多 Agent（成本上限）",
        ],
        "本文 PA-OPSD",
        [
            "相对 B4：+k,m,ρ + L_action",
            "相对 B3：动作中间表示，省 token",
            "相对 B5：压缩为单模型，报压缩损失",
            "消融：去 k / 去 m / 去 ρ / 无动作头",
        ],
    )

    add_bullets(
        prs,
        "8. 数据集：必做 vs 可选（开题主体）",
        [
            "【必做·线A】LongTutor — 长期数学；h、k、教学动作三任务；底层已含 XES3G5M 人标集",
            "【必做·线B】Eedi ~2k 对话 — 真人 tutor–student；L_action、多轮 r",
            "【必做·配套】Eedi Misconception Graph — Teacher 特权 m",
            "【不必重复】整库 XES3G5M — 仅当自训 KT/状态模型、routing 的 s_t（后期创新3/4）",
            "【可选】EdNet — 大规模 LG 代理、Pareto；不替代 LongTutor Gold",
            "【可选】EduAgentBench / MMTutorBench / ASTRA — 路由泛化、多模态、合成拓扑（非开题必需）",
        ],
    )

    add_bullets(
        prs,
        "9. Benchmark 怎么用（别和数据集混）",
        [
            "LongTutor = Benchmark + 数据：评 Evidence / Diagnosis / Teaching Action（主数学长期）",
            "Eedi = 数据为主：评动作分类、对话质量、误概念相关干预（主 action 蒸馏）",
            "LGPT 需在论文写清 ΔG 操作定义（如：下一答对错、m 是否消除、LongTutor 子任务 success）",
            "同一研究勿把 XES3G5M 与 LongTutor 写成两个平行主 bench",
        ],
    )

    add_bullets(
        prs,
        "10. 全文章节 vs 当前阶段",
        [
            "Ch4 PA-OPSD + Ch7.1–7.2（B1–B5、消融）← 开题后 4 个月主攻",
            "Ch5 自进化（EvolveR：principle → 再蒸馏）← 2027 上半年",
            "Ch6 BudgetMem 式 routing + Pareto ← 后期",
            "Ch8 多模态 / EdNet 状态模型 ← 展望",
        ],
    )

    add_bullets(
        prs,
        "11. 近期执行清单（思路落地）",
        [
            "读：OPSD 论文 + 跑通 D:\\OPSD 短训（B4）",
            "数据：Eedi 一条样本映射 q,h,r,z,a_T；LongTutor 看三任务评测脚本",
            "实现顺序：B4 collator 加 z 文本 → L_action 头 → Teacher 条件 a_T 的 L_token",
            "论文：绪论用 LongTutor / ScaffoldLM / Tutor CoPilot / OPSD / AgentArk 五段（已整理）",
        ],
    )

    add_bullets(
        prs,
        "12. 答辩可能追问 & 简短答法",
        [
            "Q：Student 没有 k,m 怎么准？ A：学的是 π(a,话|x) 逼近 π(·|x,z)；x 含错话/历史时部分可学；用 ΔG 实证",
            "Q：LongTutor 和 XES3G5M？ A：LongTutor 在其日志上做人标 benchmark，不另做一套主实验",
            "Q：和 OPSD 区别？ A：教育 z + 动作级；机制仍是 on-policy + prefill + JSD",
        ],
    )

    add_title_slide(prs, "谢谢 · 讨论", "Q & A")

    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
