# -*- coding: utf-8 -*-
from pptx import Presentation
from pptx.util import Inches, Pt

OUT = r"C:\Users\whm\Desktop\PA-OPSD_思路数据集任务_20260324.pptx"


def title_slide(prs, title, sub=""):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = title
    if sub and len(s.placeholders) > 1:
        s.placeholders[1].text = sub


def bullets(prs, title, items, font=18):
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    tf = s.placeholders[1].text_frame
    tf.clear()
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t
        p.font.size = Pt(font)


def main():
    prs = Presentation()

    title_slide(
        prs,
        "PA-OPSD 研究思路梳理",
        "数据集 · Benchmark · 论文 · 实现 · 待办\n开题稿 + EduEvolve 全文 · 2026-03-24 上课用",
    )

    bullets(prs, "1. 两套文档：现在做到哪", [
        "开题 PA-OPSD（2026.11–2027.02）：必须交付 — 三数据集 + B1–B5 + PA-OPSD",
        "1016 EduEvolve 全文：自进化 + 预算路由 + 更多数据 → 2027 下半年学位论文",
        "北星：训练贵（Team+z）→ PA-OPSD 压成 Student(x) → 后期进化/路由",
        "主指标：LGPT = ΔG / (N_token/1000)",
    ])

    bullets(prs, "2. 核心问题（开题一句）", [
        "训练期：多 Agent 教师 + 特权 z，先定教学动作 a，再说辅导话",
        "部署期：轻量 Student，只看 x=(q,h,r)，无 z、无 Team",
        "RQ1：z(k,m,y*,ρ) 能否被蒸馏进仅 x 的 Student？",
        "RQ2：动作级是否比整段文本蒸馏更省 token 且保质？",
        "RQ3：Team 分工能否压进单模型（vs B5）？",
    ])

    bullets(prs, "3. 符号 q / h / r（Student 可见 x）", [
        "q：当前题目",
        "h：长期学习史摘要（证据，LongTutor 任务1）",
        "r：当前 session 多轮对话",
        "x = (q, h, r) — 部署 Student 全部输入",
    ])

    bullets(prs, "4. 符号 z（仅 Teacher 训练）", [
        "k：知识状态诊断（LongTutor 四类：Recall/Conceptual/Procedural/Transfer）",
        "m：误概念（Eedi Misconception Graph）",
        "y*：标答/参考解（OPSD B4 已有）",
        "ρ：量规（禁泄漏答案、先追问等）",
        "注意：Student 不输出 k,m 表；学的是动作+话术 ≈ Teacher(·|x,z)",
    ])

    bullets(prs, "5. PA-OPSD 方法（两旋钮）", [
        "旋钮A：特权 z 扩展 OPSD 的「仅 y*」→ k,m,ρ,y*",
        "旋钮B：L_action(CE) + L_token(JSD)；六类动作 A_ped",
        "轨迹：Student generate 后缀 S；Teacher prefill 同一 S（同 OPSD）",
        "L = λ_a L_action + λ_t L_token；只更新 Student",
        "代码基线 B4：D:\\OPSD（opsd_trainer + data_collator）",
    ])

    bullets(prs, "6. 训练一步（5 步）", [
        "① Student prompt(x) → generate → 后缀 S",
        "② Teacher prompt(x + z文本 + 可选 a_T) + S → forward no_grad",
        "③ Student 整段 forward → π_S",
        "④ 每位置 JSD + 动作 CE",
        "⑤ 部署：仅 Student(x)，Team/z 全去掉",
    ])

    bullets(prs, "7. 开题【必做】三数据集", [
        "P0-A LongTutor：主 Benchmark（数学·长期）；Gold 1000 条；三级 Q1–Q3 / 诊断 / 教学",
        "P0-B Eedi ~2k 对话：真人 tutor；L_action、多轮 r",
        "P0-C Eedi Misconception 图：Teacher 特权 m",
        "≠ 再单独做整库 XES3G5M 当第二个主 bench（LongTutor 已含 XES 人标）",
    ])

    bullets(prs, "8. LongTutor 评什么（Benchmark 不是训练算法）", [
        "Evidence：memory 固定 3 问 Q1 抽取 / Q2 跨会话 / Q3 幻觉 → Memory SA",
        "Diagnosis：四类标签 + strategy 映射 → Acc / Macro-F1",
        "Teaching：content → LLM Judge 四维 + ROUGE-L",
        "用法：蒸馏后 Student 当「被测 Tutor」跑 eval_ai_tutor.py",
        "工程：liano3/LongTutor；human_an_updated.jsonl + history_features 对齐",
    ], font=16)

    bullets(prs, "9. Eedi 做什么", [
        "对话：1971 interventions，68717 messages → 映射六类动作、训练 rollout",
        "误概念图：8117 misconceptions → 写入 Teacher prompt 的 m",
        "合规：仅学术脱敏版；非商业",
        "ΔG 代理：干预后下一答对错 / misconception 是否消除",
    ])

    bullets(prs, "10. 全文【可选】数据集（开题不做）", [
        "XES3G5M 全量：KT 状态模型、routing 的 s_t（后期）",
        "EdNet：大规模 LG 代理、Pareto（后期）",
        "EduAgentBench：Agent+routing 泛化（建议后期）",
        "MMTutorBench / ASTRA：多模态、合成拓扑（可选/展望）",
    ])

    bullets(prs, "11. 基线 B1–B5", [
        "B1：7B + 固定 prompt，无训练",
        "B2：对话 SFT，仅 x，无动作",
        "B3：蒸馏 Team 整段文本，无 on-policy/无 a",
        "B4：原版 OPSD（y* + JSD）= 本仓库",
        "B5：部署真跑 Team+z（质量上限、成本对照）",
        "PA-OPSD：相对 B4 加 z 与 L_action；相对 B3 换蒸馏对象",
    ])

    bullets(prs, "12. 论文：开题必用（拿什么）", [
        "OPSD+ D:\\OPSD：on-policy、prefill、JSD、fixed_teacher → B4/L_token",
        "LongTutor：三级尺子 → 主表",
        "Eedi+ScaffoldLM：动作空间、状态→动作 → L_action 定义",
        "LUPI：train (x,z) / deploy x → Ch3",
        "AgentArk：MAS→Single 叙事 → B5；不做 GRPO/PAD 管线",
        "Tutor CoPilot：专家贵、要 scale → 动机",
    ], font=16)

    bullets(prs, "13. 论文：学位论文后期", [
        "EvolveR：轨迹→principle→dedup→prune → Ch5 技能库",
        "Mem²Evolve：experience/asset、专化角色",
        "BudgetMem：L/M/H routing、Pareto → Ch6",
        "SciEducator：related 划界（已有 evolve MAS+教育）",
    ])

    bullets(prs, "14. 实现：要改/新建什么", [
        "data：Eedi/LT 字段 → q,h,r,z,a 映射 v1",
        "collator：Teacher 拼 z 文本；Student 无 z；action slot",
        "trainer：action head + L_action；Teacher 条件 a_T",
        "scripts/team_offline.py：三角色 prompt 产 a_T",
        "eval：LongTutor 官方脚本 + Eedi action acc + LGPT 计数",
    ], font=16)

    bullets(prs, "15. Backbone（开题可行）", [
        "Student：Qwen2.5-7B + LoRA（AutoDL 24–80G）",
        "Team 离线：同 7B 三 prompt 或 API 大模型一次性",
        "LongTutor Judge：与被测 Student 独立的强模型",
    ])

    bullets(prs, "16. 待办 · 2026.11–12", [
        "开题答辩；克隆 LongTutor，跑通 Gold baseline（B1 数字）",
        "下载 Eedi 对话+图；50 条动作映射 v1",
        "Team 离线 trajectories_v1.jsonl（≥5k 轮目标）",
        "读 OPSD 论文 + 冒烟 opsd_train 50 step",
    ])

    bullets(prs, "17. 待办 · 2027.01–02（核心）", [
        "实现 PA-OPSD v1（L_action + z + L_token）",
        "跑通 B1–B5 + 消融（−k,−m,−ρ,−L_action, Team vs 单教师）",
        "主表：LongTutor 三级 + Eedi + LGPT",
        "论文稿 → ACL 2027 滚动（目标）",
    ])

    bullets(prs, "18. 待办 · 2027.03 以后", [
        "03–05：补实验、人工抽检、改投",
        "06–08：EvolveR 技能库 + Budget 路由（学位论文，非开题）",
        "09–10：学位论文整合、盲审",
    ])

    bullets(prs, "19. 风险（答辩预备）", [
        "ΔG 仅代理 → 写死公式 + LongTutor 留一 + 人工 50 条",
        "动作蒸馏不赢 B3 → 可发表负结果消融",
        "Student 无 k,m 输入 → 学行为非复现 k；用 ΔG/动作率验证",
    ])

    title_slide(prs, "谢谢 · 讨论", "Q & A")

    prs.save(OUT)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
