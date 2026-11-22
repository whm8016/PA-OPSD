"""Figures for the BUPT-template proposal (v4). Writes new files; never overwrites v2/v3."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_LANDSCAPE = FIG_DIR / "proposal_v4_fig1-1_landscape.png"
FIG_FRAMEWORK = FIG_DIR / "proposal_v4_fig2-1_framework.png"
FIG_PAOPSD = FIG_DIR / "proposal_v4_fig2-2_pa_opsd.png"
FIG_LOOP = FIG_DIR / "proposal_v4_fig2-3_dual_loop.png"

EDGE = "#315A7D"
FILL = "#EAF2F8"
TEXT = "#17324D"
ACCENT = "#8A3B12"
FILL_ACCENT = "#FBEEE6"
FILL_GREEN = "#EAF5EA"
EDGE_GREEN = "#3B7A3B"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fs=10.5, fill=FILL, edge=EDGE, bold=False, color=TEXT, lw=1.3, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                weight="bold" if bold else "normal", linespacing=1.45)


def frame(ax, x, y, w, h, title, edge=EDGE, fs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                linewidth=1.4, edgecolor=edge, facecolor="none", linestyle=(0, (5, 3))))
    ax.text(x + 0.012, y + h - 0.012, title, ha="left", va="top", fontsize=fs, color=edge, weight="bold")


def arrow(ax, start, end, color=EDGE, lw=1.5, text=None, fs=9, rad=0.0, text_xy=None):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=13,
                                connectionstyle=f"arc3,rad={rad}"))
    if text:
        tx, ty = text_xy if text_xy else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(tx, ty, text, ha="center", va="center", fontsize=fs, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def fig_landscape():
    fig, ax = canvas(13, 7.2)
    ax.text(0.14, 0.965, "研究方向与代表工作", ha="center", fontsize=12.5, weight="bold", color=TEXT)
    ax.text(0.47, 0.965, "现有研究的不足", ha="center", fontsize=12.5, weight="bold", color=TEXT)
    ax.text(0.84, 0.965, "本论文研究内容", ha="center", fontsize=12.5, weight="bold", color=TEXT)
    rows = [
        ("大语言模型智能导师", "MathDial · ScaffoldLM\nLongTutor · DICE",
         "教学行为依赖提示与推理时流程\n长期历史难以转化为诊断与教学决策"),
        ("特权信息与在策略蒸馏", "LUPI · GKD · OPSD\nPIDistill · PreSO · TurnOPD",
         "特权信息多为标准答案\n面向数学推理与通用智能体，缺少教学决策层"),
        ("智能体经验记忆与技能", "ReasoningBank · EvolveR\nTrace2Skill · REAT",
         "经验停留在外部检索，上下文持续膨胀\n未经验证的经验可能被反复放大"),
        ("参数化进化与能力压缩", "OPD-Evolver · EvoRouter\nAgentArk · Selection≠Learning",
         "面向通用任务，缺少教学效用验证\n缺少跨代继承与回滚约束"),
    ]
    ys = [0.75, 0.53, 0.31, 0.09]
    for (title, works, gap), y in zip(rows, ys):
        box(ax, 0.01, y, 0.26, 0.16, f"{title}\n{works}", fs=10.5)
        box(ax, 0.32, y, 0.30, 0.16, gap, fs=10.2, fill=FILL_ACCENT, edge=ACCENT)
        arrow(ax, (0.27, y + 0.08), (0.32, y + 0.08))
    box(ax, 0.68, 0.53, 0.31, 0.38,
        "研究点一\n\n教育特权教学动作级\n在策略自蒸馏\n\n教师团队构造特权 z\n动作 + token 联合蒸馏\n部署期单模型、仅见 x",
        fs=10.8, fill=FILL_GREEN, edge=EDGE_GREEN)
    box(ax, 0.68, 0.09, 0.31, 0.38,
        "研究点二\n\n经验证教学技能的\n快慢双循环自进化\n\n技能提炼与独立准入\n技能作特权回流 OPSD\n继承闸门与去检索保持",
        fs=10.8, fill=FILL_GREEN, edge=EDGE_GREEN)
    arrow(ax, (0.62, 0.83), (0.68, 0.76), color=EDGE_GREEN)
    arrow(ax, (0.62, 0.61), (0.68, 0.68), color=EDGE_GREEN)
    arrow(ax, (0.62, 0.39), (0.68, 0.32), color=EDGE_GREEN)
    arrow(ax, (0.62, 0.17), (0.68, 0.24), color=EDGE_GREEN)
    arrow(ax, (0.835, 0.47), (0.835, 0.53), color=EDGE_GREEN, text="技能作为新特权", fs=9, text_xy=(0.90, 0.50))
    ax.annotate("", xy=(0.99, 0.015), xytext=(0.01, 0.015),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=ACCENT, mutation_scale=16))
    ax.text(0.5, -0.025, "发展趋势：推理时提示与多角色协作  →  训练期特权监督与在策略蒸馏  →  经验证经验的参数化沉降与持续进化",
            ha="center", va="center", fontsize=11, color=ACCENT, weight="bold")
    save(fig, FIG_LANDSCAPE)


def fig_framework():
    fig, ax = canvas(13, 7.6)
    box(ax, 0.01, 0.60, 0.19, 0.30, "教师特权信息 z\n（仅训练期可见）\n\n标准解与解析\n误概念候选\n教学评价量规\n已验证教学技能\n（来自研究点二）",
        fs=10.0, fill=FILL_ACCENT, edge=ACCENT)
    box(ax, 0.01, 0.30, 0.19, 0.24, "学生可见输入 x\n\n长期作答历史 H\n当前题目 q\n学生当前作答 r", fs=10.3)

    frame(ax, 0.235, 0.20, 0.375, 0.72, "研究点一：教育特权教学动作级在策略自蒸馏", edge=EDGE)
    box(ax, 0.255, 0.70, 0.335, 0.12, "教师团队：Diagnoser → Pedagogue → Verifier\n证据定位 · 诊断 · 动作建议 · 质检", fs=10)
    box(ax, 0.255, 0.46, 0.155, 0.16, "学生分支\nπθ(· | x)\n自身采样 (a, y)", fs=10.2)
    box(ax, 0.435, 0.46, 0.155, 0.16, "特权教师分支\nπθ(· | x, z)\n同一参数 · 停止梯度", fs=10.2, fill=FILL_ACCENT, edge=ACCENT)
    box(ax, 0.255, 0.25, 0.335, 0.13, "动作级 + token 级联合蒸馏\nL = λa·L_act + λt·L_tok + λs·L_SFT", fs=10.3)
    arrow(ax, (0.20, 0.76), (0.255, 0.76), color=ACCENT)
    arrow(ax, (0.20, 0.45), (0.255, 0.52))
    arrow(ax, (0.5125, 0.70), (0.5125, 0.62), color=ACCENT, text="构造结构化 z", fs=8.8, text_xy=(0.545, 0.66))
    arrow(ax, (0.41, 0.54), (0.435, 0.54), text=None)
    ax.text(0.4225, 0.565, "同一前缀", ha="center", fontsize=8, color=EDGE)
    arrow(ax, (0.3325, 0.46), (0.3325, 0.38))
    arrow(ax, (0.5125, 0.46), (0.5125, 0.38), color=ACCENT)

    frame(ax, 0.635, 0.20, 0.355, 0.72, "研究点二：经验证教学技能的快慢双循环自进化", edge=EDGE_GREEN)
    box(ax, 0.655, 0.70, 0.145, 0.12, "第 t 代导师 M_t\n产生新辅导轨迹", fs=10, fill=FILL_GREEN, edge=EDGE_GREEN)
    box(ax, 0.825, 0.70, 0.145, 0.12, "成功 / 失败归因\n技能候选提炼", fs=10, fill=FILL_GREEN, edge=EDGE_GREEN)
    box(ax, 0.825, 0.48, 0.145, 0.13, "互斥验证集准入\n通过才进技能库 S_t", fs=10, fill=FILL_GREEN, edge=EDGE_GREEN)
    box(ax, 0.655, 0.48, 0.145, 0.13, "技能作为特权 z_skill\n再次执行 OPSD", fs=10, fill=FILL_GREEN, edge=EDGE_GREEN)
    box(ax, 0.655, 0.25, 0.315, 0.13, "继承闸门：新鲜确认集上质量不降、风险不升才继承\n继承后进入下一代；否则回滚至 M_t", fs=9.8,
        fill=FILL_GREEN, edge=EDGE_GREEN)
    arrow(ax, (0.80, 0.76), (0.825, 0.76), color=EDGE_GREEN)
    arrow(ax, (0.8975, 0.70), (0.8975, 0.61), color=EDGE_GREEN, text="快循环", fs=9, text_xy=(0.935, 0.655))
    arrow(ax, (0.825, 0.545), (0.80, 0.545), color=EDGE_GREEN)
    arrow(ax, (0.7275, 0.48), (0.7275, 0.38), color=EDGE_GREEN, text="慢循环", fs=9, text_xy=(0.69, 0.43))

    box(ax, 0.01, 0.02, 0.98, 0.09,
        "部署阶段：单一轻量导师模型，仅输入 x，输出 证据 e · 诊断 d · 教学动作 a · 教学内容 y；不调用教师团队，不依赖技能检索",
        fs=10.8, fill="#F4F6F8", edge="#5B6B7A", bold=True)
    save(fig, FIG_FRAMEWORK)


def fig_paopsd():
    fig, ax = canvas(13, 6.2)
    box(ax, 0.01, 0.66, 0.17, 0.2, "学生输入 x\n历史 H · 题目 q · 作答 r", fs=10.3)
    box(ax, 0.01, 0.22, 0.17, 0.2, "教师输入 x ⊕ z\nz：标准解 · 误概念候选\n评价量规 · 技能", fs=10.0, fill=FILL_ACCENT, edge=ACCENT)
    box(ax, 0.23, 0.66, 0.17, 0.2, "学生分支 πθ\n（参与梯度更新）", fs=10.3)
    box(ax, 0.23, 0.22, 0.17, 0.2, "教师分支 πθ\n（同一参数，停止梯度）", fs=10.3, fill=FILL_ACCENT, edge=ACCENT)
    arrow(ax, (0.18, 0.76), (0.23, 0.76))
    arrow(ax, (0.18, 0.32), (0.23, 0.32), color=ACCENT)

    ax.text(0.695, 0.95, "学生自身采样的输出序列 ô（教师在同一前缀上逐位置打分）", ha="center", fontsize=10.5, color=TEXT, weight="bold")
    segs = [
        (0.45, 0.095, "证据 e\n历史第 12、27 条", FILL),
        (0.565, 0.095, "诊断 d\n程序性错误", FILL),
        (0.68, 0.095, "动作 a\n程序性修正", "#FFF5CC"),
        (0.795, 0.195, "教学内容 y\n“回看第二步：通分后\n分母是 12，分子该乘几？”", FILL),
    ]
    for x, w, t, f in segs:
        box(ax, x, 0.66, w, 0.2, t, fs=9.4, fill=f)
    arrow(ax, (0.40, 0.76), (0.45, 0.76), text="采样", fs=8.8, text_xy=(0.425, 0.80))
    for x, w, _, _ in segs:
        arrow(ax, (0.40, 0.36), (x + w / 2, 0.66), color=ACCENT, lw=0.9)
    ax.text(0.235, 0.15, "教师在学生采样的同一前缀 ô<j 上\n给出逐位置分布 πθ(· | x, z, ô<j)", fontsize=9.2, color=ACCENT, ha="left")

    box(ax, 0.45, 0.06, 0.21, 0.2, "SFT 锚定项 L_SFT\n稳定输出格式\n（证据、诊断字段）", fs=9.6)
    box(ax, 0.68, 0.06, 0.095, 0.2, "动作损失\nL_act\n动作集合上\n的分布匹配", fs=9.2, fill="#FFF5CC")
    box(ax, 0.795, 0.06, 0.195, 0.2, "token 损失 L_tok\n内容位置上的广义 JSD\nDβ(教师 ‖ 学生)", fs=9.6)
    ax.text(0.5, -0.03, "L = λa · L_act + λt · L_tok + λs · L_SFT      先学“下一步怎么教”，再学“怎样说出来”",
            ha="center", fontsize=11.5, color=TEXT, weight="bold")
    save(fig, FIG_PAOPSD)


def fig_loop():
    fig, ax = canvas(13, 6.6)
    ax.text(0.5, 0.97, "快循环：教学技能提炼与验证          慢循环：技能经 OPSD 沉降到模型权重", ha="center",
            fontsize=13, weight="bold", color=TEXT)
    top = [
        (0.03, "第 t 代导师 M_t\n在训练池上辅导\n产生新轨迹"),
        (0.27, "独立评价\n成功 / 失败归因"),
        (0.51, "技能候选提炼\n聚类 · 合并 · 冲突检测"),
        (0.75, "互斥验证集准入\n启用 vs 禁用对比"),
    ]
    bottom = [
        (0.75, "活动技能库 S_t\n作为教师特权 z_skill"),
        (0.51, "动作 + token\n教育特权 OPSD"),
        (0.27, "候选模型 M_t+1\n新鲜确认集评测"),
        (0.03, "继承：M_t+1 成为新一代\n关闭检索复测保持率"),
    ]
    for x, t in top:
        box(ax, x, 0.62, 0.21, 0.2, t, fs=10.5, fill=FILL_GREEN, edge=EDGE_GREEN)
    for x, t in bottom:
        box(ax, x, 0.22, 0.21, 0.2, t, fs=10.5)
    for i in range(3):
        arrow(ax, (top[i][0] + 0.21, 0.72), (top[i + 1][0], 0.72), color=EDGE_GREEN)
        arrow(ax, (bottom[i][0], 0.32), (bottom[i + 1][0] + 0.21, 0.32))
    arrow(ax, (0.855, 0.62), (0.855, 0.42), color=EDGE_GREEN, text="通过", fs=9, text_xy=(0.885, 0.52))
    arrow(ax, (0.135, 0.42), (0.135, 0.62), text="下一轮", fs=9, text_xy=(0.10, 0.52))
    box(ax, 0.80, 0.02, 0.19, 0.12, "未通过的技能\n归档拒绝理由", fs=9.5, fill=FILL_ACCENT, edge=ACCENT)
    arrow(ax, (0.96, 0.62), (0.94, 0.14), color=ACCENT, text="未通过", fs=8.8, text_xy=(0.975, 0.38), rad=-0.15)
    box(ax, 0.27, 0.02, 0.33, 0.12, "继承闸门未通过：回滚保留 M_t\n已验证技能继续以外部形式可用", fs=9.5, fill=FILL_ACCENT, edge=ACCENT)
    arrow(ax, (0.375, 0.22), (0.375, 0.14), color=ACCENT, text="不达标", fs=8.8, text_xy=(0.42, 0.18))
    save(fig, FIG_LOOP)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_landscape()
    fig_framework()
    fig_paopsd()
    fig_loop()
    for p in (FIG_LANDSCAPE, FIG_FRAMEWORK, FIG_PAOPSD, FIG_LOOP):
        print(p)
