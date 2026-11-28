"""Compare matched dev cases across training sizes and draw the defense figure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scripts.experiments.action_distillation.analyze import (
    DIMENSIONS,
    LABELS,
    pair,
    rows,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    base = args.folder
    ref = rows(base / "dev.jsonl")
    keys = [r["id"] for r in ref]
    clusters = [r["qid"] for r in ref]
    groups = [
        ("Gold SFT", base, "gold_sft"),
        ("PA · 72", base, "v3_pa"),
        ("PA · 373", base / "scale", "v3_pa"),
        ("教师 SFT · 72", base, "v3_teacher_sft"),
        ("教师 SFT · 373", base / "scale", "v3_teacher_sft"),
        ("内容蒸馏 · 72", base, "v3_content"),
        ("内容蒸馏 · 373", base / "scale", "v3_content"),
    ]
    results = {}
    by_case = []
    for label, folder, arm in groups:
        public = rows(folder / "evaluation" / (arm + ".jsonl"))
        native = rows(folder / "official" / (arm + ".full.jsonl"))
        ix = {r["id"]: r for r in public}
        predictions = {r["_key"]: r for r in native}
        if (
            len(native) != len(public)
            or len(native) != len(keys)
            or set(keys) != set(ix)
            or set(keys) != set(predictions)
        ):
            raise ValueError("Different dev cases for " + label)
        results[label] = {"joint": [], "formal_diagnosis": []}
        for r in ref:
            k = r["id"]
            answer = ix[k]
            native_correct = predictions[k]["diagnosis"] == r["gold"]
            joint = (
                LABELS["RCPT".index(answer["diagnosis"])] == r["gold"]
                and answer["action"] in r["compatible"]
            )
            results[label]["joint"].append(int(joint))
            results[label]["formal_diagnosis"].append(int(native_correct))
            by_case.append(
                {
                    "group": label,
                    "id": k,
                    "qid": r["qid"],
                    "gold": r["gold"],
                    "structured_action": answer["action"],
                    "joint": joint,
                    "formal_prediction": predictions[k]["diagnosis"],
                    "formal_correct": native_correct,
                }
            )
    out = base / "规模配对"
    out.mkdir(exist_ok=True)
    (out / "逐题对照.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in by_case),
        encoding="utf-8",
    )
    pairs = {}
    for method in ("PA", "教师 SFT", "内容蒸馏"):
        small, large = (results[f"{method} · {n}"] for n in (72, 373))
        pairs[method] = {k: pair(small[k], large[k], clusters) for k in small}
    (out / "同题规模差异.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def load_scores(folder, model):
        return {
            r["_key"]: r["content_scores"]
            for r in rows(folder / "official_dk" / (model + ".metrics.jsonl"))
        }

    target = load_scores(base / "scale", "v3_pa")
    formal = {}
    for tag, folder, arm in [
        ("gold_sft", base, "gold_sft"),
        ("pa72", base, "v3_pa"),
        ("teacher373", base / "scale", "v3_teacher_sft"),
    ]:
        if not (folder / "official_dk" / (arm + ".metrics.jsonl")).exists():
            continue
        scores = load_scores(folder, arm)
        common = scores.keys() & target.keys()
        valid = {
            k
            for k in common
            if all(
                isinstance(collection[k], dict)
                and {*DIMENSIONS, "overall_score"} <= collection[k].keys()
                for collection in (scores, target)
            )
        }
        ids = [r["id"] for r in ref if r["id"] in valid]
        qids = [r["qid"] for r in ref if r["id"] in valid]
        formal[tag] = {
            "n": len(ids),
            "paired_rows": len(common),
            "complete": len(ids) == len(keys),
            "dimensions": {},
        }
        if not ids:
            continue
        for dim in (*DIMENSIONS, "overall_score", "four_mean"):

            def vals(records, dim=dim, ids=ids):
                return [
                    sum(records[k][j] for j in DIMENSIONS) / 4
                    if dim == "four_mean"
                    else records[k][dim]
                    for k in ids
                ]

            a, b = vals(scores), vals(target)
            comparison = {
                "baseline_mean": sum(a) / len(a),
                "pa373_mean": sum(b) / len(b),
            }
            if len(ids) == len(keys):
                comparison.update(pair(a, b, qids))
            formal[tag]["dimensions"][dim] = comparison
    (out / "教学同题差异.json").write_text(
        json.dumps(formal, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    font = next(
        (
            n
            for n in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"]
            if n in font_manager.get_font_names()
        ),
        None,
    )
    if font:
        plt.rcParams["font.family"] = font
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, metric, title, lim in zip(
        axes,
        ("joint", "formal_diagnosis"),
        ("结构化协议 · 诊断与动作同时命中", "LongTutor 官方生成 · 诊断正确"),
        (42, 90),
        strict=True,
    ):
        for i, (label, entries) in enumerate(results.items()):
            count = sum(entries[metric])
            color = (
                "#247471"
                if label.startswith("PA")
                else "#7994aa"
                if "373" in label
                else "#adb8c1"
            )
            ax.barh(i, count, height=0.58, color=color)
            ax.text(count + 0.6, i, str(count), va="center", fontsize=10)
        ax.set(
            yticks=range(len(results)),
            yticklabels=list(results),
            xlim=(0, lim),
            xlabel="答对题数 / 140",
        )
        ax.invert_yaxis()
        ax.set_title(title, fontsize=13, loc="left", pad=12)
        ax.grid(axis="x", alpha=0.16)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.8, bottom=0.2, wspace=0.52)
    fig.suptitle(
        "增加训练样本后，两种评测给出了什么结果？",
        x=0.04,
        y=0.95,
        ha="left",
        fontsize=18,
    )
    fig.text(
        0.04,
        0.035,
        "同一固定开发集，训练均从原 Gold SFT 出发。72 题按四类均衡抽样，373 题使用全部合格样本；规模与类别比例同时改变。",
        fontsize=10,
    )
    fig.savefig(out / "开题答辩实验对照.png", dpi=180)
    plt.close(fig)
    scores = formal["gold_sft"]["dimensions"]
    items = [
        ("four_mean", "四维均分"),
        ("overall_score", "独立总体评分"),
        ("history_score", "历史运用"),
        ("strategy_score", "策略适切"),
        ("coherence_score", "连贯性"),
        ("comprehension_score", "易懂性"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, (key, label) in enumerate(items):
        r = scores[key]
        x = r["diff"]
        lo, hi = r["ci95"]
        ax.errorbar(
            x,
            i,
            xerr=[[x - lo], [hi - x]],
            fmt="o",
            capsize=4,
            color="#206966",
            markersize=7,
        )
        ax.text(0.475, i, f"{x:+.3f}", va="center", fontsize=10)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
    ax.set(
        yticks=range(len(items)),
        yticklabels=[v for k, v in items],
        xlim=(-0.15, 0.59),
        xlabel="373 题 PA 减原 Gold SFT（评分差）",
    )
    ax.invert_yaxis()
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", alpha=0.16)
    fig.subplots_adjust(left=0.20, right=0.96, top=0.81, bottom=0.23)
    fig.suptitle("教学评分的改善主要来自历史运用", x=0.06, ha="left", fontsize=18)
    fig.text(
        0.06,
        0.045,
        "固定开发集 140 题、92 个当前题号。线段为按题号整簇重抽样的逐指标 95% 区间。\n四维均分由四项评分平均得出，独立总体评分由同一 DK 裁判另行给出。",
        fontsize=9,
    )
    fig.savefig(out / "373题教学评分差.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
