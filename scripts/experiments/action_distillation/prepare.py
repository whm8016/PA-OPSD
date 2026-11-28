"""Freeze public prompts, teacher-only context and references from the fixed Gold split."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from scripts.build_pa_opsd_gold_split import question_group
from scripts.experiments.action_pilot import run as gold
from scripts.experiments.teaching_skills.run import CARDS, route
from scripts.experiments.training_free_memory.prepare import sample_inputs

SYSTEM = (
    "你是数学导师。根据实际题面和历史正误选择一条历史依据、一个诊断和一个教学动作，再给学生一句能直接执行的提示。"
    "证据用序号，0 表示没有依据。诊断 R=回忆失败，C=概念缺口，P=步骤错误，T=迁移不足。"
    + "动作："
    + "；".join(k + "=" + v[0] for k, v in gold.ACTIONS.items())
    + "。勿猜测学生未展示的解法，不给最终答案。"
)
TOPICS = (
    ("分数", "fraction"),
    ("小数", "decimal"),
    ("百分", "percentage"),
    ("方程", "equation"),
    ("周长", "perimeter"),
    ("面积", "area"),
    ("体积", "volume"),
    ("角度", "angle"),
    ("平均", "mean"),
    ("比例", "ratio"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    split = gold.read_json(gold.SPLIT / "split_manifest.json")["ids"]
    train = {
        r["sample_id"]: r["teacher_privileged"]["gold_reference"]
        for r in gold.read_rows(gold.SPLIT / "development/train.jsonl")
    }
    dev = {
        r["sample_id"]: r["target"]
        for r in gold.read_rows(
            gold.ROOT
            / "outputs/pa_opsd/gold_evidence_first/inputs_v1/dev_references.jsonl"
        )
    }
    assert set(train) == set(split["train"]) and set(dev) == set(split["validation"])
    bank = gold.indexed(gold.read_rows(gold.BANK), "id")
    eedi = gold.ROOT / "data/eedi/misconceptions-graph/data/misconception-map.csv"
    with eedi.open(encoding="cp1252") as stream:
        concepts = list(csv.DictReader(stream))
    result = {"train": [], "dev": []}
    for rec in gold.read_rows(gold.REPRO / "gold_inputs.jsonl"):
        key = gold.history_key(rec)
        if key not in train and key not in dev:
            continue
        ref = train[key] if key in train else dev[key]
        group = "train" if key in train else "dev"
        curr = rec["question_info"]
        qid, stem = question_group(curr)
        info = bank[qid]["info"]
        _, retrieval = sample_inputs(dict(rec, _key=key), {"memory": []})
        pos = retrieval["temporal_positions"][-3:]
        evidence = [rec["history_info"][i - 1] for i in pos]
        card = route(curr, rec["history_info"])["card_id"]
        topic = next((en for zh, en in TOPICS if zh in curr.split("知识点:")[-1]), None)
        candidate = next(
            (c for c in concepts if topic and topic in c["ConstructName"].lower()), None
        )
        visible = (
            "\n".join(f"{i + 1}. {v}" for i, v in enumerate(evidence))
            + "\n当前题\n"
            + curr
        )
        result[group].append(
            {
                "id": key,
                "uid": rec["uid"],
                "qid": qid,
                "stem": stem,
                "system": SYSTEM,
                "public": visible,
                "evidence_positions": pos,
                "gold": ref["diagnosis"],
                "compatible": list(gold.STRATEGY_ACTIONS[ref["strategy"]]),
                "reference": ref["content"],
                "solution": str(info.get("answer") or "")
                + "\n"
                + str(info.get("analysis") or ""),
                "misconception": (candidate["MisconceptionName"] if candidate else ""),
                "misconception_id": (
                    candidate["MisconceptionId"] if candidate else None
                ),
                "rubric": "先定位当前题的一项关系或中间量。只用可核查的事实，保留学生需要完成的推导。",
                "card_id": card,
                "card": CARDS[card]["do"] if card else "",
                "image": bool(gold.IMAGE.search(curr)),
            }
        )
    for group, rows in result.items():
        rows.sort(
            key=lambda r: gold.sha256(("paopsd-v3-01|" + r["id"]).encode()).digest()
        )
        gold.write_rows(out / (group + ".jsonl"), rows)
    # Interleave diagnoses to retain rare categories before the fixed equal-budget cap.
    pools = {
        k: [
            r
            for r in result["train"]
            if r["gold"] == k and not r["image"] and r["solution"].strip()
        ]
        for k in gold.LABELS
    }
    order = [
        pool[i]["id"]
        for i in range(max(map(len, pools.values())))
        for pool in pools.values()
        if i < len(pool)
    ]
    gold.write_json(
        out / "protocol.json",
        {
            "train_ids": order[:72],
            "cohorts": {k: len(v) for k, v in result.items()},
            "train_available": {k: len(v) for k, v in pools.items()},
            "dev_classes": dict(Counter(r["gold"] for r in result["dev"])),
            "seed": 20261127,
            "sealed_used": False,
            "sha256": {g: gold.digest(out / (g + ".jsonl")) for g in result},
            "eedi_sha256": gold.digest(eedi),
            "budget": "72 on-policy examples per primary arm, 36 AdamW updates, LR 1e-5, temperature 1 in losses; greedy structured rollouts",
        },
    )


if __name__ == "__main__":
    main()
