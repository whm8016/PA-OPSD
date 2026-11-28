"""Resolve auditable factual questions against two visible scopes, with abstention."""

import argparse
import json
import re
from pathlib import Path

from scripts.experiments.diagnoser_evidence.evidence import fields

from .budget_curve import BUNDLE, MEMORY, ROOT, load_index
from .prepare import EXPLICIT_ID

FOLLOW = (
    ROOT / "outputs/pa_opsd/training_free_memory/followup_gold140_20261125_verified"
)
DATE = re.compile(r"(20\d\d)[-年](\d{1,2})[-月](\d{1,2})日?")
QUOTES = re.compile(r"[“‘\"]([^”’\"]+)[”’\"]")


def execute(question, task, events):
    ids = {a or b for a, b in EXPLICIT_ID.findall(question)}
    dates = {
        f"{int(y):04}-{int(m):02}-{int(d):02}" for y, m, d in DATE.findall(question)
    }
    if len(dates) > 1 or re.search(
        r"至|之前|最近|最后|距离|期间|时刻|\d[:：]\d|除了", question
    ):
        return None
    if re.search(r"月|年|日", DATE.sub("", question)) and task == 1:
        return None
    concepts = set(QUOTES.findall(question)) - {"正确", "错误"}
    if task == 0:
        if len(ids) != 1 or re.search(r"题干|题目原文|并写|并列|分别写", question):
            return None
        if re.search(r"知识点|考点", question) and re.search(
            r"哪些|是什么|分别是|列出|写出|有哪些", question
        ):
            operator = "concepts"
        elif re.search(r"作答结果|答题结果|回答结果", question) and re.search(
            r"正确还是错误|是什么|是正确|是错误", question
        ):
            operator = "result"
        else:
            return None
        concepts = set()
    else:
        if re.search(
            r"正确率|准确率|分别|时间戳|对应|以及|其中|按顺序写|做错的是|唯一|出现|作答结果|答题结果分别|及结果|并说明",
            question,
        ):
            return None
        count = bool(re.search(r"多少|几[道次题条个]|总数|数量", question))
        if count and re.search(r"列出|写出|说明", question):
            return None
        listing = bool(
            re.search(
                r"(?:ID|编号|题号|id).{0,6}(?:有哪些|是什么|哪一个)|列出.{0,15}(?:ID|编号|题号)|哪些.{0,5}(?:ID|编号|题号)",
                question,
            )
        )
        if count == listing or not (ids or dates or concepts):
            return None
        operator = "count" if count else "ids"
    good = bool(re.search(r"正确|答对|做对", question))
    bad = bool(re.search(r"错误|答错|做错", question))
    if task == 1 and good and bad:
        return None
    result_filter = (
        ("正确" if good else "错误") if task == 1 and (good or bad) else None
    )
    selected = []
    for pos, event in events:
        if pos == 0 and event["qid"] not in ids:
            continue
        if ids and event["qid"] not in ids:
            continue
        if dates and event["date"].date().isoformat() not in dates:
            continue
        if concepts:
            tags = set(event["concepts"])
            if not (bool(concepts & tags) if "或" in question else concepts <= tags):
                continue
        if result_filter and event["result"] != result_filter:
            continue
        selected.append((pos, event))
    if not selected:
        return None
    matches = [event for _, event in selected]
    if operator in ("result", "concepts"):
        values = {
            ", ".join(e["concepts"]) if operator == "concepts" else e["result"]
            for e in matches
        }
        if len(values) != 1:
            return None
        answer = values.pop()
    elif operator == "count":
        answer = str(len(matches))
    else:
        answer = ", ".join("[" + e["qid"] + "]" for e in matches)
    return {
        "operator": operator,
        "answer": answer,
        "positions": [p for p, _ in selected],
    }


def predict(output):
    full = load_index(BUNDLE / "inputs.jsonl")
    queries = load_index(MEMORY / "queries_only.jsonl")
    selected = load_index(MEMORY / "routing.jsonl")
    fallbacks = load_index(
        FOLLOW / "autodl_4090_49862_20261126/runs/followup_full/temporal.jsonl"
    )
    if not (
        len(full) == 140
        and full.keys() == queries.keys() == selected.keys() == fallbacks.keys()
    ):
        raise ValueError("Mismatched dev cases")
    with output.open("x", encoding="utf-8") as out:
        for key, sample in full.items():
            events = [(i, fields(s)) for i, s in enumerate(sample["history_info"], 1)]
            events.append((0, fields(sample["question_info"])))
            keep = set(selected[key]["temporal_positions"]) | {0}
            brief = [(i, e) for i, e in events if i in keep]
            for task, item in enumerate(queries[key]["memory"][:2]):
                answers = [execute(item["query"], task, x) for x in (events, brief)]
                baseline = fallbacks[key]["memory"]
                original = baseline[task]["answer"] if len(baseline) > task else ""
                agree = bool(all(answers) and answers[0] == answers[1])
                row = {
                    "key": key,
                    "task": task + 1,
                    "question": item["query"],
                    "fallback": original,
                    "full": answers[0],
                    "brief": answers[1],
                    "agree": agree,
                    "answer": answers[0]["answer"] if agree else original,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    predict(parser.parse_args().output)
