"""Build an auditable PA-OPSD pilot from reviewed queries over official student inputs."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from generate_pa_opsd_teacher import evidence_issues, history_evidence, memory_facts
from pa_opsd_contract import (
    DEFAULT_ACTION,
    audit_seeds,
    fingerprint,
    memory_queries,
    read_jsonl,
    student_messages,
    validate_target,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data/pa_opsd/v0.2_question_disjoint_weak_seed"
STRATEGIES = {
    "Recall Failure": "Retrieval Practice",
    "Conceptual Gap": "Conceptual Explanation",
    "Procedural Error": "Stepwise Scaffolding",
    "Transfer Deficit": "Analogical Transfer",
}


def visible_records(visible):
    history = history_evidence(visible)
    current = visible.split("### CURRENT QUESTION", 1)[1].split("### MEMORY QUERIES", 1)[0].strip()
    current_id = re.match(r"\[(\d+)\]", current)
    if not current_id:
        raise ValueError("Missing current question ID")
    records = []
    for record in history + [{"history_index": "current", "question_id": current_id[1], "text": current}]:
        fields = re.findall(r"^(知识点|时间|结果):[ \t]*(.+)$", record["text"], re.MULTILINE)
        parsed = dict(fields)
        if len(fields) != 3 or set(parsed) != {"知识点", "时间", "结果"}:
            raise ValueError(f"Incomplete metadata at {record['history_index']}")
        if parsed["结果"] not in {"正确", "错误"}:
            raise ValueError(f"Invalid outcome at {record['history_index']}")
        datetime.fromisoformat(parsed["时间"])
        records.append({**record, "timestamp": parsed["时间"], "result": parsed["结果"],
                        "tags": parsed["知识点"].split(", ")})
    return records


def in_scope(records, scope):
    if scope not in {"history", "current", "both"}:
        raise ValueError("Invalid record scope")
    if scope == "both":
        return records
    return [r for r in records if (r["history_index"] == "current") == (scope == "current")]


def aggregate_answer(records, plan):
    unknown = plan.keys() - {"kind", "scope", "tags", "ids", "date", "date_range", "result", "order", "distinct", "fields"}
    if unknown:
        raise ValueError(f"Unexpected query fields: {sorted(unknown)}")
    order = plan.get("order", "display")
    result = plan.get("result")
    if order not in {"display", "id", "time"} or result not in {None, "正确", "错误"}:
        raise ValueError("Invalid order or result")
    tags, ids, dates = (plan.get(key, []) for key in ("tags", "ids", "date"))
    if any(not isinstance(v, list) or any(not isinstance(s, str) or not s for s in v) for v in (tags, ids, dates)):
        raise ValueError("Tags, IDs and dates must be lists of nonempty strings")
    # Date terms are exact source dates or timestamps. Multiple terms form a union.
    for date in dates:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)?", date):
            raise ValueError(f"Invalid date or timestamp: {date}")
        datetime.fromisoformat(date)
    date_range = plan.get("date_range")
    if date_range is not None and (
        not isinstance(date_range, list) or len(date_range) != 2
        or any(not isinstance(d, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in date_range)
        or datetime.fromisoformat(date_range[0]) > datetime.fromisoformat(date_range[1])
    ):
        raise ValueError("Invalid inclusive date range")
    if not isinstance(plan.get("distinct", False), bool):
        raise TypeError("Distinct must be boolean")
    matched = [
        row for row in in_scope(records, plan.get("scope", "history"))
        if (not ids or row["question_id"] in ids)
        and (not tags or any(set(term.split("+")).issubset(row["tags"]) for term in tags))
        and (not dates or any(row["timestamp"][:10] == d if len(d) == 10
                              else datetime.fromisoformat(row["timestamp"]) == datetime.fromisoformat(d)
                              for d in dates))
        and (date_range is None or date_range[0] <= row["timestamp"][:10] <= date_range[1])
        and (not result or result == row["result"])
    ]
    if plan.get("distinct", False):
        if set(plan["fields"]) - {"total", "ids"}:
            raise ValueError("Distinct-ID counts require an explicit conflict rule for outcomes or timestamps")
        distinct = {}
        for row in matched:
            distinct.setdefault(row["question_id"], row)
        matched = list(distinct.values())
    if order == "id":
        matched.sort(key=lambda r: int(r["question_id"]))
    if order == "time":
        matched.sort(key=lambda r: datetime.fromisoformat(r["timestamp"]))
    correct = [r for r in matched if r["result"] == "正确"]
    wrong = [r for r in matched if r["result"] == "错误"]

    def listed(rows, suffix=""):
        return "、".join(f"[{row['question_id']}]" + (f" {row[suffix]}" if suffix else "") for row in rows) or "无"

    renderers = {
        "total": lambda: f"共{len(matched)}条",
        "ids": lambda: listed(matched),
        "results": lambda: listed(matched, "result"),
        "times": lambda: listed(matched, "timestamp"),
        "correct_ids": lambda: "正确：" + listed(correct),
        "wrong_ids": lambda: "错误：" + listed(wrong),
        "outcome_counts": lambda: f"正确{len(correct)}条、错误{len(wrong)}条",
        "accuracy": lambda: f"正确率{len(correct)}/{len(matched)}" if matched else "没有匹配记录，正确率不适用",
        "latest": lambda: "最近一次：" + listed(
            [max(enumerate(matched), key=lambda pair: (datetime.fromisoformat(pair[1]["timestamp"]), pair[0]))[1]]
            if matched else [], "timestamp"),
        "last": lambda: "末条：" + listed(matched[-1:], "timestamp"),
    }
    fields = plan["fields"]
    if not isinstance(fields, list) or not fields or any(field not in renderers for field in fields):
        raise ValueError("Invalid output fields")
    return "；".join(renderers[field]() for field in fields) + "。", matched


def memory_answer(records, plan):
    kind = plan["kind"]
    if kind == "aggregate":
        return aggregate_answer(records, plan)
    if kind not in {"quote", "unknown", "absence"}:
        raise ValueError("Unsupported or unresolved query plan")
    allowed = {"kind", "id", "scope", "history_index", "basis", "text", "parts", "absent_literal"}
    if plan.keys() - allowed:
        raise ValueError("Unexpected quote or absence fields")
    matches = [r for r in in_scope(records, plan.get("scope", "history"))
               if r["question_id"] == plan.get("id")]
    if "history_index" in plan:
        matches = [r for r in matches if r["history_index"] == plan["history_index"]]
    if not matches:
        raise ValueError("The reviewed source record is missing")
    if kind == "quote":
        if ("text" in plan) == ("parts" in plan):
            raise ValueError("Specify one quote or a list of quoted parts")
        parts = plan.get("parts", [plan.get("text")])
        if (len(matches) != 1 or not isinstance(parts, list) or not parts
                or any(not isinstance(p, str) or not p or p not in matches[0]["text"] for p in parts)):
            raise ValueError("Quote ambiguous or absent from the specified record")
        return "\n".join(parts), matches
    if not plan.get("basis"):
        raise ValueError("A reviewed absence requires an evidence note")
    if kind == "absence":
        literal = plan.get("absent_literal")
        if not literal or any(literal in record["text"] for record in matches):
            raise ValueError("The reviewed literal is present or unspecified")
        return "没有。", matches
    return "Unknown", matches


def build_candidate(seed, spec):
    if seed["split"] != "train":
        raise ValueError("Only train seeds can enter the pilot")
    if len(spec["memory"]) != 3:
        raise ValueError("Expected plans for all three original queries")
    messages = student_messages(seed)
    queries = memory_queries(seed)
    visible = messages[1]["content"]
    records = visible_records(visible)
    memory, trace = [], []
    for i, (query, reviewed) in enumerate(zip(queries, spec["memory"], strict=True), 1):
        if reviewed["query"] != query:
            raise ValueError(f"Q{i} plan differs from the original query text or order")
        plan = reviewed["plan"]
        named_ids = re.findall(r"\[(\d+)\]", query)
        if plan["kind"] != "aggregate" and named_ids and plan.get("id") not in named_ids:
            raise ValueError(f"Q{i} source record differs from the original question")
        answer, evidence = memory_answer(records, plan)
        memory.append({"id": f"Q{i}", "answer": answer})
        fields = ("history_index", "question_id", "timestamp", "result")
        if plan["kind"] != "aggregate":
            fields += ("text",)
        trace.append({"query": query, "plan": plan, "answer": answer,
                      "records": [{k: r[k] for k in fields} for r in evidence]})
    diagnosis, reason, content, evidence_indexes = spec["teaching"]
    index = {r["history_index"]: r for r in records if r["history_index"] != "current"}
    if not evidence_indexes or not set(evidence_indexes).issubset(index):
        raise ValueError("Diagnosis and teaching require real historical evidence")
    evidence = [index[i] for i in evidence_indexes]
    cited = set(re.findall(r"\[(\d+)\]", reason + "\n" + content))
    if not cited or not cited.issubset(r["question_id"] for r in evidence):
        raise ValueError("Teaching citation missing or outside reviewed history")
    target = {"memory": memory, "diagnosis": diagnosis, "reason": reason,
              "strategy": STRATEGIES[diagnosis], "content": content}
    # The weak annotation provides the original query text. Its answers stay out of the export.
    privileged = {k: v for k, v in seed["teacher_privileged"].items()
                  if k not in {"longtutor_weak_annotation", "action_candidates"}}
    validate_target(target, queries)
    issues = evidence_issues(target, visible, queries, privileged)
    if issues:
        raise ValueError(f"{seed['sample_id']} failed the guard: {issues}")
    candidate = {
        "sample_id": seed["sample_id"], "split": "train", "status": "review_pending", "dry_run": False,
        "messages": messages + [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
        "teacher_privileged": privileged, "action_id": DEFAULT_ACTION[diagnosis],
        "provenance": {"origin": "reviewed_visible_history", "prompt_sha256": fingerprint(messages),
                       "target_sha256": fingerprint(target), "plan_sha256": fingerprint(spec)},
    }
    return candidate, {"sample_id": seed["sample_id"], "memory": trace, "teaching_evidence": evidence}


def write_jsonl(path, rows):
    # A partial rerun cannot replace an already exported dataset.
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def coverage(seeds):
    counts = [{"program_supported": 0, "manual_queue": 0} for _ in range(3)]
    complete = 0
    for seed in seeds:
        facts = memory_facts(student_messages(seed)[1]["content"], memory_queries(seed))
        complete += all(fact["supported"] for fact in facts)
        for values, fact in zip(counts, facts, strict=True):
            values["program_supported" if fact["supported"] else "manual_queue"] += 1
    return {"seed_rows": len(seeds), "questions": 3 * len(seeds), "all_three_supported": complete,
            "by_question": {f"Q{i}": values for i, values in enumerate(counts, 1)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train = read_jsonl(SEEDS / "train.jsonl")
    validation = read_jsonl(SEEDS / "validation.jsonl")
    audit = audit_seeds(train + validation)
    specs = json.loads(args.plans.read_text(encoding="utf-8"))
    ids = [spec["sample_id"] for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate pilot IDs")
    by_id = {seed["sample_id"]: seed for seed in train}
    if not set(ids).issubset(by_id):
        raise ValueError("A plan is not linked to a train seed")
    results = []
    for spec in specs:
        sample_id = spec["sample_id"]
        try:
            results.append(build_candidate(by_id[sample_id], spec))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{sample_id}: {exc}") from exc
    report = {**coverage(train), "pilot_candidate_rows": len(results),
              "pilot_reference_answers": 3 * len(results), "split_audit": audit}
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "candidates.jsonl", (r[0] for r in results))
    write_jsonl(args.out / "evidence.jsonl", (r[1] for r in results))
    (args.out / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
