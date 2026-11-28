"""Select teacher-only history using the visible LongTutor prompt."""

import json
import re
from datetime import date

from scripts.experiments.diagnoser_evidence.evidence import build_features, fields


HISTORY = "### INTERACTION HISTORY\n"
CURRENT = "\n\n### CURRENT QUESTION\n"
QUERIES = "\n\n### MEMORY QUERIES\n"
QUESTION_ID = re.compile(r"\[(\d+)\]|(?:题目\s*ID|题号)\s*[：:]?\s*(\d+)", re.I)
DAY = re.compile(r"(20\d{2})(?:年|[-/])(\d{1,2})(?:月|[-/])(\d{1,2})(?:日|号)?")
UNCERTAIN_SCOPE = re.compile(
    r"之间|之前|之后|以前|以后|截至|以来|前后|最早|最近|首次|最后一次|范围内|其他记录|其余记录|"
    r"(?:到|至|—|–|~|～)\s*(?:20\d{2}|\[\d+\])|(?:如|例如|包括|包含)\s*(?:记录|题目)?\s*\[\d+\]|"
    r"\[\d+\]\s*(?:等|及其他|[-–—]\s*\[\d+\])"
)


def memory_positions(history, current, queries):
    """Keep every record matching an explicit ID or day. Unknown scopes use all history."""
    records = [fields(text) for text in history]
    current_id = fields(current)["qid"]
    known_ids = {record["qid"] for record in records} | {current_id}
    selected = set()
    for query in queries:
        ids = {first or second for first, second in QUESTION_ID.findall(query)}
        if UNCERTAIN_SCOPE.search(query) or re.search(r"20\d{2}|\d{1,2}月", DAY.sub("", query)):
            return None
        try:
            days = {date(*map(int, parts)) for parts in DAY.findall(query)}
        except ValueError:
            return None
        if not ids and not days:
            return None
        if not ids <= known_ids:
            return None
        if days and any(record["date"] is None for record in records):
            return None
        selected.update(i for i, record in enumerate(records, 1) if record["qid"] in ids or
                        (record["date"] is not None and record["date"].date() in days))
    return selected


def compact_history(student):
    """Return copied messages and an audit of selected positions; never read targets."""
    prefix, _, tail = student[1]["content"].partition(HISTORY)
    history_text, current_header, tail = tail.partition(CURRENT)
    current, query_header, suffix = tail.partition(QUERIES)
    if not current_header or not query_header:
        raise ValueError("Expected LongTutor history, current-question and memory-query sections")
    history = re.split(r"\n\n(?=\[\d+\]\n)", history_text)
    queries = re.findall(r"^- Q[1-3]: (.+)$", suffix, re.M)
    if len(queries) != 3:
        raise ValueError("Expected three visible memory queries")
    required = memory_positions(history, current, queries)
    if required is None:
        return student, {"fallback": True, "history_records": len(history)}
    features = build_features({"sample_id": None, "current": current, "history": history})
    positions = sorted(required | set(features["history_positions"]))
    if not positions or len(positions) == len(history):
        return student, {"fallback": True, "history_records": len(history)}
    excerpt = "\n\n".join(history[i - 1] for i in positions)
    summary = json.dumps(features["summary"], ensure_ascii=False)
    teacher = [dict(message) for message in student]
    teacher[1]["content"] = (
        f"{prefix}{HISTORY}历史统计（基于完整可见记录）: {summary}\n\n{excerpt}"
        f"{current_header}{current}{query_header}{suffix}"
    )
    return teacher, {"fallback": False, "history_records": len(history), "kept_positions": positions,
                     "memory_positions": sorted(required)}
