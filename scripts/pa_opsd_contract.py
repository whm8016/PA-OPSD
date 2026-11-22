"""LongTutor-native contract shared by offline synthesis and student training."""
from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "data/longtutor/repo/scripts"
XES = ROOT / "data/longtutor/repo/data/XES3G5M"
CONTRACT_VERSION = "pa_opsd_long_native_v1"
STRICT_OUTPUT_PROMPT_VERSION = "pa_opsd_strict_json_v1"
PROMPT_CONFIG = {"history_mode": "long", "use_feature": False, "use_teach": False}
DEFAULT_ACTION = {
    "Recall Failure": "probe", "Conceptual Gap": "correct",
    "Procedural Error": "hint", "Transfer Deficit": "example",
}

_OUTPUT_FORMAT_HEADER = "\n### OUTPUT FORMAT\n"
_STRICT_OUTPUT_FORMAT = (
    'Return exactly one valid JSON object with the five fields below. '
    'Do not wrap it in Markdown or add any text outside the object.\n'
    '{\n'
    '  "memory": [{"id":"Q1","answer":"<answer Q1>"}, '
    '{"id":"Q2","answer":"<answer Q2>"}, {"id":"Q3","answer":"<answer Q3>"}],\n'
    '  "diagnosis": "<one exact English diagnosis from the schema>",\n'
    '  "reason": "<concise justification citing specific history or features in Simplified Chinese>",\n'
    '  "strategy": "<exact English strategy mapped to the chosen diagnosis>",\n'
    '  "content": "<concise tutoring feedback in Simplified Chinese>"\n'
    '}\n'
    'Replace the placeholders with your own responses. Always include Q1, Q2, and Q3 in that order. '
    'Each item has only id and answer. Write answers in Simplified Chinese when applicable. '
    'Use "Unknown" if the answer cannot be established. '
    'Do not leave any answer, reason, or content empty. Close the JSON object before ending.\n'
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def official_module():
    sys.path.insert(0, str(OFFICIAL))
    import eval_ai_tutor
    return eval_ai_tutor


def memory_queries(seed):
    memory = seed["teacher_privileged"]["longtutor_weak_annotation"]["memory"]
    if not isinstance(memory, list) or len(memory) != 3:
        raise ValueError("expected_three_memory_queries")
    queries = [item.get("query") for item in memory]
    if any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("missing_memory_query")
    return [q.strip() for q in queries]


def student_messages(seed):
    visible = seed["student_input"]
    sample = {
        "question_info": visible["current_question"],
        "history_info": visible["history_info"],
        "related_history": visible["related_history"],
        "features": visible.get("student_visible_statistics", {}),
    }
    tests = {"memory": [{"query": q} for q in memory_queries(seed)]}
    system, user = official_module()._build_prompts(sample, tests, **PROMPT_CONFIG)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def strict_output_messages(messages):
    """Replace the native output example on a copy of verified student messages."""
    if [message["role"] for message in messages] != ["system", "user"]:
        raise ValueError("Strict output prompt requires the native system and user messages")
    prefix, marker, original = messages[0]["content"].partition(_OUTPUT_FORMAT_HEADER)
    if not marker or not original.rstrip().endswith("Make outputs concise."):
        raise ValueError("Strict output prompt requires the original LongTutor output format section")
    return [
        {**messages[0], "content": prefix + marker + _STRICT_OUTPUT_FORMAT},
        dict(messages[1]),
    ]


def validate_target(target, queries):
    required = {"memory", "diagnosis", "reason", "strategy", "content"}
    if not isinstance(target, dict) or set(target) != required:
        raise ValueError("target_must_have_exactly_five_native_fields")
    err = official_module()._validate_output(target, queries)
    if err:
        raise ValueError(err)
    if len(target["memory"]) != len(queries):
        raise ValueError("incomplete_memory")
    for index, item in enumerate(target["memory"], 1):
        if set(item) != {"id", "answer"} or item["id"] != f"Q{index}":
            raise ValueError("memory_id_or_order")
        if not isinstance(item["answer"], str) or not item["answer"].strip():
            raise ValueError("empty_memory_answer")
    if any(not target[k].strip() for k in ("reason", "content")):
        raise ValueError("empty_reason_or_content")


def contract_metadata():
    return {
        "version": CONTRACT_VERSION, "prompt_config": PROMPT_CONFIG,
        "official_prompt_source_sha256": hashlib.sha256((OFFICIAL / "eval_ai_tutor.py").read_bytes()).hexdigest(),
        "upstream_commit": "ba40af3", "patch": "preserve_strategy_enumeration_order",
    }


def audit_seeds(rows):
    """Recheck membership against official non-Gold inputs, not seed claims."""
    gold = read_jsonl(XES / "human_an_updated.jsonl")
    full = read_jsonl(XES / "history_features_lastq.jsonl")
    non_gold = read_jsonl(XES / "history_features_lastq_scale.jsonl")
    annotations = read_jsonl(XES / "pipeline_an_scale.jsonl")
    def key(row):
        digest = hashlib.sha256(str(row["question_info"]).encode()).hexdigest()
        return f"{row['uid']}||{digest}"
    def question(row):
        import re
        match = re.search(r"\[(\d+)\]", row["question_info"])
        if not match:
            raise ValueError("missing_question_id")
        return match.group(1)
    gold_keys = {r["_key"] for r in gold}
    gold_hist = [r for r in full if key(r) in gold_keys]
    if len(gold) != 1000 or len(gold_hist) != 1000:
        raise ValueError("gold_alignment")
    gold_uids = {str(r["uid"]) for r in gold_hist}
    gold_questions = {question(r) for r in gold_hist}
    official = {key(r): r for r in non_gold}
    annotation_map = {r["_key"]: r for r in annotations}
    seen = set()
    groups = {"train": {"uids": set(), "questions": set()}, "validation": {"uids": set(), "questions": set()}}
    for row in rows:
        k = row["longtutor_key"]
        if k in seen or k not in official or k in gold_keys:
            raise ValueError("duplicate_or_non_training_key")
        seen.add(k)
        history = official[k]
        uid, qid = str(history["uid"]), question(history)
        if uid in gold_uids or qid in gold_questions:
            raise ValueError("gold_uid_or_question_overlap")
        if row["split"] not in groups:
            raise ValueError("unknown_split")
        expected = {
            "current_question": history["question_info"], "history_info": history.get("history_info", []),
            "related_history": history.get("related_history", []),
            "student_visible_statistics": history.get("features", {}),
        }
        if any(row["student_input"].get(k) != v for k, v in expected.items()):
            raise ValueError("student_input_not_official")
        if str(row["student_input"]["uid"]) != uid or str(row["student_input"]["question_id"]) != qid:
            raise ValueError("seed_identity_mismatch")
        expected_queries = [m["query"].strip() for m in annotation_map[k]["memory"]]
        if memory_queries(row) != expected_queries:
            raise ValueError("memory_queries_not_official")
        groups[row["split"]]["uids"].add(uid)
        groups[row["split"]]["questions"].add(qid)
    for field in ("uids", "questions"):
        if groups["train"][field] & groups["validation"][field]:
            raise ValueError(f"train_validation_{field}_overlap")
    return {"rows": len(rows), "gold_key_overlap": 0, "gold_uid_overlap": 0,
            "gold_question_overlap": 0, "train_validation_uid_overlap": 0,
            "train_validation_question_overlap": 0, "contract": contract_metadata()}
