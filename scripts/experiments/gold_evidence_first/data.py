"""Prepare and verify the development-only Gold training inputs."""

import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path

from pa_opsd_collator import load_standard_solutions, standard_solution_messages
from scripts.build_pa_opsd_gold_split import (
    question_group,
    stratified_split,
    write_json,
    write_rows,
)
from scripts.pa_opsd_contract import (
    XES,
    fingerprint,
    read_jsonl,
    strict_output_messages,
)

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "outputs/pa_opsd/gold_split_20261120"
LEGACY = ("diagnosis", "reason", "strategy", "content", "memory")
MEMORY_FIRST = ("memory", *LEGACY[:-1])
GROUP_FIELDS = ("uid", "qid", "stem_hash", "history_hash")
TRAIN_FILES = {
    "legacy": ("sft_legacy.jsonl", LEGACY),
    "memory_first": ("sft_memory_first.jsonl", MEMORY_FIRST),
    "stage2": ("stage2.jsonl", MEMORY_FIRST),
}
DEV_PROMPTS = "dev_prompts.jsonl"
DEV_REFERENCES = "dev_references.jsonl"
BANK = "standard_solutions.jsonl"
MANIFEST = "manifest.json"


def digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assistant_target(row):
    return json.loads(row["messages"][2]["content"])


def reorder_target(row, order):
    target = assistant_target(row)
    if set(target) != set(order):
        raise ValueError("The JSON target differs from the five-field schema")
    assistant = {**row["messages"][2], "content": json.dumps(
        {key: target[key] for key in order}, ensure_ascii=False)}
    return {**row, "messages": [*row["messages"][:2], assistant]}


def group_metadata(row):
    uid, separator, suffix = row["sample_id"].rpartition("||")
    user = row["messages"][1]["content"]
    history_header = "### INTERACTION HISTORY\n"
    history, marker, tail = user.removeprefix(history_header).rpartition("\n\n### CURRENT QUESTION\n")
    current, queries, _ = tail.partition("\n\n### MEMORY QUERIES\n")
    if (not separator or not uid or not re.fullmatch(r"[0-9a-f]{64}", suffix)
            or not user.startswith(history_header) or not marker or not queries):
        raise ValueError("Cannot recover student and current question from the frozen prompt")
    qid, stem = question_group(current)
    return {"uid": uid, "qid": qid, "stem_hash": stem,
            "history_hash": fingerprint(history) if history else None}


def overlaps(left, right):
    left, right = ([group_metadata(row) for row in rows] for rows in (left, right))
    return {field: len(({row[field] for row in left} & {row[field] for row in right}) - {None})
            for field in GROUP_FIELDS}


def history_question_exposure(bundle):
    first = read_jsonl(bundle / TRAIN_FILES["legacy"][0])
    seen = set()
    for row in first:
        history = row["messages"][1]["content"].rpartition("\n\n### CURRENT QUESTION\n")[0]
        seen.update(re.findall(r"(?m)^\[(\d+)\][ \t]*$", history))
    report = {}
    for arm, path in (("stage2", TRAIN_FILES["stage2"][0]), ("development", DEV_PROMPTS)):
        current = {group_metadata(row)["qid"] for row in read_jsonl(bundle / path)}
        report[arm] = {"unique_current_questions": len(current),
                       "seen_in_sft_history": len(current & seen)}
    return report


def frozen_rows(frozen, manifest, name, split):
    relative = f"development/{name}"
    path = frozen / relative
    if digest(path) != manifest["files"][relative]:
        raise ValueError(f"Frozen file hash mismatch for {relative}")
    rows = read_jsonl(path)
    if [row["sample_id"] for row in rows] != manifest["ids"][split]:
        raise ValueError(f"Frozen IDs differ for {split} {name}")
    return rows


def label_free_prompt(row):
    messages = strict_output_messages(row["messages"][:2])
    return {"sample_id": row["sample_id"], "messages": messages, "split": row["split"]}


def training_row(row, order):
    revised = reorder_target(row, order)
    return {**revised, "messages": [*label_free_prompt(row)["messages"], revised["messages"][2]]}


def prepare(out, frozen=FROZEN, bank=XES / "questions.jsonl"):
    original = read_json(frozen / "split_manifest.json")
    train = frozen_rows(frozen, original, "train.jsonl", "train")
    prompts = frozen_rows(frozen, original, "prompts.jsonl", "validation")
    references = frozen_rows(frozen, original, "references.jsonl", "validation")
    records = [{**row, **group_metadata(row), "target": assistant_target(row)} for row in train]
    split = stratified_split(records, sizes={"sft": 420, "stage2": 140})
    index = {row["sample_id"]: row for row in train}
    first, second = ([index[row["sample_id"]] for row in split[name]] for name in ("sft", "stage2"))
    payloads = {
        "sft_legacy.jsonl": [training_row(row, LEGACY) for row in first],
        "sft_memory_first.jsonl": [training_row(row, MEMORY_FIRST) for row in first],
        "stage2.jsonl": [training_row(row, MEMORY_FIRST) for row in second],
        DEV_PROMPTS: [label_free_prompt(row) for row in prompts],
        DEV_REFERENCES: references,
    }
    solution_rows = second + prompts
    solutions = load_standard_solutions(solution_rows, bank)
    subset = {}
    for row in solution_rows:
        reference = solutions[row["sample_id"]]
        subset[group_metadata(row)["qid"]] = {
            "answer": reference["standard_answer"], "analysis": reference["analysis"]}
    payloads[BANK] = [{"id": qid, "info": info} for qid, info in sorted(subset.items())]
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in payloads.items():
        write_rows(out / name, rows)
    manifest = {
        "version": 1,
        "source_manifest_sha256": digest(frozen / "split_manifest.json"),
        "source_bank_sha256": digest(bank),
        "ids": {name: [row["sample_id"] for row in rows]
                for name, rows in (("sft", first), ("stage2", second), ("dev", prompts))},
        "counts": {name: dict(Counter(assistant_target(row)["diagnosis"] for row in rows))
                   for name, rows in (("sft", first), ("stage2", second))},
        "files": {name: digest(out / name) for name in payloads},
    }
    write_json(out / MANIFEST, manifest)
    return verify(out)


def verify(bundle):
    manifest = read_json(bundle / MANIFEST)
    names = {name for name, _ in TRAIN_FILES.values()} | {DEV_PROMPTS, DEV_REFERENCES, BANK}
    if manifest["version"] != 1 or set(manifest["files"]) != names:
        raise ValueError("Unexpected prepared input manifest")
    for name in names:
        if digest(bundle / name) != manifest["files"][name]:
            raise ValueError(f"Prepared file hash mismatch: {name}")
    train = {}
    for arm, (name, order) in TRAIN_FILES.items():
        rows = train[arm] = read_jsonl(bundle / name)
        group = "stage2" if arm == "stage2" else "sft"
        if [row["sample_id"] for row in rows] != manifest["ids"][group] or any(
                tuple(assistant_target(row)) != order for row in rows):
            raise ValueError(f"Sample IDs or serialized target order differ for {arm}")
        diagnosis = Counter(assistant_target(row)["diagnosis"] for row in rows)
        if diagnosis != manifest["counts"][group]:
            raise ValueError(f"Diagnosis counts differ for {arm}")
    if [reorder_target(row, MEMORY_FIRST) for row in train["legacy"]] != train["memory_first"]:
        raise ValueError("SFT arms must differ only in serialized target order")
    prompts = read_jsonl(bundle / DEV_PROMPTS)
    refs = read_jsonl(bundle / DEV_REFERENCES)
    for rows in (prompts, refs):
        if [row["sample_id"] for row in rows] != manifest["ids"]["dev"]:
            raise ValueError("Development IDs differ")
    partitions = [train["legacy"], train["stage2"], prompts]
    if [len(rows) for rows in partitions] != [420, 140, 140]:
        raise ValueError("Expected 420 SFT, 140 second-stage and 140 development rows")
    ids = [row["sample_id"] for rows in partitions for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated sample IDs between partitions or within a partition")
    for i, rows in enumerate(partitions):
        for other in partitions[i+1:]:
            if any(overlaps(rows, other).values()):
                raise ValueError("Student, question or history crosses experiment partitions")
    expected_keys = {"sample_id", "messages", "split"}
    if any(set(row) != expected_keys or len(row["messages"]) != 2 or row["split"] != "validation"
           for row in prompts):
        raise ValueError("Development inputs must carry only the label-free prompt")
    for row in (item for rows in partitions for item in rows):
        if any("### TEACHER-ONLY REFERENCE SOLUTION" in msg["content"] for msg in row["messages"][:2]):
            raise ValueError("Teacher-only context is present in a student prompt")
    return manifest


def teacher_prompts(bundle):
    rows = read_jsonl(bundle / DEV_PROMPTS)
    solutions = load_standard_solutions(rows, bundle / BANK)
    return [{**row, "messages": standard_solution_messages(
        row["messages"], solutions[row["sample_id"]], include_guidance=False)}
            for row in rows]
