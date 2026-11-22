"""Freeze LongTutor Gold 560/140/300 without sharing students or current questions."""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from pa_opsd_contract import (
    PROMPT_CONFIG,
    XES,
    fingerprint,
    official_module,
    read_jsonl,
    validate_target,
)

SIZES = {"train": 560, "validation": 140, "test": 300}
SEED = 42


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def question_group(question):
    header, text = question.split("\n", 1)
    qid = re.fullmatch(r"\[(\d+)\]", header).group(1)
    stem = re.split(r"\n(?:知识点|时间|结果):", text, maxsplit=1)[0]
    # Image placeholders identify different images, so preserve them in the comparison.
    normalized = "".join(unicodedata.normalize("NFKC", stem).split())
    return qid, fingerprint(normalized)


def source_rows():
    official = official_module()
    gold = read_jsonl(XES / "human_an_updated.jsonl")
    history = read_jsonl(XES / "history_features_lastq.jsonl")
    keys = {annotation["_key"] for annotation in gold}
    from gpt_memory_diagnose import _sample_key

    aligned = {}
    for sample in history:
        key = _sample_key(sample)
        if key in keys:
            if key in aligned:
                raise ValueError("Duplicate Gold history key")
            aligned[key] = sample
    if len(aligned) != len(keys) or len(keys) != len(gold) or len(gold) != sum(SIZES.values()):
        raise ValueError("Gold and history do not align one-to-one")
    records = []
    for annotation in gold:
        key = annotation["_key"]
        sample = aligned[key]
        queries = [item["query"] for item in annotation["memory"]]
        target = {field: annotation[field] for field in ("diagnosis", "reason", "strategy", "content")}
        target["memory"] = [{"id": f"Q{i}", "answer": item["answer"]}
                            for i, item in enumerate(annotation["memory"], 1)]
        validate_target(target, queries)
        # Only the benchmark's questions are shared with the student.
        test = {"memory": [{"query": query} for query in queries]}
        system, user = official._build_prompts(sample, test, **PROMPT_CONFIG)
        qid, stem_hash = question_group(sample["question_info"])
        records.append({
            "sample_id": key, "uid": str(sample["uid"]), "qid": qid, "stem_hash": stem_hash,
            "history_hash": fingerprint(sample["history_info"]) if sample["history_info"] else None,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "target": target,
        })
    return records


def components(records):
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(records):
        for field in ("uid", "qid", "stem_hash", "history_hash"):
            if row[field] is not None:
                first = seen.setdefault((field, row[field]), index)
                parents[root(index)] = root(first)
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[root(index)].append(row)
    return list(groups.values())


def stratified_split(records):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    rng = random.Random(SEED)
    groups = components(records)
    diagnoses = sorted({row["target"]["diagnosis"] for row in records})
    profiles = [Counter(row["target"]["diagnosis"] for row in group) for group in groups]
    totals = Counter(row["target"]["diagnosis"] for row in records)
    splits = tuple(SIZES)
    width = len(groups)
    n = width * len(splits)
    matrix = np.zeros((width + len(splits) * (1 + len(diagnoses)), n))
    lower = [1] * width
    upper = [1] * width
    for j, split in enumerate(splits):
        cols = slice(j * width, (j + 1) * width)
        matrix[:width, cols] = np.eye(width)
        matrix[len(lower), cols] = [len(group) for group in groups]
        lower.append(SIZES[split])
        upper.append(SIZES[split])
        for diagnosis in diagnoses:
            matrix[len(lower), cols] = [profile[diagnosis] for profile in profiles]
            desired = totals[diagnosis] * SIZES[split] / len(records)
            lower.append(np.floor(desired))
            upper.append(np.ceil(desired))
    objective = [rng.random() for _ in range(n)]
    result = milp(objective, integrality=np.ones(n), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper), options={"time_limit": 60, "mip_rel_gap": 0.02})
    if result.x is None:
        raise ValueError(f"Could not find a stratified group split: {result.message}")
    assignment = np.rint(result.x)
    observed = matrix @ assignment
    if np.any(observed < lower) or np.any(observed > upper):
        raise ValueError("Solver did not return a valid integer assignment")
    output = {split: [row for i, group in enumerate(groups) if assignment[j * width + i] for row in group]
              for j, split in enumerate(splits)}
    for rows in output.values():
        rng.shuffle(rows)
    return output


def training_row(record):
    return {
        "sample_id": record["sample_id"], "split": "train", "status": "accepted", "dry_run": False,
        "messages": [*record["messages"], {"role": "assistant",
                                          "content": json.dumps(record["target"], ensure_ascii=False)}],
        "teacher_privileged": {"gold_reference": record["target"]},
    }


def export(splits, out):
    development, sealed = out / "development", out / "sealed"
    development.mkdir(parents=True, exist_ok=False)
    sealed.mkdir()
    manifest = {"seed": SEED, "sizes": SIZES, "diagnosis": {}, "ids": {}, "split_overlap": {}, "files": {}}
    from hashlib import sha256

    for split, rows in splits.items():
        manifest["diagnosis"][split] = dict(Counter(row["target"]["diagnosis"] for row in rows))
        manifest["ids"][split] = [row["sample_id"] for row in rows]
        folder = sealed if split == "test" else development
        if split == "train":
            files = {"train.jsonl": [training_row(row) for row in rows]}
        else:
            # Internally the shared inference helper uses "validation" for all evaluation prompts.
            files = {"prompts.jsonl": [{"sample_id": r["sample_id"], "messages": r["messages"],
                                        "split": "validation"} for r in rows],
                     "references.jsonl": [{"sample_id": r["sample_id"], "target": r["target"]} for r in rows]}
        for name, payload in files.items():
            path = folder / name
            write_rows(path, payload)
            manifest["files"][f"{folder.name}/{name}"] = sha256(path.read_bytes()).hexdigest()
    for field in ("uid", "qid", "stem_hash", "history_hash"):
        groups = [{r[field] for r in rows if r[field] is not None} for rows in splits.values()]
        overlap = sum(len(left & right) for i, left in enumerate(groups) for right in groups[i + 1:])
        if overlap:
            raise ValueError(f"Split is not disjoint on {field}")
        manifest["split_overlap"][field] = overlap
    write_json(out / "split_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"ids", "files"}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    export(stratified_split(source_rows()), parser.parse_args().output)
