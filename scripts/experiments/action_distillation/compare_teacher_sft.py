"""Summarize the completed 373-example comparison on all fixed development cases."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from ..training_free_memory.prepare import digest, write_json, write_rows
from .analyze import DIMENSIONS, LABELS, pair, rows
from .judge import CacheAwareMeter

ARMS = ("v3_teacher_sft", "v3_pa")


def indexed(path, field, expected):
    data = rows(path)
    by_id = {row[field]: row for row in data}
    if len(data) != len(expected) or by_id.keys() != expected:
        raise ValueError(f"Missing, duplicate or unexpected IDs: {path}")
    return by_id


def run(root):
    scale = root / "scale"
    out = scale / "teacher_sft_completion"
    ref = rows(root / "dev.jsonl")
    ids = {r["id"] for r in ref}
    if len(ref) != 140 or len(ids) != 140:
        raise ValueError("Expected 140 unique development cases")
    detail = {r["id"]: {"id": r["id"], "qid": r["qid"], "gold": r["gold"]} for r in ref}
    sources = {}
    actions = {}
    for arm, grades in zip(ARMS, (out, scale / "official_dk"), strict=True):
        files = [
            grades / f"{arm}.metrics.jsonl",
            scale / "evaluation" / f"{arm}.jsonl",
            scale / "official" / f"{arm}.full.jsonl",
        ]
        scores, structured, predictions = [
            indexed(p, field, ids)
            for p, field in zip(files, ("_key", "id", "_key"), strict=True)
        ]
        sources.update({str(p.relative_to(root)): digest(p) for p in files})
        actions[arm] = dict(Counter(v["action"] for v in structured.values()))
        for r in ref:
            key = r["id"]
            scored = scores[key]
            values = scored["content_scores"] or {}
            act = structured[key]["action"]
            predicted = LABELS["RCPT".index(structured[key]["diagnosis"])]
            detail[key][arm] = {
                **values,
                "four_mean": mean(values[d] for d in DIMENSIONS) if values else None,
                "diagnosis_correct": int(scored["diagnosis_correct"]),
                **{f"Q{i}": int(b) for i, b in enumerate(scored["memory_correct"], 1)},
                "structured_joint": int(
                    predicted == r["gold"] and act in r["compatible"]
                ),
                "structured_action_compatible": int(act in r["compatible"]),
                "structured_action": act,
                "structured_diagnosis": predicted,
                "official_content": predictions[key]["content"],
                "judge_reason": scored["content_reason"],
            }
    summary = {}
    for field in (
        "four_mean",
        *DIMENSIONS,
        "overall_score",
        "diagnosis_correct",
        "Q1",
        "Q2",
        "Q3",
        "structured_joint",
        "structured_action_compatible",
    ):
        common = [
            r for r in detail.values() if all(r[a].get(field) is not None for a in ARMS)
        ]
        if not common:
            summary[field] = {"paired_n": 0}
            continue
        sft, opsd = [[r[arm][field] for r in common] for arm in ARMS]
        comparison = pair(sft, opsd, [r["qid"] for r in common])
        summary[field] = {
            "paired_n": len(common),
            "teacher_sft": mean(sft),
            "opsd": mean(opsd),
            **comparison,
            "ties": len(sft) - comparison["wins"] - comparison["losses"],
        }
    ledger = out / (ARMS[0] + ".usage.jsonl")
    events = rows(ledger)
    meter = CacheAwareMeter(ledger)
    write_rows(out / "paired_by_id.jsonl", detail.values())
    write_json(
        out / "paired_summary.json",
        {
            "n": len(ref),
            "sources_sha256": sources,
            "paired": summary,
            "actions": actions,
            "cost": {
                "confirmed_calls": sum(e["type"] == "settle" for e in events),
                "unconfirmed_calls": len(meter.pending),
                "yuan_estimate_confirmed": sum(
                    e.get("estimate_microyuan", 0) for e in events
                )
                / 1e6,
                "yuan_upper_including_pending": (
                    meter.spent + sum(meter.pending.values())
                )
                / 1e6,
            },
        },
    )
    print(json.dumps(summary["four_mean"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    run(parser.parse_args().folder)
