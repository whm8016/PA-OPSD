"""Apply frozen executor to a disjoint, existing training pool without model predictions."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from . import evidence_executor as executor
from .prepare import ROOT, SPLIT, sample_inputs


def run(output):
    path = ROOT / "outputs/pa_opsd/gold_split_20261120/development/train.jsonl"
    allowed = set(json.loads(SPLIT.read_text(encoding="utf-8"))["ids"]["train"])
    seen = set()
    output.mkdir(parents=True, exist_ok=False)
    accepted = []
    with (
        path.open(encoding="utf-8") as src,
        (output / "rows.jsonl").open("x", encoding="utf-8") as target,
        (output / "full_q2_audit.jsonl").open("x", encoding="utf-8") as audit_out,
    ):
        for line in src:
            row = json.loads(line)
            key = row["sample_id"]
            if key not in allowed or key in seen:
                raise ValueError("Invalid training case")
            seen.add(key)
            prompt = row["messages"][1]["content"]
            history, tail = prompt.split("### CURRENT QUESTION\n")
            current, tail = tail.split("### MEMORY QUERIES\n")
            records = re.split(
                r"\n\n(?=\[\d+\]\n)",
                history.removeprefix("### INTERACTION HISTORY\n").strip(),
            )
            questions = re.findall(r"^- Q[123]: (.+)$", tail, re.MULTILINE)
            if len(questions) != 3 or len(records) != 199:
                raise ValueError(f"Cannot reconstruct prompt {key}")
            sample = {
                "_key": key,
                "uid": key.split("||")[0],
                "question_info": current.strip(),
                "history_info": records,
            }
            _, audit = sample_inputs(
                sample, {"memory": [{"query": q} for q in questions]}
            )
            events = [(i, executor.fields(r)) for i, r in enumerate(records, 1)] + [
                (0, executor.fields(current.strip()))
            ]
            keep = set(audit["temporal_positions"]) | {0}
            for task, q in enumerate(questions[:2]):
                full = executor.execute(q, task, events)
                brief = executor.execute(
                    q, task, [(i, e) for i, e in events if i in keep]
                )
                agree = bool(full and full == brief)
                # Reference is first accessed after both answers have been produced.
                gold = json.loads(row["messages"][2]["content"])["memory"][task][
                    "answer"
                ]
                result = {
                    "key": key,
                    "task": task + 1,
                    "question": q,
                    "reference": gold,
                    "full": full,
                    "brief": brief,
                    "agree": agree,
                }
                target.write(json.dumps(result, ensure_ascii=False) + "\n")
                if task == 1 and full:
                    review = {
                        "id": 2 * len(seen) - 1,
                        "question": q,
                        "reference": gold,
                        "answer": full["answer"],
                    }
                    audit_out.write(json.dumps(review, ensure_ascii=False) + "\n")
                if agree:
                    accepted.append(result)
    if seen != allowed or len(allowed) != 560:
        raise ValueError("Incomplete train split")
    selected = random.Random(27512).sample(
        [r for r in accepted if r["task"] == 2],
        min(40, sum(r["task"] == 2 for r in accepted)),
    )
    with (output / "audit.jsonl").open("x", encoding="utf-8") as target:
        for i, r in enumerate(selected, 1):
            target.write(
                json.dumps(
                    {
                        "id": i,
                        "question": r["question"],
                        "answer": r["full"]["answer"],
                        "reference": r["reference"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "cases": len(seen),
        "accepted": [sum(r["task"] == t for r in accepted) for t in (1, 2)],
        "audit_seed": 27512,
        "audit_size": len(selected),
        "sample_ids": [r["key"] for r in selected],
        "executor_sha256": hashlib.sha256(
            Path(executor.__file__).read_bytes()
        ).hexdigest(),
        "train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(summary["cases"], summary["accepted"], summary["audit_size"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    run(p.parse_args().output)
