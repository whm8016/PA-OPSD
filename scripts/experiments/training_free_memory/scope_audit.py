"""Audit how the original annotation-side related-history slice changes executable Q2 answers."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from .evidence_executor import execute, fields
from .packet_followup import STUDY
from .paired_report import read
from .prepare import ROOT, digest, write_json, write_rows

SOURCE = ROOT / "data/longtutor/repo/data/XES3G5M/history_features_lastq.jsonl"


def audit(out):
    targets = {
        r["key"]: r
        for r in read(STUDY / "transfer560/rows.jsonl")
        if r["task"] == 2 and r["full"]
    }
    if len(targets) != 131:
        raise ValueError("Expected the frozen 131 executable Q2s")
    rng = random.Random(73041)
    results, blind, identity = [], [], []
    with SOURCE.open(encoding="utf8") as stream:
        for line in stream:
            sample = json.loads(line)
            current = sample["question_info"]
            key = (
                str(sample["uid"]) + "||" + hashlib.sha256(current.encode()).hexdigest()
            )
            if key not in targets:
                continue
            question = targets[key]["question"]
            history = sample["history_info"]
            full = [(i, fields(s)) for i, s in enumerate(history, 1)]
            # Match the official slice as an ordered subsequence, preserving repeated attempts.
            iterator = iter(enumerate(history, 1))
            selected = [
                next(i for i, s in iterator if s == record)
                for record in sample["related_history"]
            ]
            positions = set(selected)
            short = [event for event in full if event[0] in positions]
            current_event = (0, fields(current))
            full_result = execute(question, 1, [*full, current_event])
            reduced = execute(question, 1, [*short, current_event])
            if full_result != targets[key]["full"]:
                raise ValueError("Full-scope execution drifted")
            delta = full_result["answer"] != (reduced or {}).get("answer")
            row = {
                "key": key,
                "question": question,
                "reference": targets[key]["reference"],
                "changed": delta,
                "related_scope_positions": selected,
                "full": full_result,
                "related": reduced,
            }
            if delta:
                evidence = set(full_result["positions"]) | set(
                    (reduced or {}).get("positions", [])
                )
                row["support"] = [
                    {
                        "position": i,
                        "id": e["qid"],
                        "date": e["date"].isoformat(),
                        "result": e["result"],
                        "concepts": e["concepts"],
                        "in_related_scope": i in positions,
                    }
                    for i, e in [*full, current_event]
                    if i in evidence
                ]
                order = ["full", "related"]
                rng.shuffle(order)
                rid = f"S{len(blind) + 1:02}"
                blind.append(
                    {
                        "id": rid,
                        "question": question,
                        "reference": row["reference"],
                        "answers": [(row[a] or {}).get("answer", "") for a in order],
                    }
                )
                identity.append({"id": rid, "key": key, "order": order})
            results.append(row)
    if {r["key"] for r in results} != set(targets) or len(results) != len(targets):
        raise ValueError("Incomplete source mapping")
    out.mkdir(exist_ok=False)
    write_rows(out / "rows.jsonl", results)
    write_rows(out / "blind.jsonl", blind)
    write_rows(out / "identity.jsonl", identity)
    write_json(
        out / "protocol.json",
        {
            "total": len(results),
            "different": len(blind),
            "selection": "all 131 executable Q2 from original training pool",
            "change": "same executor, replace historical event collection by the original stored related_history, current-question rule identical",
            "source_sha256": digest(SOURCE),
            "code_sha256": digest(Path(__file__)),
            "analysis": "post hoc scope diagnostic; no sealed-set evaluation",
        },
    )
    print(len(results), len(blind))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=STUDY / "gpu_run1/scope")
    audit(p.parse_args().output)
