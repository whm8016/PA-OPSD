"""Fill executable memory fields with source-linked answers in a generated Tutor response."""

import argparse
import json
from pathlib import Path

from .budget_curve import load_index
from .evidence_executor import execute, fields
from .packet_followup import public_input
from .prepare import write_rows


def compose(sample, queries, generated):
    if not (sample["_key"] == queries["_key"] == generated["_key"]):
        raise ValueError("Mismatched public input, questions and generated response")
    evidence = []
    # Keep original upstream format failures, including their denominator.
    if not generated.get("diagnosis") or len(generated.get("memory") or []) != 3:
        return generated, evidence
    events = [
        (pos, fields(record)) for pos, record in enumerate(sample["history_info"], 1)
    ]
    events.append((0, fields(sample["question_info"])))
    memory = [dict(item) for item in generated["memory"]]
    for t, query in enumerate(queries["memory"][:2]):
        result = execute(query["query"], t, events)
        if result is not None:
            evidence.append(
                {
                    "task": t + 1,
                    "question": query["query"],
                    "previous": memory[t].get("answer", ""),
                    **result,
                }
            )
            memory[t]["answer"] = result["answer"]
    return dict(generated, memory=memory), evidence


def materialize(predictions, output, inputs=None, queries=None, training_pool=None):
    responses = load_index(predictions)
    if training_pool:
        if inputs or queries:
            raise ValueError("Choose one public input source")
        samples, questions = {}, {}
        with training_pool.open(encoding="utf8") as stream:
            for line in stream:
                row = json.loads(line)
                key = row["sample_id"]
                if key not in responses:
                    continue
                if key in samples:
                    raise ValueError("Duplicate training-pool key")
                samples[key], text = public_input(row)
                questions[key] = {"_key": key, "memory": [{"query": q} for q in text]}
    else:
        if not inputs or not queries:
            raise ValueError("Both --inputs and --queries are required")
        samples, questions = load_index(inputs), load_index(queries)
    if (
        not responses
        or not responses.keys() <= samples.keys()
        or not responses.keys() <= questions.keys()
    ):
        raise ValueError("Missing public data for generated cases")
    result, audit = [], []
    for key, generated in responses.items():
        assembled, evidence = compose(samples[key], questions[key], generated)
        result.append(assembled)
        audit.append({"key": key, "evidence": evidence})
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "predictions.jsonl", result)
    write_rows(output / "evidence.jsonl", audit)
    print(
        len(result),
        "responses;",
        sum(len(r["evidence"]) for r in audit),
        "executable fields",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--queries", type=Path)
    p.add_argument("--training-pool", type=Path)
    args = p.parse_args()
    materialize(
        args.predictions, args.output, args.inputs, args.queries, args.training_pool
    )
