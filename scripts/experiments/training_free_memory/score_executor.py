"""Score frozen predictions and build an anonymized paired review of accepted overrides."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .budget_curve import BUNDLE, load_index


def main(path, output):
    gold = load_index(BUNDLE / "gold.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 280 or {(r["key"], r["task"]) for r in rows} != {
        (k, t) for k in gold for t in (1, 2)
    }:
        raise ValueError("Expected Q1 and Q2 for every development case")
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    review, mapping, scored = [], [], []
    rng = random.Random(27512)
    ablation_rng = random.Random(29145)
    ablation, ablation_key = [], []
    for row in rows:
        task = row["task"]
        label = gold[row["key"]]["memory"][task - 1]
        if label["query"] != row["question"]:
            raise ValueError("Gold question mismatch")
        reference = label["answer"]
        predictions = {"model": row["fallback"], "gated": row["answer"]}
        for mode in ("full", "brief"):
            predictions[mode] = row[mode]["answer"] if row[mode] else row["fallback"]
        variants = [predictions[k] for k in ("model", "full", "brief", "gated")]
        answers = list(dict.fromkeys(variants))
        if len(answers) > 1:
            ablation_rng.shuffle(answers)
            rid = f"E{len(ablation) + 1:03}"
            ablation.append(
                {
                    "id": rid,
                    "question": row["question"],
                    "reference": reference,
                    "answers": answers,
                }
            )
            ablation_key.append(
                {
                    "id": rid,
                    "key": row["key"],
                    "task": task,
                    "indices": {
                        k: answers.index(predictions[k])
                        for k in ("model", "full", "brief", "gated")
                    },
                }
            )
        scores = {
            k: " ".join(v.split()) == " ".join(reference.split())
            for k, v in predictions.items()
        }
        for k, v in scores.items():
            counts[task, k] += int(v)
        scored.append(
            {"key": row["key"], "task": task, "reference": reference, "exact": scores}
        )
        if row["agree"]:
            roles = ["model", "gated"]
            rng.shuffle(roles)
            rid = f"R{len(review) + 1:03}"
            review.append(
                {
                    "id": rid,
                    "question": row["question"],
                    "reference": reference,
                    "A": predictions[roles[0]],
                    "B": predictions[roles[1]],
                }
            )
            mapping.append(
                {
                    "id": rid,
                    "task": task,
                    "key": row["key"],
                    "A": roles[0],
                    "B": roles[1],
                }
            )
    for name, data in (
        ("paired", scored),
        ("blind", review),
        ("key", mapping),
        ("ablation_blind", ablation),
        ("ablation_key", ablation_key),
    ):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data),
            encoding="utf-8",
        )
    summary = {
        "n_per_task": 140,
        "overrides": len(review),
        "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "exact": {
            str(t): {mode: counts[t, mode] for mode in predictions} for t in (1, 2)
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.predictions, args.output)
