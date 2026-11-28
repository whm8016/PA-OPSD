"""Route executable Q2 to compact history and others to temporal retrieval; blind paired evaluation."""

import argparse
import json
import random
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .answer_with_evidence import materialize
from .budget_curve import BUNDLE, HEADER, MEMORY, load_index
from .evidence_executor import FOLLOW, execute, fields
from .packet_followup import STUDY, TRAIN
from .packet_results import answer, paired, reviewed_rows
from .paired_report import norm, read
from .prepare import digest, write_json, write_rows

ARMS = ("temporal", "adaptive_raw", "adaptive", "shared_fallback")


def prepare(out):
    full = load_index(BUNDLE / "inputs.jsonl")
    questions = load_index(MEMORY / "queries_only.jsonl")
    short = load_index(STUDY / "gpu_evidence512/full.packet_control.jsonl")
    temporal = load_index(FOLLOW / "full.temporal.jsonl")
    if not full.keys() == questions.keys() == short.keys() == temporal.keys():
        raise ValueError("Input sources must share the same sample keys")
    routed, audit = [], []
    for key, row in full.items():
        events = [(i, fields(r)) for i, r in enumerate(row["history_info"], 1)]
        events.append((0, fields(row["question_info"])))
        active = execute(questions[key]["memory"][1]["query"], 1, events) is not None
        routed.append(short[key] if active else temporal[key])
        audit.append({"key": key, "compact": active})
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("adaptive", routed),
        ("temporal", temporal.values()),
        ("queries", questions.values()),
        ("routes", audit),
    ):
        write_rows(out / (name + ".jsonl"), rows)
    paths = [out / (arm + ".jsonl") for arm in ("adaptive", "temporal", "queries")]
    write_json(
        out / "design.json",
        {
            "input_sha256": {p.name: digest(p) for p in paths},
            "n": len(full),
            "compact": sum(r["compact"] for r in audit),
            "decision": "Exploratory after initial dev comparisons. Primary all-dev Q2 versus fresh temporal. Same model and settings. Adaptive generates once. Compact if full-history Q2 executes, otherwise original temporal. Compose supported Q1/Q2 by source-linked execution; preserve format failures.",
            "secondary": "Same exact fallback response for inactive cases; raw adaptive before composition; Q1 and prompt tokens.",
        },
    )
    with ZipFile(out.parent / "adaptive.zip", "w", ZIP_DEFLATED) as z:
        for path in [*paths, out / "design.json"]:
            z.write(path, path.name)


def blind(out):
    results = out.parent / "results/adaptive/runs"
    system = out.parent / "system/adaptive"
    gold = load_index(BUNDLE / "gold.jsonl")
    groups = {
        arm: load_index(results / name)
        for arm, name in (
            ("temporal", "temporal.jsonl"),
            ("adaptive_raw", "adaptive.jsonl"),
        )
    }
    if any(g.keys() != gold.keys() for g in groups.values()):
        raise ValueError("Incomplete paired generation")
    # Materialize both files in staging. A retry after publishing the system
    # may reuse only byte-identical predictions and evidence.
    from tempfile import TemporaryDirectory

    system.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=system.parent) as temp:
        staged = Path(temp) / "system"
        materialize(
            results / "adaptive.jsonl",
            staged,
            BUNDLE / "inputs.jsonl",
            MEMORY / "queries_only.jsonl",
        )
        if system.exists():
            if any(digest(p) != digest(system / p.name) for p in staged.iterdir()):
                raise ValueError("Existing composed predictions disagree with replay")
        else:
            staged.rename(system)
    groups["adaptive"] = load_index(system / "predictions.jsonl")
    routes = {r["key"]: r["compact"] for r in read(out / "routes.jsonl")}
    inputs, queries = (
        load_index(BUNDLE / "inputs.jsonl"),
        load_index(MEMORY / "queries_only.jsonl"),
    )
    groups["shared_fallback"] = {
        k: groups["adaptive"][k] if active else groups["temporal"][k]
        for k, active in routes.items()
    }
    rng = random.Random(432719)
    review = out / "review"
    review.mkdir(exist_ok=False)
    public, mapping = [], []
    for key, record in gold.items():
        for task in (1, 2):
            values = {arm: answer(group[key], task) for arm, group in groups.items()}
            options = list(dict.fromkeys(values.values()))
            rng.shuffle(options)
            rid = f"Y{len(public) + 1:03}"
            mem = record["memory"][task - 1]
            public.append(
                {
                    "id": rid,
                    "question": queries[key]["memory"][task - 1]["query"],
                    "reference": mem["answer"],
                    "answers": options,
                }
            )
            mapping.append(
                {
                    "id": rid,
                    "key": key,
                    "task": task,
                    "qid": HEADER.match(inputs[key]["question_info"])[1],
                    "compact": routes[key],
                    "roles": {arm: options.index(v) for arm, v in values.items()},
                }
            )
    write_rows(review / "identity.jsonl", mapping)
    for i in range(0, len(public), 70):
        write_rows(review / f"blind_{i // 70 + 1}.jsonl", public[i : i + 70])


def aggregate(out):
    review = out / "review"
    keys = reviewed_rows((review,))
    if len(keys) != 280 or len({(r["key"], r["task"]) for r in keys}) != 280:
        raise ValueError("Expected both frozen questions for all 140 cases")
    summary = {}
    for task in (1, 2):
        for label, selector in (("all", None), ("compact", True), ("fallback", False)):
            subset = [
                r
                for r in keys
                if r["task"] == task and (selector is None or r["compact"] == selector)
            ]
            summary[f"Q{task}_{label}"] = {
                arm: paired(
                    [(r["scores"]["temporal"], r["scores"][arm]) for r in subset],
                    [r["qid"] for r in subset],
                )
                for arm in ARMS
            }
    write_json(review / "semantic_summary.json", summary)
    write_rows(review / "semantic_by_id.jsonl", keys)


def routing_costs(out):
    """Compare routing to temporal retrieval with the very same output executor."""
    from statistics import mean, median

    from .answer_with_evidence import compose
    from .transfer_temporal import training_inputs

    run = out.parent
    samples, questions = training_inputs()
    temporal = load_index(run / "results/transfer_temporal/runs/temporal.jsonl")
    system = load_index(run / "transfer_temporal/system/predictions.jsonl")
    compact = load_index(run / "followup/transfer_control.jsonl")
    costs = {
        (r["arm"], r["key"]): r["prompt_tokens"]
        for r in read(run / "prompt_costs.jsonl")
    }
    if (
        not samples.keys() == temporal.keys() == system.keys()
        or not compact.keys() <= samples.keys()
    ):
        raise ValueError("Incomplete transfer cost comparison")
    rows = []
    refs = {
        r["sample_id"]: json.loads(r["messages"][2]["content"])["memory"][1]["answer"]
        for r in read(TRAIN)
    }
    # Abstain on lists, explanatory prose, ranges, decimals and multiple values.
    scalar = re.compile(r"\s*([0-9]+)\s*(?:道题目|道题|道|次|个|条)?[。.]?\s*")
    for k in samples:
        simple, evidence = compose(samples[k], questions[k], temporal[k])
        arm = "transfer_control" if k in compact else "transfer_temporal"
        before, after = answer(temporal[k], 2), answer(system[k], 2)
        reference = refs[k]
        target, old, new = (scalar.fullmatch(s) for s in (reference, before, after))
        rows.append(
            {
                "key": k,
                "compact": k in compact,
                "baseline_tokens": costs["transfer_temporal", k],
                "routed_tokens": costs[arm, k],
                "temporal_plus_execution_Q2": answer(simple, 2),
                "routed_Q2": answer(system[k], 2),
                "temporal_execution_fields": len(evidence),
                "qid": HEADER.match(samples[k]["question_info"])[1],
                "exact": [int(norm(s) == norm(reference)) for s in (before, after)],
                "plain_numeric": [
                    int(bool(m) and int(m[1]) == int(target[1])) for m in (old, new)
                ]
                if target
                else None,
            }
        )
    summary = {}
    for name, subset in (("all", rows), ("compact", [r for r in rows if r["compact"]])):
        b = [r["baseline_tokens"] for r in subset]
        a = [r["routed_tokens"] for r in subset]
        summary[name] = {
            "n": len(a),
            "identical_Q2_strings": sum(
                r["temporal_plus_execution_Q2"] == r["routed_Q2"] for r in subset
            ),
            "median_temporal": median(b),
            "median_routed": median(a),
            "mean_temporal": mean(b),
            "mean_routed": mean(a),
            "total_temporal": sum(b),
            "total_routed": sum(a),
            "fraction_prompt_tokens_saved": 1 - sum(a) / sum(b),
        }
    for name in ("exact", "plain_numeric"):
        subset = [r for r in rows if r[name] is not None]
        summary[name] = paired([r[name] for r in subset], [r["qid"] for r in subset])
    write_rows(run / "analysis/routing_costs_by_id.jsonl", rows)
    write_json(run / "analysis/routing_costs.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "blind", "aggregate", "costs"))
    parser.add_argument("--output", type=Path, default=STUDY / "gpu_run1/adaptive")
    args = parser.parse_args()
    {
        "prepare": prepare,
        "blind": blind,
        "aggregate": aggregate,
        "costs": routing_costs,
    }[args.stage](args.output)
