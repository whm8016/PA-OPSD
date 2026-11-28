"""Reproduce temporal retrieval and compare the frozen adaptive tutor on all 560 training-pool cases."""

import argparse
import hashlib
import inspect
import json
import random
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from .answer_with_evidence import compose
from .budget_curve import BUNDLE, HEADER, MEMORY, load_index
from .evidence_executor import FOLLOW, execute, fields
from .packet_followup import STUDY, TRAIN, public_input
from .packet_results import answer, check_outputs, paired
from .paired_report import read
from .prepare import SPLIT, digest, sample_inputs, write_json, write_rows
from .prepare_followup import followup_inputs

DEFAULT = STUDY / "gpu_run1/transfer_temporal"
# Frozen member followup/runs/transfer_control.jsonl in the SHA-verified gpu_run1/extra.tar.gz.
COMPACT_PREDICTION_SHA256 = "f8fe751a1a4e810fc044690d54f5f7412ffa2f8e0d9f3c28a382ce42bfd56f55"


def training_inputs():
    samples, queries = {}, {}
    for row in read(TRAIN):
        sample, text = public_input(row)
        key = sample["_key"]
        if key in samples:
            raise ValueError(f"Duplicate training input {key}")
        samples[key] = sample
        queries[key] = {"_key": key, "memory": [{"query": q} for q in text]}
    ids = json.loads(SPLIT.read_text(encoding="utf8"))["ids"]["train"]
    if len(ids) != 560 or set(ids) != samples.keys():
        raise ValueError("Expected the complete fixed training pool")
    return samples, queries


def temporal_view(sample, queries):
    controls, _ = sample_inputs(sample, queries)
    views, _ = followup_inputs(sample, queries, controls["rag"], controls["temporal_rag"])
    return views["temporal"]


def compact_route(sample, queries):
    events = [(i, fields(s)) for i, s in enumerate(sample["history_info"], 1)]
    events.append((0, fields(sample["question_info"])))
    return execute(queries["memory"][1]["query"], 1, events) is not None


def prepare(out):
    dev = load_index(BUNDLE / "inputs.jsonl")
    dev_queries = load_index(MEMORY / "queries_only.jsonl")
    archived = load_index(FOLLOW / "full.temporal.jsonl")
    if len(dev) != 140 or not dev.keys() == dev_queries.keys() == archived.keys():
        raise ValueError("Incomplete development sources")
    for key in dev:
        if temporal_view(dev[key], dev_queries[key]) != archived[key]:
            raise ValueError(f"Development temporal input mismatch for {key}")
    samples, queries = training_inputs()
    frozen_path = out.parent / "followup/transfer_control.jsonl"
    frozen = load_index(frozen_path)
    active = {k for k in samples if compact_route(samples[k], queries[k])}
    if active != frozen.keys() or len(active) != 131 or samples.keys() & dev.keys():
        raise ValueError("Unexpected compact routes or overlap with development data")
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = out.parent / "transfer_temporal.zip"
    if out.exists() or archive.exists():
        raise FileExistsError("Use a new output location")
    with TemporaryDirectory(dir=out.parent) as temp:
        directory = Path(temp) / out.name
        directory.mkdir()
        paths = [directory / "temporal.jsonl", directory / "queries.jsonl"]
        write_rows(paths[0], (temporal_view(s, queries[k]) for k, s in samples.items()))
        write_rows(paths[1], queries.values())
        config = {
            "arms": ["temporal"],
            "n": len(samples),
            "compact": len(active),
            "shared_temporal": len(samples) - len(active),
            "verified_dev_temporal": len(dev),
            "scope": "Training-pool transfer of a frozen model and a method selected after development comparisons.",
            "comparison": "All 560 temporal versus 131 frozen compact generations and 429 shared temporal generations. Compose executable Q1/Q2 fields. Preserve official format failures.",
            "primary": "Paired Q2 semantic correctness across 560, clustered by actual current question ID.",
            "input_sha256": {p.name: digest(p) for p in paths},
            "source_sha256": {
                "fixed_train": digest(TRAIN), "split": digest(SPLIT),
                "frozen_compact": digest(frozen_path), "dev_temporal": digest(FOLLOW / "full.temporal.jsonl"),
                "prepare.py": digest(Path(inspect.getfile(sample_inputs))),
                "prepare_followup.py": digest(Path(inspect.getfile(followup_inputs))),
            },
            "prepare_source_sha256": hashlib.sha256("".join(inspect.getsource(fn) for fn in
                (training_inputs, temporal_view, compact_route, prepare)).encode()).hexdigest(),
        }
        write_json(directory / "design.json", config)
        staged_zip = Path(temp) / archive.name
        with ZipFile(staged_zip, "x", ZIP_DEFLATED) as zipped:
            for path in [*paths, directory / "design.json"]:
                zipped.write(path, path.name)
        directory.rename(out)
        staged_zip.rename(archive)
    print(f"Verified 140 development inputs. Packaged 560 temporal inputs at {archive}")


@contextmanager
def new_directory(destination):
    """Publish a stage only once all of its files have been written successfully."""
    if destination.exists():
        raise FileExistsError(destination)
    with TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / destination.name
        stage.mkdir()
        yield stage
        stage.rename(destination)


def verified_input(out):
    design = json.loads((out / "design.json").read_text(encoding="utf8"))
    for name in ("temporal.jsonl", "queries.jsonl"):
        if digest(out / name) != design["input_sha256"][name]:
            raise ValueError(f"Changed prepared input: {name}")
    compact = out.parent / "followup/transfer_control.jsonl"
    if (digest(compact) != design["source_sha256"]["frozen_compact"]
            or digest(TRAIN) != design["source_sha256"]["fixed_train"]):
        raise ValueError("Frozen training or compact input changed")
    return load_index(compact)


def assemble(out, temporal_path, compact_path):
    frozen = verified_input(out)
    if digest(compact_path) != COMPACT_PREDICTION_SHA256:
        raise ValueError("Compact predictions differ from the archived transfer_control arm")
    if temporal_path.name != "temporal.jsonl":
        raise ValueError("Expected upstream temporal.jsonl predictions")
    samples, queries = training_inputs()
    temporal = check_outputs(temporal_path.parent, (temporal_path.stem,), samples)[temporal_path.stem]
    compact = check_outputs(compact_path.parent, (compact_path.stem,), frozen)[compact_path.stem]
    if not frozen.keys() <= samples.keys() or len(frozen) != 131:
        raise ValueError("Incomplete or unexpected compact routes")
    predictions, evidence = [], []
    for key, sample in samples.items():
        active = key in frozen
        if compact_route(sample, queries[key]) != active:
            raise ValueError(f"Frozen routing no longer reproducible for {key}")
        generated = compact[key] if active else temporal[key]
        composed, decisions = compose(sample, queries[key], generated)
        if not active and (any(d["task"] == 2 for d in decisions)
                           or answer(composed, 2) != answer(temporal[key], 2)):
            raise ValueError(f"Non-executable Q2 changed for {key}")
        predictions.append(composed)
        evidence.append({"key": key, "qid": HEADER.match(sample["question_info"])[1],
                         "compact": active, "evidence": decisions})
    with new_directory(out / "system") as stage:
        write_rows(stage / "predictions.jsonl", predictions)
        write_rows(stage / "evidence.jsonl", evidence)
        write_json(stage / "receipt.json", {
            "logical_generation_stages_per_case": 1, "compact_reused": len(compact),
            "retry_accounting": "Actual request counts depend on upstream retries and are not measured here.",
            "shared_temporal_reused": len(samples) - len(compact),
            "paired_baseline_cases": len(temporal),
            "temporal_sha256": digest(temporal_path), "compact_sha256": digest(compact_path),
            "system_sha256": digest(stage / "predictions.jsonl"),
            "evidence_sha256": digest(stage / "evidence.jsonl"),
        })


def blind(out, temporal_path):
    system = out / "system"
    receipt = json.loads((system / "receipt.json").read_text(encoding="utf8"))
    if (digest(temporal_path) != receipt["temporal_sha256"] or
            digest(system / "predictions.jsonl") != receipt["system_sha256"] or
            digest(system / "evidence.jsonl") != receipt["evidence_sha256"]):
        raise ValueError("Prediction or evidence archive changed after assembly")
    frozen = verified_input(out)
    inputs = load_index(out / "temporal.jsonl")
    queries = load_index(out / "queries.jsonl")
    baseline, composed = (load_index(p) for p in (temporal_path, system / "predictions.jsonl"))
    if not baseline.keys() == composed.keys() == inputs.keys() == queries.keys() or len(inputs) != 560:
        raise ValueError("Incomplete paired input")
    # References are first read here, once the full composed prediction archive exists.
    refs = {r["sample_id"]: json.loads(r["messages"][2]["content"])["memory"][1]["answer"]
            for r in read(TRAIN)}
    rng = random.Random(432720)
    queue, mapping = {"compact": [], "shared": []}, []
    for key, sample in inputs.items():
        active = key in frozen
        before, after = answer(baseline[key], 2), answer(composed[key], 2)
        if not active and before != after:
            raise ValueError(f"Non-executable Q2 changed for {key}")
        options = list(dict.fromkeys((before, after)))
        rng.shuffle(options)
        rid = f"U{len(mapping)+1:03}"
        queue["compact" if active else "shared"].append({
            "id": rid, "question": queries[key]["memory"][1]["query"],
            "reference": refs[key], "answers": options,
        })
        mapping.append({"id": rid, "key": key, "qid": HEADER.match(sample["question_info"])[1],
                        "compact": active, "roles": {"temporal": options.index(before),
                                                     "adaptive": options.index(after)}})
    with new_directory(out / "review") as stage:
        write_rows(stage / "identity.jsonl", mapping)
        for group, cases in queue.items():
            if group == "compact":
                cases.sort(key=lambda case: len(case["answers"]), reverse=True)
            for offset in range(0, len(cases), 35):
                write_rows(stage / f"blind_{group}_{offset // 35 + 1}.jsonl", cases[offset:offset + 35])
        write_json(stage / "dispatch.json", {
            "sha256": {p.name: digest(p) for p in sorted(stage.glob("*.jsonl"))}
        })


def count_scores(mapping, public, judgments):
    ids = {row["id"] for row in mapping}
    if len(ids) != len(mapping) or ids != set(public) or not judgments.keys() <= ids:
        raise ValueError("Inconsistent case IDs across mapping, blind files and verdicts")
    detail, differences = [], []
    for row in mapping:
        rid = row["id"]
        options = public[rid]["answers"]
        roles = row["roles"]
        if (not options or set(roles) != {"temporal", "adaptive"}
                or set(roles.values()) != set(range(len(options)))):
            raise ValueError(f"Invalid blind answer mapping for {rid}")
        if not row["compact"] and len(options) != 1:
            raise ValueError(f"Unshared fallback for {rid}")
        if rid in judgments:
            grades = judgments[rid]
            if len(grades) != len(options) or any(type(g) is not int or g not in (0, 1) for g in grades):
                raise ValueError(f"Invalid binary verdict for {rid}")
            scores = {a: grades[i] for a, i in roles.items()}
            delta = scores["adaptive"] - scores["temporal"]
            detail.append({**row, "scores": scores})
        elif len(options) == 1:
            delta = 0
        else:
            delta = None
        differences.append(delta)
    summary = {"n": len(mapping), "reviewed": len(detail),
               "ungraded_distinct": differences.count(None)}
    if None not in differences:
        # Identical strings have known zero difference even when their absolute score is pending.
        delta = paired([(int(d < 0), int(d > 0)) for d in differences], [r["qid"] for r in mapping])
        summary["paired_difference"] = {k: v for k, v in delta.items() if k not in ("before", "after")}
    for name, active in (("all", None), ("compact", True), ("shared", False)):
        selected = [r for r in detail if active is None or r["compact"] == active]
        expected = sum(active is None or r["compact"] == active for r in mapping)
        if selected and len(selected) == expected:
            pairs = [(r["scores"]["temporal"], r["scores"]["adaptive"]) for r in selected]
            summary[name] = {**paired(pairs, [r["qid"] for r in selected]),
                             "temporal_correct": sum(a for a, _ in pairs),
                             "adaptive_correct": sum(b for _, b in pairs)}
    return detail, summary


def review_index(directory, pattern):
    rows = [r for path in sorted(directory.glob(pattern)) for r in read(path)]
    indexed = {r["id"]: r for r in rows}
    if len(rows) != len(indexed):
        raise ValueError(f"Duplicate review ID in {pattern}")
    return indexed


def aggregate(out):
    review = out / "review"
    dispatch = json.loads((review / "dispatch.json").read_text(encoding="utf8"))["sha256"]
    sources = [review / "identity.jsonl", *review.glob("blind_*.jsonl")]
    if {p.name: digest(p) for p in sources} != dispatch:
        raise ValueError("Blind answers or role mapping changed since dispatch")
    mapping = read(review / "identity.jsonl")
    if len(mapping) != 560 or len({r["key"] for r in mapping}) != 560 or sum(r["compact"] for r in mapping) != 131:
        raise ValueError("Incomplete frozen 560-case mapping")
    public = review_index(review, "blind_*.jsonl")
    verdicts = {k: row["scores"] for k, row in review_index(review, "verdict_*.jsonl").items()}
    detail, summary = count_scores(mapping, public, verdicts)
    suffix = "" if len(detail) == len(mapping) else "_interim"
    write_rows(review / f"semantic_by_id{suffix}.jsonl", detail)
    write_json(review / f"semantic_summary{suffix}.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "assemble", "blind", "aggregate"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--temporal-predictions", type=Path)
    parser.add_argument("--compact-predictions", type=Path)
    args = parser.parse_args()
    temporal = args.temporal_predictions or args.output.parent / "results/transfer_temporal/runs/temporal.jsonl"
    compact = args.compact_predictions or args.output.parent / "results/followup/runs/transfer_control.jsonl"
    if args.stage == "assemble":
        assemble(args.output, temporal, compact)
    elif args.stage == "blind":
        blind(args.output, temporal)
    elif args.stage == "aggregate":
        aggregate(args.output)
    else:
        prepare(args.output)
