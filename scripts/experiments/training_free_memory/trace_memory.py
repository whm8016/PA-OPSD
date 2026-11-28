"""Project executable historical queries to compact, source-linked event tables."""

import argparse
import json
import random
from itertools import groupby
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from tokenizers import Tokenizer

from .budget_curve import BUNDLE, ROOT, load_index
from .evidence_executor import execute, fields
from .packet_followup import STUDY, TRAIN, clip_history, public_input
from .paired_report import read
from .prepare import digest, write_json, write_rows

ARMS = ("trace_control", "scalar_matched", "trace_only", "trace_and_scalar")


def render_trace(events, positions, question):
    by_position = dict(events)
    ordered = [(p, by_position[p]) for p in positions]
    concept_fields = any(
        word in question
        for word in ("标签", "哪些知识点", "知识点有哪些", "几个知识点")
    )
    lines = ["Q2 候选事件；每项为 原位置/题号/正误，+正确，-错误。"]
    for day, group in groupby(ordered, key=lambda pe: pe[1]["date"].date().isoformat()):
        parts = []
        for pos, event in group:
            flag = {"正确": "+", "错误": "-"}.get(event["result"], "?")
            text = f"{pos}/{event['qid']}/{flag}"
            if concept_fields:
                text += "(" + "、".join(event["concepts"]) + ")"
            parts.append(text)
        lines.append(day + " " + " ".join(parts))
    return "\n".join(lines)


def prepare(out, tokenizer_path):
    pin = json.loads((BUNDLE / "base_model.json").read_text())[
        "config_and_tokenizer_sha256"
    ]["tokenizer.json"]
    if digest(tokenizer_path) != pin:
        raise ValueError("Unexpected Qwen tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def cost(text):
        return len(tokenizer.encode(text + "\n\n", add_special_tokens=False).ids)

    followup = STUDY / "gpu_run1/followup"
    target = load_index(followup / "transfer_control.jsonl")
    variants = {arm: [] for arm in ARMS}
    audit = []
    for line in TRAIN.read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        if row["sample_id"] not in target:
            continue
        sample, queries = public_input(row)
        events = [(p, fields(r)) for p, r in enumerate(sample["history_info"], 1)]
        events.append((0, fields(sample["question_info"])))
        execution = execute(queries[1], 1, events)
        if execution is None:
            raise ValueError("Expected reproducible nonempty Q2 execution")
        positions = execution["positions"]
        visible = list(positions)
        scalar = f"Q2 计算候选：{execution['answer']}（请按题意核对）"
        trace = render_trace(events, visible, queries[1])
        while cost(trace + "\n" + scalar) > 512 and len(visible) > 1:
            visible.pop(len(visible) // 2)
            trace = render_trace(events, visible, queries[1]) + "\n仅列部分事件。"
        if cost(trace + "\n" + scalar) > 512:
            raise ValueError("One projected event exceeds the fixed budget")
        packets = {
            "trace_control": None,
            "scalar_matched": scalar,
            "trace_only": trace,
            "trace_and_scalar": trace + "\n" + scalar,
        }
        remaining = 512 - max(cost(v) for v in packets.values() if v)
        raw = []
        for record in clip_history(sample, queries, tokenizer)["history_info"]:
            size = cost(record)
            if size <= remaining:
                raw.append(record)
                remaining -= size
        for arm, packet in packets.items():
            records = [*raw, packet] if packet else raw
            variants[arm].append(dict(sample, history_info=records))
        audit.append(
            {
                "key": sample["_key"],
                "source_positions": positions,
                "shown_positions": visible,
                "operator": execution["operator"],
                "scalar": execution["answer"],
                "raw_count": len(raw),
                "history_tokens": {
                    a: sum(cost(t) for t in variants[a][-1]["history_info"])
                    for a in ARMS
                },
            }
        )
    if len(target) != 131 or len(audit) != 131:
        raise ValueError("Expected all previously executable Q2 cases")
    out.mkdir(parents=True, exist_ok=False)
    paths = []
    for name, rows in variants.items():
        paths.append(out / f"{name}.jsonl")
        write_rows(paths[-1], rows)
    write_rows(out / "audit.jsonl", audit)
    query_path = out / "queries.jsonl"
    queries = load_index(followup / "queries.jsonl")
    write_rows(query_path, (queries[key] for key in target))
    paths.append(query_path)
    write_json(
        out / "design.json",
        {
            "arms": ARMS,
            "budget": 512,
            "n": 131,
            "primary": "trace_only versus scalar_matched on Q2, matched raw history, common upper budget",
            "secondary": "trace_and_scalar versus scalar_matched; all four versus no addendum",
            "selection": "same frozen full-executable transfer set; no filtering by correctness",
            "generation": "pinned LongTutor original script; shared Qwen2.5-7B base; no references",
            "input_sha256": {p.name: digest(p) for p in paths},
            "code_sha256": digest(Path(__file__)),
        },
    )
    with ZipFile(out / "trace.zip", "x", compression=ZIP_DEFLATED) as z:
        for p in [*paths, out / "design.json"]:
            z.write(p, p.name)
    print(
        "cases",
        len(audit),
        "complete traces",
        sum(r["source_positions"] == r["shown_positions"] for r in audit),
    )


def blind(out):
    from .packet_results import answer, check_outputs

    cases = {
        r["key"]: r
        for r in read(STUDY / "transfer560/rows.jsonl")
        if r["task"] == 2 and r["full"]
    }
    generated = check_outputs(out.parent / "results/trace/runs", ARMS, cases)
    inputs = load_index(out / "trace_control.jsonl")
    review = out / "review"
    review.mkdir(exist_ok=False)
    public, keyrows = [], []
    rng = random.Random(612151)
    for i, (key, case) in enumerate(cases.items(), 1):
        predictions = {arm: answer(group[key], 2) for arm, group in generated.items()}
        distinct = list(dict.fromkeys(predictions.values()))
        rng.shuffle(distinct)
        rid = f"X{i:03}"
        public.append(
            {
                "id": rid,
                "question": case["question"],
                "reference": case["reference"],
                "answers": distinct,
            }
        )
        keyrows.append(
            {
                "id": rid,
                "key": key,
                "qid": inputs[key]["question_info"].splitlines()[0],
                "roles": {arm: distinct.index(a) for arm, a in predictions.items()},
            }
        )
    write_rows(review / "identity.jsonl", keyrows)
    for i in range(0, len(public), 35):
        write_rows(review / f"blind_{i // 35 + 1}.jsonl", public[i : i + 35])


def aggregate(out):
    from .packet_results import paired, reviewed_rows

    review = out / "review"
    detail = reviewed_rows((review,))
    if len(detail) != 131 or len({r["key"] for r in detail}) != 131:
        raise ValueError("Expected all 131 trace cases")
    write_rows(review / "semantic_by_id.jsonl", detail)
    summary = {
        base: {
            arm: paired(
                [(r["scores"][base], r["scores"][arm]) for r in detail],
                [r["qid"] for r in detail],
            )
            for arm in ARMS
        }
        for base in ("trace_control", "scalar_matched")
    }
    write_json(review / "semantic_summary.json", summary)
    print({arm: sum(r["scores"][arm] for r in detail) for arm in ARMS})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=STUDY / "gpu_run1/trace")
    p.add_argument(
        "--stage", choices=("prepare", "blind", "aggregate"), default="prepare"
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / ".codex_tmp/opd_qwen_teacher_tokenizer/tokenizer.json",
    )
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.output, args.tokenizer)
    else:
        {"blind": blind, "aggregate": aggregate}[args.stage](args.output)
