"""Build fixed public inputs for disjoint Q2 transfer and packet-value interventions."""

import argparse
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from tokenizers import Tokenizer

from .budget_curve import HEADER, bm25, load_index
from .evidence_executor import fields
from .package_budget import execution_views
from .prepare import BUNDLE, EXPLICIT_ID, ROOT, SPLIT, digest, write_json, write_rows

STUDY = ROOT / "outputs/pa_opsd/memory_budget_paper"
TRAIN = ROOT / "outputs/pa_opsd/gold_split_20261120/development/train.jsonl"


def public_input(row):
    history, tail = row["messages"][1]["content"].split("### CURRENT QUESTION\n")
    current, tail = tail.split("### MEMORY QUERIES\n")
    records = re.split(
        r"\n\n(?=\[\d+\]\n)", history.removeprefix("### INTERACTION HISTORY\n").strip()
    )
    questions = re.findall(r"^- Q[123]: (.+)$", tail, re.MULTILINE)
    if len(records) != 199 or len(questions) != 3:
        raise ValueError("Invalid public training-pool prompt")
    key = row["sample_id"]
    return {
        "_key": key,
        "uid": key.split("||")[0],
        "question_info": current.strip(),
        "history_info": records,
    }, questions


def clip_history(sample, questions, tokenizer):
    query = "\n".join(questions)
    mentioned = {a or b for a, b in EXPLICIT_ID.findall(query)} - {
        HEADER.match(sample["question_info"])[1]
    }
    records = sample["history_info"]
    parsed = [fields(s) for s in records]
    score = bm25(records, query)
    order = sorted(
        range(len(records)),
        key=lambda i: (parsed[i]["qid"] in mentioned, score[i], parsed[i]["date"], i),
        reverse=True,
    )
    lengths = tokenizer.encode_batch(
        [r + "\n\n" for r in records], add_special_tokens=False
    )
    remaining = 512
    keep = []
    for i in order:
        size = len(lengths[i].ids)
        if size <= remaining:
            remaining -= size
            keep.append(i)
    return dict(sample, history_info=[records[i] for i in sorted(keep)])


def altered(operator, value):
    if operator == "result":
        return {"正确": "错误", "错误": "正确"}[value]
    if operator == "count":
        return str(int(value) + 1)
    if operator == "ids":
        return re.sub(r"\d+", lambda m: str(int(m[0]) + 1), value, count=1)
    if operator == "concepts":
        return "统计" if value != "统计" else "测量"
    raise ValueError(operator)


def prepare(out, tokenizer_path):
    if (
        digest(tokenizer_path)
        != json.loads((BUNDLE / "base_model.json").read_text())[
            "config_and_tokenizer_sha256"
        ]["tokenizer.json"]
    ):
        raise ValueError("Unexpected tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    target = {
        r["key"]
        for r in map(
            json.loads,
            (STUDY / "transfer560/rows.jsonl").read_text(encoding="utf8").splitlines(),
        )
        if r["task"] == 2 and r["full"]
    }
    if len(target) != 131 or not target <= set(
        json.loads(SPLIT.read_text())["ids"]["train"]
    ):
        raise ValueError("Expected all 131 previously executable Q2")
    arms = {
        name: []
        for name in (
            "transfer_control",
            "transfer_packet",
            "repeat_control",
            "repeat_packet",
            "masked",
            "altered",
        )
    }
    queries = []
    for line in TRAIN.read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        if row["sample_id"] not in target:
            continue
        sample, text = public_input(row)
        query = {"_key": sample["_key"], "memory": [{"query": q} for q in text]}
        short = clip_history(sample, text, tokenizer)
        control, packet = execution_views(sample, query, short, tokenizer)
        arms["transfer_control"].append(control)
        arms["transfer_packet"].append(packet)
        queries.append(query)
    src = STUDY / "gpu_evidence512"
    controls = load_index(src / "full.packet_control.jsonl")
    dev_queries = load_index(
        ROOT
        / "outputs/pa_opsd/training_free_memory/gold140_20261125_verified/queries_only.jsonl"
    )
    interventions = []
    for key, sample in load_index(src / "full.packet.jsonl").items():
        control = controls[key]
        if sample == control:
            continue
        queries.append(dev_queries[key])
        prefix, line = sample["history_info"][-1].split("\n")
        operator, answer, positions = re.fullmatch(
            r"(Q[12] \w+) = (.+)（原记录位置 (.+)）", line
        ).groups()
        replacement = altered(operator.split()[1], answer)
        arms["repeat_control"].append(control)
        arms["repeat_packet"].append(sample)
        for arm, value in (("masked", "?"), ("altered", replacement)):
            fact = f"{prefix}\n{operator} = {value}（原记录位置 {positions}）"
            arms[arm].append(
                dict(sample, history_info=[*control["history_info"], fact])
            )
        interventions.append(
            {
                "key": key,
                "task": int(operator[1]),
                "operator": operator[3:],
                "original": answer,
                "replacement": replacement,
            }
        )
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / "interventions.jsonl", interventions)
    paths = []
    max_cost = {}
    for name, rows in arms.items():
        expected = 131 if name.startswith("transfer") else 30
        if len(rows) != expected or len({r["_key"] for r in rows}) != expected:
            raise ValueError(name)
        max_cost[name] = max(
            sum(
                len(e.ids)
                for e in tokenizer.encode_batch(
                    [s + "\n\n" for s in r["history_info"]], add_special_tokens=False
                )
            )
            for r in rows
        )
        if max_cost[name] > 512:
            raise ValueError(f"Record budget exceeded in {name}")
        paths.append(out / f"{name}.jsonl")
        write_rows(paths[-1], rows)
    paths.append(out / "queries.jsonl")
    write_rows(paths[-1], queries)
    config = {
        "arms": list(arms),
        "input_sha256": {p.name: digest(p) for p in paths},
        "max_history_tokens": max_cost,
        "parent_sha256": digest(src / "inference.zip"),
        "transfer_selection": "all 131 executable Q2; no correctness filter",
        "primary": "paired semantic correctness of Q2 on 131 transfer cases, packet minus matched control",
        "mechanism": "repeat packet and control; mask or alter computed value with sources fixed on 30 active dev cases",
        "tokenizer_sha256": digest(tokenizer_path),
        "code_sha256": digest(Path(__file__)),
    }
    write_json(out / "design.json", config)
    with ZipFile(out / "followup.zip", "x", compression=ZIP_DEFLATED) as z:
        for p in [*paths, out / "design.json"]:
            z.write(p, p.name)
    print(max_cost)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / ".codex_tmp/opd_qwen_teacher_tokenizer/tokenizer.json",
    )
    args = p.parse_args()
    prepare(args.output, args.tokenizer)
