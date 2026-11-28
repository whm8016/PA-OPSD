"""Package three frozen 512-token histories for the official LongTutor generator."""

import argparse
from pathlib import Path

from tokenizers import Tokenizer

from . import prepare
from .budget_curve import MEMORY, load_index
from .evidence_executor import execute, fields

ARMS = ("bm25", "id_recent", "id_bm25")


def execution_views(sample, questions, short, tokenizer):
    """Reserve space for results computed on full history. Both arms retain identical raw records."""
    events = [(i, fields(s)) for i, s in enumerate(sample["history_info"], 1)]
    events.append((0, fields(sample["question_info"])))
    lines = []
    for task, question in enumerate(questions["memory"][:2]):
        result = execute(question["query"], task, events)
        if result:
            positions = ",".join(map(str, result["positions"]))
            lines.append(
                f"Q{task + 1} {result['operator']} = {result['answer']}（原记录位置 {positions}）"
            )
    if not lines:
        return short, short
    packet = "历史字段计算（候选，请核对题意）\n" + "\n".join(lines)

    def cost(text):
        return len(tokenizer.encode(text + "\n\n", add_special_tokens=False).ids)

    remaining = 512 - cost(packet)
    if remaining < 0:
        return short, short
    kept = []
    for record in short["history_info"]:
        length = cost(record)
        if length <= remaining:
            kept.append(record)
            remaining -= length
    return dict(short, history_info=kept), dict(short, history_info=[*kept, packet])


def package(run, output, tokenizer_path=None):
    source = prepare.BUNDLE
    read, digest = prepare.read_json, prepare.digest
    protocol = read(source / "manifest.json")
    original = load_index(source / "inputs.jsonl")
    ids = protocol["ids"]
    if set(original) != set(ids) or len(ids) != 140:
        raise ValueError("Invalid fixed development split")
    source_queries = MEMORY / "queries_only.jsonl"
    manifest = read(run / "manifest.json")
    for path in (source / "inputs.jsonl", source_queries):
        if (
            digest(path)
            != manifest["input_sha256"][str(path.relative_to(prepare.ROOT))]
        ):
            raise ValueError(f"Source data changed: {path}")
    queries = load_index(source_queries)
    prepare.verify_queries(source, queries)
    selections = {arm: {} for arm in ARMS}
    for row in prepare.rows(run / "rows.jsonl"):
        if row["budget"] != 512 or row["arm"] not in ARMS:
            continue
        arm, key = row["arm"], row["key"]
        positions = row["positions"]
        sample = original[key]
        if (
            key in selections[arm]
            or positions != sorted(set(positions))
            or any(
                position < 1 or position > len(sample["history_info"])
                for position in positions
            )
        ):
            raise ValueError("Duplicate row or invalid history positions")
        selections[arm][key] = dict(
            sample, history_info=[sample["history_info"][i - 1] for i in positions]
        )
    if any(set(group) != set(ids) for group in selections.values()):
        raise ValueError("Incomplete arm")
    packet_count = 0
    if tokenizer_path is not None:
        expected = read(source / "base_model.json")["config_and_tokenizer_sha256"][
            "tokenizer.json"
        ]
        if digest(tokenizer_path) != expected:
            raise ValueError("Unexpected tokenizer")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        selections["packet_control"], selections["packet"] = {}, {}
        for key in ids:
            control, packet = execution_views(
                original[key], queries[key], selections["id_bm25"][key], tokenizer
            )
            selections["packet_control"][key] = control
            selections["packet"][key] = packet
            packet_count += int(control != packet)
    output.mkdir(parents=True, exist_ok=False)
    members = {
        "upstream/" + name: source / "upstream" / name
        for name in prepare.UPSTREAM_FILES
    }
    for name, path in members.items():
        if digest(path) != protocol["files"][name]:
            raise ValueError(f"Upstream source changed: {name}")
    members.update(
        {
            name: Path(__file__).with_name(name)
            for name in ("run_inference.sh", "service.py")
        }
    )
    members.update(
        {
            "base_model.json": source / "base_model.json",
            "queries_only.jsonl": source_queries,
        }
    )
    for arm, group in selections.items():
        for subset, keys in (("smoke", ids[:2]), ("full", ids)):
            path = output / f"{subset}.{arm}.jsonl"
            prepare.write_rows(path, (group[key] for key in keys))
            members[path.name] = path
    base = read(MEMORY / "manifest.json")
    settings = {
        key: base[key]
        for key in (
            "base_model",
            "base_revision",
            "inference",
            "vllm_version",
            "upstream_commit",
        )
    }
    settings.update(
        arm_order=list(selections),
        packet_cases=packet_count,
        workers=4,
        ids=ids,
        smoke_ids=ids[:2],
        history_budget=512,
        source_manifest_sha256=digest(run / "manifest.json"),
        selections_sha256=digest(run / "rows.jsonl"),
        status="prepared_on_cpu",
    )
    prepare.write_inference_bundle(output, members, settings)
    print(
        f"Packed {len(selections)} arms, each with 140 development inputs and 2 smoke inputs; packets: {packet_count}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--packet-tokenizer",
        type=Path,
        help="Add computed evidence and matched raw-record control",
    )
    args = parser.parse_args()
    package(args.run, args.output, args.packet_tokenizer)
