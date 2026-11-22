"""Prepare and validate a direct LongTutor Silver SFT control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from pa_opsd_contract import (
    XES,
    audit_seeds,
    fingerprint,
    memory_queries,
    read_jsonl,
    student_messages,
    validate_target,
)
from train_pa_opsd_sft import (
    DEFAULT_SEED_MANIFEST,
    OverlengthError,
    encode_row,
    inference_rows,
    load_tokenizer,
    validate_native_output,
)

MAX_LENGTH = 32768
MAX_NEW = 1536
SEED = 42


def hash_file(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def silver_row(seed, annotation):
    queries = memory_queries(seed)
    if annotation["_key"] != seed["longtutor_key"]:
        raise ValueError("Silver annotation key differs from the official sample")
    if [item["query"].strip() for item in annotation["memory"]] != queries:
        raise ValueError("Silver questions differ from the official input")
    target = {key: annotation[key] for key in ("diagnosis", "reason", "strategy", "content")}
    target["memory"] = [{key: entry[key] for key in ("id", "answer")} for entry in annotation["memory"]]
    validate_target(target, queries)
    return {
        "sample_id": seed["sample_id"], "split": "train", "status": "accepted", "dry_run": False,
        "messages": student_messages(seed) + [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
        "teacher_privileged": {"diagnosis": target["diagnosis"], "strategy": target["strategy"]},
        "provenance": {"source": "LongTutor pipeline_an_scale", "label_quality": "published_silver",
                       "human_signoff": False, "longtutor_key": seed["longtutor_key"],
                       "annotation_sha256": fingerprint(annotation)},
    }


def prompt_signatures(validation_path, tokenizer):
    rows = inference_rows(read_jsonl(validation_path), DEFAULT_SEED_MANIFEST)
    if len(rows) != 12 or any(row["split"] != "validation" for row in rows):
        raise ValueError("Expected the fixed 12 validation rows")
    signatures = {}
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        length = len(ids)
        if length + MAX_NEW > MAX_LENGTH:
            raise ValueError(f"Validation prompt exceeds context budget: {row['sample_id']}")
        signatures[row["sample_id"]] = {
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "input_tokens": length,
        }
    return signatures


def prepare(out, tokenizer_path, validation_path, count):
    seeds = read_jsonl(DEFAULT_SEED_MANIFEST)
    audit = audit_seeds(seeds)
    annotations = {row["_key"]: row for row in read_jsonl(XES / "pipeline_an_scale.jsonl")}
    tokenizer = load_tokenizer(tokenizer_path, local_files_only=True)
    groups = defaultdict(list)
    rejects = []
    usable = []
    for seed in seeds:
        if seed["split"] != "train":
            continue
        try:
            row = silver_row(seed, annotations[seed["longtutor_key"]])
            encoded = encode_row(row, tokenizer, MAX_LENGTH)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            rejects.append({"sample_id": seed["sample_id"],
                            "kind": "overlength" if isinstance(exc, OverlengthError) else "invalid",
                            "reason": str(exc)})
            continue
        target = json.loads(row["messages"][-1]["content"])
        item = (row, len(encoded["input_ids"]), sum(token != -100 for token in encoded["labels"]))
        groups[target["diagnosis"]].append(item)
        usable.append(row)
    rng = random.Random(SEED)
    for diagnosis in sorted(groups):
        rng.shuffle(groups[diagnosis])
    chosen = []
    while len(chosen) < count:
        previous = len(chosen)
        for diagnosis in sorted(groups):
            if groups[diagnosis] and len(chosen) < count:
                chosen.append(groups[diagnosis].pop())
        if previous == len(chosen):
            raise ValueError("Insufficient usable Silver rows")
    rng.shuffle(chosen)
    signatures = prompt_signatures(validation_path, tokenizer)
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / "train.jsonl", [row for row, _, _ in chosen])
    write_rows(out / "probe.jsonl", [max(chosen, key=lambda item: item[1])[0]])
    write_rows(out / "usable507.jsonl", usable)
    write_rows(out / "rejected.jsonl", rejects)
    shutil.copyfile(validation_path, out / "validation12.jsonl")
    result = {
        "source_file_sha256": hash_file(XES / "pipeline_an_scale.jsonl"),
        "split_audit": audit, "available_train": len(usable) + len(rejects), "usable_train": len(usable),
        "rejected": dict(Counter(row["kind"] for row in rejects)), "pilot_rows": len(chosen),
        "pilot_diagnosis": dict(Counter(row["teacher_privileged"]["diagnosis"] for row, _, _ in chosen)),
        "seed": SEED, "pilot_max_tokens": max(length for _, length, _ in chosen),
        "pilot_completion_tokens": sum(tokens for _, _, tokens in chosen),
        "validation_prompts": signatures, "max_length": MAX_LENGTH, "max_new_tokens": MAX_NEW,
        "input_files_sha256": {name: hash_file(out / name)
                               for name in ("train.jsonl", "probe.jsonl", "validation12.jsonl")},
    }
    save_json(out / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "validation_prompts"}, ensure_ascii=False))


def check_model(model, pin_path, validation_path, audit_path):
    audit = read_json(audit_path)
    pilot = audit_path.parent
    for name, digest in audit["input_files_sha256"].items():
        if hash_file(pilot / name) != digest:
            raise ValueError(f"Pilot input differs from frozen preparation: {name}")
    training = read_jsonl(pilot / "train.jsonl")
    probe = read_jsonl(pilot / "probe.jsonl")
    if len(training) != audit["pilot_rows"] or len(probe) != 1 or probe[0] not in training:
        raise ValueError("Unexpected pilot/probe row count or membership")
    pin = read_json(pin_path)
    config = read_json(model / "config.json")
    if config.get("model_type") != "qwen2" or config.get("num_hidden_layers") != 28:
        raise ValueError("Model config differs from Qwen2.5-7B")
    for name, digest in pin["config_and_tokenizer_sha256"].items():
        if fingerprint(read_json(model / name)) != digest:
            raise ValueError(f"Model or tokenizer JSON differs on {name}")
    weights = set(read_json(model / "model.safetensors.index.json")["weight_map"].values())
    if weights != set(pin["weights"]):
        raise ValueError("Base-model weight index differs from the pinned model")
    for name, expected in pin["weights"].items():
        path = model / name
        if path.stat().st_size != expected["bytes"] or hash_file(path) != expected["sha256"]:
            raise ValueError(f"Base model revision check failed on {name}")
    tokenizer = load_tokenizer(str(model), local_files_only=True)
    if prompt_signatures(validation_path, tokenizer) != audit["validation_prompts"]:
        raise ValueError("Base tokenizer does not reproduce the locally audited validation inputs")
    runtime_files = ("config.json", "generation_config.json", "model.safetensors.index.json", "merges.txt", "vocab.json")
    print(json.dumps({"weights_revision": pin["revision"], "runtime_file_sha256": {
        name: hash_file(model / name) for name in runtime_files if (model / name).is_file()
    }}, indent=2))


def check_checkpoint(out, train_path, trainer):
    import torch
    from safetensors.torch import load_file

    summary = read_json(out / "training_summary.json")
    rows = len(read_jsonl(train_path))
    if (summary["init_adapter"] is not None or summary["rows"] != rows
            or summary["data_sha256"] != hash_file(train_path) or summary["source_sha256"] != hash_file(trainer)
            or not math.isclose(summary["metrics"]["epoch"], 1)
            or not math.isfinite(summary["metrics"]["train_loss"])):
        raise ValueError("Unexpected training metadata")
    adapter = load_file(out / "adapter_model.safetensors")
    b_tensors = [tensor for name, tensor in adapter.items() if ".lora_B." in name]
    if (not b_tensors or not all(torch.isfinite(tensor).all().item() for tensor in adapter.values())
            or not any(torch.count_nonzero(tensor).item() for tensor in b_tensors)):
        raise ValueError("LoRA is invalid or has no nonzero B weights after training")
    print(f"Verified {rows} examples and nonzero LoRA updates")


def make_pairs(run, audit_path, validation_path, trainer, model):
    audit = read_json(audit_path)
    sig = audit["validation_prompts"]
    attempts = {}
    for name in ("base", "silver"):
        rows = read_jsonl(run / f"{name}.jsonl")
        summary = read_json(run / f"{name}_summary.json")
        adapter = Path(summary["adapter"]).resolve() if summary["adapter"] else None
        expected_adapter = None if name == "base" else (run / "silver").resolve()
        if (summary["model"] != model or summary["data_sha256"] != hash_file(validation_path)
                or summary["source_sha256"] != hash_file(trainer) or summary["split"] != "validation"
                or summary["do_sample"] or summary["dtype"] != "torch.bfloat16" or not summary["load_in_4bit"]
                or summary["max_length"] != MAX_LENGTH or summary["max_new_tokens"] != MAX_NEW
                or summary["completion_no_repeat_ngram_size"] != 0 or summary["selected_rows"] != len(sig)
                or summary["native_valid_rows"] != sum(r["valid"] for r in rows)
                or adapter != expected_adapter):
            raise ValueError(f"Generation settings differ: {name}")
        if [r["sample_id"] for r in rows] != list(sig):
            raise ValueError(f"Validation membership or order differs: {name}")
        if any(any(row[key] != value for key, value in sig[row["sample_id"]].items()) for row in rows):
            raise ValueError(f"Validation input differs: {name}")
        attempts[name] = rows
    paired = []
    for base, silver in zip(attempts["base"], attempts["silver"], strict=True):
        item = {"sample_id": base["sample_id"]}
        for name, row in (("base", base), ("silver", silver)):
            item[name] = {**row, "parsed": validate_native_output(row["output"]) if row["valid"] else None}
        paired.append(item)
    result = {"format_counts": {name: sum(r["valid"] for r in rows) for name, rows in attempts.items()},
              "total": len(sig), "rows": paired,
              "scoring_note": "Review Q1-Q3 and tutoring quality separately. JSON counts measure formatting only."}
    save_json(run / "paired.json", result)
    print(result["format_counts"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("out", type=Path)
    prep.add_argument("tokenizer")
    prep.add_argument("validation", type=Path)
    prep.add_argument("--count", type=int, default=32)
    verify = commands.add_parser("model")
    verify.add_argument("model", type=Path)
    for key in ("pin", "validation", "audit"):
        verify.add_argument(key, type=Path)
    checkpoint = commands.add_parser("checkpoint")
    for key in ("out", "train", "trainer"):
        checkpoint.add_argument(key, type=Path)
    pair = commands.add_parser("pair")
    for key in ("out", "audit", "validation", "trainer"):
        pair.add_argument(key, type=Path)
    pair.add_argument("model")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.count < 1:
            parser.error("count must be positive")
        prepare(args.out, args.tokenizer, args.validation, args.count)
    elif args.stage == "model":
        check_model(args.model, args.pin, args.validation, args.audit)
    elif args.stage == "checkpoint":
        check_checkpoint(args.out, args.train, args.trainer)
    else:
        make_pairs(args.out, args.audit, args.validation, args.trainer, args.model)


if __name__ == "__main__":
    main()
