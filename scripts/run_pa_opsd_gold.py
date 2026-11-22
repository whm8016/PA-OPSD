"""Run and score a frozen LongTutor Gold experiment using the shared trainers."""

import argparse
import json
import math
import sys
import unicodedata
from collections import Counter
from hashlib import sha256
from pathlib import Path

from pa_opsd_contract import read_jsonl
from train_pa_opsd_sft import encode_row, load_tokenizer, validate_native_output

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("base", "sft", "sft_cont", "pa_opsd")
DIAGNOSES = ("Recall Failure", "Conceptual Gap", "Procedural Error", "Transfer Deficit")
MAX_LENGTH = 32768
EVAL_TOKENS = 1536
ROLLOUT_TOKENS = 512

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_rows(root, manifest, scope, filename):
    path = Path(root) / filename
    data = path.read_bytes()
    if sha256(data).hexdigest() != manifest["files"][f"{scope}/{filename}"]:
        raise ValueError(f"File differs from frozen Gold split: {path}")
    rows = [json.loads(line) for line in data.decode("utf-8").split("\n") if line]
    split = "train" if filename == "train.jsonl" else "validation" if scope == "development" else "test"
    if [row["sample_id"] for row in rows] != manifest["ids"][split]:
        raise ValueError(f"IDs differ from frozen {split} split")
    return rows


def verify_training(rows, manifest_path):
    manifest_path = Path(manifest_path)
    frozen = checked_rows(manifest_path.parent / "development", read_json(manifest_path),
                          "development", "train.jsonl")
    indexed = {row["sample_id"]: row for row in frozen}
    keys = [row["sample_id"] for row in rows]
    if not rows or len(set(keys)) != len(keys) or any(row != indexed.get(key) for row, key in zip(rows, keys)):
        raise ValueError("Training can only use the frozen Gold training rows")


def verify_inference(rows, manifest_path):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    ids = [row["sample_id"] for row in rows]
    scope = "sealed" if ids == manifest["ids"]["test"] else "development"
    folder = manifest_path.parent / scope
    if rows != checked_rows(folder, manifest, scope, "prompts.jsonl"):
        raise ValueError("Inference requires the frozen, label-free prompts")
    return rows


def verify_base(model, pin_path):
    pin = read_json(pin_path)
    for name, digest in pin["config_and_tokenizer_sha256"].items():
        if sha256((model / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Base model config/tokenizer differs from pinned revision: {name}")
    for name, info in pin["weights"].items():
        path = model / name
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != info["sha256"]:
            raise ValueError(f"Base model weight shard differs from pinned revision: {name}")


def preflight(manifest_path, model, out, pin_path):
    from pa_opsd_collator import PAOPSDDataCollator

    verify_base(model, pin_path)
    manifest = read_json(manifest_path)
    folder = manifest_path.parent / "development"
    tokenizer = load_tokenizer(str(model), local_files_only=True)
    tokenizer.model_max_length = MAX_LENGTH
    collator = PAOPSDDataCollator(tokenizer, MAX_LENGTH, ROLLOUT_TOKENS)
    train = checked_rows(folder, manifest, "development", "train.jsonl")
    validation = checked_rows(folder, manifest, "development", "prompts.jsonl")
    checked_rows(folder, manifest, "development", "references.jsonl")
    sft_sizes, completion_sizes = [], []
    for row in train:
        features = encode_row(row, tokenizer, MAX_LENGTH)
        sft_sizes.append(len(features["input_ids"]))
        completion_sizes.append(sum(token != -100 for token in features["labels"]))
    teacher_sizes = [len(collator.encode_prompt(collator.prompt_messages(row)[1])) for row in train]
    validation_sizes = [len(collator.encode_prompt(row["messages"])) for row in validation]
    if max(validation_sizes) + EVAL_TOKENS > MAX_LENGTH:
        raise ValueError("Gold validation prompt leaves insufficient generation room")
    # Probe with the worst-length row for each training objective before a GPU run.
    probes = {"sft": max(range(len(train)), key=sft_sizes.__getitem__),
              "pa_opsd": max(range(len(train)), key=teacher_sizes.__getitem__)}
    for arm, index in probes.items():
        (out / f"probe_{arm}.jsonl").write_text(json.dumps(train[index], ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    with (out / "continuation.jsonl").open("w", encoding="utf-8") as stream:
        for row in train[:140]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"train_rows": len(train), "validation_rows": len(validation),
                      "max_sft_tokens": max(sft_sizes),
                      "max_teacher_prompt_tokens": max(teacher_sizes),
                      "max_validation_prompt_tokens": max(validation_sizes),
                      "max_gold_completion_tokens": max(completion_sizes),
                      "max_length": MAX_LENGTH, "rollout_tokens": ROLLOUT_TOKENS, "eval_tokens": EVAL_TOKENS}))


def check_final(manifest_path, model, run_dir, pin_path):
    verify_base(model, pin_path)
    for arm in ARMS[1:]:
        adapter = read_json(run_dir / arm / "adapter_config.json")
        if Path(adapter["base_model_name_or_path"]).resolve() != model.resolve():
            raise ValueError(f"Base model path for {arm} differs from the development run")
    manifest = read_json(manifest_path)
    for filename in ("prompts.jsonl", "references.jsonl"):
        checked_rows(manifest_path.parent / "sealed", manifest, "sealed", filename)


def normalized(value):
    return "".join(unicodedata.normalize("NFKC", value).casefold().split()) if isinstance(value, str) else ""


def row_scores(prediction, target):
    try:
        pred = json.loads(prediction)
    except (ValueError, TypeError):
        pred = {}
    if not isinstance(pred, dict):
        pred = {}
    diagnosis = next((label for label in DIAGNOSES if normalized(pred.get("diagnosis")) == normalized(label)), None)
    memory = pred.get("memory")
    items = [item for item in memory if isinstance(item, dict)] if isinstance(memory, list) else []
    result = {"strict_json": 0, "diagnosis": diagnosis, "diagnosis_gold": target["diagnosis"],
              "strategy_correct": int(normalized(pred.get("strategy")) == normalized(target["strategy"]))}
    for truth in target["memory"]:
        found = [normalized(item.get("answer")) for item in items if item.get("id") == truth["id"]]
        result[f'{truth["id"]}_exact'] = int(found == [normalized(truth["answer"])])
    # Character overlap is a cheap lexical signal for Chinese feedback.
    actual, expected = Counter(normalized(pred.get("content"))), Counter(normalized(target["content"]))
    result["feedback_char_f1"] = 2 * sum((actual & expected).values()) / (sum(actual.values()) + sum(expected.values()))
    try:
        validate_native_output(prediction)
        result["strict_json"] = 1
    except (ValueError, TypeError):
        pass
    return result


def aggregate(rows):
    fields = ("strict_json", "strategy_correct", "Q1_exact", "Q2_exact", "Q3_exact", "feedback_char_f1")
    metrics = {field: sum(row[field] for row in rows) / len(rows) for field in fields}
    per_class = {}
    for label in DIAGNOSES:
        n_gold = sum(row["diagnosis_gold"] == label for row in rows)
        n_pred = sum(row["diagnosis"] == label for row in rows)
        tp = sum(row["diagnosis"] == row["diagnosis_gold"] == label for row in rows)
        per_class[label] = {"support": n_gold, "correct": tp, "f1": 2 * tp / (n_pred + n_gold) if n_pred + n_gold else 0}
    supported = [row["f1"] for row in per_class.values() if row["support"]]
    metrics["diagnosis_macro_f1"] = sum(supported) / len(supported)
    metrics["diagnosis_accuracy"] = sum(row["correct"] for row in per_class.values()) / len(rows)
    return {"metrics": metrics, "diagnosis_per_class": per_class}


def score(manifest_path, scope, run_dir, limit):
    manifest = read_json(manifest_path)
    folder = manifest_path.parent / scope
    references = checked_rows(folder, manifest, scope, "references.jsonl")[:limit]
    ids = [row["sample_id"] for row in references]
    prompt_hashes = None
    arms = {}
    paired = {key: {} for key in ids}
    for arm in ARMS:
        predictions = read_jsonl(run_dir / f"{arm}.jsonl")
        if [row["sample_id"] for row in predictions] != ids:
            raise ValueError(f"Evaluation IDs differ for {arm}")
        hashes = [row["prompt_sha256"] for row in predictions]
        if prompt_hashes is not None and prompt_hashes != hashes:
            raise ValueError(f"Evaluation prompts differ for {arm}")
        prompt_hashes = hashes
        scores = [row_scores(pred["output"], ref["target"]) for pred, ref in zip(predictions, references)]
        arms[arm] = aggregate(scores)
        for key, metrics in zip(ids, scores):
            paired[key][arm] = metrics
    delta = {metric: value - arms["sft_cont"]["metrics"][metric]
             for metric, value in arms["pa_opsd"]["metrics"].items()}
    report = {"scope": scope, "samples": len(ids), "arms": arms, "pa_opsd_minus_sft_cont": delta}
    (run_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "paired_metrics.jsonl").write_text(
        "".join(json.dumps({"sample_id": key, **item}, ensure_ascii=False) + "\n" for key, item in paired.items()),
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def check_update(folder, init=None):
    checkpoint = folder / "adapter_model.safetensors"
    if checkpoint.stat().st_size == 0:
        raise ValueError(f"Empty adapter: {folder}")
    loss = read_json(folder / "training_summary.json")["metrics"]["train_loss"]
    if not math.isfinite(loss):
        raise ValueError(f"Non-finite training loss: {folder}")
    import torch
    from safetensors.torch import load_file

    after = load_file(checkpoint)
    before = load_file(init / checkpoint.name) if init else None
    if not after or any(not torch.isfinite(tensor).all().item() for tensor in after.values()):
        raise ValueError(f"Invalid adapter tensors: {folder}")
    if before is None:
        changed = any(tensor.count_nonzero().item() for key, tensor in after.items() if ".lora_B." in key)
    else:
        if before.keys() != after.keys() or any(before[key].dtype != after[key].dtype for key in before):
            raise ValueError(f"Adapter tensor names or dtypes changed: {folder}")
        changed = any(not torch.equal(before[key], after[key]) for key in before)
    if not changed:
        raise ValueError(f"Training did not change adapter tensors: {folder}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("sft", "opsd", "preflight", "check-final", "score", "check-update"))
    args, rest = parser.parse_known_args(argv)
    if args.task == "sft":
        from train_pa_opsd_sft import main as trainer

        return trainer(rest, validate_manifest=verify_training, load_inference=verify_inference)
    if args.task == "opsd":
        from train_pa_opsd_distill import main as trainer

        return trainer(rest, validate_manifest=verify_training, gold_run=True)
    positional = argparse.ArgumentParser(prog=f"{parser.prog} {args.task}")
    if args.task == "check-update":
        positional.add_argument("checkpoint", type=Path)
        positional.add_argument("--initial", type=Path)
        values = positional.parse_args(rest)
        return check_update(values.checkpoint, values.initial)
    positional.add_argument("manifest", type=Path)
    if args.task in {"preflight", "check-final"}:
        positional.add_argument("model", type=Path)
        positional.add_argument("out", type=Path)
        positional.add_argument("pin", type=Path)
        values = positional.parse_args(rest)
        check = preflight if args.task == "preflight" else check_final
        return check(values.manifest, values.model, values.out, values.pin)
    positional.add_argument("scope", choices=("development", "sealed"))
    positional.add_argument("run_dir", type=Path)
    positional.add_argument("--limit", type=int)
    values = positional.parse_args(rest)
    return score(values.manifest, values.scope, values.run_dir, values.limit)


if __name__ == "__main__":
    raise SystemExit(main())
