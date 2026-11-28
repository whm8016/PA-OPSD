"""Run the Gold evidence-first experiment from a versioned input bundle."""

import argparse
import json
import shutil
import sys
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from math import sqrt
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from pa_opsd_collator import PAOPSDDataCollator, load_standard_solutions
from scripts import train_pa_opsd_distill as distill
from scripts import train_pa_opsd_sft as sft
from scripts.pa_opsd_contract import OFFICIAL, XES, read_jsonl
from scripts.run_pa_opsd_gold import check_update, verify_base

from .data import (
    BANK,
    DEV_PROMPTS,
    DEV_REFERENCES,
    FROZEN,
    MANIFEST,
    ROOT,
    TRAIN_FILES,
    assistant_target,
    digest,
    history_question_exposure,
    prepare,
    read_json,
    teacher_prompts,
    verify,
    write_json,
    write_rows,
)
from .metrics import adapter_delta, paired, score_predictions
from .teacher_context import compact_history

SEED = 42
MAX_LENGTH = 32768
EVAL_TOKENS = 3000
ROLLOUT_TOKENS = 1024
RATES = (1e-6, 5e-6, 2e-5)
FIRST_ARMS = ("legacy", "memory_first")
OPD_ARMS = ("opsd", "opsd_compact", "gkd_frozen")
TRAIN_ARMS = (*FIRST_ARMS, "sft_cont", *OPD_ARMS)
DEFAULT_PIN = FROZEN / "pa_opsd_gold_20261120/base_model.json"
HISTORY_SOURCES = ("scripts/experiments/gold_evidence_first/teacher_context.py",
                   "scripts/experiments/diagnoser_evidence/evidence.py")
SOURCE_FILES = (
    "opsd_trainer.py", "data_collator.py", "pa_opsd_collator.py", "environment.yml",
    "scripts/build_pa_opsd_gold_split.py", "scripts/pa_opsd_contract.py",
    "scripts/run_pa_opsd_gold.py", "scripts/train_pa_opsd_sft.py", "scripts/train_pa_opsd_distill.py",
    *(f"scripts/experiments/gold_evidence_first/{name}" for name in
      ("data.py", "metrics.py", "run.py", "run_stage2.sh", "README.md")),
    *HISTORY_SOURCES,
    *((OFFICIAL / name).relative_to(ROOT).as_posix()
      for name in ("eval_ai_tutor.py", "gpt_memory_diagnose.py", "openai_helper.py")),
)


def check_context(bundle, tokenizer_path):
    tokenizer = sft.load_tokenizer(str(tokenizer_path), local_files_only=True)
    tokenizer.model_max_length = MAX_LENGTH
    collator = PAOPSDDataCollator(tokenizer, MAX_LENGTH, ROLLOUT_TOKENS, standard_solution_guidance=False)
    second = read_jsonl(bundle / TRAIN_FILES["stage2"][0])
    collator.standard_solutions = load_standard_solutions(second, bundle / BANK)
    report = {}
    for arm, (name, order) in TRAIN_FILES.items():
        longest = prompt_max = teacher_max = 0
        for row in read_jsonl(bundle / name):
            encoded = sft.encode_row(row, tokenizer, MAX_LENGTH)
            labels = encoded["labels"]
            prompt_length = next(i for i, token in enumerate(labels) if token != -100)
            target_ids = encoded["input_ids"][prompt_length:]
            text = tokenizer.decode(target_ids, skip_special_tokens=True)
            if tuple(json.loads(text)) != order or json.loads(text) != assistant_target(row):
                raise ValueError("Round-tripped supervised tokens changed the target")
            longest = max(longest, len(labels))
            prompt_max = max(prompt_max, prompt_length)
            if arm == "stage2":
                student, teacher = collator.prompt_messages(row)
                if student != row["messages"][:2] or not teacher[1]["content"].startswith(student[1]["content"]):
                    raise ValueError("Teacher prompt does not preserve student-visible input")
                teacher_max = max(teacher_max, len(collator.encode_prompt(teacher)))
        report[arm] = {"max_sft_tokens": longest, "max_student_prompt_tokens": prompt_max,
                       "max_teacher_prompt_tokens": teacher_max}
    report["development"] = {}
    for mode, rows in (("student", read_jsonl(bundle / DEV_PROMPTS)), ("teacher", teacher_prompts(bundle))):
        maximum = max(len(collator.encode_prompt(row["messages"])) for row in rows)
        if maximum + EVAL_TOKENS > MAX_LENGTH:
            raise ValueError(f"Development {mode} prompt exceeds generation budget")
        report["development"][mode] = maximum
    return report


def source_config():
    import platform
    import subprocess
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ("torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True,
                            check=False).stdout.strip() if shutil.which("git") else None
    return {"source_sha256": {name: digest(ROOT / name) for name in SOURCE_FILES},
            "checkout_head": commit or None, "python": platform.python_version(), "packages": packages}


def adapter_identity(path):
    return {name: digest(path / name) for name in
            ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json")}


def check_receipt(path, manifest_hash, *, anchor=False):
    receipt = read_json(path / "run_receipt.json")
    if (receipt["manifest_sha256"] != manifest_hash or receipt["adapter"] != adapter_identity(path)
            or receipt["smoke"]):
        raise ValueError("Checkpoint does not match a completed full run of this bundle")
    if anchor and (receipt["arm"] not in FIRST_ARMS or receipt["rows"] != 420):
        raise ValueError("A full first-stage SFT checkpoint from this bundle is required")
    return receipt


def check_teacher_report(report_path, manifest_hash, anchor_id, arm="opsd", teacher_pin=None):
    report = read_json(report_path)
    if (report["manifest_sha256"] != manifest_hash or report["adapter"] != anchor_id
            or report["rows"] != 140 or report["smoke"] or not report["teacher_gain"]):
        raise ValueError("OPSD requires a full paired teacher report with a positive diagnostic signal")
    if report.get("teacher_arm", "opsd") != arm or report.get("teacher_pin_sha256") != teacher_pin:
        raise ValueError("Teacher report uses a different prompt variant or frozen model")
    if arm == "opsd_compact" and any(
        report.get("environment", {}).get("source_sha256", {}).get(name) != digest(ROOT / name)
        for name in HISTORY_SOURCES
    ):
        raise ValueError("Compact teacher report was generated with a different history implementation")


def frozen_teacher_pin(args, arm):
    if arm == "gkd_frozen":
        if args.teacher_model is None or args.teacher_pin is None:
            raise ValueError("Fixed-teacher GKD requires --teacher-model and --teacher-pin")
        verify_base(args.teacher_model, args.teacher_pin)
        return digest(args.teacher_pin)
    if args.teacher_model is not None or args.teacher_pin is not None:
        raise ValueError("Frozen teacher options apply only to gkd_frozen")
    return None


def train_args(args, data_path, steps, lr, initial, out):
    shared = ["--output-dir", str(out), "--max-length", str(MAX_LENGTH),
              "--batch-size", "1", "--gradient-accumulation-steps", str(args.accumulation),
              "--max-steps", str(steps), "--learning-rate", str(lr), "--seed", str(SEED),
              "--gradient-checkpointing", "--load-in-4bit"]
    if args.arm in OPD_ARMS:
        options = ["--train-jsonl", str(data_path), "--sft-checkpoint", str(initial),
                "--teacher-mode", "local-same-model-privileged", "--precision", "bfloat16",
                "--standard-solution-bank", str(args.bundle / BANK), "--solution-only-addendum",
                "--max-completion-length",
                str(ROLLOUT_TOKENS), "--validate-rollouts", "--checkpoint-steps", "5", *shared]
        if args.arm == "gkd_frozen":
            options += ["--frozen-teacher", str(args.teacher_model)]
        return options
    result = ["train", "--data", str(data_path), "--model", str(args.model), "--lora-r", "8",
              "--assistant-logits-only", "--offload-saved-tensors", "--local-files-only", *shared]
    return [*result, "--init-adapter", str(initial), "--checkpoint-steps", "5"] if initial else result


def longest_prompts(rows, model, count):
    tokenizer = sft.load_tokenizer(str(model), local_files_only=True)
    tokenizer.model_max_length = MAX_LENGTH
    collator = PAOPSDDataCollator(tokenizer, MAX_LENGTH, ROLLOUT_TOKENS)
    return sorted(rows, key=lambda row: len(collator.encode_prompt(row["messages"][:2])),
                  reverse=True)[:count]


def diagnosis_class_balance(rows):
    counts = Counter(assistant_target(row)["diagnosis"] for row in rows)
    # Keep all examples, but interpolate between the empirical class prior and a uniform prior.
    normalizer = len(rows) / sum(sqrt(count) for count in counts.values())
    return {"method": "inverse_sqrt_frequency", "counts": dict(counts),
            "weights": {label: normalizer / sqrt(count) for label, count in counts.items()}}


def command_train(args, manifest_hash):
    import torch
    from transformers import set_seed

    first = args.arm in FIRST_ARMS
    diagnosis_multiplier = getattr(args, "diagnosis_token_multiplier", None)
    balance_diagnosis = args.balance_diagnosis_classes or diagnosis_multiplier is not None
    if balance_diagnosis and args.arm != "memory_first":
        raise ValueError("Diagnosis weighting applies only to memory_first")
    if first and (args.anchor or args.learning_rate not in (None, 2e-5)):
        raise ValueError("Both SFT order arms start at the base and use 2e-5")
    if not first and not args.anchor:
        raise ValueError("Second-stage training requires --anchor")
    if args.anchor:
        receipt = check_receipt(args.anchor, manifest_hash, anchor=True)
        if receipt["base_pin_sha256"] != digest(args.pin):
            raise ValueError("Initial checkpoint used a different base pin")
        if Path(read_json(args.anchor / "adapter_config.json")["base_model_name_or_path"]).resolve() != args.model:
            raise ValueError("Requested model does not match the anchor's base")
    anchor_id = adapter_identity(args.anchor) if args.anchor else None
    teacher_pin = frozen_teacher_pin(args, args.arm)
    is_opd = args.arm in OPD_ARMS
    if is_opd and not args.smoke:
        if not args.teacher_report:
            raise ValueError("Provide --teacher-report from the paired development probe")
        check_teacher_report(args.teacher_report, manifest_hash, anchor_id, args.arm, teacher_pin)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Training requires a CUDA GPU with BF16 support")
    verify_base(args.model, args.pin)
    full_path = args.bundle / TRAIN_FILES[args.arm if first else "stage2"][0]
    expected = read_jsonl(full_path)
    class_balance = diagnosis_class_balance(expected) if balance_diagnosis else None
    if args.smoke:
        if args.accumulation > len(expected):
            raise ValueError("Smoke accumulation exceeds the cohort")
        expected = longest_prompts(expected, args.model, args.accumulation)
    if len(expected) % args.accumulation:
        raise ValueError("Gradient accumulation must divide selected row count")
    steps = len(expected) // args.accumulation
    if args.steps is not None:
        if first or args.smoke or not 1 <= args.steps < steps:
            raise ValueError("--steps is only for a shorter second-stage learning-rate screen")
        steps = args.steps
    lr = args.learning_rate if args.learning_rate is not None else 2e-5 if first else 1e-6
    args.output.mkdir(parents=True, exist_ok=False)
    data_path = full_path
    if args.smoke:
        data_path = args.output / "smoke_input.jsonl"
        write_rows(data_path, expected)
    options = train_args(args, data_path, steps, lr, args.anchor, args.output)
    if class_balance is not None:
        options.extend(["--diagnosis-class-weights", json.dumps(class_balance["weights"], sort_keys=True),
                        "--checkpoint-steps", "5"])
    if diagnosis_multiplier is not None:
        options.extend(["--diagnosis-token-multiplier", str(diagnosis_multiplier)])
    config = {"manifest_sha256": manifest_hash, "arm": args.arm, "smoke": args.smoke,
              "screen": args.steps is not None, "rows": len(expected), "requested_steps": steps,
              "initial_adapter": anchor_id, "learning_rate": lr, "data_sha256": digest(data_path),
              "teacher_report_sha256": digest(args.teacher_report) if args.teacher_report else None,
              "teacher_pin_sha256": teacher_pin,
              "base_pin_sha256": digest(args.pin), "argv": options,
              "environment": source_config(), "gpu": torch.cuda.get_device_name()}
    if class_balance is not None:
        config["diagnosis_class_balance"] = class_balance
    if diagnosis_multiplier is not None:
        config["diagnosis_token_multiplier"] = diagnosis_multiplier
    write_json(args.output / "run_config.json", config)

    def checked(rows, _):
        if rows != expected:
            raise ValueError("Training data differs from the verified experimental arm")

    # SFT creates LoRA before constructing Trainer, so seed both before and inside the trainer.
    set_seed(SEED)
    with ((args.output / "train.log").open("w", encoding="utf-8", buffering=1) as log,
          redirect_stdout(log), redirect_stderr(log)):
        if is_opd:
            context = compact_history if args.arm == "opsd_compact" else None
            distill.main(options, validate_manifest=checked, gold_run=True, teacher_context=context)
        elif sft.main(options, validate_manifest=checked):
            raise RuntimeError("SFT trainer failed")
    summary = read_json(args.output / "training_summary.json")
    if (summary["data_sha256"] != config["data_sha256"] or summary["learning_rate"] != lr
            or summary["rows"] != len(expected)):
        raise ValueError("Trainer summary differs from the run configuration")
    if class_balance is not None and summary["diagnosis_class_weights"] != class_balance["weights"]:
        raise ValueError("Trainer class weights differ from the run configuration")
    if diagnosis_multiplier is not None and (summary["diagnosis_token_multiplier"] != diagnosis_multiplier
                                             or summary["reweighted_tokens_in_dataset"] < len(expected)):
        raise ValueError("Trainer did not reweight the expected diagnosis tokens")
    check_update(args.output, args.anchor)
    effective_steps = (summary["teacher_context"]["optimizer_steps"] if is_opd
                       else summary["optimizer_steps"])
    if not 0 < effective_steps <= steps or (not is_opd and effective_steps != steps):
        raise ValueError("Invalid effective update count")
    receipt = {**config, "adapter": adapter_identity(args.output), "effective_steps": effective_steps,
               "budget_complete": effective_steps == steps,
               "loss_tokens": (summary["rollout_validation"]["counts"]["accepted_completion_tokens"]
                               if is_opd else summary["supervised_tokens_seen"]),
               "peak_allocated_bytes": (summary["metrics"].get("peak_allocated_bytes") if is_opd
                                        else summary.get("peak_allocated_bytes")),
               "train_runtime": summary["metrics"].get("train_runtime")}
    if args.anchor:
        receipt["adapter_delta"] = adapter_delta(args.anchor, args.output)
    write_json(args.output / "run_receipt.json", receipt)


def generate(args, prompts, name):
    path = args.bundle / DEV_PROMPTS
    external = name == "teacher" and args.teacher_arm == "gkd_frozen"
    model = args.teacher_model if external else args.model
    options = ["generate", "--data", str(path), "--model", str(model), "--local-files-only",
               "--load-in-4bit", "--max-length", str(MAX_LENGTH), "--max-new-tokens", str(EVAL_TOKENS),
               "--generation-summary", str(args.output / f"{name}_generation.json")]
    if args.checkpoint and not external:
        options += ["--adapter", str(args.checkpoint)]
    if args.smoke:
        options += ["--limit", "4"]

    def checked(rows, _):
        if rows != read_jsonl(path):
            raise ValueError("Loaded prompt data differs from the verified development bundle")
        return prompts

    output_path = args.output / f"{name}.jsonl"
    with output_path.open("w", encoding="utf-8", buffering=1) as output, redirect_stdout(output):
        status = sft.main(options, load_inference=checked)
    if status not in (0, 1):
        raise RuntimeError(f"Generation failed for {name}")
    return read_jsonl(output_path)


def reuse_student_predictions(args, prompts, manifest_hash, anchor_id):
    prior = args.student_eval
    report = read_json(prior / "report.json")
    generation = read_json(prior / "student_generation.json")
    if (report["manifest_sha256"] != manifest_hash or report["adapter"] != anchor_id
            or report["base_pin_sha256"] != digest(args.pin) or report["smoke"]
            or report["rows"] != 140 or report["mode"] != "greedy_4bit"
            or generation["max_new_tokens"] != EVAL_TOKENS or generation["max_length"] != MAX_LENGTH
            or generation["do_sample"] or not generation["load_in_4bit"]
            or generation["dtype"] != "torch.bfloat16" or generation["completion_no_repeat_ngram_size"]
            or generation["data_sha256"] != digest(args.bundle / DEV_PROMPTS)):
        raise ValueError("Saved student evaluation does not match the anchor and generation protocol")
    saved = read_jsonl(prior / "student.jsonl")
    by_id = {row["sample_id"]: row for row in saved}
    if [row["sample_id"] for row in saved] != read_json(args.bundle / MANIFEST)["ids"]["dev"]:
        raise ValueError("Saved student evaluation uses a different development cohort")
    tokenizer = sft.load_tokenizer(str(args.checkpoint), local_files_only=True)
    rows = []
    for item in prompts:
        prompt = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)
        row = by_id[item["sample_id"]]
        if row["prompt_sha256"] != sha256(prompt.encode()).hexdigest():
            raise ValueError(f"Saved student prompt differs: {item['sample_id']}")
        rows.append(row)
    return rows


def evaluate(args, manifest_hash):
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Evaluation requires a CUDA GPU with BF16 support")
    verify_base(args.model, args.pin)
    training = check_receipt(args.checkpoint, manifest_hash) if args.checkpoint else None
    if args.teacher_pair and (training is None or training["arm"] not in FIRST_ARMS):
        raise ValueError("Teacher probe requires a completed first-stage checkpoint")
    teacher_pin = frozen_teacher_pin(args, args.teacher_arm if args.teacher_pair else None)
    if training and (training["base_pin_sha256"] != digest(args.pin)
                     or Path(read_json(args.checkpoint / "adapter_config.json")["base_model_name_or_path"]
                             ).resolve() != args.model):
        raise ValueError("Requested model and pin do not match the evaluation checkpoint's base")
    prompts = read_jsonl(args.bundle / DEV_PROMPTS)
    if args.probe_manifest:
        ids = read_json(args.probe_manifest)["ids"]
        wanted = set(ids)
        prompts = [row for row in prompts if row["sample_id"] in wanted]
        if not prompts or len(prompts) != len(ids):
            raise ValueError("Teacher probe IDs must be a unique subset of the frozen development set")
    context = compact_history if args.teacher_arm == "opsd_compact" else None
    arms = {"student": prompts}
    if args.teacher_pair:
        selected_ids = {row["sample_id"] for row in prompts}
        arms["teacher"] = [row for row in teacher_prompts(args.bundle, context)
                           if row["sample_id"] in selected_ids]
    reused = (reuse_student_predictions(args, prompts[:4] if args.smoke else prompts, manifest_hash,
                                        training["adapter"]) if args.student_eval else None)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"manifest_sha256": manifest_hash, "training": training,
              "adapter": training["adapter"] if training else None, "smoke": args.smoke,
              "max_new_tokens": EVAL_TOKENS, "mode": "greedy_4bit",
              "base_pin_sha256": digest(args.pin), "environment": source_config(),
              "gpu": torch.cuda.get_device_name(),
              "scoring": "offline exact labels, normalized memory exact match, and feedback character overlap"}
    if args.teacher_pair:
        report.update(teacher_arm=args.teacher_arm, teacher_pin_sha256=teacher_pin)
    if args.student_eval:
        previous = read_json(args.student_eval / "report.json")
        report["student_eval"] = {"gpu": previous["gpu"], "environment": previous["environment"],
                                  "predictions_sha256": digest(args.student_eval / "student.jsonl")}
    if args.probe_manifest:
        report["probe_manifest_sha256"] = digest(args.probe_manifest)
    write_json(args.output / "evaluation_config.json", report)
    if reused is not None:
        write_rows(args.output / "student.jsonl", reused)
    results = {arm: reused if arm == "student" and reused is not None else generate(args, rows, arm)
               for arm, rows in arms.items()}
    selected_ids = {row["sample_id"] for row in prompts}
    references = [row for row in read_jsonl(args.bundle / DEV_REFERENCES) if row["sample_id"] in selected_ids]
    if args.smoke:
        references = references[:4]
    scores = {}
    report["arms"] = {}
    for arm, predictions in results.items():
        report["arms"][arm], scores[arm] = score_predictions(predictions, references)
        write_rows(args.output / f"{arm}_scores.jsonl", scores[arm])
    report["rows"] = len(references)
    if args.teacher_pair:
        report["paired"] = paired(scores["student"], scores["teacher"])
        report["teacher_gain"] = (
            report["paired"]["right_only"] > report["paired"]["left_only"]
            and report["arms"]["teacher"]["metrics"]["diagnosis_macro_f1"]
            > report["arms"]["student"]["metrics"]["diagnosis_macro_f1"]
        )
        if context:
            audits = [{"sample_id": row["sample_id"], **context(row["messages"])[1]}
                      for row in prompts[:report["rows"]]]
            write_rows(args.output / "teacher_history.jsonl", audits)
            report["teacher_history_fallbacks"] = sum(audit["fallback"] for audit in audits)
    write_json(args.output / "report.json", report)


def compare(left, right, output, manifest_hash, manifest, *, allow_different_gpus=False):
    l_report, r_report = (read_json(path / "report.json") for path in (left, right))
    if l_report["manifest_sha256"] != manifest_hash:
        raise ValueError("Comparison does not match the supplied development bundle")
    for key in ("manifest_sha256", "rows", "smoke", "max_new_tokens", "mode", "base_pin_sha256",
                "environment"):
        if l_report[key] != r_report[key]:
            raise ValueError(f"Evaluation protocol differs: {key}")
    if l_report["gpu"] != r_report["gpu"] and not allow_different_gpus:
        raise ValueError("Evaluation GPUs differ; pass --allow-different-gpus to report this deviation")
    l_train, r_train = l_report["training"], r_report["training"]
    for proof in (l_train, r_train):
        if proof:
            data_file = TRAIN_FILES[proof["arm"] if proof["arm"] in FIRST_ARMS else "stage2"][0]
            if proof["data_sha256"] != manifest["files"][data_file]:
                raise ValueError("Reported training data does not match its experimental arm")
    if l_train and r_train:
        left_arm, right_arm = l_train["arm"], r_train["arm"]
        matched = (left_arm in FIRST_ARMS) == (right_arm in FIRST_ARMS)
        budget_keys = ("initial_adapter", "effective_steps", "requested_steps", "rows", "environment")
        if matched and (any(l_train[k] != r_train[k] for k in budget_keys)
                        or not l_train["budget_complete"] or not r_train["budget_complete"]):
            raise ValueError("Training provenance or effective update budgets differ")
        if matched and l_train["gpu"] != r_train["gpu"] and not allow_different_gpus:
            raise ValueError("Training GPUs differ; pass --allow-different-gpus to report this deviation")
        if matched and left_arm not in FIRST_ARMS and l_train["data_sha256"] != r_train["data_sha256"]:
            raise ValueError("Second-stage training inputs differ")
        if matched and left_arm != right_arm and l_train["learning_rate"] != r_train["learning_rate"]:
            raise ValueError("Training arms must use the same learning rate")
    l_scores, r_scores = (read_jsonl(path / "student_scores.jsonl") for path in (left, right))

    def prompt_keys(directory):
        return [(row["sample_id"], row["prompt_sha256"]) for row in read_jsonl(directory / "student.jsonl")]

    left_keys, right_keys = prompt_keys(left), prompt_keys(right)
    if left_keys != right_keys or [key for key, _ in left_keys] != manifest["ids"]["dev"][:l_report["rows"]]:
        raise ValueError("Evaluation student prompts or IDs differ")
    l_metrics, r_metrics = (r["arms"]["student"]["metrics"] for r in (l_report, r_report))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"left": str(left), "right": str(right), "paired": paired(l_scores, r_scores),
                   "evaluation_gpus": [l_report["gpu"], r_report["gpu"]],
                   "training_gpus": [proof["gpu"] if proof else None for proof in (l_train, r_train)],
                   "right_minus_left": {key: r_metrics[key] - value for key, value in l_metrics.items()}},
                  stream, indent=2)


def pack(bundle, archive, pin):
    manifest = verify(bundle)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "x", ZIP_DEFLATED) as package:
        for name in SOURCE_FILES:
            package.write(ROOT / name, name)
        package.write(pin, "base_model.json")
        for name in (*manifest["files"], MANIFEST):
            package.write(bundle / name, f"inputs/{name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    modes = parser.add_subparsers(dest="command", required=True)
    build = modes.add_parser("prepare")
    build.add_argument("--frozen", type=Path, default=FROZEN)
    build.add_argument("--question-bank", type=Path, default=XES / "questions.jsonl")
    check = modes.add_parser("check")
    check.add_argument("--tokenizer", type=Path)
    check.add_argument("--output", type=Path)
    packing = modes.add_parser("pack")
    packing.add_argument("--output", type=Path, required=True)
    packing.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    training = modes.add_parser("train")
    training.add_argument("arm", choices=TRAIN_ARMS)
    training.add_argument("--model", type=Path, required=True)
    training.add_argument("--pin", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--anchor", type=Path)
    training.add_argument("--teacher-report", type=Path)
    training.add_argument("--teacher-model", type=Path)
    training.add_argument("--teacher-pin", type=Path)
    training.add_argument("--learning-rate", type=float, choices=RATES)
    training.add_argument("--accumulation", type=int, default=4)
    training.add_argument("--steps", type=int)
    training.add_argument("--smoke", action="store_true")
    weighting = training.add_mutually_exclusive_group()
    weighting.add_argument("--balance-diagnosis-classes", action="store_true",
                           help="Apply inverse square-root class weights to the full memory-first SFT reply")
    weighting.add_argument("--diagnosis-token-multiplier", type=float,
                           help="Apply inverse square-root class weights times this factor only at diagnosis tokens")
    evaluation = modes.add_parser("evaluate")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--pin", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--teacher-pair", action="store_true")
    evaluation.add_argument("--teacher-arm", choices=OPD_ARMS, default="opsd")
    evaluation.add_argument("--teacher-model", type=Path)
    evaluation.add_argument("--teacher-pin", type=Path)
    evaluation.add_argument("--probe-manifest", type=Path, help="Use a frozen development subset for a teacher probe")
    evaluation.add_argument("--student-eval", type=Path, help="Reuse the verified anchor's full development predictions")
    evaluation.add_argument("--smoke", action="store_true")
    comparison = modes.add_parser("compare")
    comparison.add_argument("left", type=Path)
    comparison.add_argument("right", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--allow-different-gpus", action="store_true")
    args = parser.parse_args(argv)
    if (args.command == "train" and args.diagnosis_token_multiplier is not None
            and not 1.0 <= args.diagnosis_token_multiplier <= 100.0):
        parser.error("--diagnosis-token-multiplier must be between 1 and 100")
    args.bundle = args.bundle.resolve()
    if args.command == "prepare":
        manifest = prepare(args.bundle, args.frozen, args.question_bank)
        print(json.dumps(manifest["counts"], indent=2))
        return
    manifest = verify(args.bundle)
    identity = digest(args.bundle / MANIFEST)
    if args.command == "check":
        audit = {"manifest_sha256": identity, "counts": manifest["counts"],
                 "history_question_exposure": history_question_exposure(args.bundle)}
        if args.tokenizer:
            audit["token_lengths"] = check_context(args.bundle, args.tokenizer)
        if args.output:
            write_json(args.output, audit)
        print(json.dumps(audit, indent=2))
    elif args.command == "pack":
        pack(args.bundle, args.output, args.pin)
    elif args.command == "compare":
        compare(args.left, args.right, args.output, identity, manifest, allow_different_gpus=args.allow_different_gpus)
    else:
        args.model = args.model.resolve(strict=True)
        args.output = args.output.resolve()
        if args.teacher_model is not None:
            args.teacher_model = args.teacher_model.resolve(strict=True)
        if args.command == "train":
            if args.accumulation < 1:
                parser.error("--accumulation must be positive")
            args.anchor = args.anchor.resolve(strict=True) if args.anchor else None
            command_train(args, identity)
        else:
            if (args.teacher_arm != "opsd" or args.probe_manifest or args.student_eval) and not args.teacher_pair:
                parser.error("Teacher variant, subset and cached student predictions require --teacher-pair")
            args.checkpoint = args.checkpoint.resolve(strict=True) if args.checkpoint else None
            evaluate(args, identity)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
