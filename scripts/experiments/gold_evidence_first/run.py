"""Run the Gold evidence-first experiment from a versioned input bundle."""

import argparse
import json
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
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

SEED = 42
MAX_LENGTH = 32768
EVAL_TOKENS = 3000
ROLLOUT_TOKENS = 1024
RATES = (1e-6, 5e-6, 2e-5)
FIRST_ARMS = ("legacy", "memory_first")
TRAIN_ARMS = (*FIRST_ARMS, "sft_cont", "opsd")
DEFAULT_PIN = FROZEN / "pa_opsd_gold_20261120/base_model.json"
SOURCE_FILES = (
    "opsd_trainer.py", "data_collator.py", "pa_opsd_collator.py", "environment.yml",
    "scripts/build_pa_opsd_gold_split.py", "scripts/pa_opsd_contract.py",
    "scripts/run_pa_opsd_gold.py", "scripts/train_pa_opsd_sft.py", "scripts/train_pa_opsd_distill.py",
    *(f"scripts/experiments/gold_evidence_first/{name}" for name in ("data.py", "metrics.py", "run.py", "README.md")),
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


def check_teacher_report(report_path, manifest_hash, anchor_id):
    report = read_json(report_path)
    if (report["manifest_sha256"] != manifest_hash or report["adapter"] != anchor_id
            or report["rows"] != 140 or report["smoke"] or not report["teacher_gain"]):
        raise ValueError("OPSD requires a full paired teacher report with a positive diagnostic signal")


def train_args(args, data_path, steps, lr, initial, out):
    shared = ["--output-dir", str(out), "--max-length", str(MAX_LENGTH),
              "--batch-size", "1", "--gradient-accumulation-steps", str(args.accumulation),
              "--max-steps", str(steps), "--learning-rate", str(lr), "--seed", str(SEED),
              "--gradient-checkpointing", "--load-in-4bit"]
    if args.arm == "opsd":
        return ["--train-jsonl", str(data_path), "--sft-checkpoint", str(initial),
                "--teacher-mode", "local-same-model-privileged", "--precision", "bfloat16",
                "--standard-solution-bank", str(args.bundle / BANK), "--solution-only-addendum",
                "--max-completion-length",
                str(ROLLOUT_TOKENS), "--validate-rollouts", "--checkpoint-steps", "5", *shared]
    result = ["train", "--data", str(data_path), "--model", str(args.model), "--lora-r", "8",
              "--assistant-logits-only", "--offload-saved-tensors", "--local-files-only", *shared]
    return [*result, "--init-adapter", str(initial)] if initial else result


def longest_prompts(rows, model, count):
    tokenizer = sft.load_tokenizer(str(model), local_files_only=True)
    tokenizer.model_max_length = MAX_LENGTH
    collator = PAOPSDDataCollator(tokenizer, MAX_LENGTH, ROLLOUT_TOKENS)
    return sorted(rows, key=lambda row: len(collator.encode_prompt(row["messages"][:2])),
                  reverse=True)[:count]


def command_train(args, manifest_hash):
    import torch
    from transformers import set_seed

    first = args.arm in FIRST_ARMS
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
    if args.arm == "opsd" and not args.smoke:
        if not args.teacher_report:
            raise ValueError("Provide --teacher-report from the paired development probe")
        check_teacher_report(args.teacher_report, manifest_hash, anchor_id)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Training requires a CUDA GPU with BF16 support")
    verify_base(args.model, args.pin)
    full_path = args.bundle / TRAIN_FILES[args.arm if first else "stage2"][0]
    expected = read_jsonl(full_path)
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
    config = {"manifest_sha256": manifest_hash, "arm": args.arm, "smoke": args.smoke,
              "screen": args.steps is not None, "rows": len(expected), "requested_steps": steps,
              "initial_adapter": anchor_id, "learning_rate": lr, "data_sha256": digest(data_path),
              "teacher_report_sha256": digest(args.teacher_report) if args.teacher_report else None,
              "base_pin_sha256": digest(args.pin), "argv": options,
              "environment": source_config(), "gpu": torch.cuda.get_device_name()}
    write_json(args.output / "run_config.json", config)

    def checked(rows, _):
        if rows != expected:
            raise ValueError("Training data differs from the verified experimental arm")

    # SFT creates LoRA before constructing Trainer, so seed both before and inside the trainer.
    set_seed(SEED)
    with ((args.output / "train.log").open("w", encoding="utf-8", buffering=1) as log,
          redirect_stdout(log), redirect_stderr(log)):
        if args.arm == "opsd":
            distill.main(options, validate_manifest=checked, gold_run=True)
        elif sft.main(options, validate_manifest=checked):
            raise RuntimeError("SFT trainer failed")
    summary = read_json(args.output / "training_summary.json")
    if (summary["data_sha256"] != config["data_sha256"] or summary["learning_rate"] != lr
            or summary["rows"] != len(expected)):
        raise ValueError("Trainer summary differs from the run configuration")
    check_update(args.output, args.anchor)
    effective_steps = (summary["teacher_context"]["optimizer_steps"] if args.arm == "opsd"
                       else summary["optimizer_steps"])
    if not 0 < effective_steps <= steps or (args.arm != "opsd" and effective_steps != steps):
        raise ValueError("Invalid effective update count")
    receipt = {**config, "adapter": adapter_identity(args.output), "effective_steps": effective_steps,
               "budget_complete": effective_steps == steps,
               "loss_tokens": (summary["rollout_validation"]["counts"]["accepted_completion_tokens"]
                               if args.arm == "opsd" else summary["supervised_tokens_seen"])}
    if args.anchor:
        receipt["adapter_delta"] = adapter_delta(args.anchor, args.output)
    write_json(args.output / "run_receipt.json", receipt)


def generate(args, prompts, name):
    path = args.bundle / DEV_PROMPTS
    options = ["generate", "--data", str(path), "--model", str(args.model), "--local-files-only",
               "--load-in-4bit", "--max-length", str(MAX_LENGTH), "--max-new-tokens", str(EVAL_TOKENS),
               "--generation-summary", str(args.output / f"{name}_generation.json")]
    if args.checkpoint:
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


def evaluate(args, manifest_hash):
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Evaluation requires a CUDA GPU with BF16 support")
    verify_base(args.model, args.pin)
    training = check_receipt(args.checkpoint, manifest_hash) if args.checkpoint else None
    if args.teacher_pair and (training is None or training["arm"] not in FIRST_ARMS):
        raise ValueError("Teacher probe requires a completed first-stage checkpoint")
    if training and (training["base_pin_sha256"] != digest(args.pin)
                     or Path(read_json(args.checkpoint / "adapter_config.json")["base_model_name_or_path"]
                             ).resolve() != args.model):
        raise ValueError("Requested model and pin do not match the evaluation checkpoint's base")
    prompts = read_jsonl(args.bundle / DEV_PROMPTS)
    arms = {"student": prompts, **({"teacher": teacher_prompts(args.bundle)} if args.teacher_pair else {})}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"manifest_sha256": manifest_hash, "training": training,
              "adapter": training["adapter"] if training else None, "smoke": args.smoke,
              "max_new_tokens": EVAL_TOKENS, "mode": "greedy_4bit",
              "base_pin_sha256": digest(args.pin), "environment": source_config(),
              "gpu": torch.cuda.get_device_name(),
              "scoring": "offline exact labels, normalized memory exact match, and feedback character overlap"}
    write_json(args.output / "evaluation_config.json", report)
    results = {arm: generate(args, rows, arm) for arm, rows in arms.items()}
    references = read_jsonl(args.bundle / DEV_REFERENCES)
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
    write_json(args.output / "report.json", report)


def compare(left, right, output, manifest_hash, manifest):
    l_report, r_report = (read_json(path / "report.json") for path in (left, right))
    if l_report["manifest_sha256"] != manifest_hash:
        raise ValueError("Comparison does not match the supplied development bundle")
    for key in ("manifest_sha256", "rows", "smoke", "max_new_tokens", "mode", "base_pin_sha256",
                "environment", "gpu"):
        if l_report[key] != r_report[key]:
            raise ValueError(f"Evaluation protocol differs: {key}")
    l_train, r_train = l_report["training"], r_report["training"]
    for proof in (l_train, r_train):
        if proof:
            data_file = TRAIN_FILES[proof["arm"] if proof["arm"] in FIRST_ARMS else "stage2"][0]
            if proof["data_sha256"] != manifest["files"][data_file]:
                raise ValueError("Reported training data does not match its experimental arm")
    if l_train and r_train:
        left_arm, right_arm = l_train["arm"], r_train["arm"]
        matched = (left_arm in FIRST_ARMS) == (right_arm in FIRST_ARMS)
        budget_keys = ("initial_adapter", "effective_steps", "requested_steps", "rows", "environment", "gpu")
        if matched and (any(l_train[k] != r_train[k] for k in budget_keys)
                        or not l_train["budget_complete"] or not r_train["budget_complete"]):
            raise ValueError("Training provenance or effective update budgets differ")
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
    training.add_argument("--learning-rate", type=float, choices=RATES)
    training.add_argument("--accumulation", type=int, default=4)
    training.add_argument("--steps", type=int)
    training.add_argument("--smoke", action="store_true")
    evaluation = modes.add_parser("evaluate")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--pin", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--teacher-pair", action="store_true")
    evaluation.add_argument("--smoke", action="store_true")
    comparison = modes.add_parser("compare")
    comparison.add_argument("left", type=Path)
    comparison.add_argument("right", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
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
        compare(args.left, args.right, args.output, identity, manifest)
    else:
        args.model = args.model.resolve(strict=True)
        args.output = args.output.resolve()
        if args.command == "train":
            if args.accumulation < 1:
                parser.error("--accumulation must be positive")
            args.anchor = args.anchor.resolve(strict=True) if args.anchor else None
            command_train(args, identity)
        else:
            args.checkpoint = args.checkpoint.resolve(strict=True) if args.checkpoint else None
            evaluate(args, identity)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
