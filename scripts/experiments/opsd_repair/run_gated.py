"""Package and run the fixed Gold standard-solution OPSD rollout-filter comparison."""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[3]
ENTRY = "scripts/experiments/opsd_repair/run_gated.py"
DRIVER = "scripts/run_pa_opsd_gold.py"
GOLD = ROOT / "outputs/pa_opsd/gold_split_20261120"
ANCHOR = GOLD / "autodl_4090d_50861/formal/sft"
EVAL_BUNDLE = ROOT / "outputs/pa_opsd/longtutor_reeval_gold140/longtutor_reeval_gold140.zip"
FROZEN = {
    "continuation.jsonl": "677d7714d1d0710fca8070b7253a395bafb786b3afedf258f78b524cc0c836fd",
    "questions.jsonl": "e48ee4d7846964b03876437acfbf5ea965578ba851451f672c856651749d5657",
    "anchor/adapter_model.safetensors": "40be07306e9070de6b47d9dfdd372ae7a6183b81aad1d728ae27141f4f37313b",
}
STEPS = {"smoke": 2, "formal": 35}
ACCUMULATION = 4
COMPLETION_TOKENS = 1024


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(output):
    sources = {name: ROOT / name for name in (
        ENTRY, DRIVER, "data_collator.py", "opsd_trainer.py", "pa_opsd_collator.py",
        "scripts/pa_opsd_contract.py", "scripts/train_pa_opsd_sft.py", "scripts/train_pa_opsd_distill.py",
        "data/longtutor/repo/scripts/eval_ai_tutor.py", "data/longtutor/repo/scripts/gpt_memory_diagnose.py",
        "data/longtutor/repo/scripts/openai_helper.py",
    )}
    sources.update({
        "split_manifest.json": GOLD / "split_manifest.json",
        "development/train.jsonl": GOLD / "development/train.jsonl",
        "questions.jsonl": ROOT / "data/longtutor/repo/data/XES3G5M/questions.jsonl",
    })
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "chat_template.jinja", "added_tokens.json",
                 "special_tokens_map.json", "training_summary.json"):
        sources[f"anchor/{name}"] = ANCHOR / name
    training = sources["development/train.jsonl"].read_bytes()
    split = read_json(sources["split_manifest.json"])
    if sha256(training).hexdigest() != split["files"]["development/train.jsonl"]:
        raise ValueError("Frozen Gold training file changed")
    rows = training.decode("utf-8").splitlines()[:140]
    continuation = "".join(json.dumps(json.loads(row), ensure_ascii=False) + "\n" for row in rows).encode()
    with ZipFile(EVAL_BUNDLE) as reference:
        base_pin = reference.read("base_model.json")
        reference_manifest = json.loads(reference.read("manifest.json"))
    if sha256(base_pin).hexdigest() != reference_manifest["files"]["base_model.json"]:
        raise ValueError("Official evaluation base model pin differs from its manifest")
    hashes = {name: sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    hashes["continuation.jsonl"] = sha256(continuation).hexdigest()
    hashes["base_model.json"] = sha256(base_pin).hexdigest()
    if any(hashes[name] != digest for name, digest in FROZEN.items()):
        raise ValueError("Continuation data, question bank or SFT adapter differs from the original comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for name, source in sources.items():
            archive.write(source, name)
        archive.writestr("continuation.jsonl", continuation)
        archive.writestr("base_model.json", base_pin)
        archive.writestr("inputs.sha256.json", json.dumps(hashes, indent=2) + "\n")


def training_args(stage, output):
    args = [sys.executable, "-u", DRIVER, "opsd", "--train-jsonl", "continuation.jsonl",
            "--seed-manifest", "split_manifest.json", "--sft-checkpoint", "anchor",
            "--standard-solution-bank", "questions.jsonl", "--teacher-mode", "local-same-model-privileged",
            "--output-dir", str(output), "--load-in-4bit", "--precision", "bfloat16", "--max-length", "32768",
            "--max-completion-length", str(COMPLETION_TOKENS), "--batch-size", "1",
            "--gradient-accumulation-steps", str(ACCUMULATION), "--max-steps", str(STEPS[stage]),
            "--learning-rate", "0.000001", "--seed", "42", "--gradient-checkpointing", "--validate-rollouts"]
    if stage == "formal":
        args += ["--checkpoint-steps", "3"]
    return args


def verify_bundle():
    hashes = read_json(ROOT / "inputs.sha256.json")
    if any(hashes.get(name) != digest for name, digest in FROZEN.items()):
        raise ValueError("Bundle does not match the frozen comparison")
    for name, digest in hashes.items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Bundle input changed: {name}")
    return hashes


def rollout_totals(rows, frozen_ids):
    attempts = {}
    statuses = {"valid", "missing_eos", "invalid_json", "invalid_native_schema"}
    for row in rows:
        if row["sample_id"] not in frozen_ids:
            raise ValueError("Logged rollout is outside the frozen continuation data")
        key = (row["step"], row["sample_id"])
        history = attempts.setdefault(key, [])
        if (row["attempt"] != len(history) + 1 or row["attempt"] > 2 or
                row["status"] not in statuses or not 1 <= row["completion_tokens"] <= COMPLETION_TOKENS or
                (history and history[-1]["status"] == "valid")):
            raise ValueError(f"Invalid rollout retry trace for {key}")
        history.append(row)
    totals = {"first_pass": 0, "saved_by_retry": 0, "skipped": 0, "accepted_tokens": 0}
    for key, history in attempts.items():
        final = history[-1]
        if final["status"] == "valid":
            totals["first_pass" if len(history) == 1 else "saved_by_retry"] += 1
            totals["accepted_tokens"] += final["completion_tokens"]
        elif len(history) == 2:
            totals["skipped"] += 1
        else:
            raise ValueError(f"Missing rollout retry for {key}")
    return totals


def audit_run(stage, output):
    summary = read_json(output / "training_summary.json")
    validation = summary["rollout_validation"]
    expected_windows = STEPS[stage]
    effective = validation["effective_optimizer_steps"]
    frozen_ids = {json.loads(line)["sample_id"] for line in (ROOT / "continuation.jsonl").read_bytes().splitlines()}
    files = sorted((output / "generations").glob("generations_step_*.json"),
                   key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    rows = [row for path in files for row in read_json(path)["generations"]]
    totals = rollout_totals(rows, frozen_ids)
    observed = totals["first_pass"] + totals["saved_by_retry"] + totals["skipped"]
    covered = len({row["sample_id"] for row in rows if row["attempt"] == 1})
    counts = validation["counts"]
    valid_windows = len({row["step"] for row in rows if row["status"] == "valid"})
    first_by_window = Counter(row["step"] for row in rows if row["attempt"] == 1)
    if (validation["optimizer_windows"] != expected_windows or covered != observed or effective != valid_windows or
            first_by_window != {step: ACCUMULATION for step in range(expected_windows)} or
            any(counts.get(key, 0) != totals[field] for key, field in
                (("attempt_1_valid", "first_pass"), ("attempt_2_valid", "saved_by_retry"),
                 ("skipped_rollouts", "skipped")))):
        raise ValueError("Rollout logs do not reconcile with the training summary and fixed sample budget")
    write_json(output / "rollout_audit.json", {**totals, "effective_optimizer_steps": effective,
                                             "attempted_optimizer_windows": expected_windows,
                                             "unique_train_ids": covered,
                                             "by_gold_diagnosis": validation["skipped_by_gold_diagnosis"]})
    if stage == "formal" and effective != expected_windows:
        raise ValueError(f"Only {effective}/{expected_windows} actual updates; keep as a partial experiment")


def run(stage, output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"stage": stage, "status": "running"})
    command = training_args("smoke" if stage == "check" else stage, output)
    if stage == "check":
        command.append("--check-data-only")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8",
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TRL_EXPERIMENTAL_SILENCE": "1"}
    try:
        hashes = verify_bundle()
        write_json(output / "config.json", {"stage": stage, "command": command, "inputs": hashes})
        with (output / "train.log").open("wb") as log:
            if stage == "check":
                subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            else:
                # The Gold CLI imports sibling modules as a standalone script.
                sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
                from scripts.run_pa_opsd_gold import verify_base

                packages = ("torch", "transformers", "peft", "accelerate", "datasets", "trl", "bitsandbytes")
                write_json(output / "runtime.json", {"python": sys.version,
                                                     "packages": {name: version(name) for name in packages}})
                model = Path(read_json(ROOT / "anchor/adapter_config.json")["base_model_name_or_path"])
                pin_path = ROOT / "base_model.json"
                pin = read_json(pin_path)
                for name, key in (("generation_config.json", "generation_config_sha256"),
                                  ("model.safetensors.index.json", "weight_index_sha256")):
                    if sha256((model / name).read_bytes()).hexdigest() != pin[key]:
                        raise ValueError(f"Base model file differs from pinned revision: {name}")
                verify_base(model, pin_path)
                with (output / "gpu_monitor.csv").open("wb") as gpu_log:
                    monitor = subprocess.Popen([
                        "nvidia-smi", "--query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits", "-l", "3",
                    ], stdout=gpu_log, stderr=subprocess.STDOUT)
                    try:
                        subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
                    finally:
                        monitor.terminate()
                        monitor.wait()
                audit_run(stage, output)
                subprocess.run([sys.executable, DRIVER, "check-update", str(output), "--initial", "anchor"],
                               cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    except Exception as error:
        write_json(output / "status.json", {"stage": stage, "status": "failed", "error": str(error)})
        raise
    except KeyboardInterrupt:
        write_json(output / "status.json", {"stage": stage, "status": "interrupted"})
        raise
    write_json(output / "status.json", {"stage": stage, "status": "completed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", *STEPS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output.resolve())
    else:
        run(args.stage, args.output.resolve())


if __name__ == "__main__":
    main()
