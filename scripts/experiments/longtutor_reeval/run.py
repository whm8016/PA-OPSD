"""Re-evaluate the three frozen Gold adapters with the published LongTutor protocol."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else Path.cwd()
GOLD = Path("outputs/pa_opsd/gold_split_20261120")
REPRO = Path("outputs/longtutor_b1/official_ba40af3")
REPRO_RUN = REPRO / "autodl_5090_20765"
ARMS = ("base", "sft", "sft_cont", "pa_opsd")
LABELS = ("Recall Failure", "Conceptual Gap", "Procedural Error", "Transfer Deficit")
DIMENSIONS = ("history_score", "strategy_score", "coherence_score", "comprehension_score")
BASE_NAME = "Qwen2.5-7B-Instruct"
VLLM_VERSION = "0.9.2"
GENERATION_CONFIG_SHA256 = "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f"
WEIGHT_INDEX_SHA256 = "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028"
EVAL = {"temperature": 0, "max_tokens": 3000, "max_model_len": 32768, "dtype": "bfloat16"}
GOLD_FILE = "data/XES3G5M/human_an_updated.jsonl"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def digest(path):
    checksum = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def index(records, key):
    indexed = {}
    for record in records:
        identifier = record[key]
        if identifier in indexed:
            raise ValueError(f"Duplicate ID: {identifier}")
        indexed[identifier] = record
    return indexed


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_upstream(directory):
    scripts = str(directory / "upstream")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import eval_ai_tutor as upstream

    if Path(upstream.__file__).parent != Path(scripts):
        raise ValueError("Loaded a different LongTutor source tree")
    return upstream


def source_bundle(workspace, directory, opsd_checkpoint=None):
    repro = workspace / REPRO
    split = workspace / GOLD
    baseline = workspace / REPRO_RUN
    preflight = read_json(repro / "preflight.json")
    manifest = read_json(split / "split_manifest.json")
    ids = manifest["ids"]["validation"]
    validation_ids = set(ids)
    if len(ids) != 140 or len(validation_ids) != 140 or validation_ids & set(manifest["ids"]["test"]):
        raise ValueError("Expected the 140 frozen validation IDs")
    prompts = split / "development/prompts.jsonl"
    references = split / "development/references.jsonl"
    for path in (prompts, references):
        if digest(path) != manifest["files"][f"development/{path.name}"]:
            raise ValueError(f"Frozen file hash mismatch: {path.name}")
    for path, expected in (
        (repro / "gold_inputs.jsonl", preflight["gold_input_sha256"]),
        (repro / "upstream" / GOLD_FILE, preflight["gold_annotations_sha256"]),
        (baseline / "preds.jsonl", read_json(baseline / "reproduction_results.json")["prediction_sha256"]),
    ):
        if digest(path) != expected:
            raise ValueError(f"Source file hash mismatch: {path.name}")
    (directory / "upstream").mkdir()
    for source in sorted((repro / "upstream/scripts").glob("*.py")):
        expected = subprocess.check_output([
            "git", "-C", str(workspace / "data/longtutor/repo"), "cat-file", "blob",
            f"{preflight['upstream_commit']}:scripts/{source.name}",
        ])
        if source.read_bytes() != expected:
            raise ValueError(f"Upstream source differs from pinned commit: {source.name}")
        shutil.copyfile(source, directory / "upstream" / source.name)

    upstream = load_upstream(directory)
    histories = index(
        (dict(row, _key=key) for row in rows(repro / "gold_inputs.jsonl")
         if (key := upstream._sample_key(row)) in validation_ids),
        "_key",
    )
    gold = index((row for row in rows(repro / "upstream" / GOLD_FILE) if row["_key"] in validation_ids), "_key")
    cached = index((row for row in rows(baseline / "preds.jsonl") if row["_key"] in validation_ids), "_key")
    saved_hashes = index(
        (row for row in rows(repro / "prompt_checks.jsonl") if row["_key"] in validation_ids), "_key"
    )
    original_prompts = index(rows(prompts), "sample_id")
    targets = index(rows(references), "sample_id")
    if list(original_prompts) != ids or list(targets) != ids:
        raise ValueError("Frozen validation files do not follow manifest order")
    prompt_hashes = {}
    changed_system_prompts = 0
    for key in ids:
        target = targets[key]["target"]
        if target["diagnosis"] != gold[key]["diagnosis"]:
            raise ValueError(f"Gold diagnosis mismatch: {key}")
        if [item["answer"] for item in target["memory"]] != [item["answer"] for item in gold[key]["memory"]]:
            raise ValueError(f"Gold evidence answers differ: {key}")
        system, user = upstream._build_prompts(histories[key], gold[key])
        prompt_hash = sha256(json.dumps([system, user], ensure_ascii=False).encode()).hexdigest()
        if prompt_hash != saved_hashes[key]["prompt_sha256"]:
            raise ValueError(f"Prompt differs from cached base inference: {key}")
        prompt_hashes[key] = prompt_hash
        old_system, old_user = original_prompts[key]["messages"]
        if user != old_user["content"]:
            raise ValueError(f"Frozen user prompt differs from official input: {key}")
        changed_system_prompts += system != old_system["content"]

    write_rows(directory / "inputs.jsonl", (histories[key] for key in ids))
    write_rows(directory / "gold.jsonl", (gold[key] for key in ids))
    write_rows(directory / "base.jsonl", (cached[key] for key in ids))
    judged = index((row for row in rows(baseline / "metrics.jsonl") if row["_key"] in validation_ids), "_key")
    submitted = index((row for row in rows(baseline / "judge_input.jsonl") if row["_key"] in validation_ids), "_key")
    calibration = []
    for label in LABELS:
        eligible = [key for key in ids if gold[key]["diagnosis"] == label and key in judged
                    and cached[key]["content"].strip()
                    and valid_judgment(judged[key])]
        if len(eligible) < 8:
            raise ValueError(f"Fewer than eight calibration rows for {label}")
        calibration.extend(sorted(eligible, key=lambda key: sha256(key.encode()).digest())[:8])
    if any(submitted.get(key) != cached[key] for key in calibration):
        raise ValueError("The Gemini reference scored a different base prediction")
    write_rows(directory / "gemini_calibration.jsonl", (judged[key] for key in calibration))

    base_pin = read_json(split / "pa_opsd_gold_20261120/base_model.json")
    base_pin["generation_config_sha256"] = GENERATION_CONFIG_SHA256
    base_pin["weight_index_sha256"] = WEIGHT_INDEX_SHA256
    write_json(directory / "base_model.json", base_pin)
    shutil.copyfile(baseline / "runtime_install.lock", directory / "runtime_install.lock")
    adapter_root = split / "autodl_4090d_50861/formal"
    opsd_context = {"mode": "legacy_gold_reference"}
    if opsd_checkpoint is not None:
        summary = read_json(opsd_checkpoint / "training_summary.json")
        legacy = read_json(adapter_root / "pa_opsd" / "training_summary.json")
        opsd_context = summary.get("teacher_context", {})
        anchor = digest(adapter_root / "sft" / "adapter_model.safetensors")
        if (summary.get("stage") != "opsd" or opsd_context.get("mode") != "standard_solution"
                or summary.get("data_sha256") != legacy["data_sha256"]
                or opsd_context.get("optimizer_steps") != 35 or opsd_context.get("initial_adapter_sha256") != anchor):
            raise ValueError("Expected 35 standard-solution OPSD steps on frozen continuation from the Gold SFT anchor")
        shutil.copyfile(opsd_checkpoint / "training_summary.json", directory / "opsd_training_summary.json")
    for arm in ARMS[1:]:
        destination = directory / "adapters" / arm
        destination.mkdir(parents=True)
        source = opsd_checkpoint if arm == "pa_opsd" and opsd_checkpoint is not None else adapter_root / arm
        config = read_json(source / ADAPTER_FILES[0])
        if (config["peft_type"] != "LORA" or config["r"] != 8 or
            Path(config["base_model_name_or_path"]).name != BASE_NAME or
            set(config["target_modules"]) != {"q_proj", "k_proj", "v_proj", "o_proj"}):
            raise ValueError(f"Unexpected adapter settings: {arm}")
        for name in ADAPTER_FILES:
            shutil.copyfile(source / name, destination / name)
    return {"ids": ids, "prompt_sha256": prompt_hashes, "calibration_ids": calibration,
            "changed_system_prompts_vs_early_gold": changed_system_prompts,
            "upstream_commit": preflight["upstream_commit"], "inference": EVAL,
            "base_revision": read_json(directory / "base_model.json")["revision"],
            "vllm_version": VLLM_VERSION, "scope": "validation", "opsd_teacher_context": opsd_context}


def prepare(args):
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("Start prepare with PYTHONHASHSEED=0 to match the published base run")
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=out.parent) as temp:
        directory = Path(temp) / "bundle"
        directory.mkdir()
        manifest = source_bundle(args.workspace.resolve(), directory, args.opsd_checkpoint)
        for name in ("run.py", "README.md"):
            shutil.copyfile(Path(__file__).with_name(name), directory / name)
        files = [path for path in sorted(directory.rglob("*"))
                 if path.is_file() and "__pycache__" not in path.parts]
        manifest["files"] = {path.relative_to(directory).as_posix(): digest(path) for path in files}
        manifest_path = directory / "manifest.json"
        write_json(manifest_path, manifest)
        archive = Path(temp) / "bundle.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=3) as zipped:
            for path in files + [manifest_path]:
                zipped.write(path, path.relative_to(directory).as_posix())
        archive.replace(out)
    print(json.dumps({"validation_rows": 140, "calibration_rows": 32, "bundle": str(out)}))


def verify_bundle(directory):
    manifest = read_json(directory / "manifest.json")
    if manifest["scope"] != "validation" or len(set(manifest["ids"])) != 140 or manifest["inference"] != EVAL:
        raise ValueError("Unexpected validation manifest")
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Bundle file hash mismatch: {name}")
    return manifest


def command_serve(args):
    directory = args.bundle.resolve()
    verify_bundle(directory)
    if importlib.metadata.version("vllm") != VLLM_VERSION:
        raise ValueError(f"Use vLLM {VLLM_VERSION} to match the base run")
    model = args.model.resolve()
    pin = read_json(directory / "base_model.json")
    hashes = dict(pin["config_and_tokenizer_sha256"],
                  **{name: info["sha256"] for name, info in pin["weights"].items()})
    hashes["generation_config.json"] = pin["generation_config_sha256"]
    hashes["model.safetensors.index.json"] = pin["weight_index_sha256"]
    for name, expected in hashes.items():
        if digest(model / name) != expected:
            raise ValueError(f"Pinned base model file hash mismatch: {name}")
    executable = Path(sys.executable).with_name("vllm")
    if any((model / name).exists() for name in ("chat_template.jinja", "chat_template.json", "chat_templates")):
        raise ValueError("The cached base run used the template embedded in tokenizer_config.json")
    command = [
        str(executable), "serve", str(model), "--served-model-name", BASE_NAME,
        "--host", "127.0.0.1", "--port", str(args.port), "--dtype", EVAL["dtype"],
        "--max-model-len", str(EVAL["max_model_len"]), "--max-num-seqs", "4",
        "--gpu-memory-utilization", "0.91", "--enable-lora", "--max-lora-rank", "8",
        "--max-cpu-loras", "3", "--max-log-len", "48", "--generation-config", "auto", "--lora-modules",
        *(json.dumps({"name": arm, "path": str(directory / "adapters" / arm), "base_model_name": BASE_NAME})
          for arm in ARMS[1:]),
    ]
    os.execve(executable, command, dict(os.environ, HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
                                        VLLM_ALLOW_RUNTIME_LORA_UPDATING="0"))


def bind_output(directory, name, value):
    marker = directory / name
    if marker.exists():
        if read_json(marker) != value:
            raise ValueError(f"Existing run has different settings: {marker}")
    else:
        write_json(marker, value)


@contextmanager
def lock_output(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prediction_rows(path, ids, complete=True):
    indexed = index(rows(path), "_key")
    if indexed.keys() - set(ids) or (complete and indexed.keys() != set(ids)):
        raise ValueError(f"Predictions differ from expected IDs: {path}")
    return indexed


def inference_failure(row):
    reason = row["reason"]
    return not row["diagnosis"] and reason.startswith("ERROR:") and not (
        reason.startswith("ERROR: RuntimeError: invalid_output_after_retries:")
        or "cannot access local variable 'parsed'" in reason
    )


def verified_predictions(directory, manifest, source, arm):
    path = (directory if arm == "base" else source) / f"{arm}.jsonl"
    if arm != "base":
        settings = read_json(source / "inference_settings.json")
        if settings["manifest_sha256"] != digest(directory / "manifest.json") or settings["smoke"]:
            raise ValueError("Prediction settings do not match the full validation run")
    pred = prediction_rows(path, manifest["ids"])
    if any(inference_failure(row) for row in pred.values()):
        raise ValueError(f"Unresolved inference infrastructure failure: {path}")
    return path, pred


def recover_prediction_tail(path):
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    prefix, separator, tail = data.rpartition(b"\n")
    try:
        json.loads(tail)
    except (ValueError, UnicodeDecodeError):
        path.with_suffix(".interrupted.fragment").write_bytes(tail)
        data = prefix + separator
    else:
        data += b"\n"
    temp = path.with_suffix(".repairing")
    temp.write_bytes(data)
    temp.replace(path)


def command_check_prompts(args):
    directory = args.bundle.resolve()
    manifest = verify_bundle(directory)
    upstream = load_upstream(directory)
    gold = index(rows(directory / "gold.jsonl"), "_key")
    for key, row in zip(manifest["ids"], rows(directory / "inputs.jsonl"), strict=True):
        if upstream._sample_key(row) != key:
            raise ValueError(f"Input identity changed: {key}")
        system, user = upstream._build_prompts(row, gold[key])
        prompt_hash = sha256(json.dumps([system, user], ensure_ascii=False).encode()).hexdigest()
        if prompt_hash != manifest["prompt_sha256"][key]:
            raise ValueError(f"Runtime prompt differs from cached base: {key}")


def command_infer(args):
    directory = args.bundle.resolve()
    manifest = verify_bundle(directory)
    api = f"http://127.0.0.1:{args.port}/v1"
    with urlopen(f"{api}/models", timeout=15) as response:
        registered = index(json.load(response)["data"], "id")
    if not {*args.arms, BASE_NAME} <= registered.keys():
        raise ValueError(f"Missing model or adapters: {list(registered)}")
    for arm in args.arms:
        model = registered[arm]
        if Path(model["root"]).resolve() != directory / "adapters" / arm or model["parent"] != BASE_NAME:
            raise ValueError(f"Registered LoRA belongs to a different model or bundle: {arm}")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stamp = out / "inference_settings.json"
    if not stamp.exists() and any((out / f"{arm}.jsonl").exists() for arm in ARMS[1:]):
        raise ValueError("The inference output directory contains predictions without metadata")
    bind_output(out, stamp.name, {"manifest_sha256": digest(directory / "manifest.json"),
                                  "smoke": args.smoke, "api": api, "base_root": registered[BASE_NAME]["root"]})
    expected_ids = manifest["ids"][:2] if args.smoke else manifest["ids"]
    env = dict(os.environ, OPENAI_BASE_URL=api, OPENAI_API_KEY="EMPTY",
               PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "check-prompts", "--bundle", str(directory)],
                   env=env, check=True)
    for arm in args.arms:
        pred = out / f"{arm}.jsonl"
        if pred.exists():
            recover_prediction_tail(pred)
            prediction_rows(pred, expected_ids, complete=False)
        command = [sys.executable, "-u", str(directory / "upstream/eval_ai_tutor.py"),
                   "--input", str(directory / "inputs.jsonl"), "--tests", str(directory / "gold.jsonl"),
                   "--output", str(pred), "--model", arm, "--temperature", str(EVAL["temperature"]),
                   "--max_tokens", str(EVAL["max_tokens"]), "--workers", "2" if args.smoke else "4",
                   "--limit", str(len(expected_ids))]
        subprocess.run(command, env=env, check=True)
        indexed = prediction_rows(pred, expected_ids)
        infrastructure_errors = {key for key, row in indexed.items() if inference_failure(row)}
        if infrastructure_errors:
            write_rows(out / f"{arm}_inference_failures.jsonl", (indexed[key] for key in sorted(infrastructure_errors)))
            retry = pred.with_suffix(".retry")
            write_rows(retry, (row for key, row in indexed.items() if key not in infrastructure_errors))
            retry.replace(pred)
            raise ValueError(f"{arm}: {len(infrastructure_errors)} infrastructure failures saved for retry")
        print(json.dumps({"arm": arm, "rows": len(indexed),
                          "model_rejections": sum(not row["diagnosis"] for row in indexed.values())}), flush=True)


def scorer_module(directory):
    load_upstream(directory)
    import compute_ai_tutor_eval_metrics as grader

    if Path(grader.__file__).parent != directory / "upstream":
        raise ValueError("Loaded a different LongTutor scorer")
    return grader


def command_score(args):
    directory = args.bundle.resolve()
    manifest = verify_bundle(directory)
    ids = manifest["ids"]
    gold = index(rows(directory / "gold.jsonl"), "_key")
    grader = scorer_module(directory)
    rouge = grader.rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = {}
    diagnosis_correct = {}
    for arm in args.arms:
        _, pred = verified_predictions(directory, manifest, args.predictions, arm)
        truth = Counter(gold[key]["diagnosis"] for key in ids)
        guessed = Counter(grader._norm_text(pred[key]["diagnosis"]) for key in ids)
        correct_ids = {key for key in ids if grader._norm_text(pred[key]["diagnosis"]) == gold[key]["diagnosis"]}
        diagnosis_correct[arm] = correct_ids
        correct = Counter(gold[key]["diagnosis"] for key in correct_ids)
        per_class = {label: 2 * correct[label] / (truth[label] + guessed[label]) for label in LABELS}
        rouges = [rouge.score(grader._norm_text(gold[key]["content"]),
                             grader._norm_text(pred[key]["content"]))["rougeL"].fmeasure for key in ids]
        results[arm] = {"rows": len(ids), "diagnosis_correct": sum(correct.values()),
                        "diagnosis_accuracy": sum(correct.values()) / len(ids), "per_class_f1": per_class,
                        "macro_f1_fixed_four_classes": sum(per_class.values()) / len(LABELS),
                        "macro_f1_repository": sum(per_class.values()) / len(truth.keys() | guessed.keys()),
                        "empty_diagnosis": guessed[""], "rouge_l_repository": sum(rouges) / len(ids)}
    report = {"scope": "validation", "diagnosis_metric": "exact_labels", "arms": results}
    if "pa_opsd" in results:
        report["opsd_teacher_context"] = manifest.get("opsd_teacher_context", {"mode": "legacy_gold_reference"})
    if {"sft_cont", "pa_opsd"} <= diagnosis_correct.keys():
        sft, opsd = diagnosis_correct["sft_cont"], diagnosis_correct["pa_opsd"]
        report["paired_sft_cont_vs_pa_opsd"] = {
            "both_correct": len(sft & opsd), "both_incorrect": len(ids) - len(sft | opsd),
            "sft_cont_only_correct": len(sft - opsd), "pa_opsd_only_correct": len(opsd - sft),
        }
    write_json(args.output, report)


def valid_judgment(row):
    memory = row.get("memory_correct")
    content = row.get("content_scores")
    return (isinstance(memory, list) and len(memory) == 3 and all(isinstance(value, bool) for value in memory)
            and isinstance(content, dict)
            and all(type(content.get(dim)) is int and 1 <= content[dim] <= 5 for dim in DIMENSIONS))


def calibration_report(candidate, reference):
    pairs = [(row, reference[row["_key"]]) for row in candidate]
    evidence = {}
    for i in range(3):
        votes = Counter((a["memory_correct"][i], b["memory_correct"][i]) for a, b in pairs)
        evidence[f"Q{i + 1}"] = {
            "agreement": (votes[True, True] + votes[False, False]) / len(pairs),
            "both_correct": votes[True, True], "both_incorrect": votes[False, False],
            "candidate_only_correct": votes[True, False], "gemini_only_correct": votes[False, True],
        }
    return {
        "rows": len(pairs),
        "evidence_judgment_agreement": evidence,
        "teaching_mean_absolute_error": {
            dim: sum(abs(a["content_scores"][dim] - b["content_scores"][dim]) for a, b in pairs) / len(pairs)
            for dim in DIMENSIONS
        },
        "teaching_signed_difference_candidate_minus_gemini": {
            dim: sum(a["content_scores"][dim] - b["content_scores"][dim] for a, b in pairs) / len(pairs)
            for dim in DIMENSIONS
        },
    }


def command_judge(args):
    directory = args.bundle.resolve()
    manifest = verify_bundle(directory)
    ids = manifest["calibration_ids"] if args.calibration else manifest["ids"]
    if args.calibration and args.arm != "base":
        raise ValueError("The paired Gemini calibration set uses base predictions")
    endpoint = urlsplit(os.environ["OPENAI_BASE_URL"])
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or (
        endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
    ):
        raise ValueError("OPENAI_BASE_URL must be a credential-free HTTP(S) API base URL")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the environment")
    pred_path, pred = verified_predictions(directory, manifest, args.predictions, args.arm)
    gold = index(rows(directory / "gold.jsonl"), "_key")
    grader = scorer_module(directory)
    cfg = grader.GradeConfig(model=args.model, temperature=0, max_completion_tokens=2500,
                             max_retries=1, debug=False)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "rows.jsonl"
    if target.exists() and not (out / "settings.json").exists():
        raise ValueError("Scoring rows exist without run settings")
    bind_output(out, "settings.json", {
        "manifest_sha256": digest(directory / "manifest.json"), "predictions_sha256": digest(pred_path),
        "arm": args.arm, "model": args.model, "calibration": args.calibration,
        "endpoint": endpoint.geturl(), "temperature": 0, "max_tokens": 2500,
    })
    saved = prediction_rows(target, ids, complete=False) if target.exists() else {}
    if any(not valid_judgment(row) for row in saved.values()):
        raise ValueError("Existing judge output contains malformed rows")
    failures = ContextVar("judge_failures")
    published_match = grader._judge_match_with_gpt

    def guarded_match(*match_args, **kwargs):
        if kwargs["task"] == "diagnosis":
            return False  # Diagnosis is an exact four-label classification task here.
        try:
            result = published_match(*match_args, **kwargs)
        except Exception as error:
            failures.get().append(f"{kwargs['task']}: {type(error).__name__}")
            raise
        if result is None:
            failures.get().append(f"{kwargs['task']}: no usable semantic verdict")
        return result

    def grade(key):
        problems = []
        context_token = failures.set(problems)
        try:
            row = grader._process_one_sample(key=key, pred=pred[key], gold=gold[key], grade_cfg=cfg)
        finally:
            failures.reset(context_token)
        if not valid_judgment(row):
            problems.append("incomplete evidence or teaching scores")
        return key, row, problems

    pending = [key for key in ids if key not in saved]
    grader._judge_match_with_gpt = guarded_match
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(pending), args.workers):
                batch = list(pool.map(grade, pending[start:start + args.workers]))
                problems = {key: errors for key, _, errors in batch if errors}
                saved.update({key: row for key, row, errors in batch if not errors})
                temp = target.with_suffix(".tmp")
                write_rows(temp, (saved[key] for key in ids if key in saved))
                temp.replace(target)
                if problems:
                    write_json(out / "failures.json", problems)
                    raise ValueError(f"Judge batch incomplete; kept {len(saved)} successful rows")
    finally:
        grader._judge_match_with_gpt = published_match
    (out / "failures.json").unlink(missing_ok=True)
    complete = [saved[key] for key in ids]
    if args.calibration:
        reference = prediction_rows(directory / "gemini_calibration.jsonl", ids)
        report = calibration_report(complete, reference)
    else:
        evidence = [sum(row["memory_correct"][i] for row in complete) / len(complete) for i in range(3)]
        teaching = {dim: sum(row["content_scores"][dim] for row in complete) / len(complete)
                    for dim in DIMENSIONS}
        report = {"rows": len(complete), "evidence": dict(zip(grader.MEM_TYPES, evidence, strict=True)),
                  "evidence_average": sum(evidence) / len(evidence), "teaching": teaching,
                  "teaching_average": sum(teaching.values()) / len(teaching)}
    write_json(out / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--workspace", type=Path, default=ROOT)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--opsd-checkpoint", type=Path, help="Standard-solution adapter for the pa_opsd evaluation slot")
    build.set_defaults(handler=prepare)
    for name, handler in (("check-prompts", command_check_prompts), ("serve", command_serve), ("infer", command_infer),
                          ("score", command_score), ("judge", command_judge)):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.set_defaults(handler=handler)
        if name in {"serve", "infer"}:
            command.add_argument("--port", type=int, default=8000)
        if name == "serve":
            command.add_argument("--model", type=Path, required=True)
        elif name != "check-prompts":
            command.add_argument("--output", type=Path, required=True)
        if name == "infer":
            command.add_argument("--smoke", action="store_true")
            command.add_argument("--arms", nargs="+", choices=ARMS[1:], default=ARMS[1:])
        if name in {"score", "judge"}:
            command.add_argument("--predictions", type=Path, default=Path("."))
        if name == "score":
            command.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
        if name == "judge":
            command.add_argument("--arm", choices=ARMS, required=True)
            command.add_argument("--model", required=True)
            command.add_argument("--calibration", action="store_true")
            command.add_argument("--workers", choices=range(1, 9), type=int, default=4)
    args = parser.parse_args()
    if args.command in {"infer", "judge"}:
        with lock_output(args.output.resolve()):
            args.handler(args)
    else:
        args.handler(args)


if __name__ == "__main__":
    main()
