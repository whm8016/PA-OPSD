"""Compare standard-solution OPSD and SFT continuation on the full Gold development set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.experiments.longtutor_reeval import run as official

REEVAL = Path("outputs/pa_opsd/longtutor_reeval_gold140")
OLD_OPSD = Path("outputs/pa_opsd/opsd_repair/autodl_5090_41483_1024")


def read_reference(archive_path, directory):
    with ZipFile(archive_path) as archive:
        manifest_bytes = archive.read("manifest.json")
        manifest = json.loads(manifest_bytes)
        for name, checksum in manifest["files"].items():
            path = Path(name)
            if name != "gold.jsonl" and (path.parent != Path("upstream") or path.suffix != ".py"):
                continue
            extracted = archive.read(name)
            if sha256(extracted).hexdigest() != checksum:
                raise ValueError(f"Reference file hash mismatch: {name}")
            destination = directory / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(extracted)
    return manifest, sha256(manifest_bytes).hexdigest()


def verify_protocol(reference, comparison):
    keys = ("scope", "ids", "prompt_sha256", "upstream_commit", "base_revision", "vllm_version", "inference")
    for key in keys:
        if reference[key] != comparison[key]:
            raise ValueError(f"Incompatible official evaluation protocol: {key}")
    for name, checksum in reference["files"].items():
        if (name in {"inputs.jsonl", "gold.jsonl"} or name.startswith("upstream/")) and \
                checksum != comparison["files"].get(name):
            raise ValueError(f"Incompatible official evaluation file: {name}")


def read_predictions(directory, arm, manifest_sha, ids):
    settings_path = directory / "inference_settings.json"
    settings = official.read_json(settings_path)
    if settings["manifest_sha256"] != manifest_sha or settings["smoke"] is not False:
        raise ValueError(f"Predictions do not belong to the full official bundle: {directory}")
    path = directory / f"{arm}.jsonl"
    records = official.prediction_rows(path, ids)
    if any(official.inference_failure(row) for row in records.values()):
        raise ValueError(f"Unresolved inference infrastructure failure: {path}")
    return records, {"path": str(path), "sha256": official.digest(path), "manifest_sha256": manifest_sha}


def score_rows(ids, gold, predictions, grader, strategy_by_diagnosis):
    normalize = grader._norm_text
    rouge = grader.rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for key in ids:
        target = gold[key]
        results = {}
        for arm, records in predictions.items():
            pred = records[key]
            diagnosis = normalize(pred["diagnosis"])
            strategy = normalize(pred["strategy"])
            results[arm] = {
                "diagnosis": diagnosis,
                "diagnosis_correct": diagnosis == target["diagnosis"],
                "strategy": strategy,
                "strategy_correct": strategy == normalize(target["strategy"]),
                "diagnosis_strategy_compatible": strategy_by_diagnosis.get(diagnosis) == strategy,
                "official_model_rejection": not diagnosis,
                "rouge_l_repository": rouge.score(normalize(target["content"]),
                                                  normalize(pred["content"]))["rougeL"].fmeasure,
            }
        yield {"_key": key, "gold_diagnosis": target["diagnosis"], "gold_strategy": target["strategy"],
               "arms": results}


def summarize(records, arms):
    support = Counter(row["gold_diagnosis"] for row in records)
    result = {}
    correct_sets = {}
    for arm in arms:
        scored = [row["arms"][arm] for row in records]
        correct = Counter(row["gold_diagnosis"] for row in records if row["arms"][arm]["diagnosis_correct"])
        predicted = Counter(row["diagnosis"] for row in scored)
        per_class = {label: {"support": support[label], "predicted": predicted[label], "correct": correct[label],
                             "f1": 2 * correct[label] / (support[label] + predicted[label])}
                     for label in official.LABELS}
        correct_sets[arm] = {row["_key"] for row in records if row["arms"][arm]["diagnosis_correct"]}
        result[arm] = {
            "diagnosis_correct": sum(correct.values()),
            "diagnosis_accuracy": sum(correct.values()) / len(records),
            "macro_f1_fixed_four_classes": sum(item["f1"] for item in per_class.values()) / len(official.LABELS),
            "per_class": per_class,
            "strategy_correct": sum(row["strategy_correct"] for row in scored),
            "diagnosis_strategy_compatible": sum(row["diagnosis_strategy_compatible"] for row in scored),
            "official_model_rejections": sum(row["official_model_rejection"] for row in scored),
            "rouge_l_repository": sum(row["rouge_l_repository"] for row in scored) / len(records),
        }
    paired = {f"{left}_vs_{right}": {
        "both_correct": len(correct_sets[left] & correct_sets[right]),
        "both_wrong": len(records) - len(correct_sets[left] | correct_sets[right]),
        "first_only_correct": len(correct_sets[left] - correct_sets[right]),
        "second_only_correct": len(correct_sets[right] - correct_sets[left]),
    } for left, right in combinations(arms, 2)}
    return result, paired


def compare(args, directory):
    workspace = args.workspace.resolve()
    old_run = workspace / OLD_OPSD
    reference, reference_sha = read_reference(workspace / REEVAL / "longtutor_reeval_gold140.zip", directory)
    ids = reference["ids"]
    split = official.read_json(workspace / official.GOLD / "split_manifest.json")
    if reference["scope"] != "validation" or reference["inference"] != official.EVAL or len(set(ids)) != 140 \
            or ids != split["ids"]["validation"]:
        raise ValueError("Expected the frozen Gold development IDs and official inference settings")
    old_manifest_path = old_run / "candidate_bundle/manifest.json"
    old_manifest = official.read_json(old_manifest_path)
    verify_protocol(reference, old_manifest)
    sources = {"sft_cont": (workspace / REEVAL / "autodl_5090_35989/predictions", "sft_cont", reference_sha),
               "opsd_1024": (old_run / "official_predictions", "pa_opsd", official.digest(old_manifest_path))}
    if args.candidate_bundle is not None:
        bundle = args.candidate_bundle.resolve()
        candidate = official.verify_bundle(bundle)
        verify_protocol(reference, candidate)
        if candidate["opsd_teacher_context"] != old_manifest["opsd_teacher_context"]:
            raise ValueError("Candidate and old OPSD have different teacher context or optimizer step counts")
        old_training = official.read_json(old_run / "formal/training_summary.json")
        training = official.read_json(bundle / "opsd_training_summary.json")
        rollout_validation = training.get("rollout_validation", {})
        matched = ("data_sha256", "max_completion_length", "max_length", "teacher_mode", "precision", "load_in_4bit")
        if rollout_validation.get("effective_optimizer_steps") != 35 or \
                any(training.get(key) != old_training[key] for key in matched) or \
                "output_prompt_version" in training:
            raise ValueError("Expected gated OPSD with 35 actual updates, matched data and legacy prompt")
        sources["gated_opsd_1024"] = (args.candidate_predictions.resolve(), "pa_opsd",
                                     official.digest(bundle / "manifest.json"))
    predictions, provenance = {}, {}
    for label, (path, arm, manifest_sha) in sources.items():
        predictions[label], provenance[label] = read_predictions(path, arm, manifest_sha, ids)
    gold = official.index(official.rows(directory / "gold.jsonl"), "_key")
    if list(gold) != ids or {row["diagnosis"] for row in gold.values()} != set(official.LABELS):
        raise ValueError("Reference Gold rows do not match the validation manifest and diagnosis categories")
    grader = official.scorer_module(directory)
    strategy_by_diagnosis = official.load_upstream(directory).strategy_MAP
    per_sample = list(score_rows(ids, gold, predictions, grader, strategy_by_diagnosis))
    arms, paired = summarize(per_sample, predictions)
    return per_sample, {"scope": "frozen_gold140_development", "rows": len(ids), "protocol": reference["inference"],
                        "raw_first_pass_format_metric": "unavailable_in_official_normalized_predictions",
                        "sources": provenance, "arms": arms, "paired_diagnosis": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--candidate-bundle", type=Path, help="Extracted official Gold140 gated checkpoint bundle")
    parser.add_argument("--candidate-predictions", type=Path, help="Full official inference output directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for scored artifacts")
    args = parser.parse_args()
    if (args.candidate_bundle is None) != (args.candidate_predictions is None):
        parser.error("provide both --candidate-bundle and --candidate-predictions for a three-arm comparison")
    with TemporaryDirectory() as temp:
        per_sample, report = compare(args, Path(temp))
    args.output.mkdir(parents=True, exist_ok=False)
    official.write_rows(args.output / "per_sample.jsonl", per_sample)
    official.write_json(args.output / "summary.json", report)
    print(json.dumps({"rows": report["rows"], "arms": list(report["arms"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
