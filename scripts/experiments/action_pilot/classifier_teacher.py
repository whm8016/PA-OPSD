"""Test the added value of a held-out diagnosis in the teacher's first hint."""

import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.experiments.action_pilot import evidence_probe as base
from scripts.experiments.action_pilot import run as gold
from scripts.experiments.action_pilot import score_evidence_probe as scoring

SCOPE = "gold_train_classifier_teacher"
ROOT = gold.ROOT / "outputs/pa_opsd/action_pilot"
PREVIOUS = {
    "teacher_evidence_20261126_frozen": "9a31b6e59c52e66ecda54b2b009da93966a9cd66cc80a168f5d9e66bd3060b6f",
    "grounded_teacher_20261126_frozen": "d21fa085b9fa561eeb68646cdc70578cb9de4c86910c05ca28591d5032296c35",
}
REFERENCES = gold.ROOT / "outputs/pa_opsd/diagnosis_probe_20261125/bundle_v1/references.jsonl"
REFERENCE_SHA = "d49023f11e9f26396895e66dbab89ca2b0628f071b73438cf5af686a81ee6648"
CONTROL, TREATMENT, REPEAT = "temporal_teacher", "classifier_teacher", "classifier_teacher_repeat"
ARMS = (CONTROL, TREATMENT)
COHORT = {"Recall Failure": 3, "Conceptual Gap": 2, "Procedural Error": 19}
LABELS_ZH = dict(zip(gold.LABELS, ("回忆失败", "概念缺口", "步骤错误", "迁移不足"), strict=True))
GATE = {"min_safe": 17, "net_over_control": 3, "repeat_both_safe": 6, "repeat_same_action": 7}
SYSTEM = base.SYSTEM.split("如果使用历史", 1)[0] + (
    "标准解和历史索引只供内部核对。有分类器预测时，把它当作可能出错的线索，用题目和作答记录核对。"
    "预测类别不能证明学生用了哪种解法或哪一步出错。只在看到明确步骤时才指出该步。"
    "不要在给学生的话里提历史位置、预测类别或标准解。\n"
    + base.SYSTEM.split("\n", 1)[1].split('仅输出 JSON 对象', 1)[0]
    + '仅输出 JSON 对象，字段 "action" 为 A 到 H 的一个字母，"first_move" 为你对学生说的第一句话'
    '（若选择 A 不干预则填空字符串）。'
)


def prompts(row, arm):
    system, user = base.messages(row, "solution_evidence")
    system["content"] = SYSTEM
    if arm in (TREATMENT, REPEAT):
        user["content"] += "\n\n## 分类器预测（仅为线索）\n" + LABELS_ZH[row["classifier_prediction"]]
    return [system, user]


def parse(raw, row, arm):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != {"action", "first_move"}:
        return None
    action, first = value["action"], value["first_move"]
    if not isinstance(action, str) or action not in gold.ACTIONS or not isinstance(first, str):
        return None
    return value if (action == "A") != bool(first.strip()) else None


def priority(key, suffix=""):
    return gold.sha256((SCOPE + suffix + "|" + key).encode()).digest()


def overlaps(identity, excluded):
    return any(value in seen for value, seen in zip(identity, excluded, strict=True))


def remember(identity, excluded):
    for value, seen in zip(identity, excluded, strict=True):
        seen.add(value)


def prior_inputs():
    previous, excluded = None, set()
    for name, expected_hash in PREVIOUS.items():
        manifest_path = ROOT / name / "manifest.json"
        audit_path = manifest_path.with_name("audit.jsonl")
        if gold.digest(manifest_path) != expected_hash:
            raise ValueError(f"Teacher manifest changed: {name}")
        manifest = gold.read_json(manifest_path)
        if gold.digest(audit_path) != manifest["audit_sha256"]:
            raise ValueError(f"Teacher audit changed: {name}")
        ids = [row["sample_id"] for row in gold.read_rows(audit_path)]
        if ids != manifest["ids"]:
            raise ValueError(f"Teacher audit IDs changed: {name}")
        excluded.update(ids)
        if previous is None:
            previous = manifest
    return previous, excluded


def freeze_cohort(train, histories, identities, bank, prior_ids):
    excluded = (set(), set(), set())
    for key in prior_ids:
        remember(identities[key], excluded)
    selected = []
    for label, size in COHORT.items():
        group = []
        for key in sorted(train, key=priority):
            identity = identities[key]
            if label != train[key]["teacher_privileged"]["gold_reference"]["diagnosis"] or overlaps(identity, excluded):
                continue
            if gold.IMAGE.search(histories[key]["question_info"]):
                continue
            solution = bank[identity[1]]["info"]
            if not solution.get("answer") or not str(solution.get("analysis") or "").strip():
                continue
            remember(identity, excluded)
            group.append(key)
            if len(group) == size:
                break
        if len(group) != size:
            raise ValueError(f"Too few new pure-text cases for {label}")
        selected.extend(group)
    return selected


def classifier(train, histories, identities, selected):
    from sklearn import __version__ as sklearn_version

    from scripts.experiments.diagnosis_probe import offline

    if gold.digest(REFERENCES) != REFERENCE_SHA:
        raise ValueError("Original classifier reference data changed")
    refs = gold.indexed((row for row in gold.read_rows(REFERENCES) if row["split"] == "stage2"), "sample_id")
    if len(refs) != 140 or not refs.keys() <= train.keys():
        raise ValueError("The classifier's original fit cohort must be inside Gold train")

    def statistics(key):
        row = histories[key]
        return offline.history_statistics(row["history_info"], row["question_info"])

    excluded = (set(), set(), set())
    for key in selected:
        remember(identities[key], excluded)
    fit = {key: row for key, row in refs.items() if not overlaps(identities[key], excluded)}
    feature_names = list(next(iter(refs.values()))["statistics"])
    for key, ref in refs.items():
        if (ref["gold"] != train[key]["teacher_privileged"]["gold_reference"]["diagnosis"]
                or ref["statistics"] != statistics(key)):
            raise ValueError(f"Original feature values or label changed: {key}")
    if {row["gold"] for row in fit.values()} != set(gold.LABELS):
        raise ValueError("The isolated classifier fit lacks a diagnosis class")
    model = offline.fit_logistic([[row["statistics"][name] for name in feature_names] for row in fit.values()],
                                 [row["gold"] for row in fit.values()])
    features = {key: statistics(key) for key in selected}
    matrix = [[features[key][name] for name in feature_names] for key in selected]
    predictions = model.predict(matrix).tolist()
    probabilities = model.predict_proba(matrix).tolist()
    details = {key: {"prediction": label, "probabilities": dict(zip(model.classes_, scores, strict=True)),
                     "statistics": features[key]}
               for key, label, scores in zip(selected, predictions, probabilities, strict=True)}
    return details, {"fit_ids": list(fit), "removed_fit_ids": [key for key in refs if key not in fit],
                     "fit_counts": dict(Counter(row["gold"] for row in fit.values())),
                     "reference_sha256": REFERENCE_SHA, "source_sha256": gold.digest(Path(offline.__file__)),
                     "scikit_learn": sklearn_version, "feature_names": feature_names,
                     "fit_settings": {"class_weight": "balanced", "C": 1.0}}


def prepare(output, tokenizer_path):
    from transformers import AutoTokenizer

    from scripts.build_pa_opsd_gold_split import question_group
    from scripts.experiments.training_free_memory import prepare as temporal

    previous, prior_ids = prior_inputs()
    history_path = gold.REPRO / "gold_inputs.jsonl"
    split_path = gold.SPLIT / "split_manifest.json"
    train_path = gold.SPLIT / "development/train.jsonl"
    split = gold.read_json(split_path)
    if (gold.digest(split_path) != previous["split_sha256"]
            or gold.digest(train_path) != split["files"]["development/train.jsonl"]
            or gold.digest(history_path) != previous["canonical_history_sha256"]
            or gold.digest(gold.BANK) != gold.read_json(gold.ROOT / "outputs/pa_opsd/action_pilot_gold64/manifest.json")[
                "source_files_sha256"]["question_bank"]
            or previous["adapter_sha256"] != {name: gold.digest(gold.GOLD_SFT / name) for name in gold.ADAPTER_FILES}):
        raise ValueError("Frozen teacher sources changed")
    train = gold.indexed(gold.read_rows(train_path), "sample_id")
    if list(train) != split["ids"]["train"] or len(train) != 560 or not prior_ids <= train.keys():
        raise ValueError("Unexpected Gold train cohort")
    histories = gold.indexed((dict(row, sample_id=key) for row in gold.read_rows(history_path)
                              if (key := gold.history_key(row)) in train), "sample_id")
    if histories.keys() != train.keys() or any(len(row["history_info"]) != 199 for row in histories.values()):
        raise ValueError("Gold histories do not match the frozen training data")
    identities = {key: (row["uid"], *question_group(row["question_info"])) for key, row in histories.items()}
    bank = gold.indexed(gold.read_rows(gold.BANK), "id")
    selected = freeze_cohort(train, histories, identities, bank, prior_ids)
    details, classifier_receipt = classifier(train, histories, identities, selected)
    pin = temporal.verify_pins(temporal.BUNDLE, gold.read_json(temporal.BUNDLE / "manifest.json"), tokenizer_path)
    if pin["revision"] != previous["base_revision"]:
        raise ValueError("Base model revision changed")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    rows, audit = [], []
    for key in selected:
        uid, qid, stem = identities[key]
        raw = histories[key]
        current, history = raw["question_info"], raw["history_info"]
        _, route = temporal.sample_inputs(dict(raw, _key=key), {"memory": []})
        positions = route["temporal_positions"]
        solution = bank[qid]["info"]
        row = {"sample_id": key, "current": current, "history": [(i, history[i - 1]) for i in positions],
               "solution": {"standard_answer": solution["answer"], "analysis": solution["analysis"]},
               "facts": base.facts(current, history, positions), "classifier_prediction": details[key]["prediction"]}
        tokens = {arm: len(tokenizer.apply_chat_template(prompts(row, arm), tokenize=True,
                                                        add_generation_prompt=True, return_dict=False)) for arm in ARMS}
        if max(tokens.values()) + base.MAX_COMPLETION > base.PROBE_CONTEXT_BUDGET:
            raise ValueError(f"Prompt exceeds fixed context budget: {key}")
        rows.append(row)
        audit.append({"sample_id": key, "uid": uid, "question_id": qid, "stem_sha256": stem,
                      "diagnosis": train[key]["teacher_privileged"]["gold_reference"]["diagnosis"],
                      "classifier": details[key], "history_positions": positions, "facts": row["facts"],
                      "prompt_tokens": tokens})
    smoke = [max((row for row in audit if row["diagnosis"] == label),
                 key=lambda row: max(row["prompt_tokens"].values()))["sample_id"] for label in COHORT]
    smoke += [max((row for row in audit if row["sample_id"] not in smoke),
                  key=lambda row: max(row["prompt_tokens"].values()))["sample_id"]]
    repeats = [row["sample_id"] for row in audit if row["diagnosis"] != "Procedural Error"]
    repeats += sorted(set(selected) - set(repeats), key=lambda key: priority(key, "_repeat"))[:3]
    output.mkdir(parents=True, exist_ok=False)
    gold.write_rows(output / "inputs.jsonl", rows)
    gold.write_rows(output / "audit.jsonl", audit)
    gold.write_json(output / "classifier_fit.json", classifier_receipt)
    model_fields = ("model_alias", "model_description", "base_model", "base_revision", "vllm_version", "inference",
                    "adapter_sha256", "probe_context_budget", "temperature", "max_completion_tokens", "sampling_seed",
                    "split_sha256", "canonical_history_sha256")
    gold.write_json(output / "manifest.json", dict(
        {field: previous[field] for field in model_fields}, scope=SCOPE, ids=selected, smoke_ids=smoke,
        repeat_ids=repeats, conditions=ARMS, repeat_arm=REPEAT, gate=GATE, cohort=COHORT, previous=PREVIOUS,
        source_sha256=gold.digest(Path(base.__file__)), classifier_teacher_source_sha256=gold.digest(Path(__file__)),
        inputs_sha256=gold.digest(output / "inputs.jsonl"), audit_sha256=gold.digest(output / "audit.jsonl"),
        classifier_fit_sha256=gold.digest(output / "classifier_fit.json"),
        question_bank_sha256=gold.digest(gold.BANK), train_sha256=gold.digest(train_path),
    ))
    files = {"base_model.json": temporal.BUNDLE / "base_model.json",
             "service.py": Path(temporal.__file__).with_name("service.py"),
             "scripts/experiments/longtutor_reeval/run.py": gold.ROOT / "scripts/experiments/longtutor_reeval/run.py",
             **{name: output / name for name in ("inputs.jsonl", "manifest.json")}}
    for module in (base, gold, scoring):
        path = Path(module.__file__)
        files[path.relative_to(gold.ROOT).as_posix()] = path
    for path in (Path(__file__), Path(__file__).with_name("run_classifier_teacher.sh")):
        files[path.relative_to(gold.ROOT).as_posix()] = path
    with ZipFile(output / "inference.zip", "x", compression=ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("SHA256SUMS", "".join(f"{gold.digest(path)}  {name}\n" for name, path in files.items()))


def verify_bundle(bundle):
    manifest = gold.read_json(bundle / "manifest.json")
    source = bundle / "inputs.jsonl"
    if (manifest["scope"] != SCOPE or manifest["gate"] != GATE or manifest["cohort"] != COHORT
            or manifest["classifier_teacher_source_sha256"] != gold.digest(Path(__file__))
            or manifest["conditions"] != list(ARMS) or manifest["repeat_arm"] != REPEAT
            or manifest["inputs_sha256"] != gold.digest(source)):
        raise ValueError("Unexpected classifier teacher inputs")
    inputs = gold.indexed(gold.read_rows(source), "sample_id")
    if (list(inputs) != manifest["ids"] or len(inputs) != sum(COHORT.values())
            or any(len(items) != size or len(set(items)) != size or not set(items) <= inputs.keys()
                   for items, size in ((manifest["smoke_ids"], 4), (manifest["repeat_ids"], 8)))
            or any(row["classifier_prediction"] not in LABELS_ZH for row in inputs.values())):
        raise ValueError("Unexpected classifier teacher cohort")
    return manifest, inputs


def check_smoke(output):
    rows = list(gold.read_rows(output))
    if (len(rows) != 4
            or any(row[arm]["finish_reason"] != "stop" for row in rows for arm in ARMS)
            or any(sum(row[arm]["decision"] is not None for row in rows) < 3 for arm in ARMS)):
        raise ValueError("Smoke needs at least three valid replies per arm and no truncated replies")


def score(bundle, review, ratings, output):
    manifest, inputs = verify_bundle(bundle)
    outcomes = scoring.adjudicate(bundle, review, ratings, manifest, REPEAT)
    audit_path = bundle / "audit.jsonl"
    if gold.digest(audit_path) != manifest["audit_sha256"]:
        raise ValueError("Local classifier audit changed")
    audit = gold.indexed(gold.read_rows(audit_path), "sample_id")
    if list(audit) != list(inputs) or any(
        row["classifier"]["prediction"] != inputs[key]["classifier_prediction"] for key, row in audit.items()
    ):
        raise ValueError("Local classifier audit does not match the predictions shown to the teacher")

    def count(keys):
        return {arm: {field: sum(outcomes[key][arm][field] for key in keys) for field in
                      ("passed", "usable", *scoring.RISK, "protocol_valid", "label_matches")} for arm in ARMS}

    counts = count(list(inputs))
    wins = [key for key, row in outcomes.items() if row[TREATMENT]["passed"] and not row[CONTROL]["passed"]]
    losses = [key for key, row in outcomes.items() if row[CONTROL]["passed"] and not row[TREATMENT]["passed"]]
    repeats = [outcomes[key] for key in manifest["repeat_ids"]]
    safe = sum(row[TREATMENT]["passed"] and row[REPEAT]["passed"] for row in repeats)
    same = sum(bool(row[TREATMENT]["observed_action"]
                    and row[TREATMENT]["observed_action"] == row[REPEAT]["observed_action"]) for row in repeats)
    safe_labeled = sum(row[TREATMENT]["passed"] and row[TREATMENT]["label_matches"] for row in outcomes.values())
    checks = {"usable": counts[TREATMENT]["passed"] >= GATE["min_safe"],
              "safe_and_action_matches": safe_labeled >= GATE["min_safe"],
              "net_over_control": len(wins) - len(losses) >= GATE["net_over_control"],
              "repeat_safe": safe >= GATE["repeat_both_safe"], "repeat_same": same >= GATE["repeat_same_action"],
              "format": counts[TREATMENT]["protocol_valid"] == len(inputs),
              "risk": all(counts[TREATMENT][risk] <= counts[CONTROL][risk] for risk in scoring.RISK)}
    correct = {key for key, row in audit.items() if row["classifier"]["prediction"] == row["diagnosis"]}
    detail = output.with_suffix(".rows.jsonl")
    if output.exists() or detail.exists():
        raise FileExistsError(output)
    gold.write_json(output, {"pilot_gate_passed": all(checks.values()), "checks": checks, "thresholds": GATE,
                             "counts": counts, "wins": wins, "losses": losses, "net": len(wins) - len(losses),
                             "repeat_both_safe": safe, "repeat_same_action": same, "safe_and_action_matches": safe_labeled,
                             "by_gold_diagnosis": {label: count([key for key, row in audit.items()
                                                                if row["diagnosis"] == label]) for label in COHORT},
                             "classifier_correct": len(correct),
                             "by_classifier_correctness": {"correct": count(sorted(correct)),
                                                           "incorrect": count(sorted(audit.keys() - correct))},
                             "ratings_sha256": gold.digest(ratings)})
    gold.write_rows(detail, ({"sample_id": key, "diagnosis": audit[key]["diagnosis"],
                              "classifier_prediction": inputs[key]["classifier_prediction"], **outcomes[key]}
                             for key in inputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "probe", "blind", "score"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--tokenizer", type=Path, required=True)
        else:
            command.add_argument("--bundle", type=Path, required=True)
        if name == "probe":
            command.add_argument("--phase", choices=("smoke", "full"), required=True)
            command.add_argument("--port", type=int, default=8808)
        elif name == "blind":
            command.add_argument("--predictions", type=Path, required=True)
        elif name == "score":
            command.add_argument("--review", type=Path, required=True)
            command.add_argument("--ratings", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.tokenizer)
    elif args.command == "probe":
        verify_bundle(args.bundle)
        base.probe(args.bundle, args.output, args.phase, args.port, arms=ARMS, repeat=REPEAT, prompt=prompts,
                   parse=parse, strict_smoke=False)
        if args.phase == "smoke":
            check_smoke(args.output)
    elif args.command == "blind":
        scoring.make_blind(args.bundle, args.predictions, args.output, verify=verify_bundle, repeat=REPEAT, parse=parse)
    else:
        score(args.bundle, args.review, args.ratings, args.output)


if __name__ == "__main__":
    main()
