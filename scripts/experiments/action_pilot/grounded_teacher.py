"""Compare an evidence-constrained teacher against three fixed controls."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.experiments.action_pilot import evidence_probe as base
from scripts.experiments.action_pilot import run as gold
from scripts.experiments.action_pilot import score_evidence_probe as scoring

ORIGINAL = gold.ROOT / "outputs/pa_opsd/action_pilot/teacher_evidence_20261126_frozen"
ORIGINAL_SHA = "9a31b6e59c52e66ecda54b2b009da93966a9cd66cc80a168f5d9e66bd3060b6f"
SCOPE = "gold_train_grounded_teacher"
GUARDED = "grounded"
REPEAT = "grounded_repeat"
ARMS = (*base.CONDITIONS, GUARDED)
COHORT = {"Recall Failure": 8, "Conceptual Gap": 8, "Procedural Error": 8}
GATE = {"min_safe": 17, "net_over_public": 3, "net_over_solution": 2, "net_over_unconstrained": 2,
        "repeat_both_safe": 6, "repeat_same_action": 7}
RULE = (
    "\n本组的历史是内部选提示的依据。给学生的 first_move 不叙述他以前怎么做、会什么、做对或做错了什么，"
    "也不报旧题、历史位置或时间。history_positions 至多填一个你确实参考过的历史事实索引原始位置；没有就填 []。"
    "只知道当前题答错时，选一个能用题面核对的具体子目标，或请学生展示关键一步。"
    "有明确的错误步骤才指出该步。只说一个可执行的动作，不复问题目求什么，不直接报题目任一空的答案。"
)


def prompts(row, arm):
    if arm not in {GUARDED, REPEAT}:
        return base.messages(row, arm)
    message = base.messages(row, "solution_evidence")
    message[0]["content"] += RULE
    return message


def parse(raw, row, arm):
    decision = base.parse_arm(raw, row, arm)
    if decision is None or arm not in {GUARDED, REPEAT}:
        return decision
    citations = decision["history_positions"]
    allowed = {fact["原始位置"] for fact in row["facts"]}
    return decision if len(citations) <= 1 and set(citations) <= allowed else None


def priority(key, salt):
    return gold.sha256((salt + "|" + key).encode()).digest()


def prepare(output, tokenizer_path):
    from transformers import AutoTokenizer

    from scripts.experiments.training_free_memory import prepare as temporal

    old_manifest = ORIGINAL / "manifest.json"
    if gold.digest(old_manifest) != ORIGINAL_SHA:
        raise ValueError("Previous experiment manifest changed")
    old = gold.read_json(old_manifest)
    old_audit = ORIGINAL / "audit.jsonl"
    history_path = gold.REPRO / "gold_inputs.jsonl"
    train_path = gold.SPLIT / "development/train.jsonl"
    split_path = gold.SPLIT / "split_manifest.json"
    split = gold.read_json(split_path)
    if (gold.digest(old_audit) != old["audit_sha256"] or gold.digest(history_path) != old["canonical_history_sha256"]
            or gold.digest(split_path) != old["split_sha256"]
            or gold.digest(train_path) != split["files"]["development/train.jsonl"]
            or gold.digest(gold.BANK) != gold.read_json(gold.ROOT / "outputs/pa_opsd/action_pilot_gold64/manifest.json")[
                "source_files_sha256"]["question_bank"]
            or old["adapter_sha256"] != {name: gold.digest(gold.GOLD_SFT / name) for name in gold.ADAPTER_FILES}):
        raise ValueError("Frozen sources changed")
    old_rows = list(gold.read_rows(old_audit))
    if [row["sample_id"] for row in old_rows] != old["ids"] or len(old_rows) != 64:
        raise ValueError("Previous exclusion cohort changed")
    excluded_uid = {row["uid"] for row in old_rows}
    excluded_qid = {row["question_id"] for row in old_rows}
    train = gold.indexed(gold.read_rows(train_path), "sample_id")
    if list(train) != split["ids"]["train"] or len(train) != 560:
        raise ValueError("Frozen Gold train membership changed")
    histories = gold.indexed((dict(row, sample_id=key) for row in gold.read_rows(history_path)
                              if (key := gold.history_key(row)) in train), "sample_id")
    bank = gold.indexed(gold.read_rows(gold.BANK), "id")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    pin = temporal.verify_pins(temporal.BUNDLE, gold.read_json(temporal.BUNDLE / "manifest.json"), tokenizer_path)
    if pin["revision"] != old["base_revision"]:
        raise ValueError("Original Gold base revision changed")
    rows, audit, smoke, repeats = [], [], [], []
    for label, size in COHORT.items():
        group = []
        for key in sorted(train, key=lambda item: priority(item, SCOPE)):
            if train[key]["teacher_privileged"]["gold_reference"]["diagnosis"] != label:
                continue
            raw = histories[key]
            current, history, uid = raw["question_info"], raw["history_info"], raw["uid"]
            qid = gold.QUESTION_ID.match(current)[1]
            solution = bank[qid]["info"]
            if (uid in excluded_uid or qid in excluded_qid or gold.IMAGE.search(current)
                    or not solution.get("answer") or not str(solution.get("analysis") or "").strip()):
                continue
            if len(history) != 199:
                raise ValueError("Invalid official history size")
            _, route = temporal.sample_inputs(dict(raw, _key=key), {"memory": []})
            positions = route["temporal_positions"]
            row = {"sample_id": key, "current": current, "history": [(i, history[i - 1]) for i in positions],
                   "solution": {"standard_answer": solution["answer"], "analysis": solution["analysis"]},
                   "facts": base.facts(current, history, positions)}
            tokens = {arm: len(tokenizer.apply_chat_template(prompts(row, arm), tokenize=True,
                                                            add_generation_prompt=True, return_dict=False))
                      for arm in ARMS}
            if max(tokens.values()) + base.MAX_COMPLETION > base.PROBE_CONTEXT_BUDGET:
                raise ValueError("Prompt exceeds frozen context budget")
            rows.append(row)
            audit.append({"sample_id": key, "uid": uid, "question_id": qid, "diagnosis": label,
                          "prompt_tokens": tokens, "history_positions": positions, "facts": row["facts"]})
            excluded_uid.add(uid)
            excluded_qid.add(qid)
            group.append((key, max(tokens.values())))
            if len(group) == size:
                break
        if len(group) != size:
            raise ValueError(f"Too few disjoint training examples for {label}")
        smoke.append(max(group, key=lambda pair: pair[1])[0])
        repeats.extend(sorted((key for key, _ in group), key=lambda item: priority(item, SCOPE + "_repeat"))[:2])
    smoke.append(max((row for row in audit if row["sample_id"] not in smoke),
                     key=lambda row: max(row["prompt_tokens"].values()))["sample_id"])
    repeats.extend(sorted((row["sample_id"] for row in audit if row["sample_id"] not in repeats),
                          key=lambda item: priority(item, SCOPE + "_repeat_extra"))[:2])
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs.jsonl"
    gold.write_rows(inputs, rows)
    gold.write_rows(output / "audit.jsonl", audit)
    info = {field: old[field] for field in ("model_alias", "model_description", "base_model", "base_revision",
                                          "vllm_version", "inference", "adapter_sha256", "probe_context_budget",
                                          "temperature", "max_completion_tokens", "sampling_seed", "split_sha256")}
    gold.write_json(output / "manifest.json", dict(
        info, scope=SCOPE, ids=[row["sample_id"] for row in rows], smoke_ids=smoke, repeat_ids=repeats,
        repeat_arm=REPEAT, conditions=ARMS, gate=GATE, cohort=COHORT, inputs_sha256=gold.digest(inputs),
        facts_coverage=sum(bool(row["facts"]) for row in rows), source_sha256=gold.digest(Path(base.__file__)),
        grounded_source_sha256=gold.digest(Path(__file__)), canonical_history_sha256=gold.digest(history_path),
        exclusion_manifest_sha256=ORIGINAL_SHA, exclusion_audit_sha256=gold.digest(old_audit),
        audit_sha256=gold.digest(output / "audit.jsonl"),
    ))
    files = {"base_model.json": (temporal.BUNDLE / "base_model.json").read_bytes(),
             "service.py": Path(temporal.__file__).with_name("service.py").read_bytes(),
             "scripts/experiments/longtutor_reeval/run.py":
             (gold.ROOT / "scripts/experiments/longtutor_reeval/run.py").read_bytes()}
    files.update({name: (output / name).read_bytes() for name in ("inputs.jsonl", "manifest.json")})
    for module in (base, gold, scoring):
        path = Path(module.__file__)
        files[path.relative_to(gold.ROOT).as_posix()] = path.read_bytes()
    files[Path(__file__).relative_to(gold.ROOT).as_posix()] = Path(__file__).read_bytes()
    with ZipFile(output / "inference.zip", "x", compression=ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
        archive.writestr("SHA256SUMS", "".join(f"{gold.sha256(value).hexdigest()}  {name}\n"
                                            for name, value in files.items()))


def verify_bundle(bundle):
    manifest = gold.read_json(bundle / "manifest.json")
    source = bundle / "inputs.jsonl"
    inputs = gold.indexed(gold.read_rows(source), "sample_id")
    if (manifest["scope"] != SCOPE or manifest["gate"] != GATE or manifest["grounded_source_sha256"] !=
            gold.digest(Path(__file__)) or manifest["conditions"] != list(ARMS) or manifest["repeat_arm"] != REPEAT
            or manifest["inputs_sha256"] != gold.digest(source) or manifest["ids"] != list(inputs)
            or len(inputs) != 24 or len(manifest["repeat_ids"]) != 8 or len(set(manifest["repeat_ids"])) != 8
            or len(manifest["smoke_ids"]) != 4 or len(set(manifest["smoke_ids"])) != 4
            or not set(manifest["repeat_ids"] + manifest["smoke_ids"]) <= inputs.keys()):
        raise ValueError("Unexpected grounded teacher bundle")
    return manifest, inputs


def score(bundle, review, ratings, output):
    manifest, inputs = verify_bundle(bundle)
    paired = scoring.adjudicate(bundle, review, ratings, manifest, REPEAT)
    counts = {arm: {field: sum(row[arm][field] for row in paired.values()) for field in
                    ("passed", "usable", *scoring.RISK, "protocol_valid", "label_matches")} for arm in ARMS}
    comparisons = {}
    for arm in base.CONDITIONS:
        wins = [key for key, row in paired.items() if row[GUARDED]["passed"] and not row[arm]["passed"]]
        losses = [key for key, row in paired.items() if row[arm]["passed"] and not row[GUARDED]["passed"]]
        comparisons[arm] = {"wins": wins, "losses": losses, "net": len(wins) - len(losses)}
    repeats = [paired[key] for key in manifest["repeat_ids"]]
    safe = sum(row[GUARDED]["passed"] and row[REPEAT]["passed"] for row in repeats)
    same = sum(bool(row[GUARDED]["observed_action"] and
                    row[GUARDED]["observed_action"] == row[REPEAT]["observed_action"]) for row in repeats)
    checks = {
        "usable": counts[GUARDED]["passed"] >= GATE["min_safe"],
        "over_public": comparisons["public"]["net"] >= GATE["net_over_public"],
        "over_solution": comparisons["solution"]["net"] >= GATE["net_over_solution"],
        "over_unconstrained": comparisons["solution_evidence"]["net"] >= GATE["net_over_unconstrained"],
        "repeat_safe": safe >= GATE["repeat_both_safe"], "repeat_same": same >= GATE["repeat_same_action"],
        "format": counts[GUARDED]["protocol_valid"] == len(inputs),
        "risk": all(counts[GUARDED][risk] <= counts[arm][risk] for arm in base.CONDITIONS for risk in scoring.RISK),
    }
    indexed = [paired[key] for key, inp in inputs.items() if inp["facts"]]
    detail = output.with_suffix(".rows.jsonl")
    if output.exists() or detail.exists():
        raise FileExistsError(output)
    gold.write_json(output, {"pilot_gate_passed": all(checks.values()), "checks": checks, "thresholds": GATE,
                             "counts": counts, "comparisons": comparisons, "repeat_both_safe": safe,
                             "repeat_same_action": same, "evidence_rows": len(indexed),
                             "evidence_passes": {arm: sum(row[arm]["passed"] for row in indexed) for arm in ARMS},
                             "ratings_sha256": gold.digest(ratings)})
    gold.write_rows(detail, ({"sample_id": key, **paired[key]} for key in manifest["ids"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="command", required=True)
    prep = phases.add_parser("prepare")
    prep.add_argument("--tokenizer", type=Path, required=True)
    for command in ("probe", "blind", "score"):
        phase = phases.add_parser(command)
        phase.add_argument("--bundle", type=Path, required=True)
        phase.add_argument("--output", type=Path, required=True)
        if command == "probe":
            phase.add_argument("--phase", choices=("smoke", "full"), required=True)
            phase.add_argument("--port", type=int, default=8808)
        elif command == "blind":
            phase.add_argument("--predictions", type=Path, required=True)
        else:
            phase.add_argument("--review", type=Path, required=True)
            phase.add_argument("--ratings", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.tokenizer)
    elif args.command == "probe":
        verify_bundle(args.bundle)
        base.probe(args.bundle, args.output, args.phase, args.port, arms=ARMS, repeat=REPEAT, prompt=prompts,
                   parse=parse, strict_smoke=False)
    elif args.command == "blind":
        scoring.make_blind(args.bundle, args.predictions, args.output, verify=verify_bundle, repeat=REPEAT, parse=parse)
    else:
        score(args.bundle, args.review, args.ratings, args.output)


if __name__ == "__main__":
    main()
