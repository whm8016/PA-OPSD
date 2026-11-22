"""Prepare and run a paired, no-training probe for the eight pedagogical actions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "outputs/pa_opsd/gold_split_20261120"
REPRO = ROOT / "outputs/longtutor_b1/official_ba40af3"
BANK = ROOT / "data/longtutor/repo/data/XES3G5M/questions.jsonl"
DRAFTS = ROOT / "data/pa_opsd/action_pilot_gold64/action_drafts.jsonl"
GOLD_SFT = SPLIT / "autodl_4090d_50861/formal/sft"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
LABELS = ("Recall Failure", "Conceptual Gap", "Procedural Error", "Transfer Deficit")
ACTIONS = {
    "A": ("不干预", "保留学生当前尝试，等待或继续观察，无具体教学提示。"),
    "B": ("确认并推进", "确认一个已展示的正确思路，顺势让学生继续。"),
    "C": ("追问", "让学生自己回忆、解释或展示一个关键判断。"),
    "D": ("概念提示", "提供可迁移的概念、关系或法则。"),
    "E": ("程序性修正", "指出已观察到的错误步骤，并指导如何纠正这一环。"),
    "F": ("举例", "用另一个具体情境或题例说明相关关系。"),
    "G": ("重新聚焦子目标", "把注意力转到一个可独立处理的关键中间目标。"),
    "H": ("部分解答", "给出当前题的一段推导、具体算式或中间结果，保留关键工作。"),
}
STRATEGY_ACTIONS = {
    "Retrieval Practice": ("C", "D"),
    "Conceptual Explanation": ("D", "F"),
    "Stepwise Scaffolding": ("E", "G", "H"),
    "Analogical Transfer": ("F", "C", "D", "G"),
}
RUBRIC = (
    "以作答记录中可核对的事实判断。仅有正误时，保留对具体错因的推断性。",
    "结合当前题的解题结构，区分忘记知识、概念误解、步骤失误与新情境迁移困难。",
    "选择足够且最小的支持动作，优先让学生完成仍能独立完成的部分。",
)
CONDITIONS = ("student_visible", "teacher_privileged")
SAMPLING_TAG = "pa-opsd-action-pilot-v1|"
IMAGE = re.compile(r"question_\d+-image_\d+", re.IGNORECASE)
QUESTION_ID = re.compile(r"^\[(\d+)\]")
SYSTEM = (
    "你是小学数学导师。根据可核对的学生历史和当前题作答情况判断诊断类别，"
    "并选择你下一步会采取的一个主要教学动作。只有正误记录时，勿臆造具体作答。"
    "若收到额外教学依据，请结合学生记录使用。"
    f" 诊断标签仅可从以下选择：{', '.join(LABELS)}。"
    + "\n教学动作：\n"
    + "\n".join(f"{code} {name}。{rule}" for code, (name, rule) in ACTIONS.items())
    + '\n只输出 JSON 对象，字段为 "diagnosis"、"action"。action 填一个大写英文字母。'
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def indexed(records, field):
    result = {}
    for row in records:
        key = row[field]
        if key in result:
            raise ValueError(f"Duplicate {field}: {key}")
        result[key] = row
    return result


def digest(path):
    checksum = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_rows(path, records):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def history_key(row):
    return row["uid"] + "||" + sha256(row["question_info"].encode()).hexdigest()


def select(train, histories, questions):
    by_class = {label: [] for label in LABELS}
    for key, row in train.items():
        history = histories[key]
        qid = QUESTION_ID.match(history["question_info"]).group(1)
        question = questions[qid]["info"]
        gold = row["teacher_privileged"]["gold_reference"]
        if (question.get("answer") and str(question.get("analysis") or "").strip()
                and not IMAGE.search(history["question_info"]) and gold["content"].strip()):
            by_class[gold["diagnosis"]].append((key, qid, history, question, gold))
    counts = {label: len({item[1] for item in group}) for label, group in by_class.items()}
    chosen = {}
    seen_students, seen_questions = set(), set()
    for label in sorted(LABELS, key=counts.__getitem__):
        group = sorted(by_class[label], key=lambda item: sha256((SAMPLING_TAG + item[0]).encode()).digest())
        chosen[label] = []
        # Give smaller classes first pick of distinct questions. Fill a shortfall with new students.
        for allow_repeated_question in (False, True):
            for item in group:
                _, qid, history, _, _ = item
                if len(chosen[label]) == 16:
                    break
                if history["uid"] in seen_students or (qid in seen_questions and not allow_repeated_question):
                    continue
                chosen[label].append(item)
                seen_students.add(history["uid"])
                seen_questions.add(qid)
        if len(chosen[label]) != 16:
            raise ValueError(f"Fewer than 16 eligible students for {label}")
    return [item for label in LABELS for item in chosen[label]]


def evidence_positions(history):
    related = set(history["related_history"])
    public_history = history["history_info"]
    matching = [position for position, item in enumerate(public_history, 1) if item in related]
    return matching[-5:]


def build(args):
    split = read_json(SPLIT / "split_manifest.json")
    train_path = SPLIT / "development/train.jsonl"
    history_path = REPRO / "gold_inputs.jsonl"
    if digest(train_path) != split["files"]["development/train.jsonl"]:
        raise ValueError("Frozen training file changed")
    if digest(history_path) != read_json(REPRO / "preflight.json")["gold_input_sha256"]:
        raise ValueError("Official history input changed")
    train = indexed(read_rows(train_path), "sample_id")
    if list(train) != split["ids"]["train"] or len(train) != 560:
        raise ValueError("Training IDs changed")
    histories = indexed(
        (dict(row, sample_id=key) for row in read_rows(history_path) if (key := history_key(row)) in train),
        "sample_id",
    )
    questions = indexed(read_rows(BANK), "id")
    selected = select(train, histories, questions)
    drafts = indexed(read_rows(args.drafts), "sample_id")
    ids = [item[0] for item in selected]
    if set(drafts) != set(ids):
        raise ValueError("Draft annotations do not cover the exact 64 selected rows")

    probe, review = [], []
    for key, qid, history, item, gold in selected:
        draft = drafts[key]
        if draft["status"] not in {"draft", "review_required", "reviewed"}:
            raise ValueError(f"Unknown annotation status: {key}")
        if draft["action"] not in ACTIONS and not (
            draft["action"] is None and draft["status"] == "review_required"
        ):
            raise ValueError(f"Invalid draft action: {key}")
        if not draft["quote"].strip() or draft["quote"] not in gold["content"]:
            raise ValueError(f"Missing verbatim evidence from the Gold feedback: {key}")
        matches = evidence_positions(history)
        candidates = STRATEGY_ACTIONS[gold["strategy"]]
        strategy_review = draft["action"] is not None and draft["action"] not in candidates
        probe.append({
            "sample_id": key,
            "x": {"current_question": history["question_info"], "history": history["history_info"]},
            "z": {"z_sol": {"standard_answer": item["answer"], "analysis": item["analysis"]},
                  "z_mis": [], "z_rub": list(RUBRIC), "z_skill": [],
                  "history_evidence_positions": matches},
        })
        review.append({
            "sample_id": key, "question_id": qid, "diagnosis": gold["diagnosis"],
            "longtutor_strategy": gold["strategy"], "reference_feedback": gold["content"],
            "action_annotation": draft, "strategy_action_candidates": list(candidates),
            "strategy_review": strategy_review,
            "evidence": [{"history_position": i, "record": history["history_info"][i - 1]} for i in matches],
        })
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "probe_inputs.jsonl", probe)
    write_rows(out / "review_targets.jsonl", review)
    write_json(out / "manifest.json", {
        "scope": "frozen Gold training pilot", "ids": ids, "sampling_tag": SAMPLING_TAG,
        "class_counts": dict(Counter(row["diagnosis"] for row in review)),
        "unique_current_questions": len({row["question_id"] for row in review}),
        "source_files_sha256": {"train": digest(train_path), "history": digest(history_path),
                                "question_bank": digest(BANK), "action_drafts": digest(args.drafts)},
        "gold_sft_adapter_sha256": {name: digest(GOLD_SFT / name) for name in ADAPTER_FILES},
        "files_sha256": {name: digest(out / name) for name in ("probe_inputs.jsonl", "review_targets.jsonl")},
        "privilege_status": {"standard_solutions": 64,
                             "rows_with_history_evidence_positions": sum(bool(r["evidence"]) for r in review),
                             "eedi_candidates": 0, "verified_skills": 0},
        "annotation_counts": dict(Counter(row["action_annotation"]["status"] for row in review)),
        "strategy_review_drafts": sum(row["strategy_review"] and row["action_annotation"]["status"] == "draft"
                                      for row in review),
        "pilot_design": "Same checkpoint and public history, structured privileges added only to teacher",
    })
    report = ["# 64 条教学动作初标", "", ("先按 LongTutor 策略查看候选动作，再从反馈原文判断实际动作。"
              "策略候选用于复核，可以依据原文保留候选外动作。待复核的完整案例附在表后。"), "",
              "| 序号 | 当前题 ID | Gold 教学策略 | 候选动作 | 动作初标 | 原文依据 | 待核对 |",
              "|---:|---|---|---|---|---|---|"]
    cases = []
    for i, row in enumerate(review, 1):
        draft = row["action_annotation"]
        quote = " ".join(draft["quote"].split()).replace("|", "\\|")
        note = draft.get("note", "").replace("|", "\\|")
        action = f"{draft['action']} {ACTIONS[draft['action']][0]}" if draft["action"] else "未定"
        if row["strategy_review"] and draft["status"] != "reviewed":
            note = "；".join(filter(None, (note, "初标超出策略候选")))
        candidates = "、".join(row["strategy_action_candidates"])
        report.append(f"| {i} | {row['question_id']} | {row['longtutor_strategy']} | {candidates} | "
                      f"{action} | {quote} | {note} |")
        if draft["status"] == "review_required" or (draft["status"] == "draft" and row["strategy_review"]):
            current = probe[i - 1]["x"]["current_question"]
            cases.extend((f"### 第 {i} 条 · 题 {row['question_id']}", "", f"样本 ID `{row['sample_id']}`", "",
                          f"**{row['longtutor_strategy']}** · 候选 {candidates} · 初标 {action}", "",
                          f"**待核对**　{note or '确认反馈中最先推进学习的实质动作。'}", "",
                          "**当前题与可见作答结果**", "", "```text", current, "```", "",
                          "**完整 Gold 反馈**", "", row["reference_feedback"], ""))
    (out / "动作标注初稿.md").write_text(
        "\n".join([*report, "", "## 待复核案例", "", *cases]) + "\n", encoding="utf-8"
    )


def load_pilot(directory, sources):
    manifest = read_json(directory / "manifest.json")
    if len(manifest["ids"]) != 64 or len(set(manifest["ids"])) != 64 or manifest["scope"] != "frozen Gold training pilot":
        raise ValueError("Unexpected pilot manifest")
    result = []
    for name in sources:
        path = directory / name
        if digest(path) != manifest["files_sha256"][name]:
            raise ValueError(f"Pilot file hash mismatch: {name}")
        data = indexed(read_rows(path), "sample_id")
        if list(data) != manifest["ids"]:
            raise ValueError(f"Pilot ID/order mismatch: {name}")
        result.append(data)
    return manifest, result


def messages(row, teacher):
    visible = row["x"]
    history = "\n\n".join(f"历史第 {i} 条\n{record}" for i, record in enumerate(visible["history"], 1))
    public = f"## 学生历史\n{history}\n\n## 当前题\n{visible['current_question']}"
    if teacher:
        public += "\n\n## 教师补充依据\n" + json.dumps(row["z"], ensure_ascii=False, sort_keys=True)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": public}]


def parse_decision(text):
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    diagnosis, action = parsed.get("diagnosis"), parsed.get("action")
    diagnosis = diagnosis if diagnosis in LABELS else None
    action = action if isinstance(action, str) and action in ACTIONS else None
    return {"diagnosis": diagnosis, "action": action,
            "protocol_valid": set(parsed) == {"diagnosis", "action"} and bool(diagnosis and action)}


def checkpoint_root(client, alias, manifest):
    models = [item for item in client.models.list().data if item.id == alias]
    if len(models) != 1 or not getattr(models[0], "root", None):
        raise ValueError(f"Could not identify the adapter served as {alias}")
    directory = Path(models[0].root).resolve(strict=True)
    for name, expected in manifest["gold_sft_adapter_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"The served adapter differs from frozen Gold SFT: {name}")
    return directory


def probe(args):
    from openai import OpenAI

    manifest, (inputs,) = load_pilot(args.pilot, ("probe_inputs.jsonl",))
    ids = manifest["ids"][:args.limit] if args.limit else manifest["ids"]
    if not ids:
        raise ValueError("Pilot limit must select at least one row")
    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)
    binding = args.output.with_suffix(".settings.json")
    settings = {"model": args.model, "port": args.port, "ids": ids, "temperature": 0,
                "max_completion_tokens": 96, "input_sha256": manifest["files_sha256"]["probe_inputs.jsonl"],
                "runner_sha256": digest(Path(__file__)), "response_format": "json_object",
                "checkpoint_sha256": manifest["gold_sft_adapter_sha256"]}
    with OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY",
                max_retries=0, timeout=90) as client:
        settings["checkpoint_root"] = str(checkpoint_root(client, args.model, manifest))
        if binding.exists():
            if read_json(binding) != settings:
                raise ValueError("Output belongs to a different run")
        elif args.output.exists():
            raise ValueError("Predictions exist without run settings")
        else:
            write_json(binding, settings)
        saved = indexed(read_rows(args.output), "sample_id") if args.output.exists() else {}
        if saved.keys() - set(ids) or any(set(row) != {"sample_id", *CONDITIONS} for row in saved.values()):
            raise ValueError("Saved predictions contain unexpected samples or incomplete pairs")
        for key in ids:
            if key in saved:
                continue
            result = {"sample_id": key}
            order = CONDITIONS if sha256(key.encode()).digest()[0] % 2 == 0 else CONDITIONS[::-1]
            for condition in order:
                response = client.chat.completions.create(
                    model=args.model, messages=messages(inputs[key], condition == "teacher_privileged"),
                    temperature=0, max_completion_tokens=96, response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                result[condition] = dict(parse_decision(raw), raw=raw)
            saved[key] = result
            write_rows(args.output, (saved[identifier] for identifier in ids if identifier in saved))


def score(args):
    manifest, (targets,) = load_pilot(args.pilot, ("review_targets.jsonl",))
    settings = read_json(args.predictions.with_suffix(".settings.json"))
    ids = settings["ids"]
    if (not ids or ids != manifest["ids"][:len(ids)]
            or settings["input_sha256"] != manifest["files_sha256"]["probe_inputs.jsonl"]
            or settings["checkpoint_sha256"] != manifest["gold_sft_adapter_sha256"]):
        raise ValueError("Prediction settings differ from the pilot")
    predictions = indexed(read_rows(args.predictions), "sample_id")
    if set(predictions) != set(ids):
        raise ValueError("Predictions do not cover all selected pairs")
    report = {"rows": len(ids), "model": settings["model"], "probe_settings": settings,
              "review_targets_sha256": manifest["files_sha256"]["review_targets.jsonl"], "conditions": {}}
    correct, action_matches = {}, {}
    known_actions = [key for key in ids if targets[key]["action_annotation"]["action"] in ACTIONS
                     and (targets[key]["action_annotation"]["status"] == "reviewed"
                          or (targets[key]["action_annotation"]["status"] == "draft"
                              and not targets[key].get("strategy_review", False)))]
    reviewed = [key for key in known_actions if targets[key]["action_annotation"]["status"] == "reviewed"]
    for condition in CONDITIONS:
        predicted = {key: parse_decision(predictions[key][condition]["raw"]) for key in ids}
        gold = Counter(targets[key]["diagnosis"] for key in ids)
        guessed = Counter(row["diagnosis"] for row in predicted.values())
        correct[condition] = {key for key in ids if predicted[key]["diagnosis"] == targets[key]["diagnosis"]}
        matches = Counter(targets[key]["diagnosis"] for key in correct[condition])
        per_class = {label: {"gold": gold[label], "correct": matches[label], "predicted": guessed[label]}
                     for label in LABELS}
        f1 = [2 * matches[label] / (gold[label] + guessed[label]) for label in LABELS if gold[label]]
        action_matches[condition] = {key for key in known_actions
                                     if predicted[key]["action"] == targets[key]["action_annotation"]["action"]}
        report["conditions"][condition] = {
            "diagnosis_accuracy": len(correct[condition]) / len(ids),
            "diagnosis_macro_f1": sum(f1) / len(f1),
            "diagnosis_by_class": per_class, "invalid_diagnosis_outputs": guessed[None],
            "invalid_action_outputs": sum(row["action"] is None for row in predicted.values()),
            "invalid_protocol_outputs": sum(not row["protocol_valid"] for row in predicted.values()),
            "candidate_action_agreement": {"rows": len(known_actions), "matching": len(action_matches[condition])},
            "reviewed_action_agreement": {"rows": len(reviewed),
                                          "matching": len(action_matches[condition] & set(reviewed))},
        }
    student, teacher = (correct[condition] for condition in CONDITIONS)
    report["paired_diagnosis"] = paired_counts(student, teacher, len(ids))
    student_action, teacher_action = (action_matches[condition] for condition in CONDITIONS)
    report["paired_candidate_action"] = paired_counts(student_action, teacher_action, len(known_actions))
    report["candidate_action_counts"] = dict(Counter(targets[key]["action_annotation"]["action"]
                                                    for key in known_actions))
    write_json(args.output, report)


def paired_counts(student, teacher, total):
    return {"both_matching": len(student & teacher), "student_only_matching": len(student - teacher),
            "teacher_only_matching": len(teacher - student), "neither_matching": total - len(student | teacher)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_cmd = commands.add_parser("build")
    build_cmd.add_argument("--drafts", type=Path, default=DRAFTS)
    build_cmd.set_defaults(handler=build)
    probe_cmd = commands.add_parser("probe")
    probe_cmd.add_argument("--port", type=int, default=8000)
    probe_cmd.add_argument("--model", required=True)
    probe_cmd.add_argument("--limit", type=int, choices=range(1, 65))
    probe_cmd.set_defaults(handler=probe)
    score_cmd = commands.add_parser("score")
    score_cmd.add_argument("--predictions", type=Path, required=True)
    score_cmd.set_defaults(handler=score)
    for command in (probe_cmd, score_cmd):
        command.add_argument("--pilot", type=Path, required=True)
    for command in (build_cmd, probe_cmd, score_cmd):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
