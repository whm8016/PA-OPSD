"""Test whether verified history facts improve an SFT teacher's real first move."""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.experiments.action_pilot import run as gold

CONDITIONS = ("public", "solution", "solution_evidence")
REPEAT = "solution_evidence_repeat"
MAX_COMPLETION = 384
PROBE_CONTEXT_BUDGET = 8192
MODEL = "memory_first"  # Serving alias only. The adapter SHA pins the original Gold SFT.
SYSTEM = (
    "你是小学数学导师。判断学生现在应该做的第一步，并直接对学生说出这一步。"
    "只给一条具体的提示或追问，给学生留出需要完成的工作，不直接给出当前题答案或完整解法。"
    "历史只记正误时，不猜测学生采用的解法、具体错误步骤或图像内容。"
    "如果使用历史，仅引用给定的原始位置。额外的标准解只供内部核对。\n"
    "教学动作：\n"
    + "\n".join(f"{code} {name}。{rule}" for code, (name, rule) in gold.ACTIONS.items())
    + '\n仅输出 JSON 对象，字段 "action" 为 A 到 H 的一个字母，"first_move" 为你对学生说的第一句话'
    '（若选择 A 不干预则为空字符串），'
    '"history_positions" 为这一步实际用到且在原始历史中可见的位置列表，未用到则给空列表。'
)


def facts(current_text, history, positions):
    from scripts.experiments.diagnoser_evidence.evidence import fields, recency

    current = fields(current_text)
    leaf = current["concepts"][-1:]
    supported = {}
    for position in positions:
        past = fields(history[position - 1])
        same = past["qid"] == current["qid"]
        gap = recency(current["date"], past["date"])
        if gap is None or gap < 0 or not past["qid"] or past["result"] not in {"正确", "错误"}:
            raise ValueError(f"Unverified visible history record at position {position}")
        if same or (leaf and leaf[0] in past["concepts"]):
            supported[position] = {
                "原始位置": position, "题号": past["qid"], "作答时间": past["date"].isoformat(),
                "结果": past["result"], "距当前天数": round(gap, 2),
                "匹配": "同题" if same else f"共同知识点 {leaf[0]}",
            }
    # Prioritize an exact reattempt plus the latest relevant correct and incorrect records.
    newest = sorted(supported, key=lambda position: (supported[position]["作答时间"], position), reverse=True)
    chosen = {
        next((p for p in newest if supported[p][field] == value), None)
        for field, value in (("匹配", "同题"), ("结果", "正确"), ("结果", "错误"))
    } - {None}
    return [supported[p] for p in sorted(chosen)]


def messages(row, condition):
    history = "\n\n".join(f"历史第 {i} 条\n{text}" for i, text in row["history"])
    visible = f"## 学生历史\n{history}\n\n## 当前题\n{row['current']}"
    if condition != "public":
        visible += "\n\n## 教师专用标准解\n" + json.dumps(row["solution"], ensure_ascii=False)
    if condition in {"solution_evidence", REPEAT} and row["facts"]:
        visible += ("\n\n## 教师专用历史事实索引\n"
                    "以下仅索引已给出的原始记录。知识点相同不能证明解法相同，正误不能证明具体步骤。\n"
                    + json.dumps(row["facts"], ensure_ascii=False))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": visible}]


def prepare(pilot, tokenizer_dir, output):
    from transformers import AutoTokenizer

    from scripts.experiments.action_pilot.score_evidence_probe import GATE
    from scripts.experiments.training_free_memory import prepare as prior

    manifest, (inputs, targets) = gold.load_pilot(pilot, ("probe_inputs.jsonl", "review_targets.jsonl"))
    split_path = gold.SPLIT / "split_manifest.json"
    split = gold.read_json(split_path)["ids"]
    pin = prior.verify_pins(prior.BUNDLE, gold.read_json(prior.BUNDLE / "manifest.json"), tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if not set(inputs) <= set(split["train"]) or set(inputs) & (set(split["validation"]) | set(split["test"])):
        raise ValueError("Teacher probe must stay inside the frozen training split")
    history_path = gold.REPRO / "gold_inputs.jsonl"
    history_sha = gold.digest(history_path)
    if (history_sha != gold.read_json(gold.REPRO / "preflight.json")["gold_input_sha256"]
            or history_sha != manifest["source_files_sha256"]["history"]):
        raise ValueError("Canonical Gold histories have changed")
    histories = gold.indexed((dict(row, sample_id=key) for row in gold.read_rows(history_path)
                              if (key := gold.history_key(row)) in inputs), "sample_id")
    adapter_sha = {name: gold.digest(gold.GOLD_SFT / name) for name in gold.ADAPTER_FILES}
    if adapter_sha != manifest["gold_sft_adapter_sha256"]:
        raise ValueError("Gold SFT adapter has changed")
    if gold.digest(gold.BANK) != manifest["source_files_sha256"]["question_bank"]:
        raise ValueError("Question bank has changed")
    bank = gold.indexed(gold.read_rows(gold.BANK), "id")
    prepared, audit = [], []
    for key, item in inputs.items():
        current = item["x"]["current_question"]
        history = item["x"]["history"]
        uid = key.split("||", 1)[0]
        source = histories[key]
        if (gold.history_key({"uid": uid, "question_info": current}) != key or len(history) != 199
                or current != source["question_info"] or history != source["history_info"]):
            raise ValueError(f"Unexpected fixed training record: {key}")
        qid = gold.QUESTION_ID.match(current)[1]
        official = bank[qid]["info"]
        solution = {"standard_answer": official["answer"], "analysis": official["analysis"]}
        if item["z"]["z_sol"] != solution:
            raise ValueError(f"Standard solution changed: {qid}")
        canonical = {"_key": key, "uid": uid, "question_info": current, "history_info": history}
        # This standalone action probe has no evidence questions. Reuse the temporal selection with no named IDs.
        _, route = prior.sample_inputs(canonical, {"memory": []})
        positions = route["temporal_positions"]
        selected = list(zip(positions, (history[i - 1] for i in positions)))
        evidence = facts(current, history, positions)
        row = {"sample_id": key, "current": current, "history": selected, "solution": solution, "facts": evidence}
        if messages(row, "solution_evidence") != messages(row, REPEAT):
            raise ValueError("Repeated prompts differ")
        counts = {arm: len(tokenizer.apply_chat_template(messages(row, arm), tokenize=True,
                                                        add_generation_prompt=True, return_dict=False))
                  for arm in CONDITIONS}
        if max(counts.values()) + MAX_COMPLETION > PROBE_CONTEXT_BUDGET:
            raise ValueError(f"Probe exceeds the fixed 8,192-token budget: {key}")
        prepared.append(row)
        audit.append({"sample_id": key, "uid": uid, "question_id": qid, "history_positions": positions,
                      "facts": evidence, "prompt_tokens": counts})
    if len({row["uid"] for row in audit}) != len(audit):
        raise ValueError("A training student occurs twice")

    ids = list(inputs)
    groups = [[key for key in ids if targets[key]["diagnosis"] == label] for label in gold.LABELS]
    if any(len(group) != 16 for group in groups):
        raise ValueError("Expected 16 training cases per diagnosis for sampling")
    indexed = gold.indexed(audit, "sample_id")
    smoke = [max(group, key=lambda key: max(indexed[key]["prompt_tokens"].values())) for group in groups]
    repeats = [key for group in groups for key in sorted(
        group, key=lambda key: gold.sha256(("teacher-evidence-repeat-v1|" + key).encode()).digest())[:4]]
    output.mkdir(parents=True, exist_ok=False)
    gold.write_rows(output / "inputs.jsonl", prepared)
    gold.write_rows(output / "audit.jsonl", audit)
    gold.write_json(output / "manifest.json", {
        "scope": "gold_train_teacher_first_move", "ids": ids, "smoke_ids": smoke, "repeat_ids": repeats,
        "conditions": CONDITIONS, "model_alias": MODEL, "model_description": "Original Gold SFT adapter",
        "base_model": prior.BASE_NAME, "base_revision": pin["revision"], "vllm_version": prior.VLLM_VERSION,
        "inference": prior.EVAL, "adapter_sha256": adapter_sha,
        "probe_context_budget": PROBE_CONTEXT_BUDGET,
        "temperature": 0, "max_completion_tokens": MAX_COMPLETION, "sampling_seed": 42, "gate": GATE,
        "inputs_sha256": gold.digest(output / "inputs.jsonl"),
        "audit_sha256": gold.digest(output / "audit.jsonl"),
        "pilot_manifest_sha256": gold.digest(pilot / "manifest.json"), "split_sha256": gold.digest(split_path),
        "source_sha256": gold.digest(Path(__file__)), "facts_coverage": sum(bool(row["facts"]) for row in prepared),
        "canonical_history_sha256": history_sha,
        "history_size": dict(Counter(len(row["history"]) for row in prepared)),
    })
    members = {
        "inputs.jsonl": output / "inputs.jsonl", "manifest.json": output / "manifest.json",
        "base_model.json": prior.BUNDLE / "base_model.json",
        "service.py": Path(prior.__file__).with_name("service.py"),
        str(Path(__file__).relative_to(gold.ROOT).as_posix()): Path(__file__),
        str(Path(gold.__file__).relative_to(gold.ROOT).as_posix()): Path(gold.__file__),
        "scripts/experiments/longtutor_reeval/run.py": gold.ROOT / "scripts/experiments/longtutor_reeval/run.py",
    }
    with ZipFile(output / "inference.zip", "x", compression=ZIP_DEFLATED) as zipped:
        for name, path in members.items():
            zipped.write(path, name)
        zipped.writestr("SHA256SUMS", "".join(f"{gold.digest(path)}  {name}\n" for name, path in members.items()))
    print(f"Prepared {len(ids)} pairs, {sum(bool(row['facts']) for row in prepared)} with indexed facts")


def parse_response(raw, positions):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != {"action", "first_move", "history_positions"}:
        return None
    action, first, citations = (value[key] for key in ("action", "first_move", "history_positions"))
    if (not isinstance(action, str) or action not in gold.ACTIONS or not isinstance(first, str)
            or (action == "A") == bool(first.strip())
            or not isinstance(citations, list) or any(type(i) is not int or i not in positions for i in citations)
            or len(set(citations)) != len(citations) or (action == "A" and citations)):
        return None
    return value


def verify_service(directory, port):
    receipt = directory / f"service.{port}.json"
    with TemporaryDirectory() as temporary:
        paths = {name: Path(temporary) / f"{name}.json" for name in ("models", "version")}
        for name, path in paths.items():
            route = "/v1/models" if name == "models" else "/version"
            with urlopen(f"http://127.0.0.1:{port}{route}", timeout=10) as response:
                path.write_bytes(response.read())
        subprocess.run([sys.executable, str(directory / "service.py"), "verify", "--port", str(port),
                        "--model", MODEL, "--receipt", str(receipt), "--models", str(paths["models"]),
                        "--version", str(paths["version"])], check=True)
    verified = gold.read_json(receipt)
    return {key: value for key, value in verified.items() if key != "pid"}


def parse_arm(raw, row, arm):
    return parse_response(raw, {position for position, _ in row["history"]})


def probe(directory, output, phase, port, *, arms=CONDITIONS, repeat=REPEAT, prompt=messages,
          parse=parse_arm, strict_smoke=True):
    from openai import OpenAI

    from scripts.experiments.longtutor_reeval.run import lock_output

    if not 1 <= port <= 65535 or output.suffix != ".jsonl":
        raise ValueError("Expected a valid local TCP port and .jsonl predictions path")
    manifest = gold.read_json(directory / "manifest.json")
    if (gold.digest(directory / "inputs.jsonl") != manifest["inputs_sha256"]
            or gold.digest(Path(__file__)) != manifest["source_sha256"] or manifest["model_alias"] != MODEL):
        raise ValueError("Probe input, source or model alias changed")
    inputs = gold.indexed(gold.read_rows(directory / "inputs.jsonl"), "sample_id")
    if list(inputs) != manifest["ids"] or tuple(manifest.get("conditions", arms)) != arms:
        raise ValueError("Probe IDs or arms differ from the fixed manifest")
    ids = manifest["smoke_ids"] if phase == "smoke" else manifest["ids"]
    repeats = set(manifest["repeat_ids"]) if phase == "full" else set()
    settings = output.with_suffix(".settings.json")
    progress = output.with_suffix(".progress.jsonl")
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    with (lock_output(output.with_suffix(output.suffix + ".lockdir")),
          OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY", max_retries=0, timeout=120) as client):
        service = verify_service(directory, port)
        adapter = gold.checkpoint_root(client, MODEL, {"gold_sft_adapter_sha256": manifest["adapter_sha256"]})
        binding = {"phase": phase, "ids": ids, "base_revision": manifest["base_revision"],
                   "manifest_sha256": gold.digest(directory / "manifest.json"), "service": service,
                   "adapter_sha256": manifest["adapter_sha256"], "adapter_root": str(adapter),
                   "temperature": 0, "sampling_seed": 42, "max_completion_tokens": MAX_COMPLETION}
        if output.exists() or (settings.exists() and gold.read_json(settings) != binding):
            raise ValueError("Output is complete or belongs to a different run")
        if not settings.exists():
            if progress.exists():
                raise ValueError("Progress exists without its settings")
            gold.write_json(settings, binding)
        saved = gold.indexed(gold.read_rows(progress), "sample_id") if progress.exists() else {}
        if saved.keys() - set(ids):
            raise ValueError("Saved predictions contain unexpected samples")
        invalid = 0
        for key in ids:
            row = inputs[key]
            shift = gold.sha256(key.encode()).digest()[0] % len(arms)
            conditions = arms[shift:] + arms[:shift] + ((repeat,) if key in repeats else ())
            result = saved.setdefault(key, {"sample_id": key})
            if result.keys() - {"sample_id", *conditions}:
                raise ValueError("Saved predictions contain unexpected arms")
            for arm in conditions:
                if arm not in result:
                    response = client.chat.completions.create(
                        model=MODEL, messages=prompt(row, arm), temperature=0, seed=42,
                        max_completion_tokens=MAX_COMPLETION, response_format={"type": "json_object"},
                    )
                    choice = response.choices[0]
                    raw = choice.message.content
                    decision = parse(raw, row, arm) if choice.finish_reason == "stop" else None
                    result[arm] = {"decision": decision, "raw": raw, "finish_reason": choice.finish_reason,
                                   "usage": response.usage.to_dict() if response.usage else None}
                    gold.write_rows(progress, saved.values())
                invalid += result[arm]["decision"] is None
        progress.replace(output)
        if phase == "smoke" and invalid:
            message = f"{invalid} probe outputs failed smoke validation; raw outputs preserved"
            if strict_smoke:
                raise ValueError(message)
            print(message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("prepare")
    build.add_argument("--pilot", type=Path, required=True)
    build.add_argument("--tokenizer", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    run = subcommands.add_parser("probe")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--phase", choices=("smoke", "full"), required=True)
    run.add_argument("--port", type=int, default=8808)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.pilot, args.tokenizer, args.output)
    else:
        probe(args.bundle, args.output, args.phase, args.port)


if __name__ == "__main__":
    main()
