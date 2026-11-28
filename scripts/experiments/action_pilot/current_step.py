"""Paired teaching-hint pilot with a current-problem instruction and a solution ablation."""

import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.experiments.action_pilot import classifier_teacher as prior
from scripts.experiments.action_pilot import evidence_probe
from scripts.experiments.action_pilot import run as gold
from scripts.experiments.action_pilot import score_evidence_probe as scoring

ARMS = ("original", "step_solution", "step_public")
REPEAT = "step_solution_repeat"
PARENT = prior.ROOT / "classifier_teacher_20261127_run1"
PARENT_SHA = "a23ebd83811216f58e77a5bbd9768879eb3f5fa0a05c4cec828083523bce69dd"
INSTRUCTION = (
    "实际观测只有题目、时间、作答正误，当前学生没有展示计算过程。请只对当前题说一句可以立刻执行的提示。"
    "找出题目中的一个具体中间量、关系或关键判断，让学生先写出它或展示相关一步；"
    "用当前题中的对象命名这个小任务。不要提及过去的表现、掌握程度或假定的解法。"
    "不要猜测错在哪里，不给最终答案，不把完整解法变成反问句。"
    "先确定实际话术，再按话术填最贴切的动作字母。询问一个明确中间量属于 G；"
    "请学生自己解释或展示已有想法属于 C；E 需要题中出现可核对的错误步骤。"
)


def messages(row, arm):
    system, user = prior.prompts(row, prior.CONTROL)
    if arm != "original":
        system = dict(system, content=system["content"] + "\n" + INSTRUCTION)
    if arm == "step_public":
        user = evidence_probe.messages(row, "public")[1]
        if row["facts"]:
            _, extra = prior.prompts(row, prior.CONTROL)[1]["content"].split(
                "## 教师专用历史事实索引", 1
            )
            user = dict(
                user, content=user["content"] + "\n\n## 教师专用历史事实索引" + extra
            )
    return [system, user]


def prepare(out):
    from scripts.build_pa_opsd_gold_split import question_group
    from scripts.experiments.training_free_memory import prepare as retrieval

    if gold.digest(PARENT / "manifest.json") != PARENT_SHA:
        raise ValueError("Unexpected prior study")
    manifest, old = prior.verify_bundle(PARENT)
    _, used = prior.prior_inputs()
    used.update(old)
    train = gold.indexed(
        gold.read_rows(gold.SPLIT / "development/train.jsonl"), "sample_id"
    )
    source = gold.REPRO / "gold_inputs.jsonl"
    if (
        gold.digest(source) != manifest["canonical_history_sha256"]
        or gold.digest(gold.SPLIT / "development/train.jsonl")
        != manifest["train_sha256"]
        or gold.digest(gold.BANK) != manifest["question_bank_sha256"]
    ):
        raise ValueError("Frozen data changed")
    histories = {
        gold.history_key(row): row
        for row in gold.read_rows(source)
        if gold.history_key(row) in train
    }
    bank = gold.indexed(gold.read_rows(gold.BANK), "id")
    ident = {
        k: (row["uid"], *question_group(row["question_info"]))
        for k, row in histories.items()
    }
    excluded = (set(), set(), set())
    for key in used:
        prior.remember(ident[key], excluded)
    inputs = list(old.values())
    fresh = []
    for key in sorted(
        train, key=lambda k: gold.sha256(("current_step_v1|" + k).encode()).digest()
    ):
        row = histories[key]
        if prior.overlaps(ident[key], excluded) or gold.IMAGE.search(
            row["question_info"]
        ):
            continue
        sol = bank[ident[key][1]]["info"]
        if not sol.get("answer") or not str(sol.get("analysis") or "").strip():
            continue
        prior.remember(ident[key], excluded)
        _, route = retrieval.sample_inputs(dict(row, _key=key), {"memory": []})
        pos = route["temporal_positions"]
        inputs.append(
            {
                "sample_id": key,
                "current": row["question_info"],
                "history": [(p, row["history_info"][p - 1]) for p in pos],
                "facts": evidence_probe.facts(
                    row["question_info"], row["history_info"], pos
                ),
                "solution": {
                    "standard_answer": sol["answer"],
                    "analysis": sol["analysis"],
                },
            }
        )
        fresh.append(key)
    if not fresh:
        raise ValueError("No fresh training questions available")
    # No predicted or gold labels are needed by any arm.
    inputs = [
        {k: v for k, v in row.items() if k != "classifier_prediction"} for row in inputs
    ]
    ids = [row["sample_id"] for row in inputs]
    groups = {"regression": list(old), "fresh": fresh}
    repeats = manifest["repeat_ids"] + fresh[::2]
    out.mkdir(parents=True, exist_ok=False)
    gold.write_rows(out / "inputs.jsonl", inputs)
    gold.write_json(out / "cohorts.json", groups)
    keep = (
        "model_alias",
        "model_description",
        "base_model",
        "base_revision",
        "vllm_version",
        "inference",
        "adapter_sha256",
        "probe_context_budget",
        "temperature",
        "max_completion_tokens",
        "sampling_seed",
        "split_sha256",
        "canonical_history_sha256",
        "source_sha256",
    )
    config = {k: manifest[k] for k in keep}
    config.update(
        scope="current_step_v1",
        conditions=ARMS,
        repeat_arm=REPEAT,
        ids=ids,
        smoke_ids=manifest["smoke_ids"] + fresh[:1],
        repeat_ids=repeats,
        groups=groups,
        cohort_labels={
            name: dict(
                Counter(
                    train[k]["teacher_privileged"]["gold_reference"]["diagnosis"]
                    for k in keys
                )
            )
            for name, keys in groups.items()
        },
        inputs_sha256=gold.digest(out / "inputs.jsonl"),
        parent_sha256=PARENT_SHA,
        intervention_sha256=gold.digest(Path(__file__)),
        primary="paired safe first move on original vs step_solution; report cohorts separately",
        privilege="step_solution vs step_public; same instruction, history and visible fact index",
        training_gate="This repair pilot cannot authorize training by itself. Requires independent multi-class confirmation of safer hints AND a privilege-specific advantage.",
    )
    gold.write_json(out / "manifest.json", config)
    with (
        ZipFile(PARENT / "inference.zip") as source_zip,
        ZipFile(out / "inference.zip", "x", ZIP_DEFLATED) as dest,
    ):
        files = {
            n: source_zip.read(n)
            for n in source_zip.namelist()
            if n
            not in (
                "SHA256SUMS",
                "manifest.json",
                "inputs.jsonl",
                "scripts/experiments/action_pilot/run_classifier_teacher.sh",
            )
        }
        files.update(
            {n: (out / n).read_bytes() for n in ("manifest.json", "inputs.jsonl")}
        )
        files[Path(__file__).relative_to(gold.ROOT).as_posix()] = Path(
            __file__
        ).read_bytes()
        runner = Path(__file__).with_name("run_current_step.sh")
        files[runner.relative_to(gold.ROOT).as_posix()] = runner.read_bytes()
        for name, content in files.items():
            dest.writestr(name, content)
        dest.writestr(
            "SHA256SUMS",
            "".join(
                gold.sha256(v).hexdigest() + "  " + k + "\n" for k, v in files.items()
            ),
        )


def verify(bundle):
    cfg = gold.read_json(bundle / "manifest.json")
    if (
        cfg["scope"] != "current_step_v1"
        or cfg["intervention_sha256"] != gold.digest(Path(__file__))
        or list(ARMS) != cfg["conditions"]
        or gold.digest(bundle / "inputs.jsonl") != cfg["inputs_sha256"]
    ):
        raise ValueError("Input or intervention changed")
    sample = gold.indexed(gold.read_rows(bundle / "inputs.jsonl"), "sample_id")
    if (
        list(sample) != cfg["ids"]
        or cfg["groups"]["regression"] + cfg["groups"]["fresh"] != cfg["ids"]
    ):
        raise ValueError("Cohort mismatch")
    return cfg, sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "smoke", "full", "blind", "score"))
    p.add_argument("bundle", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--port", type=int, default=8808)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.bundle)
        return
    cfg, _ = verify(args.bundle)
    if args.command in ("smoke", "full"):
        evidence_probe.probe(
            args.bundle,
            args.out,
            args.command,
            args.port,
            arms=ARMS,
            repeat=REPEAT,
            prompt=messages,
            parse=prior.parse,
            strict_smoke=False,
        )
        if args.command == "smoke":
            rows = list(gold.read_rows(args.out))
            if any(
                sum(r[a]["decision"] is not None for r in rows) < len(rows) - 1
                or any(r[a]["finish_reason"] != "stop" for r in rows)
                for a in ARMS
            ):
                raise ValueError("Paired smoke failed")
    elif args.command == "blind":
        scoring.make_blind(
            args.bundle,
            args.predictions,
            args.out,
            verify=verify,
            repeat=REPEAT,
            parse=prior.parse,
        )
    else:
        review = args.out
        outcomes = scoring.adjudicate(
            args.bundle, review, review / "ratings.jsonl", cfg, REPEAT
        )
        result = {}
        for group, keys in cfg["groups"].items():
            part = {}
            for a in ARMS:
                part[a] = {
                    f: sum(outcomes[k][a][f] for k in keys)
                    for f in (
                        "passed",
                        "label_matches",
                        "protocol_valid",
                        *scoring.RISK,
                    )
                }
            for control in ("original", "step_public"):
                diffs = [
                    int(outcomes[k]["step_solution"]["passed"])
                    - int(outcomes[k][control]["passed"])
                    for k in keys
                ]
                part["versus_" + control] = {
                    "wins": diffs.count(1),
                    "losses": diffs.count(-1),
                    "net": sum(diffs),
                }
            repeat_keys = [k for k in keys if k in cfg["repeat_ids"]]
            part["repeat"] = {
                "n": len(repeat_keys),
                "both_safe": sum(
                    outcomes[k]["step_solution"]["passed"]
                    and outcomes[k][REPEAT]["passed"]
                    for k in repeat_keys
                ),
            }
            result[group] = part
        gold.write_rows(
            review / "scored.rows.jsonl",
            ({"sample_id": k, **v} for k, v in outcomes.items()),
        )
        gold.write_json(review / "summary.json", result)


if __name__ == "__main__":
    main()
