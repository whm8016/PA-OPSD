"""Generate separate diagnoser, pedagogue and verifier records using the frozen Gold SFT."""

import argparse
import json
import time
from pathlib import Path


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-only", action="store_true")
    for name in ("model", "adapter", "bundle", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    cfg = json.loads((a.bundle / "protocol.json").read_text())
    keys = set(cfg["train_ids"])
    mapped = json.loads((a.bundle / "eedi_candidates.json").read_text())
    cohorts = {}
    for name in ("train", "dev"):
        rows = [
            json.loads(line)
            for line in (a.bundle / (name + ".jsonl")).read_text().splitlines()
        ]
        for row in rows:
            row["misconception"] = mapped.get(row["id"], {}).get("candidate", "")
        cohorts[name] = [r for r in rows if name == "dev" or r["id"] in keys]
    primary = cohorts["train"] + ([] if a.train_only else cohorts["dev"])
    rows = primary + (
        []
        if a.train_only
        else [dict(r, misconception="") for r in primary if r["misconception"]]
    )
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    engine = LLM(
        model=str(a.model),
        enable_lora=True,
        max_lora_rank=8,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=32,
        gpu_memory_utilization=0.89,
        enforce_eager=True,
    )
    adapter = LoRARequest("gold", 1, str(a.adapter))
    a.out.mkdir(exist_ok=False)
    results = [
        {"id": r["id"], "split": "train" if r["id"] in keys else "dev"} for r in rows
    ]
    stages = [
        (
            "diagnoser",
            "你是诊断教师。依据学生可见历史与题面，从 R 回忆失败、C 概念缺口、P 步骤错误、T 迁移不足中选一码并给出一条证据，不能断言未展示的思路。",
        ),
        (
            "pedagogue",
            "你是教学策略教师。参考诊断者意见和标准解，为学生选择最适合的一个教学动作并写一句可执行提示。",
        ),
        (
            "verifier",
            "你是审核教师。检查数学、历史证据、是否泄题，必要时独立更正诊断、动作与话术。只输出 JSON，diagnosis 为 R/C/P/T 中一个，action 为 A 到 H 中一个，first_move 为给学生的中文提示。",
        ),
    ]
    for name, role in stages:
        prompts = []
        for row, record in zip(rows, results, strict=True):
            extra = (
                row["public"]
                + "\n标准解："
                + row["solution"][:1200]
                + "\n量规："
                + row["rubric"]
            )
            if row["misconception"]:
                extra += "\n备选误概念，需结合当前题核查：" + row["misconception"]
            for prior in ("diagnoser", "pedagogue"):
                if prior in record:
                    extra += "\n" + prior + "：" + record[prior]
            prompts.append(
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": row["system"] + "\n" + role},
                        {"role": "user", "content": extra},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        start = time.monotonic()
        answers = engine.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=200, seed=20261127),
            lora_request=adapter,
            use_tqdm=False,
        )
        for rec, answer in zip(results, answers, strict=True):
            o = answer.outputs[0]
            rec[name] = o.text
            rec[name + "_usage"] = [
                len(answer.prompt_token_ids),
                len(o.token_ids),
                o.finish_reason,
            ]
        ordinary = results[: len(primary)]
        no_eedi = {r["id"]: r for r in results[len(primary) :]}
        for suffix, items in [
            ("", ordinary),
            ("_no_eedi", [no_eedi.get(r["id"], r) for r in ordinary]),
        ]:
            if a.train_only and suffix:
                continue
            (a.out / (name + suffix + ".jsonl")).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items)
            )
        print(name, len(answers), round(time.monotonic() - start, 1), flush=True)


if __name__ == "__main__":
    main()
