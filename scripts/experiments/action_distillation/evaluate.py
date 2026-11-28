"""Batch public deployment and external prompts for all saved LoRA variants."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def write(path, rows):
    tmp = path.with_suffix(".partial")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", action="store_true")
    for name in ("bundle", "model", "adapter", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if __package__:
        from . import provenance
    else:
        import provenance
    base_info = provenance.verified_base(a.bundle, a.model)
    cfg = json.loads((a.bundle / "protocol.json").read_text())
    if (
        hashlib.sha256((a.bundle / "dev.jsonl").read_bytes()).hexdigest()
        != cfg["sha256"]["dev"]
    ):
        raise ValueError("Wrong development pool")
    rows = load(a.bundle / "dev.jsonl")
    external = load(a.bundle / "external_inputs.jsonl")
    extcfg = json.loads((a.bundle / "external_protocol.json").read_text())
    assert (
        hashlib.sha256((a.bundle / "external_inputs.jsonl").read_bytes()).hexdigest()
        == extcfg["sha256"]
    )
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    llm = LLM(
        model=str(a.model),
        enable_lora=True,
        max_lora_rank=8,
        max_loras=1,
        max_cpu_loras=2,
        dtype="bfloat16",
        max_model_len=6144,
        max_num_seqs=48,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    a.out.mkdir(exist_ok=True)
    arms = {} if a.scale else {"base": None, "gold_sft": a.adapter}
    variants = (
        "v3_pa",
        "v3_teacher_sft",
        "v3_action_kl",
        "v3_content",
        "offpolicy",
        "v3_solution_only",
        "v3_wrong",
        "v3_neutral",
        "v3_no_eedi",
        "joint",
        "action",
        "token",
        "sft",
        "solution",
        "mismatch",
        "irrelevant",
    )
    for n in ("v3_pa", "v3_teacher_sft", "v3_content") if a.scale else variants:
        cp = a.bundle / n / "adapter"
        if not (cp / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(cp)
        arms[n] = cp
    for n in [] if a.scale else range(1, 4):
        arms["cycle_" + str(n)] = a.bundle / "cycles" / str(n) / "candidate"
    for j, (name, cp) in enumerate(arms.items(), 1):
        start = time.monotonic()
        lora = LoRARequest(name, j, str(cp.resolve())) if cp else None
        dest = a.out / (name + ".jsonl")
        extdest = a.out / (name + ".external.jsonl")
        binding = {
            "model": name,
            "base": base_info,
            "adapter": provenance.digest(cp / "adapter_model.safetensors")
            if cp
            else None,
            "config": provenance.digest(cp / "adapter_config.json") if cp else None,
            "dev": cfg["sha256"]["dev"],
            "external": extcfg["sha256"],
            "external_commit": extcfg["pin"],
            "external_max_tokens": 128,
            "structured_max_tokens": 96,
            "temperature": 0,
            "wrapper": provenance.digest(Path(__file__)),
        }
        provenance.bind(a.out / (name + ".start.json"), binding, [dest, extdest])
        if not dest.exists():
            inputs = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": r["system"]},
                        {"role": "user", "content": r["public"]},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for r in rows
            ]
            outputs = [
                {"id": r["id"], "model": name, "public_tokens": len(ids)}
                for r, ids in zip(rows, inputs, strict=True)
            ]
            for field, label, space in [
                ("evidence", "证据=", "0123"),
                ("diagnosis", "\n诊断=", "RCPT"),
                ("action", "\n动作=", "ABCDEFGH"),
            ]:
                pieces = tok.encode(label, add_special_tokens=False)
                params = []
                for idx, r in enumerate(rows):
                    inputs[idx] = inputs[idx] + pieces
                    letters = (
                        space[: len(r["evidence_positions"]) + 1]
                        if field == "evidence"
                        else space
                    )
                    allowed = [
                        tok.encode(c, add_special_tokens=False)[0] for c in letters
                    ]
                    params.append(
                        SamplingParams(
                            temperature=0, max_tokens=1, allowed_token_ids=allowed
                        )
                    )
                resp = llm.generate(
                    [{"prompt_token_ids": ids} for ids in inputs],
                    params,
                    lora_request=lora,
                    use_tqdm=False,
                )
                for i, ans in enumerate(resp):
                    token = ans.outputs[0].token_ids
                    outputs[i][field] = tok.decode(token)
                    inputs[i] = inputs[i] + list(token)
            for i, ids in enumerate(inputs):
                inputs[i] = ids + tok.encode("\n给学生的话=", add_special_tokens=False)
            resp = llm.generate(
                [{"prompt_token_ids": ids} for ids in inputs],
                SamplingParams(temperature=0, max_tokens=96),
                lora_request=lora,
                use_tqdm=False,
            )
            for r, ans in zip(outputs, resp, strict=True):
                item = ans.outputs[0]
                r.update(
                    text=item.text,
                    finish_reason=item.finish_reason,
                    text_tokens=len(item.token_ids),
                )
            write(dest, outputs)
        if {r["id"] for r in load(dest)} != {r["id"] for r in rows} or len(
            load(dest)
        ) != len(rows):
            raise ValueError("Incomplete evaluation")
        extdest = a.out / (name + ".external.jsonl")
        if (
            not a.scale
            and name in {"base", "gold_sft", "v3_pa", "v3_teacher_sft"}
            and not extdest.exists()
        ):
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in external
            ]
            opts = [
                SamplingParams(temperature=0, max_tokens=128, stop=r["stop"])
                for r in external
            ]
            answers = llm.generate(prompts, opts, lora_request=lora, use_tqdm=False)
            results = [
                dict(
                    r,
                    model=name,
                    text=v.outputs[0].text,
                    finish=v.outputs[0].finish_reason,
                    prompt_tokens=len(v.prompt_token_ids),
                    output_tokens=len(v.outputs[0].token_ids),
                )
                for r, v in zip(external, answers, strict=True)
            ]
            write(extdest, results)
        if extdest.exists():
            cached = load(extdest)
            if len(cached) != len(external) or any(
                r["model"] != name
                or any(r[k] != s[k] for k in ("task", "index", "prompt", "stop"))
                for r, s in zip(cached, external, strict=True)
            ):
                raise ValueError("External output does not match frozen inputs")
        proof = {
            "start": binding,
            "structured_sha256": provenance.digest(dest),
            "external_sha256": provenance.digest(extdest) if extdest.exists() else None,
        }
        target = a.out / (name + ".proof.json")
        if target.exists() and json.loads(target.read_text()) != proof:
            raise ValueError("Modified completed predictions")
        target.write_text(json.dumps(proof))
        print(name, len(rows), round(time.monotonic() - start, 1), flush=True)


if __name__ == "__main__":
    main()
