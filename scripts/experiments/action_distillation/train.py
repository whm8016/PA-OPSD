"""Single-GPU LoRA comparisons on aligned public and teacher continuations."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

LABELS = ("Recall Failure", "Conceptual Gap", "Procedural Error", "Transfer Deficit")
ARMS = (
    "action",
    "token",
    "joint",
    "solution",
    "offpolicy",
    "sft",
    "mismatch",
    "irrelevant",
)
GOLD_SHA = "40be07306e9070de6b47d9dfdd372ae7a6183b81aad1d728ae27141f4f37313b"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def jsd(s, t):
    ls, lt = s.float().log_softmax(-1), t.detach().float().log_softmax(-1)
    mixture = torch.logaddexp(ls, lt) - 0.6931471805599453
    return (
        (ls.exp() * (ls - mixture) + lt.exp() * (lt - mixture)).sum(-1) * 0.5
    ).mean()


def context(tok, row, view, donor=None, skill=""):
    user = row["public"]
    if view != "public":
        if view == "skill_only":
            extra = skill
        else:
            ref = donor if view == "mismatch" else row
            extra = "标准解与解析：" + ref["solution"]
            if view != "solution":
                extra += "\n评价量规：" + ref["rubric"]
                if ref["misconception"]:
                    extra += (
                        "\n按知识点召回的备选误概念（需核实）：" + ref["misconception"]
                    )
            if skill:
                extra += "\n教学经验：" + skill
        ids = tok.encode(extra, add_special_tokens=False)[:640]
        if view == "irrelevant":
            # Same exact count in token space, with a neutral unrelated sentence repeated.
            filler = tok.encode("档案依次装入灰色文件夹。", add_special_tokens=False)
            ids = (filler * (len(ids) // len(filler) + 1))[: len(ids)]
        # Build a separate token segment, so the paired neutral view has exactly equal length.
        user += "\n仅供教师内部参考："
        head = tok.apply_chat_template(
            [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        before, tail = head.rsplit("仅供教师内部参考：", 1)
        return (
            tok.encode(before + "仅供教师内部参考：", add_special_tokens=False)
            + ids
            + tok.encode(tail, add_special_tokens=False)
        )
    return tok.apply_chat_template(
        [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": user},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )


def tensor(ids):
    return torch.tensor([ids], device="cuda")


def codes(tok, letters):
    ids = [tok.encode(c, add_special_tokens=False) for c in letters]
    if any(len(i) != 1 for i in ids) or len({i[0] for i in ids}) != len(ids):
        raise ValueError("Decision alphabet must use distinct single tokens")
    return [i[0] for i in ids]


@torch.no_grad()
def trajectory(model, tok, prompt, row, limit, forced=False):
    seq, decisions = [], []
    for label, choices, target in (
        ("证据=", "0123"[: 1 + len(row["evidence_positions"])], None),
        ("\n诊断=", "RCPT", "RCPT"[LABELS.index(row["gold"])] if forced else None),
        ("\n动作=", "ABCDEFGH", row["compatible"][0] if forced else None),
    ):
        seq.extend(tok.encode(label, add_special_tokens=False))
        allowed = codes(tok, choices)
        if target is None:
            logits = model(
                tensor(prompt + seq), use_cache=False, logits_to_keep=1
            ).logits[0, -1, allowed]
            pick = int(logits.argmax())
        else:
            pick = choices.index(target)
        decisions.append((len(seq), allowed, choices[pick]))
        seq.append(allowed[pick])
    seq.extend(tok.encode("\n给学生的话=", add_special_tokens=False))
    start = len(seq)
    if forced:
        tokens = tok.encode(row["reference"], add_special_tokens=False)[:limit]
    else:
        x = tensor(prompt + seq)
        generated = model.generate(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            do_sample=False,
            max_new_tokens=limit,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
        tokens = generated[0, x.size(1) :].tolist()
    if not tokens:
        raise ValueError("Empty continuation")
    return seq + tokens, decisions, start


def aligned(model, prompt, seq):
    # The first returned row predicts the first continuation token, even when prompt lengths differ.
    return model(
        tensor(prompt + seq[:-1]), use_cache=False, logits_to_keep=len(seq)
    ).logits[0]


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "model", "adapter", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--count", type=int, default=72)
    p.add_argument("--limit", type=int, default=40)
    opts = p.parse_args()
    cfg = json.loads((opts.bundle / "protocol.json").read_text())
    assert sha(opts.bundle / "train.jsonl") == cfg["sha256"]["train"]
    assert sha(opts.adapter / "adapter_model.safetensors") == GOLD_SHA
    data = {r["id"]: r for r in read(opts.bundle / "train.jsonl")}
    keys = cfg["train_ids"][: opts.count]
    if len(keys) != opts.count or len(keys) % 2 or len(set(keys)) != len(keys):
        raise ValueError("Expected unique training cases and paired accumulation")
    torch.manual_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(opts.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        opts.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, opts.adapter, is_trainable=True)
    model.eval()  # Gradients stay enabled in the student pass; disable dropout for matched contexts.
    if opts.arm == "offpolicy":
        model.load_adapter(
            opts.adapter, adapter_name="fixed_source", is_trainable=False
        )
        model.set_adapter("default")
    opt = torch.optim.AdamW(
        [v for v in model.parameters() if v.requires_grad], lr=1e-5, weight_decay=0
    )
    out = opts.out
    out.mkdir(parents=True, exist_ok=False)
    start_time = time.monotonic()
    # Derange by solution length, selecting a nearby distinct answer from the same training cohort.
    by_len = sorted(keys, key=lambda k: len(data[k]["solution"]))
    donors = {
        k: next(
            other
            for j in range(1, len(keys))
            if (other := by_len[(i + j) % len(keys)]) != k
            and data[other]["solution"] != data[k]["solution"]
        )
        for i, k in enumerate(by_len)
    }
    with (out / "steps.jsonl").open("x", encoding="utf-8") as logs:
        for i, key in enumerate(keys):
            row = data[key]
            public = context(tok, row, "public")
            view = (
                opts.arm
                if opts.arm in {"solution", "mismatch", "irrelevant"}
                else "full"
            )
            teacher = context(tok, row, view, data[donors[key]])
            off = opts.arm == "offpolicy"
            if off:
                model.set_adapter("fixed_source")
            seq, decisions, content = trajectory(
                model,
                tok,
                teacher if off else public,
                row,
                opts.limit,
                opts.arm == "sft",
            )
            if opts.arm != "sft":
                with torch.no_grad():
                    target = aligned(model, teacher, seq)
            if off:
                model.set_adapter("default")
            s = aligned(model, public, seq)
            pos, candidates, act = decisions[-1]
            if opts.arm == "sft":
                loc = [decisions[1][0], pos, *range(content, len(seq))]
                loss = torch.nn.functional.cross_entropy(
                    s[loc].float(), tensor(seq)[0, loc]
                )
                action_loss = word_loss = None
            else:
                action_loss = jsd(s[pos, candidates], target[pos, candidates])
                word_loss = jsd(s[content:], target[content:])
                loss = {
                    "action": action_loss,
                    "token": word_loss,
                    "solution": word_loss,
                }.get(opts.arm, (action_loss + word_loss) / 2)
            (loss / 2).backward()
            grad = None
            if i % 2:
                grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                if not torch.isfinite(torch.tensor(grad)):
                    raise ValueError("Nonfinite update")
                opt.step()
                opt.zero_grad(set_to_none=True)
            record = {
                "id": key,
                "arm": opts.arm,
                "step": (i + 1) // 2,
                "public_tokens": len(public),
                "teacher_tokens": len(teacher),
                "evidence": decisions[0][2],
                "diagnosis": decisions[1][2],
                "action": act,
                "content": tok.decode(seq[content:]),
                "response_tokens": seq,
                "action_pos": pos,
                "content_start": content,
                "loss": float(loss.detach()),
                "action_jsd": float(action_loss.detach())
                if action_loss is not None
                else None,
                "token_jsd": float(word_loss.detach())
                if word_loss is not None
                else None,
                "mismatch_id": donors[key] if view == "mismatch" else None,
                "grad_norm": grad,
            }
            print(json.dumps(record, ensure_ascii=False), file=logs, flush=True)
            if (i + 1) % 12 == 0:
                print(
                    i + 1,
                    "examples",
                    round(time.monotonic() - start_time, 1),
                    flush=True,
                )
    model.save_pretrained(
        out / "adapter", selected_adapters=["default"], safe_serialization=True
    )
    saved = sha(out / "adapter/adapter_model.safetensors")
    if saved == GOLD_SHA:
        raise ValueError("No weight update")
    (out / "receipt.json").write_text(
        json.dumps(
            {
                "arm": opts.arm,
                "examples": len(keys),
                "updates": len(keys) // 2,
                "initial": GOLD_SHA,
                "final": saved,
                "seconds": time.monotonic() - start_time,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "code_sha": sha(Path(__file__)),
                "packet_sha": sha(opts.bundle / "protocol.json"),
                "limit": opts.limit,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
