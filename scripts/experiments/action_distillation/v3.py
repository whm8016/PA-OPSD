"""PA-OPSD with a fixed round-start policy as teacher."""

import argparse
import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment

if __package__:
    from . import train as core
else:
    import train as core


UPSTREAM_OPSD_COMMIT = "ae7d2519e94920c4eb6206c0c26de46d9c50abae"


def load_fixed_teacher(model, checkpoint):
    # The initial policy here includes the existing SFT LoRA. Keep a frozen copy
    # of it on the shared frozen base instead of disabling it as in base-start OPSD.
    model.load_adapter(checkpoint, adapter_name="fixed_teacher", is_trainable=False)
    model.set_adapter("default")
    if any(
        p.requires_grad and "default" not in name.split(".")
        for name, p in model.named_parameters()
    ):
        raise ValueError(
            "Fixed teacher requires all trainable weights to belong to the student adapter"
        )


@contextmanager
def fixed_teacher_forward(model):
    with torch.no_grad():
        try:
            model.set_adapter("fixed_teacher")
            # PEFT set_adapter makes the active adapter trainable automatically.
            model.requires_grad_(False)
            yield
        finally:
            model.set_adapter("default")


def action_kl(student, teacher):
    target = teacher.detach().float().log_softmax(-1)
    return (target.exp() * (target - student.float().log_softmax(-1))).sum(-1).mean()


def team_extra(row, team):
    # Standard solution and hypotheses are privileged. The team only saw this same problem.
    fields = [
        ("审核意见", team["verifier"][:450]),
        ("诊断者", team["diagnoser"][:200]),
        ("策略教师", team["pedagogue"][:200]),
        ("量规", row["rubric"]),
        ("标准解", row["solution"][:700]),
    ]
    if row["misconception"]:
        fields.insert(1, ("待核对误概念", row["misconception"]))
    return "\n".join(k + "：" + v for k, v in fields)


def teacher_ids(tok, row, text, neutral=False):
    marker = "\n教师私有资料："
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["public"] + marker},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    head, tail = prompt.rsplit(marker, 1)
    ids = tok.encode(text, add_special_tokens=False)[:768]
    if neutral:
        filler = tok.encode("档案依次装入灰色文件夹。", add_special_tokens=False)
        ids = (filler * (len(ids) // len(filler) + 1))[: len(ids)]
    return (
        tok.encode(head + marker, add_special_tokens=False)
        + ids
        + tok.encode(tail, add_special_tokens=False)
    )


def parse_team(text):
    try:
        value = json.JSONDecoder().raw_decode(text[text.index("{") :])[0]
        if (
            value["diagnosis"] in list("RCPT")
            and value["action"] in list("ABCDEFGH")
            and isinstance(value["first_move"], str)
            and value["first_move"].strip()
        ):
            return value
    except (ValueError, KeyError, TypeError):
        pass
    return None


def dialogue(model, tok, prompt, row, limit, teacher_text=None, stochastic=True):
    seq, decisions = [], []
    for lb, choices, field in [
        ("证据=", "0123"[: len(row["evidence_positions"]) + 1], None),
        ("\n诊断=", "RCPT", "diagnosis"),
        ("\n动作=", "ABCDEFGH", "action"),
    ]:
        seq.extend(tok.encode(lb, add_special_tokens=False))
        ids = core.codes(tok, choices)
        if teacher_text and field:
            choice = choices.index(teacher_text[field])
        else:
            with torch.no_grad():
                logits = (
                    model(core.tensor(prompt + seq), logits_to_keep=1, use_cache=False)
                    .logits[0, -1, ids]
                    .float()
                )
                choice = (
                    int(torch.multinomial((logits / 0.7).softmax(0), 1))
                    if stochastic
                    else int(logits.argmax())
                )
        decisions.append((len(seq), ids, choices[choice]))
        seq.append(ids[choice])
    seq.extend(tok.encode("\n给学生的话=", add_special_tokens=False))
    offset = len(seq)
    if teacher_text:
        words = tok.encode(teacher_text["first_move"], add_special_tokens=False)[:limit]
    else:
        x = core.tensor(prompt + seq)
        with torch.no_grad():
            params = {
                "do_sample": stochastic,
                "max_new_tokens": limit,
                "use_cache": True,
                "pad_token_id": tok.eos_token_id,
            }
            if stochastic:
                params.update(temperature=0.7, top_p=0.9)
            gen = model.generate(
                input_ids=x, attention_mask=torch.ones_like(x), **params
            )
        words = gen[0, len(prompt) + len(seq) :].tolist()
    if not words:
        raise ValueError("Missing teaching text")
    return seq + words, decisions, offset


def losses(s, t, seq, decisions, content):
    pos, ids, _ = decisions[-1]
    act = action_kl(s[pos, ids], t[pos, ids])
    token = core.jsd(s[content:], t[content:])
    # Canonical public scaffold is known, without any teacher label or preferred action.
    decision_positions = {entry[0] for entry in decisions}
    scaffold = [j for j in range(content) if j not in decision_positions]
    anchor = torch.nn.functional.cross_entropy(
        s[scaffold].float(), core.tensor(seq)[0, scaffold]
    )
    return act, token, anchor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "model", "adapter", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--arm",
        choices=[
            "pa",
            "action_kl",
            "content",
            "teacher_sft",
            "wrong",
            "neutral",
            "solution_only",
            "no_eedi",
        ],
        required=True,
    )
    p.add_argument(
        "--teacher-mode",
        choices=("fixed",),
        default="fixed",
        help="Freeze the initial Gold SFT policy",
    )
    p.add_argument("--count", type=int, default=72)
    p.add_argument("--limit", type=int, default=72)
    a = p.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text_sft = a.arm == "teacher_sft"
    cfg = json.loads((a.bundle / "protocol.json").read_text())
    assert core.sha(a.bundle / "train.jsonl") == cfg["sha256"]["train"]
    assert core.sha(a.adapter / "adapter_model.safetensors") == core.GOLD_SHA
    rows = {r["id"]: r for r in core.read(a.bundle / "train.jsonl")}
    mapped = json.loads((a.bundle / "eedi_candidates.json").read_text())
    for row in rows.values():
        row["misconception"] = mapped.get(row["id"], {}).get("candidate", "")
    team_file = (
        a.bundle
        / "team"
        / ("verifier_no_eedi.jsonl" if a.arm == "no_eedi" else "verifier.jsonl")
    )
    team = {r["id"]: r for r in core.read(team_file) if r["split"] == "train"}
    keys = cfg["train_ids"][: a.count]
    if (
        len(keys) != a.count
        or len(set(keys)) != len(keys)
        or not set(keys) <= team.keys()
    ):
        raise ValueError("Unverified team records or sample count")
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    text = {k: team_extra(rows[k], team[k]) for k in keys}
    lengths = [len(tok.encode(text[k], add_special_tokens=False)[:768]) for k in keys]
    cost = [
        [
            abs(lengths[i] - lengths[j])
            if rows[k]["qid"] != rows[l]["qid"]
            and rows[k]["solution"] != rows[l]["solution"]
            else 1000000
            for j, l in enumerate(keys)
        ]
        for i, k in enumerate(keys)
    ]
    ii, jj = linear_sum_assignment(cost)
    assert all(cost[i][j] < 1000000 for i, j in zip(ii, jj, strict=True))
    donors = {keys[i]: keys[j] for i, j in zip(ii, jj, strict=True)}
    a.out.mkdir(exist_ok=False)
    base = AutoModelForCausalLM.from_pretrained(
        a.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, a.adapter, is_trainable=True)
    model.eval()
    load_fixed_teacher(model, a.adapter)
    opt = torch.optim.AdamW(
        [v for v in model.parameters() if v.requires_grad], lr=1e-5, weight_decay=0
    )
    start = time.monotonic()
    with (a.out / "steps.jsonl").open("x", encoding="utf-8") as log:
        for i, key in enumerate(keys):
            torch.manual_seed(cfg["seed"] + i)
            row = rows[key]
            public = core.context(tok, row, "public")
            z = text[donors[key] if a.arm == "wrong" else key]
            if a.arm == "solution_only":
                z = row["solution"]
            if a.arm == "no_eedi":
                z = team_extra(dict(row, misconception=""), team[key])
            teacher = teacher_ids(tok, row, z, a.arm == "neutral")
            parsed = parse_team(team[key]["verifier"]) if text_sft else None
            rollout_context = (
                fixed_teacher_forward(model) if text_sft else nullcontext()
            )
            with rollout_context:
                seq, decs, off = dialogue(
                    model,
                    tok,
                    teacher if text_sft else public,
                    row,
                    a.limit,
                    parsed,
                    not text_sft,
                )
            if not text_sft:
                with fixed_teacher_forward(model):
                    target = core.aligned(model, teacher, seq)
            s = core.aligned(model, public, seq)
            details = {}
            if text_sft:
                loss = torch.nn.functional.cross_entropy(s.float(), core.tensor(seq)[0])
            else:
                act, word, anchor = losses(s, target, seq, decs, off)
                if a.arm == "action_kl":
                    loss = act + 0.05 * anchor
                elif a.arm == "content":
                    loss = word + 0.05 * anchor
                elif a.arm == "solution_only":
                    loss = word
                else:
                    loss = 0.5 * act + 0.5 * word + 0.05 * anchor
                details = {
                    "action_kl": act.item(),
                    "content_jsd": word.item(),
                    "format_ce": anchor.item(),
                }
            accumulation = min(2, len(keys) - (i // 2) * 2)
            (loss / accumulation).backward()
            grad = None
            if i % 2 or i + 1 == len(keys):
                grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1))
                if not torch.isfinite(torch.tensor(grad)):
                    raise ValueError("Nonfinite gradient")
                opt.step()
                opt.zero_grad(set_to_none=True)
            record = dict(
                id=key,
                arm=a.arm,
                loss=loss.item(),
                grad_norm=grad,
                public_tokens=len(public),
                teacher_tokens=len(teacher),
                evidence=decs[0][2],
                diagnosis=decs[1][2],
                action=decs[2][2],
                action_position=decs[2][0],
                content_start=off,
                output_tokens=seq,
                first_move=tok.decode(seq[off:], skip_special_tokens=True),
                capped=len(seq) - off == a.limit and seq[-1] != tok.eos_token_id,
                source=("team_text" if parsed else "fixed_teacher_rollout")
                if text_sft
                else "on_policy_sample",
                donor=donors[key] if a.arm == "wrong" else None,
                **details,
            )
            print(json.dumps(record, ensure_ascii=False), file=log, flush=True)
    model.save_pretrained(a.out / "adapter", selected_adapters=["default"])
    h = core.sha(a.out / "adapter/adapter_model.safetensors")
    assert h != core.GOLD_SHA
    (a.out / "receipt.json").write_text(
        json.dumps(
            {
                "arm": a.arm,
                "examples": len(keys),
                "updates": (len(keys) + 1) // 2,
                "teacher_mode": a.teacher_mode,
                "teacher_snapshot_adapter_sha256": core.GOLD_SHA,
                "upstream_opsd_reference": UPSTREAM_OPSD_COMMIT,
                "initial": core.GOLD_SHA,
                "final": h,
                "seconds": time.monotonic() - start,
                "code": core.sha(Path(__file__)),
                "shared": core.sha(Path(core.__file__)),
                "team": core.sha(team_file),
                "limit": a.limit,
                "input_hash": cfg["sha256"]["train"],
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    )


if __name__ == "__main__":
    main()
