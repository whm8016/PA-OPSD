"""Frozen historical-outcome intervention at fixed and freely selected tutoring prefixes."""

import argparse
import json
import re
from pathlib import Path

if __package__:
    from . import provenance
else:
    import provenance


def flip(public, count):
    history, marker, current = public.rpartition("\n当前题\n")
    if not marker:
        raise ValueError("Missing current-question boundary")
    history, changed = re.subn(
        r"(?m)^结果: (正确|错误)$",
        lambda m: "结果: " + ("错误" if m[1] == "正确" else "正确"),
        history,
    )
    if changed != count or not changed:
        raise ValueError("Unverified historical outcomes")
    return history + marker + current


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "model", "sft", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    bundle = args.bundle.resolve()
    base = provenance.verified_base(bundle, args.model)
    token = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dev = bundle / "dev.jsonl"
    old = bundle / "evaluation/gold_sft.jsonl"
    assert (
        provenance.digest(dev)
        == json.loads((bundle / "protocol.json").read_text())["sha256"]["dev"]
    )
    assert (
        provenance.digest(old)
        == json.loads((bundle / "evaluation/gold_sft.proof.json").read_text())[
            "structured_sha256"
        ]
    )
    refs = rows(dev)
    anchors = {r["id"]: r for r in rows(old)}
    assert set(anchors) == {r["id"] for r in refs} and len(anchors) == len(refs) == 140
    args.out.mkdir(exist_ok=True)
    groups = {
        "gold_sft": args.sft,
        "pa72": bundle / "v3_pa/adapter",
        "wrong72": bundle / "v3_wrong/adapter",
        "pa373": bundle / "scale/v3_pa/adapter",
    }
    freeze = {
        "base": base,
        "dev_sha": provenance.digest(dev),
        "prefix_source": provenance.digest(old),
        "weights": {},
        "code": provenance.digest(Path(__file__)),
        "preregistered_local_protocol": provenance.digest(args.out / "design.json"),
    }
    for arm, cp in groups.items():
        expected = (
            "40be07306e9070de6b47d9dfdd372ae7a6183b81aad1d728ae27141f4f37313b"
            if arm == "gold_sft"
            else json.loads((cp.parent / "receipt.json").read_text())["final"]
        )
        freeze["weights"][arm] = provenance.digest(cp / "adapter_model.safetensors")
        assert freeze["weights"][arm] == expected
    provenance.bind(
        args.out / "binding.json", freeze, [args.out / (n + ".jsonl") for n in groups]
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(backbone, args.sft).eval()
    seq = lambda text: token.encode(text, add_special_tokens=False)
    letters = {k: seq(k) for k in "0123RCPTABCDEFGH"}
    assert all(len(v) == 1 for v in letters.values()) and len(
        {v[0] for v in letters.values()}
    ) == len(letters)

    def probabilities(prompts, candidates):
        result = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for begin in range(0, len(order), args.batch):
            chosen = order[begin : begin + args.batch]
            width = max(len(prompts[i]) for i in chosen)
            ids = torch.full(
                (len(chosen), width),
                token.pad_token_id,
                dtype=torch.long,
                device="cuda",
            )
            mask = torch.zeros_like(ids)
            for j, i in enumerate(chosen):
                length = len(prompts[i])
                ids[j, -length:] = torch.tensor(prompts[i], device="cuda")
                mask[j, -length:] = 1
            logits = (
                model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=(mask.cumsum(1) - 1).clamp_min(0),
                    use_cache=False,
                    logits_to_keep=1,
                )
                .logits[:, -1]
                .float()
            )
            for j, i in enumerate(chosen):
                result[i] = (
                    logits[j, [letters[c][0] for c in candidates[i]]]
                    .softmax(0)
                    .cpu()
                    .tolist()
                )
        return result

    def evaluate(subset):
        cases = [(r, view) for r in subset for view in ("real", "flip")]
        prompts = []
        spaces = []
        fixed = []
        for r, view in cases:
            user = (
                r["public"]
                if view == "real"
                else flip(r["public"], len(r["evidence_positions"]))
            )
            prefix = token.apply_chat_template(
                [
                    {"role": "system", "content": r["system"]},
                    {"role": "user", "content": user},
                ],
                tokenize=True,
                add_generation_prompt=True,
            ) + seq("证据=")
            prompts.append(prefix)
            spaces.append("0123"[: len(r["evidence_positions"]) + 1])
            a = anchors[r["id"]]
            fixed.append(
                prefix
                + letters[a["evidence"]]
                + seq("\n诊断=")
                + letters[a["diagnosis"]]
                + seq("\n动作=")
            )
        ev = probabilities(prompts, spaces)
        chosen_e = [
            sp[max(range(len(probs)), key=probs.__getitem__)]
            for sp, probs in zip(spaces, ev)
        ]
        diagnosis = [v + letters[e] + seq("\n诊断=") for v, e in zip(prompts, chosen_e)]
        diag = probabilities(diagnosis, ["RCPT"] * len(cases))
        selected = ["RCPT"[max(range(4), key=p.__getitem__)] for p in diag]
        actions = [v + letters[k] + seq("\n动作=") for v, k in zip(diagnosis, selected)]
        action = probabilities(actions, ["ABCDEFGH"] * len(cases))
        direct = probabilities(fixed, ["ABCDEFGH"] * len(cases))
        return [
            {
                "id": r["id"],
                "view": v,
                "evidence": e,
                "diag_probs": d,
                "free_action_probs": a,
                "fixed_action_probs": f,
                "fixed_evidence": anchors[r["id"]]["evidence"],
                "fixed_diagnosis": anchors[r["id"]]["diagnosis"],
            }
            for (r, v), e, d, a, f in zip(
                cases, chosen_e, diag, action, direct, strict=True
            )
        ]

    with torch.inference_mode():
        for arm, cp in groups.items():
            dest = args.out / (arm + ".jsonl")
            if dest.exists():
                raise ValueError("Output already exists " + str(dest))
            if arm != "gold_sft":
                model.load_adapter(cp, adapter_name=arm, is_trainable=False)
                model.set_adapter(arm)
            smoke = evaluate(refs[:2])
            assert len(smoke) == 4 and all(
                abs(sum(r["fixed_action_probs"]) - 1) < 1e-5 for r in smoke
            )
            records = evaluate(refs)
            temp = dest.with_suffix(".partial")
            temp.write_text("".join(json.dumps(r) + "\n" for r in records))
            temp.replace(dest)
            print(arm, len(records), provenance.digest(dest), flush=True)


if __name__ == "__main__":
    main()
