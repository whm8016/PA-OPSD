"""Compare eight action probabilities on prefixes of saved public-view student rollouts."""

import argparse
import hashlib
import json
import re
from pathlib import Path

ACTIONS = list("ABCDEFGH")
VIEWS = ("public", "correct", "mismatch")


def load_rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def prepare(args):
    from transformers import AutoTokenizer

    from scripts.experiments.action_pilot import current_step

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    sources = {}
    for group, folder in (("train_unique", args.train), ("development", args.dev)):
        cfg = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        data = {r["sample_id"]: r for r in load_rows(folder / "inputs.jsonl")}
        pred_path = folder / "autodl_4090_49862/archive/runs/full.jsonl"
        preds = {r["sample_id"]: r for r in load_rows(pred_path)}
        keys = cfg["groups"]["fresh"]
        # Rotate sorted lengths to obtain an in-cohort derangement, paired without consulting outcomes.
        lens = {
            k: len(
                tok.encode(
                    json.dumps(data[k]["solution"], ensure_ascii=False),
                    add_special_tokens=False,
                )
            )
            for k in keys
        }
        order = sorted(keys, key=lambda k: (lens[k], k))
        partners = {}
        # Each pair swaps, and an odd tail uses a three-cycle.
        pos = 0
        while pos < len(order):
            width = 3 if len(order) - pos == 3 else 2
            block = order[pos : pos + width]
            partners.update(zip(block, block[1:] + block[:1], strict=True))
            pos += width
        if len(set(partners.values())) != len(keys) or any(
            k == v for k, v in partners.items()
        ):
            raise ValueError(
                "Mismatched solutions must be one-to-one without fixed points"
            )
        for k in keys:
            row = data[k]
            raw = preds[k]["step_public"]["raw"]
            m = re.match(r'^\s*\{\s*"action"\s*:\s*"([A-H])"', raw)
            entry = {"sample_id": k, "group": group, "included": False}
            if not m or preds[k]["step_public"]["decision"] is None:
                rows.append(entry)
                continue
            prefix = raw[: m.start(1)]
            observed = m.group(1)
            token_ids = {}
            mismatched = data[partners[k]]["solution"]
            if mismatched == row["solution"]:
                raise ValueError("Identical substitute solution")
            for view in VIEWS:
                obj = dict(row, solution=mismatched) if view == "mismatch" else row
                msg = current_step.messages(
                    obj, "step_public" if view == "public" else "step_solution"
                )
                text = (
                    tok.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True
                    )
                    + prefix
                )
                ids = tok.encode(text, add_special_tokens=False)
                candidates = []
                for action in ACTIONS:
                    whole = tok.encode(text + action, add_special_tokens=False)
                    if whole[:-1] != ids or len(whole) != len(ids) + 1:
                        raise ValueError(
                            "Action is not a single token on this actual prefix"
                        )
                    candidates.append(whole[-1])
                actual = tok.encode(text + raw[m.start(1) :], add_special_tokens=False)
                if actual[: len(ids) + 1] != ids + [
                    candidates[ACTIONS.index(observed)]
                ]:
                    raise ValueError(
                        "Public action token does not match the saved rollout"
                    )
                if len(set(candidates)) != 8 or (
                    entry.get("candidates") and entry["candidates"] != candidates
                ):
                    raise ValueError("Inconsistent action vocabulary")
                entry["candidates"] = candidates
                token_ids[view] = ids
            entry.update(
                included=True,
                observed=observed,
                public_prefix=prefix,
                mismatch_id=partners[k],
                reference_tokens=lens[k],
                mismatch_tokens=lens[partners[k]],
                ids=token_ids,
            )
            rows.append(entry)
        sources[group] = {
            "manifest_sha256": sha(folder / "manifest.json"),
            "pred_sha256": sha(pred_path),
            "sample_ids": keys,
            "solution_donor": partners,
        }
    with (args.out / "tokens.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(
        args.out / "config.json",
        {
            "action_codes": ACTIONS,
            "views": VIEWS,
            "sources": sources,
            "tokens_sha256": sha(args.out / "tokens.jsonl"),
            "source_sha256": sha(Path(__file__)),
            "adapter_sha256": cfg["adapter_sha256"],
            "base_revision": cfg["base_revision"],
            "estimand": "Conditional eight-action distributions on the student's observed pre-action prefix, before student-facing text. No updates. Mismatch swaps near-length solutions in the same cohort.",
            "primary": "Mean total variation for correct vs public and correct vs mismatch; count top-action changes. Distribution movement alone does not measure teaching quality.",
        },
    )


def infer(args):
    import peft
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    cfg = json.loads((args.bundle / "config.json").read_text(encoding="utf-8"))
    if (
        sha(args.bundle / "tokens.jsonl") != cfg["tokens_sha256"]
        or sha(Path(__file__)) != cfg["source_sha256"]
    ):
        raise ValueError("Unexpected probe inputs or code")
    for name, h in cfg["adapter_sha256"].items():
        if sha(args.adapter / name) != h:
            raise ValueError("Wrong adapter")
    pin = json.loads((args.bundle / "base_model.json").read_text(encoding="utf-8"))
    if pin["revision"] != cfg["base_revision"]:
        raise ValueError("Wrong base revision")
    for name, h in pin["weights"].items():
        if sha(args.model / name) != h["sha256"]:
            raise ValueError("Wrong base shard")
    args.out.mkdir(parents=True, exist_ok=False)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    model = peft.PeftModel.from_pretrained(
        base, args.adapter, is_trainable=False
    ).eval()
    rows = load_rows(args.bundle / "tokens.jsonl")
    save(
        args.out / "runtime.json",
        {
            "torch": torch.__version__,
            "peft": peft.__version__,
            "transformers": transformers.__version__,
            "mode": "eval, inference_mode, bfloat16 forward then float32 softmax",
            "gpu": torch.cuda.get_device_name(0),
            "base_revision": pin["revision"],
            "inputs": cfg["tokens_sha256"],
        },
    )
    with (
        torch.inference_mode(),
        (args.out / "distributions.jsonl").open("x", encoding="utf-8") as out,
    ):
        for row in rows:
            result = {k: v for k, v in row.items() if k != "ids"}
            if row["included"]:
                result["distribution"] = {}
                for v in VIEWS:
                    tokens = torch.tensor([row["ids"][v]], device="cuda:0")
                    logits = (
                        model(input_ids=tokens, use_cache=False, logits_to_keep=1)
                        .logits[0, -1]
                        .float()
                    )
                    subset = logits[row["candidates"]]
                    prob = subset.softmax(dim=0)
                    entry = {
                        "p": prob.cpu().tolist(),
                        "top": ACTIONS[int(prob.argmax())],
                        "action_mass": float(
                            (subset.logsumexp(0) - logits.logsumexp(0)).exp()
                        ),
                    }
                    if (
                        not torch.isfinite(prob).all()
                        or abs(sum(entry["p"]) - 1) > 1e-5
                    ):
                        raise ValueError("Invalid action probability")
                    result["distribution"][v] = entry
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            if len(rows) > 1 and row is rows[0]:
                print("First prefix verified", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    for name in ("train", "dev", "tokenizer", "out"):
        prep.add_argument("--" + name, type=Path, required=True)
    gpu = sub.add_parser("infer")
    for name in ("bundle", "model", "adapter", "out"):
        gpu.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "prepare":
        prepare(a)
    else:
        infer(a)


if __name__ == "__main__":
    main()
