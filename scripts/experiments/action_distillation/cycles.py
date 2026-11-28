"""Three stateful rounds of candidate skill extraction, independent checks and no-retrieval weight tests."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

LABELS = ("Recall Failure", "Conceptual Gap", "Procedural Error", "Transfer Deficit")


def read(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")


def partitions(rows, exclude, n):
    groups = {}
    for row in rows:
        if row["image"] or any(row[k] in exclude[k] for k in exclude):
            continue
        groups.setdefault(row["stem"], []).append(row)
    stems = sorted(
        groups, key=lambda s: hashlib.sha256(("v3-cycle-01|" + s).encode()).digest()
    )
    buckets = [[] for _ in range(n)]
    owner = {}
    for i, stem in enumerate(stems):
        for row in groups[stem]:
            b = i % n
            if any(owner.get((k, row[k]), b) != b for k in ("uid", "qid", "stem")):
                continue
            if len(buckets[b]) >= 20:
                continue
            buckets[b].append(row["id"])
            for k in ("uid", "qid", "stem"):
                owner[k, row[k]] = b
    return buckets


def freeze(args):
    cfg = json.loads((args.bundle / "protocol.json").read_text(encoding="utf-8"))
    train = read(args.bundle / "train.jsonl")
    dev = read(args.bundle / "dev.jsonl")
    main = [r for r in train if r["id"] in cfg["train_ids"]]
    keys = ("uid", "qid", "stem")
    pools = partitions(train, {k: {r[k] for r in main} for k in keys}, 9)
    old = read(args.prior)
    exclude = {
        k: {r[k] for r in train} | {r[{"stem": "stem_sha256"}.get(k, k)] for r in old}
        for k in keys
    }
    confirm = partitions(dev, exclude, 3)
    rounds = [
        {
            "source": pools[i * 3],
            "admission": pools[i * 3 + 1],
            "update": pools[i * 3 + 2],
            "confirm": confirm[i],
        }
        for i in range(3)
    ]
    if min(len(v) for r in rounds for v in r.values()) < 4:
        raise ValueError("Insufficient independent cohort")
    save(
        args.bundle / "cycles_protocol.json",
        {
            "rounds": rounds,
            "seed": 20261127,
            "source_conditions": "A source positive and counterexample are required. Repeated or conflicting cues do not enter the bank. Textual question and explicit-answer patterns are measured on actual traces.",
            "gate": "Incremental admission compares prior bank to bank+candidate; at least two net additional joint diagnosis+action hits, no reduction in diagnosis accuracy or increased explicit-answer flags. Independent no-retrieval confirmation additionally requires lower 95% student-paired bootstrap bounds >= -0.05 in macro-F1 and joint hit rate, and mean deployment tokens <=1.10x incumbent at equal calls.",
            "retain": "Only report if retrieved-minus-incumbent joint hit rate >= 0.05 on confirmation. All fail-admission cards are trained only in tagged shadow branches, never inherited.",
            "sealed": False,
        },
    )


def cues(row):
    history, _, current = row["public"].rpartition("\n当前题\n")
    qid = current.splitlines()[0]
    return {
        "all",
        "repeat" if qid in history else "new",
        "recent_correct"
        if "结果: 正确" in history.splitlines()[-1:]
        else "recent_other",
    }


def evaluate(model, tok, rows, skills=()):
    import train as core
    import v3

    out = []
    for row in rows:
        active = [s for s in skills if s["cue"] in cues(row)]
        public = core.context(tok, row, "public")
        prompt = (
            v3.teacher_ids(tok, row, "\n".join(s["instruction"] for s in active))
            if active
            else public
        )
        started = time.monotonic()
        seq, decisions, offset = v3.dialogue(
            model, tok, prompt, row, 72, stochastic=False
        )
        text = tok.decode(seq[offset:], skip_special_tokens=True)
        diag = LABELS["RCPT".index(decisions[1][2])]
        out.append(
            {
                "id": row["id"],
                "uid": row["uid"],
                "actual": row["gold"],
                "pred": diag,
                "compatible": row["compatible"],
                "action": decisions[2][2],
                "ok_diag": diag == row["gold"],
                "joint": diag == row["gold"] and decisions[2][2] in row["compatible"],
                # A literal final-answer announcement is a narrow observable safety indicator.
                "explicit_answer": bool(
                    re.search(
                        r"(?:最终答案|正确答案|所以答案|答案应该)\s*[是为：:]", text
                    )
                ),
                "text": text,
                "active": bool(active),
                "tokens": len(seq),
                "prompt_tokens": len(prompt),
                "model_invocations": 4,
                "processed_tokens": sum(len(prompt) + d[0] for d in decisions)
                + len(prompt)
                + len(seq),
                "seconds": time.monotonic() - started,
            }
        )
    return out


def extract(rows, outputs):
    by_id = {r["id"]: r for r in outputs}
    candidates = []
    actions = {
        "C": "先请学生解释一个关键判断",
        "D": "用一个概念关系引导",
        "G": "请学生独立完成一个中间量",
        "F": "选一个便于类比的小例子",
        "E": "仅在出现可核对的步骤时指出该步",
        "H": "只示范必要的一小步",
    }
    for cue in sorted(set.union(*(cues(r) for r in rows))):
        subset = [r for r in rows if cue in cues(r)]
        if len(subset) < 3:
            continue
        for action, wording in actions.items():
            gain = sum(
                int(action in r["compatible"])
                - int(by_id[r["id"]]["action"] in r["compatible"])
                for r in subset
            )
            candidates.append((gain, len(subset), cue, action, wording))
    gain, n, cue, act, wording = max(candidates)
    support = [by_id[r["id"]] for r in rows if cue in cues(r)]
    positives = [
        r
        for r in support
        if r["action"] in r["compatible"]
        and not r["explicit_answer"]
        and r["text"].strip()
    ]
    negatives = [
        r for r in support if r["action"] not in r["compatible"] or r["explicit_answer"]
    ]
    observable = []
    if positives and sum(
        "？" in r["text"] or "?" in r["text"] for r in positives
    ) * 2 >= len(positives):
        observable.append("用一个问题结束，把下一步留给学生。")
    if any(r["explicit_answer"] for r in negatives):
        observable.append(
            "历史失败例包含报出最终答案的表述，把它改成学生自己可执行的步骤。"
        )
    wording = (
        "相似作答轨迹的经验是"
        + wording
        + "。"
        + "".join(observable)
        + "先核对具体题面和正误记录；若不适用，独立选择更合适的动作。不揭示最终答案。"
    )
    return {
        "cue": cue,
        "action": act,
        "instruction": wording,
        "source_estimated_net": gain,
        "source_support": n,
        "positive_cases": [{"id": r["id"], "text": r["text"]} for r in positives[:2]],
        "negative_cases": [{"id": r["id"], "text": r["text"]} for r in negatives[:2]],
        "source_ids": [r["id"] for r in rows if cue in cues(r)],
    }


def scores(rows):
    y = [r["actual"] for r in rows]
    p = [r["pred"] for r in rows]
    f = []
    for c in LABELS:
        tp = sum(a == b == c for a, b in zip(y, p))
        den = y.count(c) + p.count(c)
        f.append(2 * tp / den if den else 0.0)
    return {
        "joint": sum(r["joint"] for r in rows),
        "diagnosis": sum(r["ok_diag"] for r in rows),
        "macro_f1": sum(f) / 4,
        "explicit": sum(r["explicit_answer"] for r in rows),
        "n": len(rows),
    }


def gate(old, new, confirm=False):
    if [r["id"] for r in old] != [r["id"] for r in new]:
        raise ValueError("Unpaired evaluation")
    a, b = scores(old), scores(new)
    accept = (
        b["joint"] >= a["joint"] + 2
        and b["diagnosis"] >= a["diagnosis"]
        and b["explicit"] <= a["explicit"]
    )
    result = {"before": a, "after": b}
    if confirm:
        rng = np.random.default_rng(20261127)
        clusters = {}
        for i, r in enumerate(old):
            clusters.setdefault(r["uid"], []).append(i)
        clusters = list(clusters.values())
        vals = []
        for _ in range(2000):
            ids = [
                j
                for k in rng.integers(len(clusters), size=len(clusters))
                for j in clusters[k]
            ]
            sa, sb = scores([old[i] for i in ids]), scores([new[i] for i in ids])
            vals.append(
                (
                    (sb["joint"] - sa["joint"]) / len(ids),
                    sb["macro_f1"] - sa["macro_f1"],
                )
            )
        ci = np.quantile(vals, [0.025, 0.975], axis=0).tolist()
        result["ci95_joint_and_macro"] = ci
        result["deployment_tokens_ratio"] = sum(
            r["processed_tokens"] for r in new
        ) / sum(r["processed_tokens"] for r in old)
        accept = (
            accept and min(ci[0]) >= -0.05 and result["deployment_tokens_ratio"] <= 1.10
        )
    return dict(result, accepted=accept)


def verified_cycle_seed(bundle):
    training = bundle / "v3_pa"
    receipt = json.loads((training / "receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("arm") != "pa"
        or receipt.get("teacher_mode") != "fixed"
        or not receipt.get("teacher_snapshot_adapter_sha256")
        or receipt["teacher_snapshot_adapter_sha256"] != receipt.get("initial")
    ):
        raise ValueError("Skill training requires a fixed-teacher V3 run receipt")
    adapter = training / "adapter"
    with (adapter / "adapter_model.safetensors").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != receipt.get("final"):
        raise ValueError("V3 adapter weights do not match the run receipt")
    return adapter


def execute(a):
    incumbent = verified_cycle_seed(a.bundle)
    import torch
    import train as core
    import v3
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads((a.bundle / "cycles_protocol.json").read_text())
    data = {
        r["id"]: r
        for name in ("train", "dev")
        for r in read(a.bundle / (name + ".jsonl"))
    }
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    a.out.mkdir(exist_ok=False)
    memory = []
    for rnd, parts in enumerate(cfg["rounds"], 1):
        folder = a.out / str(rnd)
        folder.mkdir()
        groups = {k: [data[key] for key in keys] for k, keys in parts.items()}
        base = AutoModelForCausalLM.from_pretrained(
            a.model,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(base, incumbent, is_trainable=True)
        model.eval()
        source = evaluate(model, tok, groups["source"])
        skill = extract(groups["source"], source)
        old = evaluate(model, tok, groups["admission"], memory)
        bank_candidate = [*memory, skill]
        with_card = evaluate(model, tok, groups["admission"], bank_candidate)
        admission = gate(old, with_card)
        admission["duplicate_or_conflict"] = any(
            s["cue"] == skill["cue"] for s in memory
        )
        admission["source_positive_and_counterexample"] = bool(
            skill["positive_cases"] and skill["negative_cases"]
        )
        admission["accepted"] = (
            admission["accepted"]
            and not admission["duplicate_or_conflict"]
            and admission["source_positive_and_counterexample"]
        )
        before = evaluate(model, tok, groups["confirm"])
        retrieved = evaluate(model, tok, groups["confirm"], bank_candidate)
        for name, result in [
            ("source", source),
            ("skill", skill),
            ("admission", admission),
            ("admission_public", old),
            ("admission_retrieved", with_card),
            ("confirm_public", before),
            ("confirm_retrieved", retrieved),
        ]:
            save(folder / (name + ".json"), result)
        if admission["accepted"]:
            memory.append(skill)
        teacher_snapshot_sha = core.sha(incumbent / "adapter_model.safetensors")
        v3.load_fixed_teacher(model, incumbent)
        opt = torch.optim.AdamW(
            [v for v in model.parameters() if v.requires_grad], lr=1e-5, weight_decay=0
        )
        updates = []
        for i, row in enumerate(groups["update"]):
            active = [s for s in memory if s["cue"] in cues(row)]
            if not admission["accepted"] and skill["cue"] in cues(row):
                active.append(
                    skill
                )  # Explicitly shadow-only and ineligible for inheritance.
            if not active:
                continue
            torch.manual_seed(20261127 + 100 * rnd + i)
            public = core.context(tok, row, "public")
            z = (
                "\n".join(s["instruction"] for s in active)
                + "\n量规："
                + row["rubric"]
                + "\n标准解："
                + row["solution"]
            )
            teacher = v3.teacher_ids(tok, row, z)
            seq, dec, off = v3.dialogue(model, tok, public, row, 72)
            with v3.fixed_teacher_forward(model):
                t = core.aligned(model, teacher, seq)
            s = core.aligned(model, public, seq)
            act, word, anchor = v3.losses(s, t, seq, dec, off)
            loss = 0.5 * act + 0.5 * word + 0.05 * anchor
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                raise ValueError("Nonfinite skill update")
            opt.step()
            opt.zero_grad(set_to_none=True)
            updates.append(
                {
                    "id": row["id"],
                    "loss": loss.item(),
                    "tokens": seq,
                    "grad_norm": float(gn),
                }
            )
            del s, t, act, word, anchor, loss
        model.save_pretrained(folder / "candidate", selected_adapters=["default"])
        after = evaluate(model, tok, groups["confirm"])
        inherit = gate(before, after, True)
        b = scores(before)["joint"]
        denom = scores(retrieved)["joint"] - b
        inherited = admission["accepted"] and inherit["accepted"] and bool(updates)
        save(folder / "confirm_candidate_off.json", after)
        save(folder / "updates.json", updates)
        save(
            folder / "result.json",
            {
                "source": str(incumbent),
                "teacher_mode": "fixed",
                "teacher_snapshot_adapter_sha256": teacher_snapshot_sha,
                "upstream_opsd_reference": v3.UPSTREAM_OPSD_COMMIT,
                "skill": skill,
                "admission": admission,
                "confirmation": inherit,
                "updates": len(updates),
                "inherited": inherited,
                "active_memory": len(memory),
                "shadow": not admission["accepted"],
                "keep": (scores(after)["joint"] - b) / denom
                if denom / len(before) >= 0.05
                else None,
                "model_sha": core.sha(folder / "candidate/adapter_model.safetensors"),
            },
        )
        if inherited:
            incumbent = folder / "candidate"
        print(rnd, len(updates), admission["accepted"], inherited, flush=True)
        del opt, model, base
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    save(a.out / "incumbent.json", {"path": str(incumbent), "admitted": memory})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "execute"])
    p.add_argument("--bundle", type=Path, required=True)
    for name in ("prior", "model", "out"):
        p.add_argument("--" + name, type=Path)
    args = p.parse_args()
    (freeze if args.command == "freeze" else execute)(args)


if __name__ == "__main__":
    main()
