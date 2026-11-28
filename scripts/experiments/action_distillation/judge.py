"""Run the pinned official judge with one fixed share of the new 5-yuan allowance per arm."""

import argparse
import json
import sys

import httpx
from dotenv import dotenv_values
from openai import APIError, OpenAI

from scripts.experiments.action_distillation.official import ARMS, HASHES, PIN
from scripts.experiments.longtutor_reeval.run import (
    lock_output,
    recover_prediction_tail,
)
from scripts.experiments.training_free_memory import judge_retrieval as billing


class CacheAwareMeter(billing.Meter):
    """Cap settled calls at peak tariffs, preserving every unconfirmed reservation."""

    def apply(self, event):
        if event["type"] == "settle" and "cache_hit" in event:
            prompt, hit, output = (
                event["prompt"],
                event["cache_hit"],
                event["completion"],
            )
            if not 0 <= hit <= prompt or output < 0:
                raise ValueError("Invalid usage receipt")
            peak = 2 * (prompt - hit) + (hit + 24) // 25 + 8 * output
            event = dict(event, upper_microyuan=peak)
        super().apply(event)


def remaining_for_scale(root, arm, ref):
    order = ("v3_pa", "v3_teacher_sft", "v3_content")
    if arm not in order:
        raise ValueError("Unregistered scale arm")
    prior = [(root / "official_dk", n) for n in ARMS]
    prior += [(root / "scale/official_dk", n) for n in order[: order.index(arm)]]
    spent = 0
    for directory, name in prior:
        ids = [r["_key"] for r in billing.rows(directory / (name + ".metrics.jsonl"))]
        if len(ids) != len(ref) or set(ids) != ref:
            raise ValueError("Earlier budget allocation still active")
        meter = CacheAwareMeter(directory / (name + ".usage.jsonl"))
        if not meter.spent:
            raise ValueError("No billing evidence for completed group")
        spent += meter.spent + sum(meter.pending.values())
    return max(0, 5_000_000 - spent)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=ARMS)
    p.add_argument("--scale", action="store_true")
    p.add_argument(
        "--complete-teacher-sft",
        action="store_true",
        help="Resume the frozen 56 grades in a separate directory under the additional 2-yuan cap.",
    )
    args = p.parse_args()
    arm = args.arm
    if args.complete_teacher_sft and not (args.scale and arm == "v3_teacher_sft"):
        p.error("The additional allowance applies only to the 373-case teacher SFT")
    root = billing.ROOT / "outputs/pa_opsd/v3_full_20261127_run1"
    output = root / "scale" if args.scale else root
    folder = output / "official"
    pred = folder / (arm + ".full.jsonl")
    proof = json.loads((folder / (arm + ".receipt.json")).read_text(encoding="utf-8"))
    gold = billing.BUNDLE / "gold.jsonl"
    ref = {r["_key"] for r in billing.rows(gold)}
    found = [r["_key"] for r in billing.rows(pred)]
    assert len(ref) == 140 and len(found) == 140 and set(found) == ref
    assert (
        proof["arm"] == arm
        and proof["upstream"] == PIN
        and proof["input"] == HASHES["full.temporal.jsonl"]
    )
    assert proof["pred"] == billing.digest(pred)
    start = json.loads((folder / (arm + ".start.json")).read_text(encoding="utf-8"))
    weights = (
        "40be07306e9070de6b47d9dfdd372ae7a6183b81aad1d728ae27141f4f37313b"
        if arm == "gold_sft"
        else json.loads((output / arm / "receipt.json").read_text(encoding="utf-8"))[
            "final"
        ]
    )
    assert start["weights"] == proof["weights"] == weights and start["arm"] == arm
    assert start["base"]["base_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert (
        billing.digest(gold)
        == "9bcc08c3085372c139f58184d79e2d440e9f62220df8ff5393edbf1e39151e50"
    )
    assert (
        billing.digest(billing.BUNDLE / "upstream/compute_ai_tutor_eval_metrics.py")
        == billing.SCORER_SHA
    )
    assert (
        billing.digest(billing.BUNDLE / "upstream/openai_helper.py")
        == HASHES["upstream/openai_helper.py"]
    )
    sys.path.insert(0, str(billing.BUNDLE / "upstream"))
    import compute_ai_tutor_eval_metrics as grader
    import openai_helper

    d = output / "official_dk"
    if args.complete_teacher_sft:
        original_grades = d / (arm + ".metrics.jsonl")
        seed = original_grades.read_bytes()
        if (
            billing.digest(original_grades)
            != "9ca1814e545ddc047064ec48b1b0cec7a3e8280d0845bebbcf29e299d3e11b0f"
            or len(billing.rows(original_grades)) != 56
        ):
            raise ValueError("Original 56 grades changed")
        d = output / "teacher_sft_completion"
        billing.LIMIT_MICROYUAN = 2_000_000
    else:
        billing.LIMIT_MICROYUAN = (
            remaining_for_scale(root, arm, ref)
            if args.scale
            else 5_000_000 // len(ARMS)
        )
    d.mkdir(exist_ok=True)
    path = d / (arm + ".metrics.jsonl")
    ledger = d / (arm + ".usage.jsonl")
    with lock_output(d / (arm + ".lockdir")):
        binding = {
            "pred": proof["pred"],
            "gold": billing.digest(gold),
            "model": billing.MODEL,
            "scorer": billing.SCORER_SHA,
            "upstream": PIN,
            "weights": proof["weights"],
            "cap": billing.LIMIT_MICROYUAN,
            "temperature": 0,
            "thinking": "disabled",
        }
        marker = d / (arm + ".binding.json")
        if args.complete_teacher_sft:
            old_binding = json.loads((output / "official_dk" / marker.name).read_text())
            if dict(old_binding, cap=binding["cap"]) != binding:
                raise ValueError(
                    "New judge configuration differs from the original 56 grades"
                )
        if marker.exists():
            assert json.loads(marker.read_text()) == binding
            if args.complete_teacher_sft and not ledger.exists():
                raise ValueError("Bound continuation has lost its budget ledger")
        else:
            assert not path.exists()
            if args.complete_teacher_sft:
                if ledger.exists() and ledger.stat().st_size:
                    raise ValueError("Unbound nonempty ledger")
                ledger.touch()
            else:
                assert not ledger.exists()
            marker.write_text(json.dumps(binding))
        if args.complete_teacher_sft:
            if not path.exists():
                if ledger.exists() and ledger.stat().st_size:
                    raise ValueError("Paid ledger exists without resumable results")
                staged = path.with_suffix(".tmp")
                staged.write_bytes(seed)
                staged.replace(path)
            recover_prediction_tail(path)
            if not path.read_bytes().startswith(seed):
                raise ValueError("Resume file lost the original scores")
            if not ledger.stat().st_size and path.read_bytes() != seed:
                raise ValueError("Missing budget ledger for additional scores")
        if path.exists():
            completed = [r["_key"] for r in billing.rows(path)]
            assert (
                len(completed) == len(set(completed))
                and set(completed) <= ref
                and ledger.exists()
            )
        meter = CacheAwareMeter(ledger)
        with OpenAI(
            base_url="https://api.deepseek.com",
            api_key=dotenv_values(billing.ROOT / ".env")["DEEPSEEK_API_KEY"],
            max_retries=0,
            http_client=httpx.Client(
                timeout=90,
                event_hooks={"request": [meter.request], "response": [meter.response]},
            ),
        ) as client:
            original = client.chat.completions.create

            def call(**kwargs):
                try:
                    return original(
                        **kwargs, extra_body={"thinking": {"type": "disabled"}}
                    )
                except (APIError, httpx.HTTPError) as exc:
                    chain = []
                    cause = exc
                    while cause and len(chain) < 5:
                        chain.append(type(cause).__name__)
                        cause = cause.__cause__ or cause.__context__
                    billing.stop("Judge transport failed: " + " / ".join(chain))

            client.chat.completions.create = call
            openai_helper._get_openai_client = lambda *args: client
            return grader.main(
                [
                    "--pred",
                    str(pred),
                    "--gold",
                    str(gold),
                    "--model",
                    billing.MODEL,
                    "--temperature",
                    "0",
                    "--limit",
                    "140",
                    "--workers",
                    "1",
                    "--out",
                    str(path),
                ]
            )


if __name__ == "__main__":
    raise SystemExit(main())
