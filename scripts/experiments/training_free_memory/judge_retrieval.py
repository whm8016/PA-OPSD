"""Score one frozen retrieval arm with the upstream CLI and a persistent API allowance."""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import dotenv_values
from openai import APIError, OpenAI

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = (
    ROOT
    / "outputs/pa_opsd/longtutor_reeval_gold140/official_base_vs_memory_first_20261123/bundle"
)
FOLLOWUP = (
    ROOT / "outputs/pa_opsd/training_free_memory/followup_gold140_20261125_verified"
)
PREDICTIONS = FOLLOWUP / "autodl_4090_49862_20261126/runs/followup_full"
SCORER_SHA = "44e2106d228f6597a1ab99ff8b4b0fd715026815701d5f1a90341024de808a0a"
MODEL = "deepseek-flash"
LIMIT_MICROYUAN = (
    2_500_000  # Each of the two authorized arms has its own nontransferable allowance.
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def stop(message):
    print(message, file=sys.stderr, flush=True)
    os._exit(
        70
    )  # The upstream scorer normally converts transport exceptions into incorrect answers.


class Meter:
    def __init__(self, ledger):
        self.ledger = ledger
        self.lock = threading.Lock()
        self.pending = {}
        self.spent = 0
        if ledger.exists():
            for event in rows(ledger):
                self.apply(event)

    def apply(self, event):
        key = event["id"]
        if event["type"] == "reserve":
            self.pending[key] = event["microyuan"]
        else:
            self.pending.pop(key)
            self.spent += event["upper_microyuan"]

    def log(self, event):
        with self.ledger.open("a", encoding="utf-8") as out:
            out.write(json.dumps(event) + "\n")
            out.flush()
            os.fsync(out.fileno())
        self.apply(event)

    def request(self, req):
        data = json.loads(req.content)
        if req.url.host != "api.deepseek.com" or data["model"] != MODEL:
            stop("Unapproved judge endpoint or model")
        # UTF-8 wire bytes plus framing are a conservative input-token bound.
        reserve = 2 * (len(req.content) + 512) + 8 * data["max_completion_tokens"]
        with self.lock:
            if self.spent + sum(self.pending.values()) + reserve > LIMIT_MICROYUAN:
                stop("Per-arm API allowance reached; incomplete rows remain pending")
            key = time.time_ns()
            req.extensions["billing_id"] = key
            self.log({"type": "reserve", "id": key, "microyuan": reserve})

    def response(self, resp):
        if resp.status_code != 200:
            stop(
                "Judge returned HTTP "
                + str(resp.status_code)
                + "; see pending reservation"
            )
        body = json.loads(resp.read())
        usage = body.get("usage") or {}
        if not {"prompt_tokens", "completion_tokens"} <= usage.keys():
            stop("Judge response lacks usage")
        prompt, generated = usage["prompt_tokens"], usage["completion_tokens"]
        hit = usage.get("prompt_cache_hit_tokens", 0)
        upper = prompt * 2 + generated * 8
        key = resp.request.extensions["billing_id"]
        with self.lock:
            if upper > self.pending[key]:
                stop("Unexpected usage beyond reservation")
            bj = datetime.now(timezone(timedelta(hours=8)))
            peak = bj.weekday() < 5 and (9 <= bj.hour < 12 or 14 <= bj.hour < 18)
            estimate = (prompt - hit + 0.02 * hit + 4 * generated) * (2 if peak else 1)
            self.log(
                {
                    "type": "settle",
                    "id": key,
                    "upper_microyuan": upper,
                    "estimate_microyuan": estimate,
                    "model": body.get("model"),
                    "time": bj.isoformat(),
                    "prompt": prompt,
                    "cache_hit": hit,
                    "completion": generated,
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("rag", "temporal"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if (
        args.output.resolve()
        != (ROOT / "outputs/pa_opsd/proposal_evidence_20261127/dk").resolve()
    ):
        raise ValueError(
            "Each authorized arm has one fixed ledger and output directory"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    pred = PREDICTIONS / (args.arm + ".jsonl")
    gold = BUNDLE / "gold.jsonl"
    reference = json.loads((FOLLOWUP / "manifest.json").read_text(encoding="utf-8"))
    hashes = {
        name: sha
        for sha, name in (
            line.split()
            for line in (PREDICTIONS / "predictions.sha256").read_text().splitlines()
        )
    }
    ref_ids = set(reference["ids"])
    if (
        digest(pred) != hashes[pred.name]
        or digest(BUNDLE / "upstream/compute_ai_tutor_eval_metrics.py") != SCORER_SHA
    ):
        raise ValueError("Unexpected prediction or scorer version")
    if (
        digest(gold)
        != "9bcc08c3085372c139f58184d79e2d440e9f62220df8ff5393edbf1e39151e50"
        or digest(BUNDLE / "upstream/openai_helper.py")
        != "8e878d8d2e0a7c26f7a9863cbbc3ae7b5416ded55969d2e5a3b1f65f657878ce"
    ):
        raise ValueError("Frozen gold or helper changed")
    for source in (gold, pred):
        keys = [r["_key"] for r in rows(source)]
        if len(keys) != 140 or set(keys) != ref_ids:
            raise ValueError("Unexpected keys or sample count")
    sys.path.insert(0, str(BUNDLE / "upstream"))
    import compute_ai_tutor_eval_metrics as grader
    import openai_helper

    from scripts.experiments.longtutor_reeval.run import lock_output

    arm = args.output / args.arm
    with lock_output(arm.with_suffix(".lockdir")):
        receipt = arm.with_suffix(".binding.json")
        binding = {
            "arm": args.arm,
            "pred": digest(pred),
            "gold": digest(gold),
            "scorer": SCORER_SHA,
            "helper": digest(BUNDLE / "upstream/openai_helper.py"),
            "judge": MODEL,
            "thinking": "disabled",
            "temperature": 0,
            "microyuan_limit": LIMIT_MICROYUAN,
        }
        metrics, ledger = (
            arm.with_suffix(".metrics.jsonl"),
            arm.with_suffix(".usage.jsonl"),
        )
        if receipt.exists():
            if json.loads(receipt.read_text(encoding="utf-8")) != binding or (
                metrics.exists() and not ledger.exists()
            ):
                raise ValueError("Unbound run or missing budget ledger")
        elif metrics.exists() or ledger.exists():
            raise ValueError("Existing run lacks its input binding")
        else:
            receipt.write_text(json.dumps(binding, indent=2), encoding="utf-8")
        if metrics.exists():
            found = [r["_key"] for r in rows(metrics)]
            if len(found) != len(set(found)) or not set(found) <= ref_ids:
                raise ValueError("Invalid resume sample keys")
        meter = Meter(ledger)
        api_key = dotenv_values(ROOT / ".env")["DEEPSEEK_API_KEY"]
        with OpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key,
            max_retries=0,
            http_client=httpx.Client(
                timeout=90,
                event_hooks={"request": [meter.request], "response": [meter.response]},
            ),
        ) as client:
            original_create = client.chat.completions.create

            def create(**kwargs):
                try:
                    return original_create(
                        **kwargs, extra_body={"thinking": {"type": "disabled"}}
                    )
                except (APIError, httpx.HTTPError) as exc:
                    stop("Judge transport failed: " + type(exc).__name__)

            client.chat.completions.create = create
            openai_helper._get_openai_client = lambda *args: client
            return grader.main(
                [
                    "--pred",
                    str(pred),
                    "--gold",
                    str(gold),
                    "--model",
                    MODEL,
                    "--temperature",
                    "0",
                    "--limit",
                    "140",
                    "--workers",
                    "8",
                    "--out",
                    str(arm.with_suffix(".metrics.jsonl")),
                ]
            )


if __name__ == "__main__":
    raise SystemExit(main())
