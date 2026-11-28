"""Run unchanged LongTutor generation on a locally served, identified adapter."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ARMS = (
    "v3_pa",
    "gold_sft",
    "v3_teacher_sft",
    "v3_content",
    "v3_action_kl",
    "offpolicy",
)
PIN = "ba40af3b9e976f5960c42a15a645c0e9c3a7d718"
HASHES = {
    "upstream/gpt_memory_diagnose.py": "d3791b488b83d8efad5ba98930978c0b427082f9374565d30a7c3c1277846d30",
    "upstream/eval_ai_tutor.py": "53fbda4a99660500216e8ee80be76a815da345182c25e7dae394d85b089d4212",
    "upstream/openai_helper.py": "8e878d8d2e0a7c26f7a9863cbbc3ae7b5416ded55969d2e5a3b1f65f657878ce",
    "queries_only.jsonl": "c6917953b54097650e160b64885ed8c1b6ecce6205001fc85861416bb50e8090",
    "full.temporal.jsonl": "2f8da3e90c9518a1be622b433c43a8137c1677f57afdc18ed0f97304da167487",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", action="store_true")
    for n in ("bundle", "official", "sft", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    args = p.parse_args()
    pinned = args.official
    for n, h in HASHES.items():
        if sha(pinned / n) != h:
            raise ValueError("Wrong source: " + n)
    out = args.out
    out.mkdir(exist_ok=True)
    models = json.load(
        urllib.request.urlopen(os.environ["OPENAI_BASE_URL"] + "/models")
    )
    (out / "models.json").write_text(json.dumps(models))
    listed = {r["id"]: r for r in models["data"]}
    if __package__:
        from . import provenance
    else:
        import provenance
    parent = listed["Qwen2.5-7B-Instruct"]
    base = provenance.verified_base(args.bundle, Path(parent["root"]))

    def keys(path):
        return [json.loads(l)["_key"] for l in path.read_text().splitlines()]

    full = pinned / "full.temporal.jsonl"
    for arm in ("v3_pa", "v3_teacher_sft", "v3_content") if args.scale else ARMS:
        adapter = args.sft if arm == "gold_sft" else args.bundle / arm / "adapter"
        h = sha(adapter / "adapter_model.safetensors")
        expected = (
            "40be07306e9070de6b47d9dfdd372ae7a6183b81aad1d728ae27141f4f37313b"
            if arm == "gold_sft"
            else json.loads((adapter.parent / "receipt.json").read_text())["final"]
        )
        if h != expected or Path(listed[arm]["root"]).resolve() != adapter.resolve():
            raise ValueError("Wrong live adapter mapping: " + arm)
        startup = {
            "arm": arm,
            "weights": h,
            "adapter_config": sha(adapter / "adapter_config.json"),
            "base": base,
            "upstream": PIN,
            "inputs": dict(HASHES, smoke=sha(pinned / "smoke.temporal.jsonl")),
            "temperature": 0,
            "max_tokens": 3000,
            "workers": 4,
            "base_path": parent["root"],
            "adapter_path": str(adapter.resolve()),
            "wrapper": sha(Path(__file__)),
        }
        provenance.bind(
            out / (arm + ".start.json"),
            startup,
            [out / (arm + "." + n + ".jsonl") for n in ("smoke", "full")],
        )
        complete = out / (arm + ".receipt.json")
        if complete.exists() and json.loads(complete.read_text())["pred"] != sha(
            out / (arm + ".full.jsonl")
        ):
            raise ValueError("Completed official predictions changed")
        for name in ("smoke", "full"):
            src = pinned / (name + ".temporal.jsonl")
            pred = out / (arm + "." + name + ".jsonl")
            cmd = [
                sys.executable,
                "-u",
                str(pinned / "upstream/eval_ai_tutor.py"),
                "--input",
                str(src),
                "--tests",
                str(pinned / "queries_only.jsonl"),
                "--model",
                arm,
                "--output",
                str(pred),
                "--temperature",
                "0",
                "--max_tokens",
                "3000",
                "--workers",
                "4",
            ]
            with (out / (arm + "." + name + ".log")).open("a") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = [json.loads(l) for l in pred.read_text().splitlines()]
            if len(rows) != len(keys(src)) or set(keys(pred)) != set(keys(src)):
                raise ValueError("Incomplete official predictions")
            for r in rows:
                reason = r["reason"]
                if (
                    not r["diagnosis"]
                    and reason.startswith("ERROR:")
                    and not (
                        "invalid_output_after_retries" in reason
                        or "cannot access local variable 'parsed'" in reason
                    )
                ):
                    raise ValueError(
                        "Transport or input failure in " + arm + ": " + reason[:300]
                    )
        receipt = {
            "arm": arm,
            "weights": h,
            "upstream": PIN,
            "pred": sha(out / (arm + ".full.jsonl")),
            "input": sha(full),
            "n": len(keys(full)),
            "temperature": 0,
            "max_tokens": 3000,
            "workers": 4,
        }
        (out / (arm + ".receipt.json")).write_text(json.dumps(receipt))
        print(arm, receipt, flush=True)


if __name__ == "__main__":
    main()
