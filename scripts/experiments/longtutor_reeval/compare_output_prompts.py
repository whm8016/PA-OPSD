"""Run a paired Gold development comparison of the native output instructions."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.experiments.longtutor_reeval import run as official
from scripts.experiments.opsd_repair.score_gold140 import score_rows, summarize
from scripts.pa_opsd_contract import (
    STRICT_OUTPUT_PROMPT_VERSION,
    official_module,
    strict_output_messages,
)
from scripts.train_pa_opsd_sft import validate_native_output

PROMPTS_SHA256 = "d7e9902e1b601929a4301000e9c30b18ac7ddc4540d4e4fcbb4fe4e414bfbafe"
VARIANTS = ("legacy", "strict")


def inputs(bundle, manifest, prompt_file):
    if official.digest(prompt_file) != PROMPTS_SHA256:
        raise ValueError("Expected the frozen native Gold development prompts")
    upstream = official.load_upstream(bundle)
    source = official.index(official.rows(bundle / "inputs.jsonl"), "_key")
    gold = official.index(official.rows(bundle / "gold.jsonl"), "_key")
    prompts = official.index(official.rows(prompt_file), "sample_id")
    if any(list(records) != manifest["ids"] for records in (source, gold, prompts)):
        raise ValueError("Inputs, prompts or Gold are out of order or outside the frozen development set")
    for key in manifest["ids"]:
        messages = prompts[key]["messages"]
        _, user = upstream._build_prompts(source[key], gold[key])
        if messages[1]["content"] != user or prompts[key]["split"] != "validation":
            raise ValueError(f"Unexpected held-out model input: {key}")
        strict_output_messages(messages)
    return source, gold, prompts


def infer_variant(upstream, client, model, ids, source, gold, prompts, variant, out, served_model=None):
    # Each worker uses the original parser, retry loop and normalization. Only
    # the selected prompt and the unconstrained local generation call are injected.
    active = ContextVar("active_generation")
    original_path = sys.path[:]
    try:
        if official_module() is not upstream:
            raise ValueError("Native validation must use the pinned upstream module")
    finally:
        sys.path[:] = original_path

    def build_prompts(*_args):
        messages, _ = active.get()
        return messages[0]["content"], messages[1]["content"]

    def complete(messages, model, temperature, max_completion_tokens, debug):
        _, attempts = active.get()
        response = client.chat.completions.create(
            messages=messages, model=model, temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        try:
            validate_native_output(text)
            error = None
        except ValueError as exc:
            error = str(exc)
        attempts.append({"output": text, "finish_reason": choice.finish_reason,
                         "output_tokens": response.usage.completion_tokens, "strict_error": error})
        return text

    def one_sample(key):
        messages = prompts[key]["messages"]
        if variant == "strict":
            messages = strict_output_messages(messages)
        attempts = []
        raw = {"_key": key, "attempts": attempts, "variant": variant,
               "prompt_sha256": sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()}
        reset = active.set((messages, attempts))
        try:
            prediction = upstream.process_one_sample(config, source[key], gold[key])
        except (RuntimeError, UnboundLocalError) as exc:
            # The pinned upstream raises UnboundLocalError if every JSON parse fails.
            if len(attempts) != 5 or not (str(exc).startswith("invalid_output_after_retries:") or
                                         isinstance(exc, UnboundLocalError) and "'parsed'" in str(exc)):
                return None, raw, exc
            prediction = {"_key": key, "memory": [{"answer": ""} for _ in range(3)],
                          "diagnosis": "", "reason": f"ERROR: {type(exc).__name__}: {exc}",
                          "strategy": "", "content": ""}
        except Exception as exc:  # noqa: BLE001 - preserve this trace before failing the run
            return None, raw, exc
        finally:
            active.reset(reset)
        if prediction["_key"] != key or not attempts:
            return None, raw, ValueError(f"Missing trace or wrong prediction ID: {key}")
        return prediction, raw, None

    config = upstream.RunConfig(input_path=Path("inputs.jsonl"), tests_path=Path("gold.jsonl"),
                                output_path=out / f"{model}.jsonl", model=served_model or model,
                                temperature=official.EVAL["temperature"],
                                max_completion_tokens=official.EVAL["max_tokens"],
                                limit=len(ids), resume=False, debug=False, workers=4)
    original = upstream._build_prompts, upstream.chat_completion_with_retry
    upstream._build_prompts, upstream.chat_completion_with_retry = build_prompts, complete
    try:
        with (ThreadPoolExecutor(max_workers=4) as executor,
              config.output_path.open("x", encoding="utf-8") as pred_file,
              (out / "raw_generation.jsonl").open("x", encoding="utf-8") as raw_file):
            for start in range(0, len(ids), 4):
                failures = []
                for prediction, raw, error in executor.map(one_sample, ids[start:start + 4]):
                    if error is None:
                        pred_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    else:
                        failures.append(error)
                        raw["infrastructure_error"] = type(error).__name__
                    raw_file.write(json.dumps(raw, ensure_ascii=False) + "\n")
                    pred_file.flush()
                    raw_file.flush()
                if failures:
                    raise RuntimeError("Comparison paused after an infrastructure error") from failures[0]
    finally:
        upstream._build_prompts, upstream.chat_completion_with_retry = original


def evaluate(bundle, out, ids, gold, predictions, traces):
    grader = official.scorer_module(bundle)
    upstream = official.load_upstream(bundle)
    per_sample = list(score_rows(ids, gold, predictions, grader, upstream.strategy_MAP))
    for row in per_sample:
        key = row["_key"]
        for variant, result in row["arms"].items():
            first = traces[variant][key]["attempts"][0]
            result["strict_first_pass"] = first["strict_error"] is None and first["finish_reason"] == "stop"
            result["first_pass_truncated"] = first["finish_reason"] == "length"
            pred = grader._extract_memory_items(predictions[variant][key])
            target = grader._extract_memory_items(gold[key])
            result["memory_exact"] = [i < len(pred) and grader._exact_match_memory(item, pred[i])
                                      for i, item in enumerate(target)]
    arms, paired = summarize(per_sample, predictions)
    for variant in predictions:
        rows = [row["arms"][variant] for row in per_sample]
        arms[variant].update(strict_first_pass=sum(row["strict_first_pass"] for row in rows),
                             first_pass_truncated=sum(row["first_pass_truncated"] for row in rows),
                             memory_exact=[sum(row["memory_exact"][i] for row in rows) for i in range(3)])
    official.write_rows(out / "per_sample.jsonl", per_sample)
    official.write_json(out / "summary.json", {"rows": len(ids), "arms": arms, "paired_diagnosis": paired})


def score_saved(bundle, manifest, output, arm, gold, variants=VARIANTS, match_sft=None):
    ids = manifest["ids"]
    predictions, traces = {}, {}
    for variant in variants:
        out = output / variant
        settings = official.read_json(out / "inference_settings.json")
        expected = STRICT_OUTPUT_PROMPT_VERSION if variant == "strict" else "legacy_native"
        if (settings["prompt_version"] != expected or settings["model"] != arm or
                settings["manifest_sha256"] != official.digest(bundle / "manifest.json") or settings["smoke"] or
                settings["input_prompts_sha256"] != PROMPTS_SHA256 or
                settings["source_sha256"] != official.digest(Path(__file__)) or
                settings["inference"] != official.EVAL or settings["decoder_json_constraint"] is not False):
            raise ValueError("Saved generations have incompatible settings")
        completed = official.read_json(out / "generation_complete.json")
        if completed != generation_checksums(out, arm, len(ids)):
            raise ValueError("Saved predictions, settings or raw traces changed after generation")
        predictions[variant] = official.prediction_rows(out / f"{arm}.jsonl", ids)
        if any(official.inference_failure(row) for row in predictions[variant].values()):
            raise ValueError("Unresolved inference infrastructure failure")
        trace = official.index(official.rows(out / "raw_generation.jsonl"), "_key")
        if list(trace) != ids or any(row["variant"] != variant or not row["attempts"] or
                                     "infrastructure_error" in row for row in trace.values()):
            raise ValueError("Saved generation traces are incomplete")
        traces[variant] = trace
    evaluate(bundle, output, ids, gold, predictions, traces)
    if match_sft is not None:
        reference = official.read_json(match_sft / "inference_settings.json")
        settings = official.read_json(output / "strict/inference_settings.json")
        shared = ("manifest_sha256", "prompt_version", "input_prompts_sha256", "inference", "decoder_json_constraint")
        if reference["model"] != "sft" or reference["smoke"] or any(reference[key] != settings[key] for key in shared):
            raise ValueError("Gold SFT reference differs in evaluation settings")
        if (official.read_json(match_sft / "generation_complete.json") != generation_checksums(match_sft, "sft", len(ids))):
            raise ValueError("Gold SFT reference changed after generation")
        sft_trace = official.index(official.rows(match_sft / "raw_generation.jsonl"), "_key")
        if list(sft_trace) != ids or any(sft_trace[key]["prompt_sha256"] != traces["strict"][key]["prompt_sha256"]
                                         for key in ids):
            raise ValueError("Base and Gold SFT used different prompts")
        official.write_json(output / "matched_sft_protocol.json", {"matched_prompts": len(ids),
                            "sft_settings_sha256": official.digest(match_sft / "inference_settings.json"),
                            **{key: settings[key] for key in shared}})


def generation_checksums(directory, arm, count):
    filenames = ("inference_settings.json", f"{arm}.jsonl", "raw_generation.jsonl")
    return {"rows": count, "sha256": {name: official.digest(directory / name) for name in filenames}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", choices=official.ARMS, default="sft")
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--strict-only", action="store_true")
    parser.add_argument("--match-sft", type=Path, help="Verified strict Gold SFT output directory")
    parser.add_argument("--port", type=int, default=8022)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = official.verify_bundle(bundle)
    source, gold, prompts = inputs(bundle, manifest, args.prompts)
    if args.score:
        official.scorer_module(bundle)
    if args.check:
        print(json.dumps({"validated_rows": len(prompts), "strict_prompt_version": STRICT_OUTPUT_PROMPT_VERSION}))
        return
    if args.output is None:
        parser.error("--output is required for generation and scoring")
    variants = ("strict",) if args.strict_only else VARIANTS
    if args.match_sft and (not args.score or args.arm != "base" or not args.strict_only):
        parser.error("--match-sft requires --score --arm base --strict-only")
    if args.score:
        score_saved(bundle, manifest, args.output, args.arm, gold, variants, args.match_sft)
        return
    from openai import OpenAI

    api = f"http://127.0.0.1:{args.port}/v1"
    model = official.BASE_NAME if args.arm == "base" else args.arm
    with urlopen(f"{api}/models", timeout=10) as response:
        registered = official.index(json.load(response)["data"], "id").get(model, {})
    model_root = args.base_root if args.arm == "base" else bundle / "adapters" / args.arm
    parent = None if args.arm == "base" else official.BASE_NAME
    if (not registered.get("root") or model_root is None or registered.get("parent") != parent or
            Path(registered["root"]).resolve() != model_root.resolve()):
        raise ValueError("Wrong base model or LoRA registered on the local server")
    ids = manifest["ids"][:2] if args.smoke else manifest["ids"]
    args.output.mkdir(parents=True, exist_ok=False)
    upstream = official.load_upstream(bundle)
    client = OpenAI(base_url=api, api_key="EMPTY", timeout=360, max_retries=2)
    for variant in variants:
        out = args.output / variant
        out.mkdir()
        official.write_json(out / "inference_settings.json", {
            "manifest_sha256": official.digest(bundle / "manifest.json"), "smoke": args.smoke,
            "prompt_version": STRICT_OUTPUT_PROMPT_VERSION if variant == "strict" else "legacy_native",
            "model": args.arm, "source_sha256": official.digest(Path(__file__)),
            "input_prompts_sha256": PROMPTS_SHA256, "decoder_json_constraint": False,
            "inference": official.EVAL, "api": api, "pythonhashseed": os.getenv("PYTHONHASHSEED"),
        })
        infer_variant(upstream, client, args.arm, ids, source, gold, prompts, variant, out, served_model=model)
        count = len(official.prediction_rows(out / f"{args.arm}.jsonl", ids))
        official.write_json(out / "generation_complete.json", generation_checksums(out, args.arm, count))
        print(json.dumps({"variant": variant, "completed": count}), flush=True)


if __name__ == "__main__":
    main()
