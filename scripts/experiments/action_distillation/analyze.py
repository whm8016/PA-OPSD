"""Aggregate fixed public predictions, official scores, skill gates and run receipts."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

LABELS = ("Recall Failure", "Conceptual Gap", "Procedural Error", "Transfer Deficit")
DIMENSIONS = (
    "history_score",
    "strategy_score",
    "coherence_score",
    "comprehension_score",
)


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def macro(y, p):
    return (
        sum(
            2 * sum(a == b == c for a, b in zip(y, p)) / (y.count(c) + p.count(c))
            if y.count(c) + p.count(c)
            else 0
            for c in LABELS
        )
        / 4
    )


def pair(a, b, clusters):
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    labels = np.unique(clusters)
    sums = np.array([sum(diff[np.asarray(clusters) == k]) for k in labels])
    counts = np.array([clusters.count(k) for k in labels])
    draws = np.random.default_rng(20261127).integers(
        len(labels), size=(10000, len(labels))
    )
    ci = np.quantile(sums[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975])
    return {
        "diff": float(diff.mean()),
        "ci95": ci.tolist(),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "groups": len(labels),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    a = p.parse_args().folder
    ref = rows(a / "dev.jsonl")
    ids = [r["id"] for r in ref]
    truth = [r["gold"] for r in ref]
    clusters = [r["qid"] for r in ref]
    result = {
        "structured": {},
        "official": {},
        "cost": {},
        "external": {},
        "cycles": {},
        "paired": {},
    }
    decision = {}
    team_folder = a / "team"
    if (team_folder / "verifier.jsonl").exists():
        result["team"] = {}
        mapped = json.loads((a / "eedi_candidates.json").read_text(encoding="utf-8"))
        for label, name in [
            ("full", "verifier.jsonl"),
            ("no_eedi", "verifier_no_eedi.jsonl"),
        ]:
            f = team_folder / name
            if not f.exists():
                continue
            ix = {r["id"]: r for r in rows(f) if r["split"] == "dev"}
            if not ix:
                continue
            assert set(ix) == set(ids)
            pred = []
            acts = []
            texts = []
            for k in ids:
                raw = ix[k]["verifier"]
                try:
                    obj = json.JSONDecoder().raw_decode(raw[raw.index("{") :])[0]
                    choice = obj["diagnosis"]
                    if (
                        choice not in list("RCPT")
                        or obj["action"] not in list("ABCDEFGH")
                        or not isinstance(obj["first_move"], str)
                    ):
                        raise ValueError("bad response")
                    pred.append(LABELS["RCPT".index(choice)])
                    acts.append(obj["action"])
                    texts.append(obj["first_move"])
                except (ValueError, TypeError, KeyError):
                    pred.append("")
                    acts.append("")
                    texts.append("")
            right = [int(i == j) for i, j in zip(truth, pred)]
            compat = [int(act in r["compatible"]) for act, r in zip(acts, ref)]
            both = [i * j for i, j in zip(right, compat)]
            decision["team_" + label] = {
                "correct": right,
                "compatible": compat,
                "joint": both,
            }
            result["team"][label] = {
                "n": len(ids),
                "valid": sum(bool(x) for x in pred),
                "diagnosis": sum(right),
                "compatible": sum(compat),
                "joint": sum(both),
                "macro_f1": macro(truth, pred),
                "mapped": sum(k in mapped for k in ids),
                "input_per_case": sum(
                    sum(
                        ix[k][role + "_usage"][0]
                        for role in ("diagnoser", "pedagogue", "verifier")
                    )
                    for k in ids
                )
                / len(ids),
                "generated_per_case": sum(
                    sum(
                        ix[k][role + "_usage"][1]
                        for role in ("diagnoser", "pedagogue", "verifier")
                    )
                    for k in ids
                )
                / len(ids),
                "role_calls": 3,
            }
    for f in sorted((a / "evaluation").glob("*.jsonl")):
        if ".external." in f.name:
            continue
        data = rows(f)
        ix = {r["id"]: r for r in data}
        assert set(ix) == set(ids) and len(data) == len(ids)
        pred = [LABELS["RCPT".index(ix[k]["diagnosis"])] for k in ids]
        right = [int(i == j) for i, j in zip(truth, pred)]
        compat = [int(ix[r["id"]]["action"] in r["compatible"]) for r in ref]
        both = [i * j for i, j in zip(right, compat)]
        decision[f.stem] = {"correct": right, "compatible": compat, "joint": both}
        result["structured"][f.stem] = {
            "n": len(ids),
            "diagnosis": sum(right),
            "macro_f1": macro(truth, pred),
            "compatible": sum(compat),
            "joint": sum(both),
            "actions": dict(Counter(r["action"] for r in data)),
            "diag_by_gold": {
                c: [sum(x for x, y in zip(right, truth) if y == c), truth.count(c)]
                for c in LABELS
            },
            "capped": sum(r["finish_reason"] == "length" for r in data),
            "mean_output_tokens": float(np.mean([r["text_tokens"] for r in data])),
        }
    if "v3_pa" in decision:
        for other in decision:
            result["paired"]["structured_pa_vs_" + other] = {
                k: pair(decision[other][k], seq, clusters)
                for k, seq in decision["v3_pa"].items()
            }
    official = {}
    gold_file = (
        Path(__file__).resolve().parents[3]
        / "outputs/pa_opsd/longtutor_reeval_gold140/official_base_vs_memory_first_20261123/bundle/gold.jsonl"
    )
    gold = {r["_key"]: r for r in rows(gold_file)}
    result["official_prediction_audit"] = {}
    for file in sorted((a / "official").glob("*.full.jsonl")):
        preds = {r["_key"]: r for r in rows(file)}
        if set(preds) != set(ids):
            continue
        keys = ids
        ys = [gold[k]["diagnosis"] for k in keys]
        pp = [preds[k]["diagnosis"] for k in keys]
        result["official_prediction_audit"][file.name.split(".")[0]] = {
            "n": len(keys),
            "diagnosis": sum(x == y for x, y in zip(ys, pp)),
            "macro_f1_four_labels": macro(ys, pp),
            "strategy_exact": sum(
                preds[k]["strategy"] == gold[k]["strategy"] for k in keys
            ),
            "format_failure": sum(
                not preds[k]["diagnosis"]
                and str(preds[k]["reason"]).startswith("ERROR:")
                for k in keys
            ),
        }
    for f in sorted((a / "official_dk").glob("*.metrics.jsonl")):
        data = rows(f)
        arm = f.name.split(".")[0]
        if len(data) != len(ids):
            result["official"][arm] = {"incomplete": len(data)}
            continue
        ix = {r["_key"]: r for r in data}
        assert set(ix) == set(ids)
        # The native totals are archived in *.judge.log. Paired quantities below use raw official per-item fields.
        rr = [ix[k] for k in ids]
        nums = {
            f"Q{j + 1}": [int(r["memory_correct"][j]) for r in rr] for j in range(3)
        }
        nums["diagnosis"] = [int(r["diagnosis_correct"]) for r in rr]
        valid = [
            r
            for r in rr
            if r.get("content_scores")
            and all(d in r["content_scores"] for d in DIMENSIONS)
        ]
        if len(valid) == len(ids):
            nums["teaching"] = [
                sum(r["content_scores"][d] for d in DIMENSIONS) / 4 for r in rr
            ]
        official[arm] = nums
        result["official"][arm] = {
            "n": len(ids),
            "counts": {k: sum(v) for k, v in nums.items() if k != "teaching"},
            "teaching_mean": float(
                np.mean(
                    [sum(r["content_scores"][d] for d in DIMENSIONS) / 4 for r in valid]
                )
            )
            if valid
            else None,
            "teaching_n": len(valid),
            "macro_f1_complete_denominator": macro(
                [r["diagnosis_gold"] for r in rr], [r["diagnosis_pred"] for r in rr]
            ),
        }
    for arm in official:
        log = a / "official_dk" / (arm + ".judge.log")
        if log.is_file():
            raw = log.read_text(encoding="utf-8")
            idx = raw.rfind('{\n  "pred":')
            if idx >= 0:
                summary = json.JSONDecoder().raw_decode(raw[idx:])[0]
                if summary["samples"] != len(ids):
                    raise ValueError("Official aggregate denominator changed")
                result["official"][arm]["upstream"] = summary
    if "v3_pa" in official:
        for other, nums in official.items():
            result["paired"]["official_pa_vs_" + other] = {
                k: pair(nums[k], v, clusters)
                for k, v in official["v3_pa"].items()
                if k in nums
            }
    for f in (a / "official_dk").glob("*.usage.jsonl"):
        records = rows(f)
        settled = [r for r in records if r["type"] == "settle"]
        paid = {r["id"] for r in settled}
        pending = (
            sum(
                r["microyuan"]
                for r in records
                if r["type"] == "reserve" and r["id"] not in paid
            )
            / 1e6
        )
        result["cost"][f.name.split(".")[0]] = {
            "peak_upper_with_cache_and_pending": pending
            + sum(
                2 * (r["prompt"] - r["cache_hit"])
                + (r["cache_hit"] + 24) // 25
                + 8 * r["completion"]
                for r in settled
            )
            / 1e6,
            "requests": len(settled),
            "estimate": sum(r["estimate_microyuan"] for r in settled) / 1e6,
            "upper": sum(r["upper_microyuan"] for r in settled) / 1e6,
            "pending_reserved_yuan": pending,
            "upper_including_pending": pending
            + sum(r["upper_microyuan"] for r in settled) / 1e6,
            "pending": len(records) - 2 * len(settled),
        }
    for f in sorted((a / "cycles").glob("*/result.json")):
        result["cycles"][f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
    for f in sorted((a / "external_scores").glob("*/*/*.yaml")):
        import yaml

        if not (f.parent / "pred.sha256").is_file():
            continue

        result["external"].setdefault(f.parents[1].name, {}).update(
            yaml.safe_load(f.read_text(encoding="utf-8"))
        )
    (a / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
