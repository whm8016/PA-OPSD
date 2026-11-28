"""Blind and score the fixed training-only teacher evidence probe."""

import argparse
import random
import secrets
from pathlib import Path

from scripts.experiments.action_pilot import evidence_probe as probe
from scripts.experiments.action_pilot import run as gold

# Frozen before any model output. These are engineering thresholds for entering a small training pilot.
GATE = {"minimum_teacher_passes": 45, "minimum_teacher_protocol_valid": 64,
        "net_over_public": 7, "net_over_solution_with_index": 4,
        "repeat_both_pass": 12, "repeat_same_observed_action": 13}
RISK = ("unsupported_history", "math_or_task_error", "direct_answer")


def verified_inputs(bundle):
    manifest = gold.read_json(bundle / "manifest.json")
    path = bundle / "inputs.jsonl"
    if gold.digest(path) != manifest["inputs_sha256"] or manifest["scope"] != "gold_train_teacher_first_move":
        raise ValueError("Unexpected probe inputs")
    inputs = gold.indexed(gold.read_rows(path), "sample_id")
    if (list(inputs) != manifest["ids"] or len(inputs) != 64 or len(set(manifest["repeat_ids"])) != 16
            or not set(manifest["repeat_ids"]) <= inputs.keys() or manifest["gate"] != GATE):
        raise ValueError("Unexpected fixed cohort, repeated cases or thresholds")
    return manifest, inputs


def make_blind(bundle, predictions, output, *, verify=verified_inputs, repeat=probe.REPEAT, parse=probe.parse_arm):
    manifest, inputs = verify(bundle)
    receipt = gold.read_json(predictions.with_suffix(".settings.json"))
    if (receipt["phase"] != "full" or receipt["ids"] != manifest["ids"]
            or receipt["manifest_sha256"] != gold.digest(bundle / "manifest.json")
            or receipt["adapter_sha256"] != manifest["adapter_sha256"]
            or any(receipt[key] != manifest[key] for key in ("base_revision", "temperature", "sampling_seed",
                                                            "max_completion_tokens"))):
        raise ValueError("GPU predictions do not use the fixed settings")
    indexed = gold.indexed(gold.read_rows(predictions), "sample_id")
    if indexed.keys() != inputs.keys():
        raise ValueError("GPU predictions do not cover the full cohort")
    output.mkdir(parents=True, exist_ok=False)
    (output / "internal").mkdir()
    rng = random.SystemRandom()
    packet = secrets.token_hex(6)
    blind, key_map = [], []
    for number, key in enumerate(manifest["ids"], 1):
        sample, row = inputs[key], indexed[key]
        expected = list(manifest.get("conditions", probe.CONDITIONS))
        if key in manifest["repeat_ids"]:
            expected.append(repeat)
        if set(expected) != row.keys() - {"sample_id"}:
            raise ValueError(f"Incorrect prediction arms for case {number}")
        rng.shuffle(expected)
        variants = {}
        for arm in expected:
            candidate = row[arm]
            decision = (parse(candidate["raw"], sample, arm)
                        if candidate["finish_reason"] == "stop" else None)
            if decision != candidate["decision"]:
                raise ValueError(f"Saved parsing differs for case {number}")
            text = decision["first_move"] if decision else None
            variants.setdefault(text, []).append({"arm": arm, "declared_action": decision["action"] if decision else None})
        options = {}
        for letter, (text, assignments) in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", variants.items()):
            options[letter] = text
            key_map.append({"case_option": f"{packet}-{number}{letter}", "sample_id": key, "assignments": assignments})
        blind.append({"packet": packet, "case": number, "current": sample["current"],
                      "history": sample["history"], "options": options})
    gold.write_rows(output / "blind.jsonl", blind)
    gold.write_rows(output / "internal/blind_key.jsonl", key_map)
    gold.write_json(output / "receipt.json", {
        "manifest_sha256": receipt["manifest_sha256"], "predictions_sha256": gold.digest(predictions),
        "blind_sha256": gold.digest(output / "blind.jsonl"),
        "key_sha256": gold.digest(output / "internal/blind_key.jsonl"),
    })


def adjudicate(bundle, review, ratings_path, manifest, repeat):
    receipt = gold.read_json(review / "receipt.json")
    key_path = review / "internal/blind_key.jsonl"
    if (receipt["manifest_sha256"] != gold.digest(bundle / "manifest.json")
            or receipt["key_sha256"] != gold.digest(key_path)
            or receipt["blind_sha256"] != gold.digest(review / "blind.jsonl")):
        raise ValueError("Blinded packet or key changed")
    ratings = gold.indexed(gold.read_rows(ratings_path), "case_option")
    mapping = gold.indexed(gold.read_rows(key_path), "case_option")
    if ratings.keys() != mapping.keys():
        raise ValueError("Ratings must cover every blinded reply, including missing replies")
    outcomes = {}
    for option, key in mapping.items():
        rating = ratings[option]
        action = rating["observed_action"]
        adjudicable = all(entry["declared_action"] is not None for entry in key["assignments"])
        if ((action is not None and (not isinstance(action, str) or action not in gold.ACTIONS))
                or bool(action) != adjudicable
                or any(type(rating[field]) is not bool for field in ("usable", *RISK))
                or (not action and any(rating[field] for field in ("usable", *RISK)))
                or not isinstance(rating["reason"], str) or not rating["reason"].strip()):
            raise ValueError(f"Invalid rating {option}")
        sample = outcomes.setdefault(key["sample_id"], {})
        for entry in key["assignments"]:
            arm, declared = entry["arm"], entry["declared_action"]
            if arm in sample:
                raise ValueError(f"Duplicate arm rating: {key['sample_id']}/{arm}")
            sample[arm] = dict(rating, passed=bool(action and rating["usable"] and not any(rating[field] for field in RISK)),
                               declared_action=declared, label_matches=(action == declared if action else False),
                               protocol_valid=declared is not None)
    for key in manifest["ids"]:
        required = set(manifest.get("conditions", probe.CONDITIONS))
        if key in manifest["repeat_ids"]:
            required.add(repeat)
        if outcomes.get(key, {}).keys() != required:
            raise ValueError(f"Missing or unexpected arm ratings: {key}")
    return outcomes


def score(bundle, review, ratings_path, output):
    manifest, inputs = verified_inputs(bundle)
    outcomes = adjudicate(bundle, review, ratings_path, manifest, probe.REPEAT)
    teacher = "solution_evidence"
    counts = {arm: {field: sum(outcomes[key][arm][field] for key in manifest["ids"])
                    for field in ("usable", *RISK, "passed", "label_matches", "protocol_valid")}
              for arm in probe.CONDITIONS}
    paired = {}
    for arm in ("public", "solution"):
        wins = [key for key, item in outcomes.items() if item[teacher]["passed"] and not item[arm]["passed"]]
        losses = [key for key, item in outcomes.items() if item[arm]["passed"] and not item[teacher]["passed"]]
        paired[arm] = {"wins": wins, "losses": losses, "net": len(wins) - len(losses)}
    repeats = [outcomes[key] for key in manifest["repeat_ids"]]
    covered = [key for key in manifest["ids"] if inputs[key]["facts"]]
    coverage_passes = {arm: sum(outcomes[key][arm]["passed"] for key in covered) for arm in probe.CONDITIONS}
    repeat_passes = sum(item[teacher]["passed"] and item[probe.REPEAT]["passed"] for item in repeats)
    repeat_actions = sum(bool(item[teacher]["observed_action"]
                              and item[teacher]["observed_action"] == item[probe.REPEAT]["observed_action"])
                         for item in repeats)
    checks = {
        "usable": counts[teacher]["passed"] >= GATE["minimum_teacher_passes"],
        "over_public": paired["public"]["net"] >= GATE["net_over_public"],
        "over_solution_with_index": (coverage_passes[teacher] - coverage_passes["solution"]
                                     >= GATE["net_over_solution_with_index"]),
        "repeat_pass": repeat_passes >= GATE["repeat_both_pass"],
        "repeat_same_action": repeat_actions >= GATE["repeat_same_observed_action"],
        "teacher_content_adjudicable": counts[teacher]["protocol_valid"] >= GATE["minimum_teacher_protocol_valid"],
        "unsupported_history_and_direct_answers": all(counts[teacher][risk] <= counts[arm][risk]
                                                       for arm in paired
                                                       for risk in ("unsupported_history", "direct_answer")),
    }
    if output.exists() or output.with_suffix(".rows.jsonl").exists():
        raise FileExistsError(output)
    gold.write_json(output, {"gate_passed": all(checks.values()), "checks": checks, "thresholds": GATE,
                             "counts": counts, "paired": paired, "repeat_both_pass": repeat_passes,
                             "repeat_same_action": repeat_actions, "evidence_index_rows": len(covered),
                             "evidence_index_passes": coverage_passes, "ratings_sha256": gold.digest(ratings_path)})
    gold.write_rows(output.with_suffix(".rows.jsonl"),
                    ({"sample_id": key, **outcomes[key]} for key in manifest["ids"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    blind = commands.add_parser("blind")
    blind.add_argument("--predictions", type=Path, required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--review", type=Path, required=True)
    scoring.add_argument("--ratings", type=Path, required=True)
    for command in (blind, scoring):
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    if opts.command == "blind":
        make_blind(opts.bundle, opts.predictions, opts.output)
    else:
        score(opts.bundle, opts.review, opts.ratings, opts.output)


if __name__ == "__main__":
    main()
