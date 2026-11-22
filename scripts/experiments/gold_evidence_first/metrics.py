"""Offline scores and adapter changes for paired comparisons."""

import json
import math
from collections import Counter

from scripts.run_pa_opsd_gold import DIAGNOSES, aggregate, row_scores


def adapter_delta(initial, final):
    import torch
    from safetensors.torch import load_file

    before, after = (load_file(path / "adapter_model.safetensors") for path in (initial, final))
    if not before or before.keys() != after.keys():
        raise ValueError("Different initial and final adapter tensors")
    squared_change = squared_initial = 0.0
    elements = changed = 0
    for name, tensor in before.items():
        final_tensor = after[name]
        if tensor.shape != final_tensor.shape or tensor.dtype != final_tensor.dtype:
            raise ValueError(f"Adapter shape or dtype changed: {name}")
        initial_tensor = tensor.to(torch.float64)
        delta = final_tensor.to(torch.float64) - initial_tensor
        if not torch.isfinite(delta).all():
            raise ValueError(f"Nonfinite adapter update: {name}")
        squared_change += delta.square().sum().item()
        squared_initial += initial_tensor.square().sum().item()
        elements += delta.numel()
        changed += delta.count_nonzero().item()
    if not changed or not squared_initial:
        raise ValueError("No measurable LoRA update from the starting checkpoint")
    return {"changed_elements": changed, "elements": elements,
            "relative_l2": math.sqrt(squared_change / squared_initial),
            "rms": math.sqrt(squared_change / elements)}


def first_field(output):
    try:
        parsed = json.loads(output)
    except (ValueError, TypeError):
        return "unparsed"
    return next(iter(parsed), "empty") if isinstance(parsed, dict) else "unparsed"


def score_predictions(predictions, references):
    ids = [row["sample_id"] for row in references]
    if not ids or [row["sample_id"] for row in predictions] != ids:
        raise ValueError("Prediction and reference IDs must match in order")
    scores = [{**row_scores(pred["output"], ref["target"]), "sample_id": ref["sample_id"],
               "first_field": first_field(pred["output"]), "stop_reason": pred["stop_reason"]}
              for pred, ref in zip(predictions, references)]
    report = aggregate(scores)
    report["metrics"]["diagnosis_macro_f1"] = sum(
        report["diagnosis_per_class"][label]["f1"] for label in DIAGNOSES) / len(DIAGNOSES)
    for label, stats in report["diagnosis_per_class"].items():
        stats["predicted"] = sum(row["diagnosis"] == label for row in scores)
        stats["recall"] = stats["correct"] / stats["support"] if stats["support"] else None
    report["first_field_counts"] = dict(Counter(row["first_field"] for row in scores))
    report["diagnosis_counts"] = dict(Counter(row["diagnosis"] or "invalid" for row in scores))
    report["truncated"] = sum(row["stop_reason"] == "token_limit" for row in scores)
    return report, scores


def paired(left, right):
    if not left or [(x["sample_id"], x["diagnosis_gold"]) for x in left] != [
            (x["sample_id"], x["diagnosis_gold"]) for x in right]:
        raise ValueError("Cannot pair different development references")
    counts = Counter()
    for a, b in zip(left, right):
        counts[(a["diagnosis"] == a["diagnosis_gold"], b["diagnosis"] == b["diagnosis_gold"])] += 1
    return {"both_correct": counts[True, True], "left_only": counts[True, False],
            "right_only": counts[False, True], "neither": counts[False, False]}
