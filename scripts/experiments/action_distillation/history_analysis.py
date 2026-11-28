"""Paired sensitivity and natural matched-question discrimination, with frozen denominators."""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.experiments.action_distillation.analyze import pair, rows


def distances(a, b):
    return np.abs(a - b).sum(axis=1) / 2


def eligible(a, b):
    return (
        a["qid"] == b["qid"]
        and a["stem"] == b["stem"]
        and a["uid"] != b["uid"]
        and set(a["compatible"]).isdisjoint(b["compatible"])
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    root = ap.parse_args().folder
    out = root / "history_probe"
    ref = rows(root / "dev.jsonl")
    anchors = {r["id"]: r for r in rows(root / "evaluation/gold_sft.jsonl")}
    keys = [r["id"] for r in ref]
    clusters = [r["qid"] for r in ref]
    result = {"models": {}, "paired": {}}
    matched = [
        (i, j) for i, j in combinations(range(len(ref)), 2) if eligible(ref[i], ref[j])
    ]
    result["natural_pairs"] = len(matched)
    result["natural_questions"] = len({clusters[i] for i, j in matched})
    train = rows(root / "train.jsonl")
    configs = {"available_560": train}
    for field, loc in [("primary_72", root), ("scale_373", root / "scale")]:
        ids = set(
            json.loads((loc / "protocol.json").read_text(encoding="utf-8"))["train_ids"]
        )
        configs[field] = [r for r in train if r["id"] in ids]
    result["train_pair_feasibility"] = {}
    for n, data in configs.items():
        candidates = [(i, j) for i, j in combinations(data, 2) if eligible(i, j)]
        result["train_pair_feasibility"][n] = {
            "records": len(data),
            "pairs": len(candidates),
            "questions": len({i["qid"] for i, j in candidates}),
            "students": len({r["uid"] for couple in candidates for r in couple}),
        }
    tv = {}
    normal = {}
    per_item = []
    for name, loc in [
        ("gold_sft", "evaluation/gold_sft.jsonl"),
        ("pa72", "evaluation/v3_pa.jsonl"),
        ("wrong72", "evaluation/v3_wrong.jsonl"),
        ("pa373", "scale/evaluation/v3_pa.jsonl"),
    ]:
        data = rows(out / (name + ".jsonl"))
        ix = {(r["id"], r["view"]): r for r in data}
        assert len(data) == len(ix) == 2 * len(keys)
        for r in data:
            a = anchors[r["id"]]
            if (
                r["fixed_evidence"] != a["evidence"]
                or r["fixed_diagnosis"] != a["diagnosis"]
            ):
                raise ValueError(
                    "Fixed prefix differs from frozen original student output"
                )
        orig = {r["id"]: r for r in rows(root / loc)}

        def matrix(view, field, lookup=ix):
            a = np.array([lookup[k, view][field] for k in keys])
            assert (
                np.isfinite(a).all()
                and (a >= 0).all()
                and np.allclose(a.sum(1), 1, atol=1e-5)
            )
            return a

        metrics = {}
        tv[name] = {}
        for key in ("fixed_action_probs", "free_action_probs", "diag_probs"):
            real, flip = (matrix(v, key) for v in ("real", "flip"))
            delta = distances(real, flip)
            tv[name][key] = delta
            metrics[key] = {
                "tv_mean": float(delta.mean()),
                "changed_argmax": int((real.argmax(1) != flip.argmax(1)).sum()),
                "n": len(real),
            }
            per_item.extend(
                {"id": k, "model": name, "readout": key, "tv": float(x)}
                for k, x in zip(keys, delta)
            )
        normal[name] = matrix("real", "fixed_action_probs")
        free = matrix("real", "free_action_probs")
        diag = matrix("real", "diag_probs")
        metrics["original_backend_agreement"] = {
            "action": sum(
                "ABCDEFGH"[free[i].argmax()] == orig[k]["action"]
                for i, k in enumerate(keys)
            ),
            "diagnosis": sum(
                "RCPT"[diag[i].argmax()] == orig[k]["diagnosis"]
                for i, k in enumerate(keys)
            ),
            "evidence": sum(
                ix[k, "real"]["evidence"] == orig[k]["evidence"] for k in keys
            ),
        }
        qgroup = defaultdict(list)
        for i, j in matched:
            own_i = ["ABCDEFGH".index(a) for a in ref[i]["compatible"]]
            own_j = ["ABCDEFGH".index(a) for a in ref[j]["compatible"]]
            margin = (
                (free[i] - free[j])[own_i].sum() + (free[j] - free[i])[own_j].sum()
            ) / 2
            qgroup[clusters[i]].append(float(margin))
        qm = [float(np.mean(v)) for v in qgroup.values()]
        metrics["natural_direction"] = {
            "mean_per_question": float(np.mean(qm)),
            "positive_questions": sum(v > 0 for v in qm),
            "question_means": dict(zip(qgroup, qm, strict=True)),
        }
        result["models"][name] = metrics
    for other in ("wrong72", "gold_sft", "pa373"):
        result["paired"]["pa72_minus_" + other] = {
            k: pair(tv[other][k], v, clusters) for k, v in tv["pa72"].items()
        }
    result["real_history_weight_shift_pa72_vs_gold"] = float(
        distances(normal["pa72"], normal["gold_sft"]).mean()
    )
    (out / "sensitivity_by_id.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in per_item), encoding="utf-8"
    )
    (out / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
