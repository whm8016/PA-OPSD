"""Summarize paired official outcomes and the retrieval-by-checkpoint exact-match diagnostic."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from scripts.experiments.training_free_memory.judge_retrieval import (
    BUNDLE,
    FOLLOWUP,
    PREDICTIONS,
    ROOT,
)

OUT = ROOT / "outputs/pa_opsd/proposal_evidence_20261127"
ARMS = ("rag", "temporal")
DIMENSIONS = (
    "history_score",
    "strategy_score",
    "coherence_score",
    "comprehension_score",
)


def read(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def index(path, ids):
    data = read(path)
    result = {row["_key"]: row for row in data}
    if len(result) != len(data) or set(result) != set(ids):
        raise ValueError(f"Unexpected sample keys in {path}")
    return result


def norm(text):
    return " ".join(str(text or "").split())


def exact(row, gold, pos):
    memory = row.get("memory") or []
    return int(
        pos < len(memory)
        and norm(memory[pos].get("answer")) == norm(gold["memory"][pos]["answer"])
    )


def summarize(values, groups, weights):
    a, b = np.asarray(values, dtype=float).T
    diff = b - a
    sums = np.bincount(groups, weights=diff)
    n = weights @ np.bincount(groups)
    lo, hi = np.quantile((weights @ sums) / n, (0.025, 0.975))
    return {
        "n": len(diff),
        "before": float(a.mean()),
        "after": float(b.mean()),
        "difference": float(diff.mean()),
        "improved": int((diff > 0).sum()),
        "worsened": int((diff < 0).sum()),
        "cluster_ci95": [float(lo), float(hi)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sft", type=Path, help="Directory with the two new GPU predictions"
    )
    args = parser.parse_args()
    manifest = json.loads((FOLLOWUP / "manifest.json").read_text(encoding="utf-8"))
    ids = manifest["ids"]
    gold = index(BUNDLE / "gold.jsonl", ids)
    qids = index(FOLLOWUP / "autodl_4090_49862_20261126/offline_exact_rows.jsonl", ids)
    clusters, group = np.unique([qids[k]["qid"] for k in ids], return_inverse=True)
    rng = np.random.default_rng(20261127)
    weights = rng.multinomial(
        len(clusters), np.full(len(clusters), 1 / len(clusters)), size=10000
    )
    official = [index(OUT / "dk" / (a + ".metrics.jsonl"), ids) for a in ARMS]
    preds = [index(PREDICTIONS / (a + ".jsonl"), ids) for a in ARMS]
    sft_receipt = None
    if args.sft:
        sft_receipt = json.loads((OUT / "sft_retrieval/prediction_receipt.json").read_text(encoding="utf-8"))
        if sft_receipt["adapter"] != manifest["adapter_sha256"] or sft_receipt["model"] != "memory_first":
            raise ValueError("Unexpected SFT checkpoint binding")
        for arm in ARMS:
            raw = (args.sft / (arm + ".jsonl")).read_bytes()
            if hashlib.sha256(raw).hexdigest() != sft_receipt["predictions"][arm]:
                raise ValueError("SFT predictions differ from the verified GPU archive")
    sf = [index(args.sft / (a + ".jsonl"), ids) for a in ARMS] if args.sft else []
    visible = [index(FOLLOWUP / ("full." + a + ".jsonl"), ids) for a in ARMS]
    mechanisms = []
    details = []
    for k in ids:
        row = {"_key": k, "qid": qids[k]["qid"]}
        for j in range(3):
            row["Q" + str(j + 1)] = [r[k]["memory_correct"][j] for r in official]
        row["Diagnosis"] = [r[k]["diagnosis_correct"] for r in official]
        row["Teaching_mean"] = [
            sum(r[k]["content_scores"][d] for d in DIMENSIONS) / 4 for r in official
        ]
        for d in DIMENSIONS:
            row[d] = [r[k]["content_scores"][d] for r in official]
        for name, pair in (("base", preds), ("sft", sf)):
            if pair:
                for j in range(3):
                    row[name + "_exact_Q" + str(j + 1)] = [
                        exact(p[k], gold[k], j) for p in pair
                    ]
                row[name + "_diagnosis_exact"] = [
                    int(p[k]["diagnosis"] == gold[k]["diagnosis"]) for p in pair
                ]
        if sf:
            for arm_no, arm in enumerate(ARMS):
                row["base_to_sft_" + arm + "_Q1"] = [
                    row[name + "_exact_Q1"][arm_no] for name in ("base", "sft")
                ]
        query = gold[k]["memory"][0]["query"]
        named = set(re.findall(r"\[(\d+)\]", query))
        seen = [
            {
                m.group(1)
                for text in src[k]["history_info"]
                if (m := re.match(r"\[(\d+)\]", text))
            }
            for src in visible
        ]
        pattern = (
            "no_bracket_id"
            if not named
            else "present_both"
            if all(named <= s for s in seen)
            else "added_by_temporal"
            if named <= seen[1]
            else "current_question"
            if named == {str(qids[k]["qid"])}
            else "other"
        )
        mechanisms.append(
            {
                "_key": k,
                "group": pattern,
                "q1_query": query,
                "gold": gold[k]["memory"][0]["answer"],
                "gold_ids": sorted(named),
                "pred_answers": [
                    p[k]["memory"][0]["answer"] if p[k]["memory"] else None
                    for p in preds
                ],
                "dk_q1": row["Q1"],
            }
        )
        details.append(row)
    summary = {
        "sft_receipt": sft_receipt,
        "n_qid_clusters": len(clusters),
        "seed": 20261127,
        "bootstrap": 10000,
        "arm_order": list(ARMS),
        "upstream_commit": manifest["upstream_commit"],
        "paired": {
            f: summarize([r[f] for r in details], group, weights)
            for f in details[0]
            if f not in ("_key", "qid")
        },
    }
    summary["q1_by_named_record_presence"] = {}
    for group_name in sorted({m["group"] for m in mechanisms}):
        v = [m["dk_q1"] for m in mechanisms if m["group"] == group_name]
        summary["q1_by_named_record_presence"][group_name] = {
            "n": len(v),
            "before": sum(x for x, y in v),
            "after": sum(y for x, y in v),
            "wins": sum(y and not x for x, y in v),
            "losses": sum(x and not y for x, y in v),
        }
    (OUT / "q1_evidence_coverage.rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mechanisms),
        encoding="utf-8",
    )
    if sf:
        interaction = np.asarray(
            [
                np.diff(r["sft_exact_Q1"])[0] - np.diff(r["base_exact_Q1"])[0]
                for r in details
            ]
        )
        summary["q1_retrieval_by_sft_interaction"] = summarize(
            np.column_stack([np.zeros(len(ids)), interaction]), group, weights
        )
    stem = "paired_all" if sf else "paired_official"
    (OUT / (stem + ".rows.jsonl")).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details),
        encoding="utf-8",
    )
    (OUT / (stem + ".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
