"""Cross-fit an executor reliability score without reading any new GPU prediction."""

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .budget_curve import HEADER, load_index
from .packet_followup import STUDY
from .paired_report import read
from .prepare import EXPLICIT_ID, digest, write_json, write_rows

FEATURES = (
    "count",
    "log_support",
    "log_query_length",
    "has_date",
    "explicit_id",
    "asks_multiple_fields",
    "mentions_identifier",
    "short_scope_agrees",
    "short_scope_has_result",
)
THRESHOLDS = (0.5, 0.75, 0.9)


def features(row):
    q = row["question"]
    return [
        int(row["full"]["operator"] == "count"),
        np.log1p(len(row["full"]["positions"])),
        np.log1p(len(q)),
        int(bool(re.search(r"20\d\d[-年]", q))),
        int(bool(EXPLICIT_ID.search(q))),
        int(bool(re.search(r"列出|写出|同时|以及|分别|并说明|并给出", q))),
        int(bool(re.search(r"题号|编号|ID", q, re.IGNORECASE))),
        int(row["agree"]),
        int(row["brief"] is not None),
    ]


def group_folds(rows):
    parent = {}

    def root(k):
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for r in rows:
        parent[root(("student", r["key"].split("||")[0]))] = root(
            ("question", r["qid"])
        )
    components = {}
    for i, r in enumerate(rows):
        components.setdefault(root(("question", r["qid"])), []).append(i)
    buckets = [[] for _ in range(5)]
    for indices in sorted(
        components.values(), key=lambda v: (-len(v), min(rows[i]["key"] for i in v))
    ):
        min(buckets, key=len).extend(indices)
    return buckets


def fit_predict(train, y, test):
    mean, sd = train.mean(axis=0), train.std(axis=0)
    sd[sd < 1e-8] = 1

    def standard(a):
        return np.column_stack([np.ones(len(a)), (a - mean) / sd])

    x = standard(train)

    def objective(beta):
        z = x @ beta
        loss = (np.logaddexp(0, z) - y * z).sum() + 0.5 * np.dot(beta[1:], beta[1:])
        grad = x.T @ (expit(z) - y)
        grad[1:] += beta[1:]
        return loss, grad

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return expit(standard(test) @ result.x)


def prepare(run):
    source = STUDY / "transfer560"
    labels = {r["id"]: r["correct"] for r in read(source / "full_q2_verdict.jsonl")}
    inputs = load_index(run / "followup/transfer_control.jsonl")
    rows = []
    for i, r in enumerate(read(source / "rows.jsonl")):
        if r["task"] == 2 and r["full"]:
            rows.append(
                {
                    **r,
                    "correct": labels[i],
                    "qid": HEADER.match(inputs[r["key"]]["question_info"])[1],
                }
            )
    if len(rows) != 131 or len(labels) != 131:
        raise ValueError("Incomplete labeled pool")
    x = np.asarray([features(r) for r in rows])
    y = np.array([r["correct"] for r in rows])
    dev = [
        r
        for r in read(STUDY / "four_rankers/executor_fixed.jsonl")
        if r["task"] == 2 and r["full"]
    ]
    result = []
    for j, indices in enumerate(group_folds(rows)):
        train = np.ones(len(rows), dtype=bool)
        train[indices] = False
        scores = fit_predict(x[train], y[train], x[indices])
        for i, score in zip(indices, scores, strict=True):
            result.append(
                {
                    "key": rows[i]["key"],
                    "population": "transfer",
                    "fold": j,
                    "probability": float(score),
                    "scope_agrees": rows[i]["agree"],
                    "direct_correct": int(y[i]),
                }
            )
    scores = fit_predict(x, y, np.asarray([features(r) for r in dev]))
    result.extend(
        {
            "key": r["key"],
            "population": "dev",
            "fold": None,
            "probability": float(s),
            "scope_agrees": r["agree"],
        }
        for r, s in zip(dev, scores, strict=True)
    )
    out = run / "policy"
    out.mkdir(exist_ok=False)
    write_rows(out / "decisions.jsonl", result)
    write_json(
        out / "protocol.json",
        {
            "thresholds": THRESHOLDS,
            "primary_threshold": 0.75,
            "features": FEATURES,
            "training": "binary logistic, standardized per training fold, L2 penalty coefficient 1, no hyperparameter search",
            "groups": "five folds of connected student-current-question groups; dev Q2 scored using all transfer labels",
            "target": "prior executor correctness, no access to new generator responses",
            "application": "use packet response when probability >= threshold; otherwise matched control. Analysis of frozen generated alternatives.",
            "code_sha256": digest(Path(__file__)),
            "label_sha256": digest(source / "full_q2_verdict.jsonl"),
        },
    )
    print(
        {
            str(t): sum(
                r["population"] == "transfer" and r["probability"] >= t for r in result
            )
            for t in THRESHOLDS
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=STUDY / "gpu_run1")
    prepare(p.parse_args().run)
