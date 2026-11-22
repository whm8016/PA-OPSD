#!/usr/bin/env python3
"""Verify LongTutor history inputs align with the 1,000 Gold annotations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LT_REPO = ROOT / "data" / "longtutor" / "repo"
SCRIPT_DIR = LT_REPO / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from gpt_memory_diagnose import _sample_key  # noqa: E402


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    base = LT_REPO / "data" / "XES3G5M"
    history_path = base / "history_features_lastq.jsonl"
    scale_path = base / "history_features_lastq_scale.jsonl"
    gold_path = base / "human_an_updated.jsonl"

    for path in (history_path, scale_path, gold_path):
        if not path.is_file():
            print(json.dumps({
                "ok": False,
                "error": f"missing_file: {path}",
            }, ensure_ascii=False))
            return 2

    history_rows = list(_iter_jsonl(history_path))
    gold_rows = list(_iter_jsonl(gold_path))
    scale_rows = list(_iter_jsonl(scale_path))

    input_keys = [_sample_key(row) for row in history_rows]
    gold_keys = [str(row.get("_key", "")) for row in gold_rows]
    gold_n = len(gold_keys)
    positional_matches = sum(
        left == right
        for left, right in zip(input_keys[:gold_n], gold_keys)
    )
    intersection = len(set(input_keys[:gold_n]) & set(gold_keys))
    unique_gold = len(set(gold_keys))
    ok = (
        gold_n == 1000
        and unique_gold == gold_n
        and positional_matches == gold_n
        and intersection == gold_n
        and len(history_rows) == 3437
        and len(scale_rows) == len(history_rows) - gold_n
    )

    result = {
        "ok": ok,
        "history_rows": len(history_rows),
        "scale_rows": len(scale_rows),
        "gold_rows": gold_n,
        "unique_gold_keys": unique_gold,
        "positional_matches": positional_matches,
        "key_intersection": intersection,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
