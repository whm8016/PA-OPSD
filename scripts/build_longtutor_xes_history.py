#!/usr/bin/env python3
"""Build XES3G5M history_features_lastq*.jsonl for LongTutor eval (local, no API)."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LT = ROOT / "data" / "longtutor" / "repo"
sys.path.insert(0, str(LT / "scripts"))

import compute_history_stats as chs  # noqa: E402


def _xes_concept_segments(concepts):
    """Use the XES3G5M concept path documented in compute_history_stats.py.

    The active helper in the upstream file is for MOOCRadar. For XES3G5M, each
    concept is a ``----`` separated path and only the path segments are used;
    using the raw full string changes question_info and breaks Gold _key alignment.
    """
    if not concepts:
        return []
    seen = set()
    segments = []
    for concept in concepts:
        if not concept:
            continue
        for part in str(concept).split("----")[1:]:
            value = part.strip()
            if value and value not in seen:
                seen.add(value)
                segments.append(value)
    return segments


chs._concept_segments = _xes_concept_segments


def main() -> None:
    base = LT / "data" / "XES3G5M"
    sequences_path = base / "sequences_long.jsonl"
    questions_path = base / "questions.jsonl"
    full_out = base / "history_features_lastq.jsonl"
    scale_out = base / "history_features_lastq_scale.jsonl"
    gold_n = 1000

    if not sequences_path.is_file():
        raise SystemExit(f"Missing {sequences_path}; run scripts/download_edu_datasets.py first")

    questions_map = chs.load_questions_map(questions_path)
    global_err = chs.compute_global_error_rates(sequences_path)
    with contextlib.redirect_stdout(io.StringIO()):
        chs.build_history_features(
            sequences_jsonl=sequences_path,
            questions_map=questions_map,
            global_error_rate=global_err,
            output_jsonl=full_out,
            recent_k=10,
        )

    lines = full_out.read_text(encoding="utf-8").splitlines()
    if len(lines) < gold_n:
        raise SystemExit(f"Expected >={gold_n} rows in {full_out}, got {len(lines)}")
    scale_out.write_text("\n".join(lines[gold_n:]) + "\n", encoding="utf-8")
    print(f"Wrote {full_out} ({len(lines)} rows)")
    print(f"Wrote {scale_out} ({len(lines) - gold_n} rows)")


if __name__ == "__main__":
    main()
