#!/usr/bin/env python3
"""Build the first PA-OPSD dataset from LongTutor/XES3G5M non-Gold rows.

The output is intentionally conservative:

* LongTutor Gold1000 is never included.
* ``pipeline_an_scale`` is kept as weak supervision, not as gold action labels.
* Student-visible fields and Teacher-only privileged fields are separated.
* The custom pedagogical action is only a candidate/default derived from the
  official LongTutor strategy; every row remains pending review.

Run from the repository root::

    python scripts/build_pa_opsd_seed.py --limit 100
    python scripts/build_pa_opsd_seed.py

The first command creates an approximately 90/10 smoke-test seed. The second command creates
the strict question-disjoint non-Gold seed used for the first main PA-OPSD claim. By default,
current questions that also occur in LongTutor Gold1000 are removed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LONGTUTOR_XES = ROOT / "data" / "longtutor" / "repo" / "data" / "XES3G5M"
EEDI_DIALOGUES = (
    ROOT
    / "data"
    / "eedi"
    / "dialogues"
    / "anchored-dialogues"
    / "train-00000-of-00001.parquet"
)
EEDI_MISCONCEPTIONS = (
    ROOT
    / "data"
    / "eedi"
    / "misconceptions-graph"
    / "data"
    / "misconception-map.csv"
)

ACTION_SCHEMA_VERSION = "pa_opsd_action_v0.1"
DATASET_SCHEMA_VERSION = "pa_opsd_dataset_v0.2_question_disjoint"

ACTIONS = {
    "probe": "追问",
    "hint": "提示",
    "correct": "纠错",
    "example": "举例",
    "partial_solution": "部分解答",
    "stop": "停止",
}

STRATEGY_RULES = {
    "Retrieval Practice": {
        "diagnosis": "Recall Failure",
        "default_action": "probe",
        "candidate_actions": ["probe", "hint"],
        "rule": "Prior mastery exists but retrieval is weak; first ask the student to recall the relevant rule or successful step.",
    },
    "Conceptual Explanation": {
        "diagnosis": "Conceptual Gap",
        "default_action": "correct",
        "candidate_actions": ["correct", "example", "hint"],
        "rule": "The concept itself is unstable; point out the unstable idea and use a definition or concrete example before completing the solution.",
    },
    "Stepwise Scaffolding": {
        "diagnosis": "Procedural Error",
        "default_action": "hint",
        "candidate_actions": ["hint", "partial_solution", "probe"],
        "rule": "The idea is mostly understood but a step failed; first isolate the failed step, then scaffold only that step.",
    },
    "Analogical Transfer": {
        "diagnosis": "Transfer Deficit",
        "default_action": "example",
        "candidate_actions": ["example", "hint", "probe"],
        "rule": "A related successful case exists; connect the current problem to that case before giving a direct correction.",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sample_key(row: dict[str, Any]) -> str:
    uid = str(row.get("uid", ""))
    digest = hashlib.sha256(str(row.get("question_info", "")).encode("utf-8")).hexdigest()
    return f"{uid}||{digest}"


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        question_id = str(row.get("id", ""))
        info = row.get("info", {}) if isinstance(row.get("info"), dict) else {}
        questions[question_id] = {
            "content": info.get("content", ""),
            "answer": info.get("answer", []),
            "analysis": info.get("analysis", ""),
            "question_type": info.get("type", ""),
            "options": info.get("options", {}),
            "concepts": info.get("concepts", []),
        }
    return questions


def current_question_id(question_info: str) -> str | None:
    match = re.search(r"\[(\d+)\]", question_info)
    return match.group(1) if match else None


def build_action_codebook() -> dict[str, Any]:
    definitions = {
        "probe": {
            "zh": ACTIONS["probe"],
            "definition": "Ask the student to explain a rule, step, reason, or missing intermediate result.",
            "use_when": "The diagnosis is not yet stable, or the student can probably repair the answer after retrieval.",
            "avoid_when": "The student has already explained the same missing point or is clearly stuck after repeated prompts.",
        },
        "hint": {
            "zh": ACTIONS["hint"],
            "definition": "Give a targeted cue that narrows attention without completing the key step.",
            "use_when": "The error mechanism is identified and the student needs one cue to continue.",
            "avoid_when": "The concept is absent, or the student needs an analogous case rather than another local cue.",
        },
        "correct": {
            "zh": ACTIONS["correct"],
            "definition": "Explicitly identify the incorrect idea or step, then explain why it fails.",
            "use_when": "Continued probing would reinforce a wrong rule or prolong confusion.",
            "avoid_when": "A minor hint is enough, or the student only made a transfer slip.",
        },
        "example": {
            "zh": ACTIONS["example"],
            "definition": "Use an analogous solved case, counterexample, or concrete situation to expose the shared structure.",
            "use_when": "A related successful problem exists, or the current concept is too abstract for direct correction.",
            "avoid_when": "The student lacks the prerequisite concept needed to understand the example.",
        },
        "partial_solution": {
            "zh": ACTIONS["partial_solution"],
            "definition": "Complete only one bounded substep, then hand the next step back to the student.",
            "use_when": "Repeated hints failed, or the student cannot start a multi-step procedure.",
            "avoid_when": "A hint or analogous case is still likely to trigger self-repair.",
        },
        "stop": {
            "zh": ACTIONS["stop"],
            "definition": "Close or pause the intervention after the target understanding has been demonstrated.",
            "use_when": "The student has given the expected reasoning or the current objective is complete.",
            "avoid_when": "Only the final answer is correct but the underlying reason has not been checked.",
        },
    }

    return {
        "schema_version": ACTION_SCHEMA_VERSION,
        "status": "draft_for_teacher_team_and_human_review",
        "warning": (
            "These are project-defined pedagogical actions. Eedi TalkMove labels and "
            "LongTutor strategy labels are not native gold labels for this six-action schema."
        ),
        "actions": [
            {"action_id": action_id, **definition}
            for action_id, definition in definitions.items()
        ],
        "strategy_to_action": STRATEGY_RULES,
    }


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_eedi_action_examples(max_examples_per_action: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Select deterministic, explicitly heuristic exemplars from Eedi train split.

    Eedi's TalkMovePrediction is not a gold label for PA-OPSD actions. These rows
    are only few-shot references for Teacher Team prompts and must be audited.
    """

    frame = pd.read_parquet(EEDI_DIALOGUES)
    required = {"InterventionId", "QuestionId_DQ", "MessageSequence", "IsTutor", "MessageString", "TalkMovePrediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Eedi dialogue parquet is missing columns: {sorted(missing)}")

    frame = frame.fillna({"TalkMovePrediction": "<None>", "MessageString": ""})
    frame["InterventionId"] = frame["InterventionId"].astype(str)
    frame["QuestionId_DQ"] = frame["QuestionId_DQ"].astype(str)
    frame["MessageSequence"] = pd.to_numeric(frame["MessageSequence"], errors="coerce").fillna(0).astype(int)
    frame = frame.sort_values(["InterventionId", "MessageSequence"], kind="stable")

    grouped = {str(key): group.to_dict("records") for key, group in frame.groupby("InterventionId", sort=False)}
    talk_move_counts = Counter(frame["TalkMovePrediction"].astype(str))

    patterns = {
        "correct": re.compile(
            r"^(?:no,|that(?:'s| is) (?:not|incorrect|wrong)|you(?:'re| are) (?:wrong|not right)|it(?:'s| is) (?:not|wrong))",
            re.IGNORECASE,
        ),
        "partial_solution": re.compile(
            r"\b(?:let's start by|let’s start by|first step (?:is|should be)|our first step is|we need to first|we should start by|start by calculating|let's work out the first)\b",
            re.IGNORECASE,
        ),
        "stop": re.compile(
            r"\b(?:well done|great job|you(?:'ve| have) got|that(?:'s| is) correct|perfect|brilliant)\b",
            re.IGNORECASE,
        ),
    }

    selected: dict[str, list[dict[str, Any]]] = {action: [] for action in ACTIONS}

    def add_example(
        action: str,
        row: dict[str, Any],
        reason: str,
        confidence: str,
    ) -> None:
        if len(selected[action]) >= max_examples_per_action:
            return
        message = _clean_text(row.get("MessageString"))
        if len(message) < 25 or re.search(r"\b(?:hello|hi|are you ok|goodbye)\b", message, re.IGNORECASE):
            return

        intervention_id = str(row.get("InterventionId", ""))
        messages = grouped.get(intervention_id, [])
        sequence = int(row.get("MessageSequence", 0))
        context = [
            {
                "speaker": "tutor" if int(item.get("IsTutor", 0)) == 1 else "student",
                "message": _clean_text(item.get("MessageString"))[:800],
            }
            for item in messages
            if 1 <= sequence - int(item.get("MessageSequence", 0)) <= 2
        ]

        selected[action].append(
            {
                "intervention_id": intervention_id,
                "question_id_dq": str(row.get("QuestionId_DQ", "")),
                "message_sequence": sequence,
                "talk_move": str(row.get("TalkMovePrediction", "")),
                "tutor_message": message[:1200],
                "recent_context": context,
                "mapping_reason": reason,
                "mapping_confidence": confidence,
                "review_status": "pending",
            }
        )

    for row in frame.to_dict("records"):
        if int(row.get("IsTutor", 0)) != 1:
            continue
        talk_move = str(row.get("TalkMovePrediction", ""))
        message = _clean_text(row.get("MessageString"))

        if talk_move == "<Press for Reasoning>":
            add_example("probe", row, "Eedi Press for Reasoning directly asks for justification.", "high")
        elif talk_move == "<Press for Accuracy>" and re.search(
            r"\b(?:why|how|what|do you think|can you)\b", message, re.IGNORECASE
        ):
            add_example("probe", row, "Eedi Press for Accuracy contains a probe for the student's result or rule.", "medium")

        if talk_move in {"<Keep Together>", "<Revoicing>", "<Restating>"} and len(message) >= 35:
            add_example("hint", row, f"Eedi {talk_move[1:-1]} often restates or narrows the relevant cue.", "medium")

        if talk_move == "<Getting Student to Relate>":
            add_example("example", row, "Eedi Getting Student to Relate asks the student to connect another case.", "high")

        correction_is_statement = not message.endswith("?") and not re.match(
            r"^(?:do|does|did|what|why|how|can|could|should|is|are)\b", message, re.IGNORECASE
        )
        if patterns["correct"].search(message) and correction_is_statement:
            add_example("correct", row, "Tutor message explicitly identifies an error; inferred from text, not a native Eedi action label.", "medium")

        partial_is_statement = not message.endswith("?") and not re.match(
            r"^(?:do|does|did|what|why|how|can|could|should|is|are)\b", message, re.IGNORECASE
        )
        if patterns["partial_solution"].search(message) and partial_is_statement:
            add_example("partial_solution", row, "Tutor message starts a bounded substep; inferred from text, not a native Eedi action label.", "low")

        intervention_messages = grouped.get(str(row.get("InterventionId", "")), [])
        is_late_message = int(row.get("MessageSequence", 0)) >= max(
            [int(item.get("MessageSequence", 0)) for item in intervention_messages] or [0]
        ) - 2
        if is_late_message and patterns["stop"].search(message):
            add_example("stop", row, "Positive closure occurs near the end of an intervention; inferred from context.", "low")

    return selected, dict(sorted(talk_move_counts.items()))


def load_misconception_index(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_bytes = path.read_bytes()
    text = None
    used_encoding = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeDecodeError("csv", raw_bytes, 0, 1, "unable to decode Eedi misconception CSV")

    for raw in csv.DictReader(StringIO(text)):
        rows.append(
            {
                "misconception_id": str(raw.get("MisconceptionId", "")),
                "misconception": _clean_text(raw.get("MisconceptionName")),
                "construct_id": str(raw.get("ConstructId", "")),
                "construct": _clean_text(raw.get("ConstructName")),
                "subtopic": _clean_text(raw.get("SubTopicName")),
                "topic": _clean_text(raw.get("TopicName")),
                "subject": _clean_text(raw.get("SubjectName")),
                "percentage_in_construct": raw.get("PercentageMisconceptionExistsInThisConstruct"),
                "source_encoding": used_encoding,
                "retrieval_status": "not_retrieved_cross_lingual_index_pending",
            }
        )
    return rows


def load_aligned_non_gold(allow_gold_question_overlap: bool) -> dict[str, Any]:
    paths = {
        "gold": LONGTUTOR_XES / "human_an_updated.jsonl",
        "full_history": LONGTUTOR_XES / "history_features_lastq.jsonl",
        "scale_annotations": LONGTUTOR_XES / "pipeline_an_scale.jsonl",
        "scale_history": LONGTUTOR_XES / "history_features_lastq_scale.jsonl",
        "questions": LONGTUTOR_XES / "questions.jsonl",
        "eedi_dialogues": EEDI_DIALOGUES,
        "eedi_misconceptions": EEDI_MISCONCEPTIONS,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}; run scripts/download_edu_datasets.py and scripts/build_longtutor_xes_history.py first")

    gold = read_jsonl(paths["gold"])
    full_history = read_jsonl(paths["full_history"])
    history = read_jsonl(paths["scale_history"])
    annotations = read_jsonl(paths["scale_annotations"])
    gold_keys = {str(row.get("_key", "")) for row in gold}
    history_keys = [sample_key(row) for row in history]
    annotation_keys = [str(row.get("_key", "")) for row in annotations]

    positional_matches = sum(left == right for left, right in zip(history_keys, annotation_keys))
    gold_overlap = len(set(history_keys) & gold_keys)
    raw_non_gold_history = full_history[len(gold) :]
    if len(raw_non_gold_history) != len(history):
        raise ValueError(
            f"Expected full history tail to match scale history: {len(raw_non_gold_history)} vs {len(history)}"
        )
    raw_tail_keys = [sample_key(row) for row in raw_non_gold_history]
    if raw_tail_keys != history_keys:
        raise ValueError("history_features_lastq.jsonl tail does not match history_features_lastq_scale.jsonl")

    gold_question_ids = {current_question_id(str(row.get("question_info", ""))) for row in full_history[: len(gold)]}
    gold_question_ids.discard(None)
    gold_question_overlap_rows = sum(
        current_question_id(str(row.get("question_info", ""))) in gold_question_ids
        for row in raw_non_gold_history
    )
    eligible_pairs = [
        (history_row, annotation_row)
        for history_row, annotation_row in zip(raw_non_gold_history, annotations)
        if allow_gold_question_overlap
        or current_question_id(str(history_row.get("question_info", ""))) not in gold_question_ids
    ]
    ok = (
        len(gold) == 1000
        and len(history) == 2437
        and len(annotations) == 2437
        and len(set(history_keys)) == 2437
        and positional_matches == 2437
        and gold_overlap == 0
    )
    if not ok:
        raise ValueError(
            "LongTutor/XES alignment failed: "
            f"gold={len(gold)}, history={len(history)}, annotations={len(annotations)}, "
            f"positional_matches={positional_matches}, gold_overlap={gold_overlap}"
        )

    return {
        "source_pairs": eligible_pairs,
        "ok": True,
        "split_policy": "question_uid_disjoint_stable_hash",
        "gold_question_overlap_allowed": allow_gold_question_overlap,
        "gold_rows": len(gold),
        "non_gold_rows": len(history),
        "annotation_rows": len(annotations),
        "positional_key_matches": positional_matches,
        "gold_non_gold_overlap": gold_overlap,
        "gold_unique_current_questions": len(gold_question_ids),
        "raw_non_gold_unique_current_questions": len(
            {current_question_id(str(row.get("question_info", ""))) for row in raw_non_gold_history} - {None}
        ),
        "raw_non_gold_gold_question_overlap_rows": gold_question_overlap_rows,
        "eligible_rows": len(eligible_pairs),
        "excluded_by_gold_question_overlap": len(raw_non_gold_history) - len(eligible_pairs),
        "eligible_gold_question_overlap_rows": (
            gold_question_overlap_rows if allow_gold_question_overlap else 0
        ),
    }


def build_teacher_prompt_template() -> dict[str, Any]:
    return {
        "template_version": "pa_opsd_teacher_team_prompt_v0.1",
        "system": (
            "You are a Teacher Team with four specialists: Evidence Agent, Diagnosis Agent, "
            "Action Agent, and Response Agent, followed by a Verifier. You see privileged "
            "information during training. Do not claim unsupported facts. The Eedi exemplars "
            "are references, not gold action labels."
        ),
        "user_template": (
            "### TASK\n"
            "Review the weak LongTutor annotation and produce one improved PA-OPSD target.\n\n"
            "### STUDENT-VISIBLE INPUT x\n{student_input_json}\n\n"
            "### TEACHER-PRIVILEGED INPUT z\n{teacher_privileged_json}\n\n"
            "### EEDI FEW-SHOT REFERENCES\n{eedi_examples_json}\n\n"
            "### OUTPUT JSON ONLY\n"
            "{{\n"
            '  "evidence": ["..."],\n'
            '  "diagnosis": "Recall Failure | Conceptual Gap | Procedural Error | Transfer Deficit",\n'
            '  "diagnosis_reason": "...",\n'
            '  "action_id": "probe | hint | correct | example | partial_solution | stop",\n'
            '  "action_reason": "...",\n'
            '  "response": "...",\n'
            '  "verifier": {{"pass": true, "issues": []}}\n'
            "}}\n"
        ),
        "constraints": [
            "Evidence must be grounded in the provided history or statistics.",
            "Diagnosis must be one of the four LongTutor diagnosis categories.",
            "Action must be one of the six PA-OPSD actions.",
            "Response must implement the selected action and must not leak privileged fields as metadata.",
            "Return valid JSON only.",
        ],
    }


def build_rows(
    source_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    questions: dict[str, dict[str, Any]],
    eedi_examples: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for history, annotation in source_pairs:
        key = sample_key(history)
        if key != str(annotation.get("_key", "")):
            raise ValueError(f"Unexpected key mismatch while building row: {key}")

        question_id = current_question_id(str(history.get("question_info", "")))
        question = questions.get(question_id or "", {})
        strategy = str(annotation.get("strategy", ""))
        rule = STRATEGY_RULES.get(strategy)
        if rule is None:
            raise ValueError(f"Unknown LongTutor strategy: {strategy!r}")

        default_action = rule["default_action"]
        sample_id = f"xes3g5m::{history.get('uid', '')}::q{question_id or 'unknown'}::{hashlib.sha256(key.encode()).hexdigest()[:12]}"

        student_input = {
            "source": "LongTutor/XES3G5M history_features_lastq_scale",
            "uid": history.get("uid"),
            "question_id": question_id,
            "current_question": history.get("question_info", ""),
            "history_info": history.get("history_info", []),
            "related_history": history.get("related_history", []),
            "student_visible_statistics": history.get("features", {}),
        }

        teacher_privileged = {
            "source": "LongTutor/XES3G5M question bank + pipeline_an_scale + Eedi graph",
            "knowledge_concepts": question.get("concepts", []),
            "standard_answer": question.get("answer", []),
            "expert_analysis": question.get("analysis", ""),
            "question_type": question.get("question_type", ""),
            "question_options": question.get("options", {}),
            "longtutor_weak_annotation": {
                "memory": annotation.get("memory", []),
                "diagnosis": annotation.get("diagnosis", ""),
                "reason": annotation.get("reason", ""),
                "strategy": strategy,
                "content": annotation.get("content", ""),
            },
            "misconception_retrieval": {
                "status": "pending_cross_lingual_index",
                "note": (
                    "Eedi graph is indexed separately. Per-sample candidates are intentionally "
                    "empty until a multilingual construct-to-misconception retriever is enabled."
                ),
                "candidates": [],
            },
            "action_candidates": rule["candidate_actions"],
        }

        target = {
            "target_status": "weak_pending_teacher_team_review",
            "source": "pipeline_an_scale_v1_plus_rule_action_v0.1",
            "evidence": annotation.get("memory", []),
            "diagnosis": annotation.get("diagnosis", ""),
            "diagnosis_reason": annotation.get("reason", ""),
            "strategy": strategy,
            "action_id": default_action,
            "action_label_zh": ACTIONS[default_action],
            "action_source": "deterministic_strategy_mapping_not_gold",
            "action_label_usable_for_training": False,
            "action_candidates": rule["candidate_actions"],
            "response": annotation.get("content", ""),
            "review": {
                "status": "pending",
                "reviewer": "",
                "issues": [],
                "teacher_output": None,
            },
        }

        rows.append(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "sample_id": sample_id,
                "longtutor_key": key,
                "split": "to_assign",
                "student_input": student_input,
                "teacher_privileged": teacher_privileged,
                "target": target,
                "eedi_reference_examples": {
                    "default_action": default_action,
                    "examples": eedi_examples.get(default_action, []),
                    "warning": "Heuristic references only; not action gold labels.",
                },
                "provenance": {
                    "benchmark_system": "LongTutor/XES3G5M",
                    "gold_test_used": False,
                    "eedi_role": "action_exemplars_and_misconception_graph",
                    "license_note": "Eedi dialogues are non-commercial research data; inspect official license before redistribution.",
                },
            }
        )

    return rows


def assign_splits(rows: list[dict[str, Any]], validation_ratio: float, split_seed: str) -> None:
    total = len(rows)
    if total <= 0:
        return

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in (0, 1), got {validation_ratio}")

    # A sample is simultaneously attached to a student UID and a current question.
    # Union-find keeps all rows sharing either value in the same split, preventing
    # both question leakage and student leakage between train and validation.
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in rows:
        question_node = f"question::{row['student_input']['question_id']}"
        student_node = f"student::{row['student_input']['uid']}"
        union(question_node, student_node)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_node = f"question::{row['student_input']['question_id']}"
        groups[find(question_node)].append(row)

    ordered_roots = sorted(
        groups,
        key=lambda root: hashlib.sha256(f"{split_seed}:{root}".encode()).hexdigest(),
    )
    target_validation_size = max(1, min(total - 1, round(total * validation_ratio)))
    validation_count = 0
    validation_roots: set[str] = set()
    for root in ordered_roots:
        if validation_count >= target_validation_size:
            break
        group_size = len(groups[root])
        if validation_count == 0 or validation_count + group_size <= total - 1:
            validation_roots.add(root)
            validation_count += group_size

    for root, group in groups.items():
        split = "validation" if root in validation_roots else "train"
        for row in group:
            row["split"] = split


def audit(rows: list[dict[str, Any]], input_check: dict[str, Any], eedi_counts: dict[str, int], eedi_examples: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    diagnosis_counts = Counter(row["target"]["diagnosis"] for row in rows)
    strategy_counts = Counter(row["target"]["strategy"] for row in rows)
    action_counts = Counter(row["target"]["action_id"] for row in rows)
    missing_question = sum(row["student_input"]["question_id"] is None for row in rows)
    missing_standard_answer = sum(not row["teacher_privileged"]["standard_answer"] for row in rows)
    missing_analysis = sum(not row["teacher_privileged"]["expert_analysis"] for row in rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    train_question_ids = {row["student_input"]["question_id"] for row in train_rows}
    validation_question_ids = {row["student_input"]["question_id"] for row in validation_rows}
    train_uids = {row["student_input"]["uid"] for row in train_rows}
    validation_uids = {row["student_input"]["uid"] for row in validation_rows}
    train_validation_question_overlap = len(train_question_ids & validation_question_ids)
    train_validation_uid_overlap = len(train_uids & validation_uids)
    if missing_question or train_validation_question_overlap or train_validation_uid_overlap:
        raise ValueError(
            "Split isolation failed: "
            f"missing_question={missing_question}, "
            f"question_overlap={train_validation_question_overlap}, "
            f"uid_overlap={train_validation_uid_overlap}"
        )
    if not input_check["gold_question_overlap_allowed"] and input_check["eligible_gold_question_overlap_rows"] != 0:
        raise ValueError("Strict mode unexpectedly retained a Gold question overlap")

    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "input_alignment": input_check,
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "diagnosis_counts": dict(diagnosis_counts),
        "strategy_counts": dict(strategy_counts),
        "rule_default_action_counts": dict(action_counts),
        "all_actions_have_at_least_one_eedi_heuristic_example": all(eedi_examples.values()),
        "split_isolation": {
            "policy": input_check["split_policy"],
            "train_unique_current_questions": len(train_question_ids),
            "validation_unique_current_questions": len(validation_question_ids),
            "train_validation_current_question_overlap": train_validation_question_overlap,
            "train_validation_uid_overlap": train_validation_uid_overlap,
        },
        "eedi_heuristic_example_counts": {action: len(items) for action, items in eedi_examples.items()},
        "eedi_talk_move_counts": eedi_counts,
        "missing_question_id": missing_question,
        "missing_standard_answer": missing_standard_answer,
        "missing_expert_analysis": missing_analysis,
        "quality_warnings": [
            "pipeline_an_scale is weak supervision and must not be reported as human gold.",
            "action_id is a deterministic strategy mapping and must be reviewed by Teacher Team/human.",
            "The strategy mapping currently produces only probe/hint/correct/example; partial_solution and stop require Teacher Team or human review before L_action training.",
            "Eedi TalkMove exemplars are heuristic and are not native six-action labels.",
            "Per-sample Eedi misconception candidates are pending a cross-lingual retrieval step.",
        ],
    }


def write_readme(path: Path, audit_report: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# PA-OPSD v0.2 question-disjoint seed dataset",
                "",
                "This directory contains a same-system, non-Gold LongTutor/XES3G5M seed set.",
                "",
                "## Field contract",
                "",
                "- `student_input`: fields available to the deployed Student Tutor.",
                "- `teacher_privileged`: standard answer, expert analysis, concepts, weak LongTutor annotation, action candidates, and future misconception candidates.",
                "- `target`: weak target from `pipeline_an_scale` plus a draft action; every row is `pending` review.",
                "- `eedi_reference_examples`: heuristic few-shot references, not gold action labels.",
                "",
                "## Non-negotiable checks",
                "",
                f"- Gold rows: `{audit_report['input_alignment']['gold_rows']}`.",
                f"- Non-Gold rows: `{audit_report['input_alignment']['non_gold_rows']}`.",
                f"- Gold/non-Gold overlap: `{audit_report['input_alignment']['gold_non_gold_overlap']}`.",
                f"- Raw non-Gold rows excluded for Gold question overlap: `{audit_report['input_alignment']['excluded_by_gold_question_overlap']}`.",
                f"- Eligible Gold-question overlap rows: `{audit_report['input_alignment']['eligible_gold_question_overlap_rows']}`.",
                f"- Positional key matches: `{audit_report['input_alignment']['positional_key_matches']}`.",
                f"- Train/validation current-question overlap: `{audit_report['split_isolation']['train_validation_current_question_overlap']}`.",
                f"- Train/validation student-UID overlap: `{audit_report['split_isolation']['train_validation_uid_overlap']}`.",
                "",
                "## Next step",
                "",
                "Run Teacher Team review before SFT/OPSD. Replace or confirm each `target.review.teacher_output`, then copy reviewed rows into the trainer-specific format.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "pa_opsd" / "v0.2_question_disjoint_weak_seed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all eligible strict rows; 100 creates an approximately 90/10 smoke seed",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", default="pa_opsd_v0.2")
    parser.add_argument(
        "--allow-gold-question-overlap",
        action="store_true",
        help="Secondary/ablation only: retain non-Gold rows whose current question also appears in Gold1000",
    )
    parser.add_argument("--eedi-examples-per-action", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_check = load_aligned_non_gold(args.allow_gold_question_overlap)
    source_pairs = input_check.pop("source_pairs")
    if args.limit > 0:
        source_pairs = source_pairs[: args.limit]
    questions = load_questions(LONGTUTOR_XES / "questions.jsonl")
    eedi_examples, eedi_counts = load_eedi_action_examples(args.eedi_examples_per_action)
    misconception_index = load_misconception_index(EEDI_MISCONCEPTIONS)

    rows = build_rows(source_pairs, questions, eedi_examples)
    assign_splits(rows, args.validation_ratio, args.split_seed)
    audit_report = audit(rows, input_check, eedi_counts, eedi_examples)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "all.jsonl"
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    manifest_path = output_dir / "split_manifest.jsonl"
    codebook_path = output_dir / "action_codebook.json"
    examples_path = output_dir / "eedi_action_examples.json"
    misconception_path = output_dir / "eedi_misconception_index.jsonl"
    prompt_path = output_dir / "teacher_prompt_template.json"
    audit_path = output_dir / "audit_report.json"

    all_count = write_jsonl(all_path, rows)
    train_count = write_jsonl(train_path, (row for row in rows if row["split"] == "train"))
    val_count = write_jsonl(val_path, (row for row in rows if row["split"] == "validation"))
    write_jsonl(
        manifest_path,
        (
            {
                "sample_id": row["sample_id"],
                "longtutor_key": row["longtutor_key"],
                "split": row["split"],
                "diagnosis": row["target"]["diagnosis"],
                "strategy": row["target"]["strategy"],
                "draft_action_id": row["target"]["action_id"],
                "review_status": row["target"]["review"]["status"],
            }
            for row in rows
        ),
    )
    codebook_path.write_text(json.dumps(build_action_codebook(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    examples_path.write_text(json.dumps(eedi_examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(misconception_path, misconception_index)
    prompt_path.write_text(json.dumps(build_teacher_prompt_template(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(output_dir / "README.md", audit_report)

    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(output_dir),
                "all_rows": all_count,
                "train_rows": train_count,
                "validation_rows": val_count,
                "misconception_rows": len(misconception_index),
                "audit": str(audit_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
