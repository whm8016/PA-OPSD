import json
import tempfile
import unittest
from pathlib import Path

from pa_opsd_collator import standard_solution_messages
from scripts.experiments.gold_evidence_first.data import (
    MEMORY_FIRST,
    group_metadata,
    reorder_target,
)
from scripts.experiments.gold_evidence_first.metrics import paired
from scripts.experiments.gold_evidence_first.run import check_teacher_report


class GoldEvidenceFirstTests(unittest.TestCase):
    def test_serialized_target_starts_with_memory_and_preserves_values(self):
        target = {"diagnosis": "Procedural Error", "reason": "reason", "strategy": "strategy",
                  "content": "feedback", "memory": [{"id": "Q1", "answer": "answer"}]}
        messages = [{"role": "system", "content": "task"}, {"role": "user", "content": "history"}]
        row = {"messages": [*messages, {"role": "assistant", "content": json.dumps(target)}]}
        revised = reorder_target(row, MEMORY_FIRST)
        self.assertEqual(revised["messages"][:2], messages)
        written = json.loads(revised["messages"][2]["content"])
        self.assertEqual(tuple(written), MEMORY_FIRST)
        self.assertEqual(written, target)
        self.assertEqual(next(iter(json.loads(row["messages"][2]["content"]))), "diagnosis")

    def test_group_identity_uses_current_question_and_bound_history(self):
        user = ("### INTERACTION HISTORY\n[99]\nhistory\n\n"
                "### CURRENT QUESTION\n[12]\nA question\n知识点: K\n\n### MEMORY QUERIES\nQ1")
        row = {"sample_id": "student||" + "a" * 64,
               "messages": [{"content": "task"}, {"content": user}]}
        meta = group_metadata(row)
        self.assertEqual((meta["uid"], meta["qid"]), ("student", "12"))
        self.assertIsNotNone(meta["history_hash"])

    def test_teacher_solution_preserves_history_and_student_prompt(self):
        student = [{"role": "system", "content": "task"}, {"role": "user", "content": "history"}]
        teacher = standard_solution_messages(
            student, {"standard_answer": ["12"], "analysis": "example"}, include_guidance=False)
        self.assertEqual(student[1]["content"], "history")
        self.assertEqual(student[0], teacher[0])
        self.assertTrue(teacher[1]["content"].startswith("history\n\n"))
        self.assertIn('"standard_answer": ["12"]', teacher[1]["content"])
        self.assertNotIn("Use the worked solution", teacher[1]["content"])

    def test_no_opsd_with_smoke_or_different_anchor_teacher_result(self):
        report = {"manifest_sha256": "bundle", "adapter": "anchor", "rows": 140,
                  "smoke": False, "teacher_gain": True}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            check_teacher_report(path, "bundle", "anchor")
            with self.assertRaises(ValueError):
                check_teacher_report(path, "bundle", "another_anchor")
            path.write_text(json.dumps({**report, "smoke": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                check_teacher_report(path, "bundle", "anchor")

    def test_score_pairs_require_the_same_reference(self):
        left = [{"sample_id": "q1", "diagnosis_gold": "Recall Failure", "diagnosis": None}]
        right = [{**left[0], "diagnosis": "Recall Failure"}]
        self.assertEqual(paired(left, right)["right_only"], 1)
        with self.assertRaises(ValueError):
            paired(left, [{**right[0], "sample_id": "q2"}])
