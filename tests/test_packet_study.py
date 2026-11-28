import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.experiments.training_free_memory.packet_followup import altered
from scripts.experiments.training_free_memory.packet_policy import group_folds
from scripts.experiments.training_free_memory.packet_results import (
    answer,
    paired,
    reviewed_rows,
)
from scripts.experiments.training_free_memory.prepare import write_rows


class PacketStudyTest(unittest.TestCase):
    def test_blind_mapping_rejects_duplicate_cases_and_negative_positions(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            questions = [{"id": k, "answers": ["yes"]} for k in ("a", "b")]
            write_rows(root / "blind_1.jsonl", questions)
            write_rows(
                root / "verdict_1.jsonl", [{"id": k, "scores": [1]} for k in ("a", "b")]
            )
            for ids, index in ((("a", "a"), 0), (("a", "b"), -1)):
                write_rows(
                    root / "identity.jsonl",
                    [{"id": k, "roles": {"control": index}} for k in ids],
                )
                with self.assertRaises(ValueError):
                    reviewed_rows((root,))

    def test_official_format_failure_keeps_empty_positional_answers(self):
        failed = {"memory": [{"answer": ""}] * 3}
        self.assertEqual(answer(failed, 2), "")
        self.assertEqual(answer({"memory": []}, 3), "")
        self.assertEqual(
            answer({"memory": [{"answer": "first"}, {"answer": "second"}]}, 2), "second"
        )

    def test_connected_student_and_question_stay_in_same_fold(self):
        rows = [
            {"key": key, "qid": qid}
            for key, qid in (
                ("s1||a", "q1"),
                ("s1||b", "q2"),
                ("s2||c", "q2"),
                ("s3||d", "q3"),
            )
        ]
        folds = group_folds(rows)
        self.assertTrue(any(set(x) == {0, 1, 2} for x in folds))
        self.assertEqual(sorted(i for group in folds for i in group), list(range(4)))

    def test_repeated_question_cluster_uses_paired_cases(self):
        result = paired([(0, 1), (1, 1), (0, 1)], ["q1", "q1", "q2"])
        self.assertEqual(
            (result["n"], result["clusters"], result["improved"], result["worsened"]),
            (3, 2, 2, 0),
        )

    def test_intervention_changes_value_but_preserves_other_ids(self):
        self.assertEqual(altered("ids", "[328], [335]"), "[329], [335]")
        self.assertEqual(altered("count", "7"), "8")
