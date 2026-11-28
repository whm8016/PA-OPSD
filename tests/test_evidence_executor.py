import unittest

from scripts.experiments.diagnoser_evidence.evidence import fields
from scripts.experiments.training_free_memory.evidence_executor import execute


def event(qid):
    return fields(f"[{qid}]\n知识点: 面积\n时间: 2021-01-02T13:00:00\n结果: 正确")


class EvidenceExecutorTest(unittest.TestCase):
    def test_history_query_excludes_current(self):
        answer = execute(
            "历史中包含“面积”知识点的题目共有多少道？",
            1,
            [(1, event(100)), (0, event(200))],
        )
        self.assertEqual(answer, {"operator": "count", "answer": "1", "positions": [1]})

    def test_explicit_current_is_eligible(self):
        answer = execute("记录[200]的知识点是什么？", 0, [(0, event(200))])
        self.assertEqual(answer["positions"], [0])

    def test_compound_projection_abstains(self):
        self.assertIsNone(
            execute("在记录[100]里，题干和作答结果分别是什么？", 0, [(1, event(100))])
        )

    def test_duplicate_record_preserves_two_events(self):
        answer = execute(
            "包含“面积”知识点的题目共尝试几次？", 1, [(1, event(100)), (2, event(100))]
        )
        self.assertEqual(answer["positions"], [1, 2])
