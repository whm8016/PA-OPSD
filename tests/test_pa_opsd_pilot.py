import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_pa_opsd_pilot import build_candidate, memory_answer, visible_records

VISIBLE = """### INTERACTION HISTORY
[12]
单选: 已知长度是7米．
知识点: 几何, 精确标签
时间: 2021-04-17T01:00:00
结果: 正确
[12]
单选: 已知长度是7米．
知识点: 几何, 精确标签
时间: 2021-04-17T01:00:00
结果: 错误
[18]
填空: 如图。 question_18-image_0
知识点: 精确标签拓展
时间: 2021-04-18T02:00:00.125
结果: 正确
### CURRENT QUESTION
[2]
填空: 当前题。
知识点: 几何, 精确标签
时间: 2021-04-18T02:00:00.250
结果: 错误
### MEMORY QUERIES
- Q1: ...
"""


class PilotMemoryTests(unittest.TestCase):
    def test_counts_attempts_and_excludes_current_by_default(self):
        indexed = visible_records(VISIBLE)
        plan = {"kind": "aggregate", "tags": ["精确标签"], "date": ["2021-04-17"],
                "fields": ["total", "ids"]}
        answer, records = memory_answer(indexed, plan)
        self.assertEqual(answer, "共2条；[12]、[12]。")
        self.assertEqual([r["history_index"] for r in records], [1, 2])
        plan.update(scope="both", distinct=True, order="id")
        plan.pop("date")
        self.assertEqual(memory_answer(indexed, plan)[0], "共2条；[2]、[12]。")

    def test_zero_and_unknown_have_different_evidence(self):
        indexed = visible_records(VISIBLE)
        answer, records = memory_answer(indexed, {"kind": "aggregate", "tags": ["不存在"], "fields": ["total", "ids"]})
        self.assertEqual((answer, records), ("共0条；无。", []))
        absence = {"kind": "unknown", "id": "18", "basis": "只显示图片占位符"}
        self.assertEqual(memory_answer(indexed, absence)[0], "Unknown")
        with self.assertRaises(ValueError):
            memory_answer(indexed, {**absence, "id": "99"})

    def test_exact_quote_and_disallow_ambiguous_or_fabricated_quote(self):
        indexed = visible_records(VISIBLE)
        plan = {"kind": "quote", "id": "12", "history_index": 1, "text": "长度是7米"}
        self.assertEqual(memory_answer(indexed, plan)[0], "长度是7米")
        with self.assertRaises(ValueError):
            memory_answer(indexed, {**plan, "text": "长度是8米"})
        with self.assertRaises(ValueError):
            memory_answer(indexed, {k: v for k, v in plan.items() if k != "history_index"})
        parts = {"kind": "quote", "id": "12", "history_index": 1, "parts": ["长度是7米", "结果: 正确"]}
        self.assertEqual(memory_answer(indexed, parts)[0], "长度是7米\n结果: 正确")
        with self.assertRaises(ValueError):
            memory_answer(indexed, {**parts, "parts": ["长度是7米", "结果: 错误"]})

    def test_invalid_query_fields_do_not_silently_widen_scope(self):
        indexed = visible_records(VISIBLE)
        with self.assertRaises(ValueError):
            memory_answer(indexed, {"kind": "aggregate", "tags": ["精确标签"], "data": ["2021-04-17"], "fields": ["total"]})

    def test_dates_and_times_are_inclusive_and_last_tie_uses_display_order(self):
        plan = {"kind": "aggregate", "tags": ["精确标签"], "date_range": ["2021-04-17", "2021-04-17"],
                "fields": ["latest", "last"]}
        answer, records = memory_answer(visible_records(VISIBLE.replace("[12]", "[13]", 1)), plan)
        self.assertEqual(answer, "最近一次：[12] 2021-04-17T01:00:00；末条：[12] 2021-04-17T01:00:00。")
        self.assertEqual([r["history_index"] for r in records], [1, 2])

    def test_absence_checks_the_named_literal(self):
        plan = {"kind": "absence", "id": "18", "basis": "check marker", "absent_literal": "question_18-image_0"}
        with self.assertRaisesRegex(ValueError, "present"):
            memory_answer(visible_records(VISIBLE), plan)

    def test_each_reviewed_plan_is_bound_to_original_question(self):
        plans = [{"query": f"Q{i}", "plan": {"kind": "aggregate", "fields": ["total"]}} for i in range(1, 4)]
        spec = {"memory": [plans[1], plans[0], plans[2]]}
        with (
            patch("build_pa_opsd_pilot.memory_queries", return_value=["Q1", "Q2", "Q3"]),
            patch("build_pa_opsd_pilot.student_messages", return_value=[{}, {"content": VISIBLE}]),
            self.assertRaisesRegex(ValueError, "Q1 plan differs"),
        ):
            build_candidate({"split": "train"}, spec)


if __name__ == "__main__":
    unittest.main()
