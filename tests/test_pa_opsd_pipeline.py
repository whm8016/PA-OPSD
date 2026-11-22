import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_pa_opsd_teacher import (
    evidence_issues,
    history_evidence,
    load_records,
    memory_facts,
    reflect,
    resolve_evidence,
    synthesize,
    synthesize_single,
    validate_verdict,
)
from pa_opsd_contract import (
    XES,
    audit_seeds,
    memory_queries,
    official_module,
    read_jsonl,
    student_messages,
    validate_target,
)

SEED = ROOT / "data/pa_opsd/v0.2_question_disjoint_weak_seed/all.jsonl"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(SEED)
        cls.seed = cls.rows[0]

    def target(self):
        annotation = self.seed["teacher_privileged"]["longtutor_weak_annotation"]
        facts = memory_facts(student_messages(self.seed)[1]['content'], memory_queries(self.seed))
        memory = [{'id': f['id'], 'answer': f.get('question_text') or (
            ' '.join(f['record_ids']) + f" 共{f['total']}条，正确{f['correct']}条" if 'record_ids' in f else 'Unknown')}
                  for f in facts]
        return {"memory": memory,
                "diagnosis": annotation["diagnosis"], "strategy": annotation["strategy"],
                "reason": "K/M证据待核验", "content": "先回想相关概念。"}

    def test_real_seed_isolation(self):
        audit = audit_seeds(self.rows)
        self.assertEqual(audit["rows"], 564)
        self.assertEqual(sum(r["split"] == "train" for r in self.rows), 507)
        self.assertEqual(sum(r["split"] == "validation" for r in self.rows), 57)

    def test_memory_scope_handles_repeated_attempts_and_explicit_current_record(self):
        visible = """### HISTORY
[51]
知识点: 计数模块, 树形图
时间: 2021-01-05T05:00:00
结果: 正确
[51]
知识点: 计数模块, 树形图
时间: 2021-01-06T05:00:00
结果: 错误
[52]
知识点: 计数模块, 树形图进阶
时间: 2021-01-06T05:00:00
结果: 正确
### CURRENT QUESTION
[99]
知识点: 计数模块, 树形图
时间: 2021-01-07T05:00:00
结果: 正确
### MEMORY QUERIES"""
        queries = [
            "仅根据提供的历史记录，知识点包含“树形图”的题目一共有多少道？",
            "在提供的历史记录中（包括当前记录），列出知识点包含“树形图”的题目编号及结果。",
            "请列出知识点包含“树形图”的所有正确记录编号。",
        ]
        facts = memory_facts(visible, queries)
        self.assertEqual([fact["record_ids"] for fact in facts], [["51", "51"], ["51", "51", "99"], ["51"]])
        self.assertEqual([fact["require_record_ids"] for fact in facts], [False, True, True])
        target = self.target()
        target["strategy"] = "Conceptual Explanation"
        target["memory"] = [
            {"id": "Q1", "answer": "共2道"},
            {"id": "Q2", "answer": "[51]正确、[99]正确"},
            {"id": "Q3", "answer": "[51]、[52]"},
        ]
        self.assertEqual(evidence_issues(target, visible, queries), [
            "memory_record_list_incomplete:Q2",
            "memory_record_list_outside_scope:Q3",
        ])
        target["memory"] = [
            {"id": "Q1", "answer": "共2道"},
            {"id": "Q2", "answer": "包含当前题[99]，共3次，历史共2次：[51]（两次）、[99]。"},
            {"id": "Q3", "answer": "[51]"},
        ]
        self.assertEqual(evidence_issues(target, visible, queries), [])

    def test_memory_scope_abstains_when_query_has_an_unresolved_filter(self):
        visible = ("[1]\n知识点: 计数模块, 树形图\n时间: 2021-01-06T05:00:00\n结果: 正确\n"
                   "### CURRENT QUESTION\n[2]\n结果: 错误\n### MEMORY QUERIES")
        queries = [
            "列出知识点包含“树形图”（但排除课外练习）的记录编号。",
            "列出知识点包含“树形图”的不同题目编号。",
            "列出知识点包含“树形图”的正确或错误题目的编号。",
            "请列出知识点包含“树形图”的题目编号（不包括给每边人数的题）。",
            "根据记录[1]的题干，给出图中标注的数值。",
        ]
        self.assertTrue(all(not fact["supported"] for fact in memory_facts(visible, queries)))

    def test_memory_scope_verifies_numbers_and_record_outcomes_on_real_training_seed(self):
        row = next(r for r in self.rows if r["split"] == "train" and "::q2818::" in r["sample_id"])
        visible = student_messages(row)[1]["content"]
        queries = memory_queries(row)
        facts = memory_facts(visible, queries)
        self.assertEqual(facts[1]["record_ids"], ["3843", "5013", "3844"])
        target = {"memory": [
            {"id": "Q1", "answer": facts[0].get("question_text", "Unknown")},
            {"id": "Q2", "answer": "共3道：[3843]正确、[5013]错误、[3844]正确。"},
            {"id": "Q3", "answer": "Unknown"},
        ], "strategy": "Conceptual Explanation", "content": "再想一步。", "reason": "暂定。"}
        self.assertIn("memory_record_result_mismatch:Q2", evidence_issues(target, visible, queries))

    def test_memory_scope_with_outcome_counts_and_correct_only_ids(self):
        visible = (
            "[51]\n知识点: 计数模块, 树形图\n时间: 2021-01-05T05:00:00\n结果: 正确\n"
            "[51]\n知识点: 计数模块, 树形图\n时间: 2021-01-05T05:01:00\n结果: 错误\n"
            "[52]\n知识点: 计数模块, 树形图进阶\n时间: 2021-01-05T05:02:00\n结果: 正确\n"
            "[53]\n知识点: 计数模块, 树形图\n时间: 2021-01-06T05:00:00\n结果: 正确\n"
            "### CURRENT QUESTION\n[99]\n知识点: 树形图\n时间: 2021-01-05T06:00:00\n结果: 错误\n"
            "### MEMORY QUERIES"
        )
        queries = [
            '在“计数模块, 树形图”这一特定知识点下，统计学生在 2021-01-05 提交的记录中，正确和错误的题目数量各是多少？',
            '基于当前提供的历史记录，统计你在“树形图”知识点下作答正确与错误的次数，并列出作对题目的记录编号。',
            '仅查看2021-01-05这一天且知识点包含“树形图”的记录，请问学生共尝试了多少道题？其中结果为“正确”的题目编号有哪些？',
        ]
        facts = memory_facts(visible, queries)
        self.assertEqual([(f['total'], f['correct'], f['record_ids']) for f in facts], [
            (2, 1, ['51', '51']), (3, 2, ['51', '53']), (2, 1, ['51']),
        ])
        target = {'strategy': 'Conceptual Explanation', 'reason': '暂定。', 'content': '再想一步。',
                  'memory': [{'id': f'Q{i}', 'answer': answer} for i, answer in enumerate(
                      ('正确：1，错误：1', '答对2次，答错1次，正确题号：[51]、[53]。', '共2道，正确题号：[51]。'), 1)]}
        self.assertEqual(evidence_issues(target, visible, queries), [])
        target['memory'][0]['answer'] = '正确：2，错误：1'
        target['memory'][1]['answer'] = '答对2次，[51]、[53]。'
        target['memory'][2]['answer'] = '共2道，正确题号：[51]、[99]。'
        issues = evidence_issues(target, visible, queries)
        self.assertIn('memory_count_mismatch:Q1', issues)
        self.assertIn('memory_requested_count_missing:Q2', issues)
        self.assertIn('memory_record_list_outside_scope:Q3', issues)
        target['memory'][1]['answer'] = '正确2次，错误1次，[51]错误、[53]正确。'
        self.assertIn('memory_record_result_mismatch:Q2', evidence_issues(target, visible, queries))
        queries[2] = queries[2].replace('2021-01-05', '2021-01-07')
        empty = memory_facts(visible, queries)[2]
        self.assertEqual((empty['total'], empty['record_ids']), (0, []))
        self.assertIn('memory_record_list_outside_scope:Q3', evidence_issues(target, visible, queries))

    def test_memory_scope_checks_timestamp_against_its_record_id(self):
        visible = (
            "[51]\n知识点: 一元一次方程\n时间: 2021-01-05T05:00:00\n结果: 正确\n"
            "[52]\n知识点: 一元一次方程\n时间: 2021-01-05T05:02:00\n结果: 正确\n"
            "### CURRENT QUESTION\n[99]\n结果: 错误\n### MEMORY QUERIES"
        )
        queries = ['在提供的历史记录中，列出知识点包含“一元一次方程”且结果为“正确”的所有记录编号及其对应的时间戳。']
        target = {'strategy': 'Conceptual Explanation', 'reason': '暂定。', 'content': '再想一步。',
                  'memory': [{'id': 'Q1', 'answer': '[51] 2021-01-05T05:00:00、[52]（2021-01-05 05:02:00）'}]}
        self.assertTrue(memory_facts(visible, queries)[0]['require_record_timestamps'])
        self.assertEqual(evidence_issues(target, visible, queries), [])
        target['memory'][0]['answer'] = '[51] 时间: 2021-01-05T05:00:00、[52]（2021-01-05 05:02:00）'
        self.assertEqual(evidence_issues(target, visible, queries), [])
        target['memory'][0]['answer'] = '[51] 2021-01-05T05:02:00、[52] 2021-01-05T05:00:00'
        self.assertEqual(evidence_issues(target, visible, queries), ['memory_record_timestamp_mismatch:Q1'])
        target['memory'][0]['answer'] = '[51] 2021-01-05T05:00:00+08:00、[52] 2021-01-05T05:02:00'
        self.assertEqual(evidence_issues(target, visible, queries), ['memory_record_timestamp_mismatch:Q1'])
        target['memory'][0]['answer'] = '[51] 2021-01-05T05:00:00、[52] 2021-01-05T05:02:00、另有2021-01-04T05:02:00'
        self.assertEqual(evidence_issues(target, visible, queries), ['memory_record_timestamp_mismatch:Q1'])

    def test_memory_scope_for_explicit_date_ranges_and_alternative_tags_on_train(self):
        expected = {'q447': ['4645', '3282', '2388', '1320', '4653', '2407', '2408'],
                    'q3815': ['307'], 'q653': ['3177', '1464']}
        seen = set()
        for row in self.rows:
            qid = row['sample_id'].split('::')[2]
            if qid in expected and row['split'] == 'train' and qid not in seen:
                visible = student_messages(row)[1]['content']
                queries = memory_queries(row)
                facts = memory_facts(visible, queries)
                self.assertEqual(facts[1]['record_ids'], expected[qid])
                if qid == 'q447':
                    target = self.target()
                    target['memory'][1]['answer'] = '、'.join(f'[{i}]' for i in expected[qid]) + '，共7道。'
                    target['content'] = '你是用“面积÷长=宽”求出宽的。'
                    issues = evidence_issues(target, visible, queries)
                    self.assertNotIn('memory_requested_count_missing:Q2', issues)
                    self.assertIn('unobserved_student_method:content', issues)
                    target['content'] = '如果你是用方程的方法，先想一想未知数是什么。'
                    self.assertNotIn('unobserved_student_method:content', evidence_issues(target, visible, queries))
                    answer = '共7道，编号：' + '、'.join(reversed(expected[qid]))
                    target = self.target()
                    target['memory'][1]['answer'] = answer
                    self.assertIn('memory_record_order_mismatch:Q2', evidence_issues(target, visible, queries))
                    target['memory'][1]['answer'] = '共7道，编号：' + '、'.join(expected[qid] + ['999999'])
                    self.assertIn('memory_record_list_outside_scope:Q2', evidence_issues(target, visible, queries))
                    target['memory'][1]['answer'] = '共7道，' + '、'.join(f'[{i}]' for i in expected[qid]) + '；另有编号：999999'
                    self.assertIn('memory_record_list_outside_scope:Q2', evidence_issues(target, visible, queries))
                    target['memory'][1]['answer'] = '共7道，编号：' + '、'.join(f'[{i}]' for i in expected[qid]) + '、999999'
                    self.assertIn('memory_record_list_outside_scope:Q2', evidence_issues(target, visible, queries))
                elif qid == 'q3815':
                    target = self.target()
                    target['memory'][1]['answer'] = '错误次数1次；错误记录为：[307]。'
                    self.assertFalse(any(issue.endswith(':Q2') for issue in evidence_issues(target, visible, queries)))
                    target['memory'][1]['answer'] = '错误次数1次；错误记录为：[307]（99999次）。'
                    self.assertIn('memory_record_list_outside_scope:Q2', evidence_issues(target, visible, queries))
                elif qid == 'q653':
                    target = self.target()
                    target['memory'][0]['answer'] = 'Unknown'
                    self.assertIn('memory_question_text_mismatch:Q1', evidence_issues(target, visible, queries))
                    target['reason'] = '过去两题都做对了，说明概念已具备。'
                    self.assertTrue(any(issue.startswith('unsupported_mastery:')
                                        for issue in evidence_issues(target, visible, queries)))
                    target['reason'] = '两次答对不足以说明你已具备独立解题的能力。'
                    self.assertFalse(any(issue.startswith('unsupported_mastery:')
                                         for issue in evidence_issues(target, visible, queries)))
                seen.add(qid)
        self.assertEqual(seen, set(expected))

    def test_identity_and_query_tamper_rejected(self):
        for field in ("current_question", "uid"):
            bad = copy.deepcopy(self.seed)
            bad["student_input"][field] = "TAMPER"
            with self.assertRaises(ValueError):
                audit_seeds([bad])
        bad = copy.deepcopy(self.seed)
        bad["teacher_privileged"]["longtutor_weak_annotation"]["memory"][0]["query"] = "TAMPER"
        with self.assertRaises(ValueError):
            audit_seeds([bad])

    def test_exact_official_prompt(self):
        sample = next(r for r in read_jsonl(XES / "history_features_lastq_scale.jsonl")
                      if str(r["uid"]) == str(self.seed["student_input"]["uid"]) and
                      r["question_info"] == self.seed["student_input"]["current_question"])
        annotation = {"memory": [{"query": q} for q in memory_queries(self.seed)]}
        expected = official_module()._build_prompts(sample, annotation,
                      history_mode="long", use_feature=False, use_teach=False)
        self.assertEqual([m["content"] for m in student_messages(self.seed)], list(expected))

    def test_privileged_sentinel_absent(self):
        bad = copy.deepcopy(self.seed)
        bad["teacher_privileged"]["standard_answer"] = ["SECRET_SENTINEL_991"]
        bad["teacher_privileged"]["expert_analysis"] = "SECRET_SENTINEL_991"
        bad["teacher_privileged"]["longtutor_weak_annotation"]["memory"][0]["answer"] = "SECRET_SENTINEL_991"
        self.assertNotIn("SECRET_SENTINEL_991", json.dumps(student_messages(bad)))

    def test_prompt_hash_stable_across_processes(self):
        script = ("import sys;sys.path.insert(0,'scripts');from pa_opsd_contract import *;"
                  f"print(fingerprint(student_messages(read_jsonl({str(SEED)!r})[0])))")
        hashes = []
        for seed in ("1", "27"):
            hashes.append(subprocess.check_output([sys.executable, "-c", script], cwd=ROOT,
                           env={**os.environ, "PYTHONHASHSEED": seed}, text=True).strip())
        self.assertEqual(hashes[0], hashes[1])

    def test_strict_target(self):
        target = self.target()
        validate_target(target, memory_queries(self.seed))
        for mutation in (lambda t: t["memory"].pop(),
                         lambda t: t["memory"][0].update(id="Q2"),
                         lambda t: t.update(strategy="INVALID"),
                         lambda t: t.update(query="OUTPUT_FORBIDDEN")):
            broken = copy.deepcopy(target)
            mutation(broken)
            with self.assertRaises(ValueError):
                validate_target(broken, memory_queries(self.seed))

    def test_bounded_api_failure_no_success(self):
        calls = []
        def failure(messages):
            calls.append(messages)
            raise RuntimeError("fixture_failure")
        result = synthesize(self.seed, failure, failure, 2)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("messages", result)

    def test_history_evidence_is_exact_and_excludes_current_question(self):
        visible = student_messages(self.seed)[1]["content"]
        records = history_evidence(visible)
        self.assertEqual(len(records), 199)
        self.assertEqual([r["history_index"] for r in records], list(range(1, 200)))
        self.assertTrue(all(r["text"] in visible.split("### CURRENT QUESTION")[0] for r in records))
        resolved = resolve_evidence({"evidence_indices": [1, 199]}, records)
        self.assertEqual(resolved["evidence"], [records[0]["text"], records[-1]["text"]])
        self.assertNotIn("evidence_indices", resolved)
        for invalid in ([], [0], [200], [True], ["1"], [1, 1]):
            with self.assertRaisesRegex(ValueError, "invalid_history_evidence_indices"):
                resolve_evidence({"evidence_indices": invalid}, records)
        with self.assertRaisesRegex(ValueError, "history_records_missing"):
            history_evidence("### INTERACTION HISTORY\n### CURRENT QUESTION\n[1]\nnot history")

    def test_mock_stages_only_not_real_api_evidence(self):
        annotation = self.seed["teacher_privileged"]["longtutor_weak_annotation"]
        replies = iter([{"diagnosis": annotation["diagnosis"], "reason": "K/M", "evidence_indices": [1]}, self.target()])
        teacher_calls = []
        def teacher(messages):
            request = json.loads(messages[1]["content"])
            teacher_calls.append(request)
            if len(teacher_calls) == 1:
                original = student_messages(self.seed)
                history, current = original[1]["content"].split("### CURRENT QUESTION", 1)
                self.assertEqual(request["history_records"], history_evidence(original[1]["content"]))
                self.assertEqual(request["student_messages"], [original[0],
                    {"role": "user", "content": "### CURRENT QUESTION" + current}])
                self.assertNotIn(history.strip(), request["student_messages"][1]["content"])
            else:
                self.assertEqual(request["student_messages"], student_messages(self.seed))
            return json.dumps(next(replies), ensure_ascii=False), {"model": "mock", "usage": None}
        def verifier(messages):
            request = json.loads(messages[1]["content"])
            self.assertNotIn("diagnoser", request)
            self.assertNotIn("student_messages", request)
            self.assertEqual(request["student_visible_text"], student_messages(self.seed)[1]["content"])
            matches = [r["question_id"] for r in request["visible_record_index"] if "图形认知" in r["knowledge_tags"]]
            self.assertEqual(matches, ["870", "5193", "914", "1957", "330", "331", "4613", "1022", "1022", "4643", "3281", "4644"])
            return json.dumps({"pass": True, "grounded": True, "no_early_answer_leak": True,
                               "strategy_aligned": True, "memory_grounded": True,
                               "appropriate": True, "issues": []}), {"model": "mock", "usage": None}
        result = synthesize(self.seed, teacher, verifier, 1)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(result["stages"]), 3)

    def test_verifier_cannot_pass_with_issues_or_failed_checks(self):
        valid = {"pass": True, "grounded": True, "no_early_answer_leak": True,
                 "strategy_aligned": True, "memory_grounded": True, "appropriate": True,
                 "issues": []}
        validate_verdict(valid)
        for field in ("pass", "grounded", "no_early_answer_leak", "strategy_aligned",
                      "memory_grounded", "appropriate"):
            with self.assertRaises(ValueError):
                validate_verdict({**valid, field: False})
        with self.assertRaises(ValueError):
            validate_verdict({**valid, "issues": ["confirmed_defect"]})
        with self.assertRaises(ValueError):
            validate_verdict({**valid, "grounded": 1})

    def test_false_positive_verifier_cannot_admit_unsupported_mastery(self):
        target = self.target()
        target['reason'] = '历史题正确，说明已掌握基本模型。'
        replies = iter([{'diagnosis': target['diagnosis'], 'reason': 'K/M',
                         'evidence_indices': [1]}, target])
        def teacher(messages):
            return json.dumps(next(replies), ensure_ascii=False), {}
        def verifier(messages):
            return json.dumps({'pass': True, 'grounded': True, 'no_early_answer_leak': True,
                               'strategy_aligned': True, 'memory_grounded': True,
                               'appropriate': True, 'issues': []}), {}
        result = synthesize(self.seed, teacher, verifier, 1)
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('unsupported_mastery', json.dumps(result['errors']))

    def test_evidence_gate_preserves_uncertainty_and_rejects_observed_method_claim(self):
        target = self.target()
        target.update(diagnosis='Conceptual Gap', strategy='Conceptual Explanation',
                      reason='仅有正确记录，不能认定已掌握。M为待验证假设。')
        visible = student_messages(self.seed)[1]['content']
        self.assertEqual(evidence_issues(target, visible, memory_queries(self.seed)), [])
        target['content'] = '你是先求1人1小时做几件，再乘人数。'
        self.assertTrue(any('unobserved_student_method' in x for x in evidence_issues(
            target, visible, memory_queries(self.seed))))
        for instruction in ('你把总数除以人数，能得到什么？', '请你把总数除以人数，先告诉我结果。'):
            target['content'] = instruction
            self.assertFalse(any('unobserved_student_method' in x for x in evidence_issues(
                target, visible, memory_queries(self.seed))))

    def test_single_teacher_uses_one_teacher_call_and_keeps_official_student_prompt(self):
        calls = []
        target = self.target()
        target.update(diagnosis='Conceptual Gap', strategy='Conceptual Explanation')
        def teacher(messages):
            calls.append(messages)
            payload = json.loads(messages[1]['content'])
            self.assertEqual(payload['student_messages'], student_messages(self.seed))
            self.assertNotIn('longtutor_weak_annotation', payload['privileged_z'])
            return json.dumps(target), {}
        def verifier(messages):
            return json.dumps({'pass': True, 'grounded': True, 'no_early_answer_leak': True,
                               'strategy_aligned': True, 'memory_grounded': True,
                               'appropriate': True, 'issues': []}), {}
        result = synthesize_single(self.seed, teacher, verifier, 1)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(len(calls), 1)
        self.assertEqual([s['stage'] for s in result['stages']], ['SingleTeacher', 'Verifier'])
        self.assertEqual(result['messages'][:2], student_messages(self.seed))

    def test_known_false_accepts_are_rejected_by_visible_memory_and_answer_checks(self):
        visible = ('### INTERACTION HISTORY\n[101]\n填空: 求数。\n知识点: 和差倍问题\n'
                   '时间: 2020-11-14T10:00:00\n结果: 正确\n'
                   '[102]\n填空: 求数。\n知识点: 和差倍问题\n'
                   '时间: 2020-11-14T11:00:00\n结果: 错误\n'
                   '### CURRENT QUESTION\n[103]\n填空: 分草莓。\n知识点: 盈亏问题\n'
                   '时间: 2021-05-13T12:00:00\n结果: 错误\n### MEMORY QUERIES')
        queries = ['无可见答案', '与“和差倍问题”相关的记录，2020-11-14共做了几道，答对几道？请列题号。', '无可见答案']
        target = self.target()
        target.update(diagnosis='Conceptual Gap', strategy='Conceptual Explanation',
                      reason='M是待验证假设。', content='草莓总数=24个。')
        target['memory'] = [{'id': 'Q1', 'answer': 'Unknown'},
                            {'id': 'Q2', 'answer': '共1道，答对1道，101'}, {'id': 'Q3', 'answer': 'Unknown'}]
        issues = evidence_issues(target, visible, queries, {'standard_answer': ['$$24$$']})
        self.assertIn('current_answer_exposed', issues)
        self.assertTrue(any('memory_record_list_incomplete' in issue for issue in issues))
        self.assertIn('memory_count_mismatch:Q2', issues)
        target['content'] = '比如3×8=24。现在请说说本题第一步如何想？'
        self.assertNotIn('current_answer_exposed', evidence_issues(
            target, visible, queries, {'standard_answer': ['$$24$$']}))
        target['content'] = '先核对中间量=24.5千克。'
        self.assertNotIn('current_answer_exposed', evidence_issues(
            target, visible, queries, {'standard_answer': ['$$24$$']}))

    def test_review_regressions_question_reference_and_reason_leak(self):
        visible = ('### INTERACTION HISTORY\n[101]\n填空: 求数。\n知识点: 数学\n'
                   '时间: 2020-11-14T10:00:00\n结果: 正确\n'
                   '### CURRENT QUESTION\n[103]\n填空: 求数。\n### MEMORY QUERIES')
        target = self.target()
        target.update(diagnosis='Conceptual Gap', strategy='Conceptual Explanation', reason='M为假设。',
                      content='你已经掌握这个概念了吗？')
        queries = ['未提供', '未提供', '未提供']
        self.assertEqual(evidence_issues(target, visible, queries), [])
        target.update(diagnosis='Transfer Deficit', strategy='Analogical Transfer',
                      content='回想第101题中答对的同类任务，先说说怎样找总数？')
        self.assertEqual(evidence_issues(target, visible, queries), [])
        target['reason'] = 'M为假设。本题答案=24。'
        self.assertIn('current_answer_exposed', evidence_issues(
            target, visible, queries, {'standard_answer': ['$$24$$']}))

    def test_verifier_issues_reach_both_teacher_roles_on_retry(self):
        diagnosis = self.seed["teacher_privileged"]["longtutor_weak_annotation"]["diagnosis"]
        replies = iter([{"diagnosis": diagnosis, "reason": "K/M", "evidence_indices": [1]},
                        self.target(), {"diagnosis": diagnosis, "reason": "K/M", "evidence_indices": [1]},
                        self.target()])
        requests = []
        def teacher(messages):
            requests.append(json.loads(messages[1]["content"]))
            return json.dumps(next(replies)), {}
        verdicts = iter([{"pass": False, "issues": ["fixture_missing_image_value"]},
                         {"pass": True, "grounded": True, "no_early_answer_leak": True,
                          "strategy_aligned": True, "memory_grounded": True,
                          "appropriate": True, "issues": []}])
        def verifier(messages):
            return json.dumps(next(verdicts)), {}
        result = synthesize(self.seed, teacher, verifier, 2)
        self.assertEqual(result["status"], "accepted")
        for request in requests[2:]:
            self.assertEqual(request["previous_errors"][0]["verifier_issues"], ["fixture_missing_image_value"])

    def test_interrupted_append_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "journal.jsonl"
            path.write_bytes(b'{"sample_id":"one"}\n{"sample')
            self.assertEqual(load_records(path), [{"sample_id": "one"}])
            self.assertEqual(path.read_bytes(), b'{"sample_id":"one"}\n')
            path.write_bytes(b'broken\n')
            with self.assertRaises(ValueError):
                load_records(path)

    def test_dry_run_resume_and_config_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dry.jsonl"
            cmd = [sys.executable, str(ROOT / "scripts/generate_pa_opsd_teacher.py"),
                   "--input", str(SEED), "--output", str(path), "--limit", "2", "--dry-run"]
            subprocess.run(cmd, check=True, capture_output=True)
            first = path.read_bytes()
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(first, path.read_bytes())
            records = read_jsonl(path)
            self.assertTrue(all(r["dry_run"] and r["status"] == "contract_only" and "messages" not in r for r in records))
            forbidden = subprocess.run(cmd + ["--accepted-output", str(Path(folder) / "accepted.jsonl")],
                                       capture_output=True, check=False)
            self.assertNotEqual(forbidden.returncode, 0)
            self.assertFalse((Path(folder) / "accepted.jsonl").exists())
            same_path = subprocess.run(cmd + ["--accepted-output", str(path)], capture_output=True, check=False)
            self.assertNotEqual(same_path.returncode, 0)
            changed = subprocess.run(cmd + ["--teacher-model", "changed"], capture_output=True, check=False)
            self.assertNotEqual(changed.returncode, 0)

    def test_reflection_rejects_validation_only(self):
        with self.assertRaises(ValueError):
            reflect([{"split": "validation", "status": "accepted", "sample_id": "x"}], None)


if __name__ == "__main__":
    unittest.main()
