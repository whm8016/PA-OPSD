import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.experiments.diagnoser_evidence import evidence, run


def record(qid, day, result, concepts='具体知识点, 数学'):
    return f'[{qid}]\n练习\n知识点: {concepts}\n时间: 2025-01-{day}\n结果: {result}'


class DiagnoserEvidenceTest(unittest.TestCase):
    def test_future_subsecond_record_never_enters_statistics(self):
        row = {'sample_id': 'fixture', 'current': record(3, '04T11:01:00', '错误'),
               'history': [record(2, '04T11:01:00.001', '正确')]}
        with self.assertRaisesRegex(ValueError, 'after the current exercise'):
            evidence.build_features(row)

    def test_same_directory_cannot_overwrite_frozen_inputs(self):
        with TemporaryDirectory() as folder:
            same = Path(folder)
            with self.assertRaisesRegex(ValueError, 'different directories'):
                run.build(same, same)

    def test_counts_use_all_history_and_short_prompt_preserves_original_citations(self):
        row = {'sample_id': 'fixture', 'current': record(3, '31', '错误'),
               'history': [record(i, '01', '正确') for i in range(4, 24)],
               'solution': {'answer': ['42'], 'analysis': '题库解析'}}
        features = evidence.build_features(row)
        self.assertEqual(features['summary']['共享知识点'][0]['正确数'], 20)
        self.assertEqual(features['summary']['共享知识点'][0]['错误数'], 0)
        self.assertEqual(features['summary']['共享知识点'][0]['距当前天数'], 30)
        self.assertEqual(len(features['history_positions']), 12)
        for arm in run.ARMS:
            messages = run.messages(row, features, arm)
            self.assertTrue(messages[1]['content'].endswith(run.team.INSTRUCTIONS['diagnoser']))
        compact = run.messages(row, features, run.ARMS[2])[1]['content']
        self.assertIn('历史序号 20\n[23]', compact)
        self.assertNotIn('历史序号 1\n[4]', compact)
        raw = json.dumps({'diagnosis': 'Recall Failure', 'evidence': [1, 20]})
        self.assertEqual(run.parse(raw, row, features, run.ARMS[2])['evidence'], [])
        self.assertEqual(run.parse(raw, row, features, run.ARMS[0])['evidence'], [1, 20])
