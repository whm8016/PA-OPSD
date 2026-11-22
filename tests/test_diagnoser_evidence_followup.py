import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.experiments.diagnoser_evidence import build_followup, run


class DiagnoserEvidenceFollowupTest(unittest.TestCase):
    def test_selects_fresh_development_rows_and_keeps_labels_out_of_inputs(self):
        gold = run.gold
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            for name in ('split/development', 'repro', 'adapter', 'previous'):
                (root / name).mkdir(parents=True)
            histories, questions, refs = [], [], []
            for i, label in enumerate(gold.LABELS):
                for j in range(14):
                    qid = 10 * i + j + 1 + 10 * i
                    history = {'uid': str(qid),
                               'question_info': f'[{qid}]\n练习\n知识点: 数学\n时间: 2025-02-04\n结果: 错误',
                               'history_info': ['[1]\n练习\n知识点: 数学\n时间: 2025-01-04\n结果: 正确'],
                               'related_history': []}
                    questions.append({'id': qid, 'info': {'answer': ['42'], 'analysis': '题库解析'}})
                    histories.append(history)
                    refs.append({'sample_id': gold.history_key(history), 'target': {
                        'diagnosis': label, 'strategy': 'reference-only-strategy',
                        'content': 'reference-only-feedback'}})
            ref_path = root / 'split/development/references.jsonl'
            history_path = root / 'repro/gold_inputs.jsonl'
            bank = root / 'bank.jsonl'
            prior_rows = [{'sample_id': row['sample_id']} for i, row in enumerate(refs) if i % 14 < 8]
            gold.write_rows(ref_path, refs)
            gold.write_rows(history_path, histories)
            gold.write_rows(bank, questions)
            gold.write_rows(root / 'previous/inputs.jsonl', prior_rows)
            for name in gold.ADAPTER_FILES:
                (root / 'adapter' / name).write_bytes(b'fixture-adapter')
            gold.write_json(root / 'split/split_manifest.json', {
                'ids': {'validation': [row['sample_id'] for row in refs]},
                'files': {'development/references.jsonl': gold.digest(ref_path)},
            })
            gold.write_json(root / 'repro/preflight.json', {'gold_input_sha256': gold.digest(history_path)})
            gold.write_json(root / 'previous/manifest.json', {
                'scope': 'validation', 'selection_tag': run.team.TAG,
                'ids': [row['sample_id'] for row in prior_rows],
                'files': {'inputs.jsonl': gold.digest(root / 'previous/inputs.jsonl')},
                'question_bank_sha256': gold.digest(bank),
                'gold_sft_adapter_sha256': {name: gold.digest(root / 'adapter' / name) for name in gold.ADAPTER_FILES},
            })
            with patch.multiple(gold, SPLIT=root / 'split', REPRO=root / 'repro', BANK=bank,
                                GOLD_SFT=root / 'adapter'):
                build_followup.build(root / 'previous', root / 'out')
                with self.assertRaises(FileExistsError):
                    build_followup.build(root / 'previous', root / 'out')
            output = root / 'out'
            manifest = gold.read_json(output / 'manifest.json')
            self.assertEqual(manifest['class_counts'], dict.fromkeys(gold.LABELS, 6))
            inputs = run.team.load(output, 'inputs.jsonl', manifest)
            features = run.team.load(output, 'features.jsonl', manifest)
            self.assertEqual(set(inputs), {r['sample_id'] for r in refs} - {p['sample_id'] for p in prior_rows})
            self.assertEqual(set(inputs), set(features))
            for key, row in inputs.items():
                self.assertEqual(row.keys(), {'sample_id', 'current', 'history', 'solution'})
                for arm in run.ARMS:
                    prompt = run.messages(row, features[key], arm)[1]['content']
                    self.assertIn('题库解析', prompt)
                    self.assertNotIn('reference-only', prompt)
