import json

import pytest

from scripts.experiments.diagnosis_probe import baseline, baseline_summary, offline, run


def test_prefix_only_uses_top_level_diagnosis_after_memory():
    memory = json.dumps({'memory': [{'id': 'Q1', 'answer': '示例 "diagnosis": "'}]}, ensure_ascii=False)
    prefix = '```json\n' + memory[:-1] + ',\n  "diagnosis": "'
    assert run.diagnosis_prefix(prefix + 'Recall Failure"}') == prefix
    assert run.diagnosis_prefix('{"reason": "test", "diagnosis": "') is None
    assert run.diagnosis_prefix('{"memory": {"diagnosis": "') is None


def test_baseline_memory_prefix_contains_only_answer_fields():
    memory = [{'id': 'Q1', 'answer': '含有引号 " 的回答', 'diagnosis': 'Recall Failure', 'reason': 'private'}]
    prefix = baseline.memory_prefix(memory)
    assert run.diagnosis_prefix(prefix) == prefix
    result = json.loads(prefix + 'Procedural Error"}')
    assert result['memory'] == [{'id': 'Q1', 'answer': memory[0]['answer']}]


def test_paired_diagnosis_uses_question_groups_and_keeps_missing_predictions():
    a, b = run.LABELS[:2]
    report = baseline_summary.paired_delta(['1', '1', '2'], [a, b, a], [None, b, a], [a, b, a])
    assert report['groups'] == 2
    assert report['accuracy']['delta'] == pytest.approx(1 / 3)
    assert report['macro_f1']['delta'] == pytest.approx(0.25 * (1 - 2 / 3))


def test_oracle_report_rejects_duplicate_sample_before_filtering(tmp_path):
    label = run.LABELS[0]
    rows = [{'sample_id': 's', 'gold': label, 'prediction': {'model_memory': value, 'gold_memory': label}}
            for value in (None, label)]
    path = tmp_path / 'scores.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    with pytest.raises(ValueError, match='full fixed development'):
        baseline_summary.paired_probe(path, ('model_memory', 'gold_memory'), {'s': label}, {'s': '1'}, common=True)


def test_resume_discards_only_unterminated_row(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    path.write_bytes(b'{"sample_id":"sample-a"}\n{"sample_id":"unfinished')
    assert run.resume_rows(path) == [{'sample_id': 'sample-a'}]
    assert path.read_bytes() == b'{"sample_id":"sample-a"}\n'


def test_fast_candidate_branches_equal_independent_teacher_forcing():
    torch = pytest.importorskip('torch')
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(41)
    config = Qwen2Config(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = Qwen2ForCausalLM(config).eval()
    prefix = [3, 5, 8, 9]
    candidates = dict(zip(run.LABELS, ([10, 11, 12], [10, 21], [22, 23, 24, 12], [29]), strict=True))
    with torch.inference_mode():
        scores, hidden = run.score_tokens(model, prefix, candidates, model.device, keep_hidden=True)
        assert len(hidden) == config.hidden_size
        for label, tokens in candidates.items():
            ids = torch.tensor([prefix + tokens])
            logits = model(input_ids=ids, use_cache=False).logits[0, len(prefix) - 1:-1]
            expected = logits.log_softmax(dim=-1).gather(1, torch.tensor(tokens)[:, None]).sum()
            assert scores[label] == pytest.approx(expected.item(), abs=1e-5)


def test_calibration_uses_stage2_and_counts_missing_dev(tmp_path, monkeypatch):
    frequent, rare = 'Procedural Error', 'Transfer Deficit'
    scores = {label: -3.0 for label in run.LABELS}
    scores[frequent], scores[rare] = -1.0, -1.1
    refs = [{'sample_id': str(i), 'split': split, 'gold': label} for i, (split, label) in enumerate([
        ('stage2', rare), ('dev', frequent), ('dev', rare)])]
    rows = [{'results': {'original': {'scores': scores if i < 2 else None},
                         'history_stats': {'scores': scores if i == 1 else None}}} for i in range(3)]
    local = {'sft_prior_counts': dict.fromkeys(run.LABELS, 1) | {frequent: 10}}
    inference = tmp_path / 'run'
    inference.mkdir()
    (inference / 'predictions.jsonl').touch()
    monkeypatch.setattr(offline, 'collect', lambda *_: (refs, rows, local))
    report = offline.score(None, inference, tmp_path / 'results', 'full')
    assert report['selected_tau'] == offline.TAUS[1]
    assert report['dev']['prior_corrected']['correct'] == 0
    assert report['dev']['original']['correct'] == 1
    assert report['dev']['original']['total'] == 2
    assert report['dev']['original']['missing'] == 1
    assert report['changes_from_original']['prior_corrected'][frequent]['lost'] == 1


def test_public_stats_use_only_previous_overlapping_concepts():
    def record(qid, concepts, day, result):
        return f'[{qid}]\n题干\n知识点: {concepts}\n时间: 2026-11-{day:02d}\n结果: {result}'

    history = [record(1, '分数,加法', 12, '错误'), record(2, '分数,加法', 17, '正确'),
               record(3, '几何', 22, '正确')]
    current = record(1, '分数,加法', 25, '错误')
    stats = offline.history_statistics(history, current)
    assert stats['共享至少一个知识点的记录数'] == 2
    assert stats['其中正确率'] == 0.5
    assert stats['距最近一次同知识点答对天数'] == 8
    assert stats['当前题此前作答次数'] == 1
    with pytest.raises(ValueError, match='after the current question'):
        offline.history_statistics([record(2, '分数', 26, '正确')], current)
