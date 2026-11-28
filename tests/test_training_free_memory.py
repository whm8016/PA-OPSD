import pytest

from scripts.experiments.training_free_memory.prepare import sample_inputs
from scripts.experiments.training_free_memory.prepare_followup import (
    followup_inputs,
    source_events,
)


def record(qid, day, result='正确', tags='小数, 小数加减'):
    return f'[{qid}]\n练习\n知识点: {tags}\n时间: 2026-11-{day:02d}T08:00:00\n结果: {result}'


def test_same_evidence_with_and_without_an_applicable_card():
    history = [record(2, 2), record(3, 3, '错误', tags='小数'), record(8, 4, tags='几何')]
    sample = {'uid': 'learner', '_key': 'sample', 'question_info': record(1, 15, '错误'),
              'history_info': history, 'solution': 'GOLD_CANARY'}
    inputs, audit = sample_inputs(sample, {'memory': [{'query': '记录[8]的描述是什么？'}]})
    plain, card = inputs['temporal_rag'], inputs['temporal_rag_card']

    assert audit['latest_related_correct'] == 1
    assert audit['latest_related_wrong'] == 2
    assert audit['named_positions'] == [3]
    assert audit['applied_card'] == 'recall'
    assert plain['history_info'][:-1] == history
    assert card['history_info'][:-1] == plain['history_info'][:-1]
    assert card['history_info'][-1].startswith(plain['history_info'][-1] + '\n\n')
    assert set(card) == {'uid', '_key', 'question_info', 'history_info'}
    assert card['question_info'] == sample['question_info']
    assert '回想' in card['history_info'][-1]
    assert 'Recall Failure' not in card['history_info'][-1]
    assert 'GOLD_CANARY' not in str(inputs)


def test_overflow_keeps_all_occurrences_and_abstains_from_cards():
    history = [record(8, day, '错误') for day in range(1, 14)]
    sample = {'uid': 'learner', '_key': 'sample', 'question_info': record(1, 15, '错误'),
              'history_info': history}
    inputs, audit = sample_inputs(sample, {'memory': [{'query': '题目ID为8的所有记录'}]})

    assert audit['full_history_fallback']
    assert audit['named_positions'] == list(range(1, 14))
    assert inputs['temporal_rag']['history_info'][:-1] == history
    assert audit['applied_card'] is None
    assert inputs['temporal_rag_card'] == inputs['temporal_rag']


def test_card_in_full_fallback_preserves_the_first_record_in_official_history_window():
    history = [record(2, 1), *[record(8, 2, tags='几何') for _ in range(198)]]
    sample = {'uid': 'learner', '_key': 'sample', 'question_info': record(1, 15, '错误'),
              'history_info': history}
    inputs, audit = sample_inputs(sample, {'memory': [{'query': '记录[8]'}]})
    plain = inputs['temporal_rag']['history_info']
    card = inputs['temporal_rag_card']['history_info']

    assert audit['full_history_fallback'] and audit['applied_card'] == 'recall'
    assert len(plain) == len(card) == 200
    assert card[-200:-1] == plain[-200:-1] == history


def test_followup_preserves_the_factorial_pairs_and_indexed_event_provenance():
    history = [record(8, 8, '错误'), record(2, 2), record(8, 4), record(3, 6, tags='几何')]
    sample = {'uid': 'learner', '_key': 'sample', 'question_info': record(1, 15, '错误'),
              'history_info': history, 'solution': 'GOLD_CANARY'}
    queries = {'memory': [{'query': '回忆题目'}, {'query': '记录[8]和2026-11-05至2026-11-07的作答？'},
                          {'query': '检查事实'}]}
    prior, _ = sample_inputs(sample, queries)
    arms, audit = followup_inputs(sample, queries, prior['rag'], prior['temporal_rag'])

    assert arms['rag_stats']['history_info'] == [*arms['rag']['history_info'],
                                                arms['temporal_rag']['history_info'][-1]]
    assert arms['temporal']['history_info'] == arms['temporal_rag']['history_info'][:-1]
    assert audit['q2_candidates'] == [3, 4, 1]
    assert audit['state_concepts'] == ['小数', '小数加减']
    ledger = arms['event_ledger']['history_info']
    assert ledger[:-1] == arms['temporal']['history_info']
    assert '[4,"3","2026-11-06T08:00:00","正确",["几何"]]' in ledger[-1]
    assert 'GOLD_CANARY' not in str(arms)


def test_followup_abstains_over_candidate_budget_and_rejects_future_events():
    history = [record(8, 8) for _ in range(25)]
    sample = {'uid': 'learner', '_key': 'sample', 'question_info': record(1, 15, '错误'), 'history_info': history}
    queries = {'memory': [{'query': '回忆题目'}, {'query': '查找[8]'}, {'query': '检查事实'}]}
    prior, _ = sample_inputs(sample, queries)
    arms, audit = followup_inputs(sample, queries, prior['rag'], prior['temporal_rag'])

    assert len(audit['q2_candidates']) == 25
    assert arms['event_ledger'] == prior['temporal_rag']
    with pytest.raises(ValueError, match='Invalid public event'):
        source_events(dict(sample, history_info=[record(8, 16)]))
