import copy
import json
from argparse import Namespace

import pytest

from scripts.experiments.diagnoser_evidence import evidence
from scripts.experiments.teacher_team import grounded, grounding

gold = grounded.gold


def item(qid, date, result, tags='应用题, 逆向还原'):
    return f'[{qid}]\n一题有关运算的练习。\n知识点: {tags}\n时间: {date}\n结果: {result}'


@pytest.fixture
def row():
    return {
        'sample_id': 'fake',
        'current': item(9999, '2026-11-21T12:00:00', '错误'),
        'history': [
            item(9999, '2026-11-15T12:00:00', '正确'),
            item(852, '2026-11-21T11:00:00', '错误'),
            item(852, '2026-11-21T11:00:00', '错误'),
            item(9999, '2026-11-21T11:30:00', '正确'),
            item(900, '2026-11-21T11:40:00', '正确', '应用题, 圆的周长'),
            item(9999, '2026-11-21T12:00:00', '错误'),
        ],
        'solution': {'answer': ['$$42$$'], 'analysis': '由两数相加得知。'},
    }


def response(data, finish='stop'):
    return {'raw': json.dumps(data, ensure_ascii=False), 'finish': finish,
            'prompt_tokens': 1, 'completion_tokens': 1, 'seconds': 0.1}


def claim(card, **fields):
    return dict({'id': card['id'], 'result': card['result'], 'interval': card['interval'],
                 'relation': 'same_skill', 'role': 'support', 'method': '倒推上一步'}, **fields)


def decision(card, name='Recall Failure', **fields):
    return dict({'diagnosis': name, 'alternative': 'Transfer Deficit', 'confidence': 'supported',
                 'evidence': [claim(card)], 'why_over_alternative': '核对同一方法的正误记录'}, **fields)


def test_citable_attempts_and_time_are_demarcated(row):
    built = grounding.cards(row)
    old, repeat, recent, broad = built['cards']
    assert [card['id'] for card in built['cards']] == ['H01', 'H02', 'H03', 'H04']
    assert old['interval'] == 'four_days_or_more'
    assert repeat['positions'] == [2, 3]
    assert repeat['gap_minutes'] == 60
    assert recent['gap_minutes'] == 30
    assert recent['interval'] == 'under_day'
    assert broad['tag_match'] == 'shared_tags'
    visible = json.dumps(grounding.visible(built), ensure_ascii=False)
    assert '9999' not in visible and '852' not in visible and 'positions' not in visible
    assert 'H02' in visible


def test_conflicting_attempts_are_not_two_independent_observations(row):
    row['history'].append(item(852, '2026-11-21T11:00:00', '正确'))
    assert not any(c['qid'] == '852' for c in grounding.cards(row)['cards'])
    row['history'].append(item(876, '2026-11-22T11:00:00', '正确'))
    with pytest.raises(ValueError, match='after the current'):
        grounding.cards(row)


def test_id_result_time_and_duplicate_support_gate(row):
    built = grounding.cards(row)
    old, repeat, recent, broad = built['cards']
    assert grounding.diagnosis(response(decision(old)), built)['status'] == 'eligible'
    for invalid in (821, '9999', 'H99', []):
        value = decision(old, evidence=[claim(old, id=invalid)])
        assert 'invalid_evidence_id' in grounding.diagnosis(response(value), built)['flags']
    value = decision(recent, evidence=[claim(recent, interval='four_days_or_more')])
    assert 'evidence_fact_mismatch' in grounding.diagnosis(response(value), built)['flags']
    value = decision(recent, evidence=[claim(recent, result='incorrect')])
    assert 'evidence_fact_mismatch' in grounding.diagnosis(response(value), built)['flags']
    assert 'weak_class_support' in grounding.diagnosis(response(decision(broad)), built)['flags']
    value = decision(repeat, 'Conceptual Gap', evidence=[claim(repeat), claim(repeat)])
    assert 'weak_class_support' in grounding.diagnosis(response(value), built)['flags']
    assert 'invalid_output' in grounding.diagnosis(response(decision(old), 'length'), built)['flags']


def test_blind_review_discards_invalid_first_and_preserves_credible_conflict(row):
    built = grounding.cards(row)
    old, _, recent, _ = built['cards']
    bogus = grounding.diagnosis(response(decision(old, evidence=[claim(old, id='9999')])), built)
    cautious = grounding.diagnosis(response(decision(old, confidence='uncertain')), built)
    independent = grounding.diagnosis(response(decision(recent, 'Procedural Error')), built)
    assert grounded.requires_review(bogus)
    assert grounded.reconcile(bogus, independent)['route'] == 'blind_review'
    unresolved = grounded.reconcile(cautious, independent)
    assert unresolved['status'] == 'unresolved'
    assert unresolved['label'] is None
    assert grounded.prompt(built, 'verifier', cautious) == grounded.prompt(built, 'verifier', bogus)
    assert 'unresolved' in grounded.prompt(built, 'pedagogue', unresolved)[1]['content']
    assert gold.RUBRIC[-1] in grounded.prompt(built, 'verifier')[1]['content']


def test_partial_tags_require_blind_agreement(row):
    row['history'][4] = row['history'][4].replace('2026-11-21T11:40:00', '2026-11-10T11:40:00')
    built = grounding.cards(row)
    partial = grounding.diagnosis(response(decision(built['cards'][-1])), built)
    assert set(partial['flags']) == grounding.BLIND_FLAGS
    assert grounded.requires_review(partial)
    assert grounded.reconcile(partial, None)['status'] == 'unresolved'
    assert grounded.reconcile(partial, partial)['route'] == 'blind_agreement'
    assert grounded.reconcile(partial, partial)['status'] == 'eligible'
    different = grounding.diagnosis(response(decision(built['cards'][2], 'Procedural Error')), built)
    assert grounded.reconcile(partial, different)['status'] == 'unresolved'
    independent = grounding.diagnosis(response(decision(built['cards'][0])), built)
    assert independent['status'] == 'eligible'
    assert grounded.reconcile(partial, independent)['status'] == 'unresolved'


def test_old_success_on_different_item_does_not_confirm_three_diagnoses(row):
    row['history'][0] = row['history'][0].replace('[9999]', '[821]')
    built = grounding.cards(row)
    card = built['cards'][0]
    for label in ('Recall Failure', 'Transfer Deficit'):
        guess = grounding.diagnosis(response(decision(card, label, alternative='Procedural Error')), built)
        assert guess['flags'] == ['ambiguous_support']
        assert grounded.requires_review(guess)
    guess = grounding.diagnosis(response(decision(card, 'Procedural Error')), built)
    assert 'weak_class_support' in guess['flags']


def test_missing_figure_and_explicit_reference_answer(row):
    row['current'] = row['current'].replace('一题', 'question_9999-image_1 一题')
    built = grounding.cards(row)
    first = grounding.diagnosis(response(decision(built['cards'][0])), built)
    assert not grounded.requires_review(first)
    unresolved = grounded.reconcile(first, None)
    assert grounding.lesson(response({'action': 'C', 'feedback': '能补一下题图吗？'}), built, unresolved)['status'] == 'candidate'
    for action, feedback, flag in (
        ('G', '先看图的左边。', 'needs_visible_figure'),
        ('E', '改正刚才的第一步。', 'student_steps_unavailable'),
        ('C', '正确答案是 $$42$$。', 'explicit_reference_answer'),
    ):
        assert flag in grounding.lesson(response({'action': action, 'feedback': feedback}), built, unresolved)['flags']


def test_direct_disclosure_patterns_do_not_flag_non_disclosing_questions(row):
    row['solution']['answer'] = ['C', '42']
    built = grounding.cards(row)
    resolved = {'status': 'eligible'}
    for text in ('应选（C）。', '空格里填写 42。', '最后结果为 42。', '结果是 42 吗？'):
        assert 'explicit_reference_answer' in grounding.lesson(
            response({'action': 'G', 'feedback': text}), built, resolved,
        )['flags']
    for text in ('想一想是否不应该选 C。', '你是怎么算出结果的？请先看乘法分配律。'):
        assert 'explicit_reference_answer' not in grounding.lesson(
            response({'action': 'G', 'feedback': text}), built, resolved,
        )['flags']


def test_real_failure_examples_and_paired_resume(monkeypatch, tmp_path):
    source = gold.ROOT / 'outputs/pa_opsd/teacher_team_gold32/input/validation'
    if not source.exists():
        pytest.skip('Local frozen Gold regression artifacts are unavailable')
    bundle, out = tmp_path / 'bundle', tmp_path / 'predictions'
    grounded.build('regression32', source, bundle)
    manifest = grounded.manifest_for(bundle)
    source_manifest = gold.read_json(source / 'manifest.json')
    original = grounded.previous.load(source, 'inputs.jsonl', source_manifest)
    inputs = grounded.previous.load(bundle, 'inputs.jsonl', manifest)
    smoke = {evidence.fields(original[k]['current'])['qid']: inputs[k] for k in manifest['ids'][:4]}
    assert set(smoke) == grounded.REGRESSION_SMOKE
    basket = smoke['4086']
    by_position = {i: card for card in basket['cards'] for i in card['positions']}
    assert any(card['interval'] == 'under_day' for i, card in by_position.items() if i >= 190)
    q409 = smoke['409']
    bad = decision(q409['cards'][0], evidence=[{'id': 2323}, {'id': 2322}])
    assert 'invalid_evidence_id' in grounding.diagnosis(response(bad), q409)['flags']
    q1267 = smoke['1267']
    bad = decision(q1267['cards'][0], evidence=[{'id': 1267}])
    assert grounding.diagnosis(response(bad), q1267)['status'] == 'needs_review'

    calls = []
    accepted = ['single', 'diagnoser', 'verifier', 'pedagogue']

    def fake(port, model, limit, messages):
        stage = accepted[len(calls) % 4]
        calls.append(stage)
        if stage == 'pedagogue':
            return response({'action': 'C', 'feedback': '你会先处理哪一个运算？'})
        return response({'diagnosis': 'Recall Failure', 'confidence': 'uncertain', 'action': 'C',
                         'feedback': '你会先处理哪一个运算？'})

    monkeypatch.setattr(grounded.compare, 'checkpoint_identity', lambda args: {'base_revision': 'mock'})
    monkeypatch.setattr(grounded.compare, 'generate', fake)
    args = Namespace(bundle=bundle, output=out, max_tokens=1024, port=8020, batch=1, partial=False)
    grounded.infer(args)
    with pytest.raises(ValueError, match='Missing complete pairs'):
        grounded.score(args)
    args.partial = True
    grounded.score(args)
    assert gold.read_json(out / 'score.json')['subsets']['all']['team']['unresolved'] == 1
    args.batch = 3
    grounded.infer(args)
    grounded.score(args)
    result = gold.read_json(out / 'score.json')
    assert result['rows'] == 4 and result['expected_rows'] == 32 and result['partial']
    assert result['reviews_triggered'] == 4
    assert result['subsets']['all']['team']['calls'] == 12
    assert calls == accepted * 4
    stale = copy.copy(args)
    stale.max_tokens = 512
    with pytest.raises(ValueError, match='another run'):
        grounded.infer(stale)
