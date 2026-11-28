from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.experiments.teacher_team import compare

ROW = {
    'sample_id': 's1', 'history': [{'position': 21, 'text': 'a recorded correct exercise'}],
    'history_summary': {'count': 21}, 'current': 'exercise', 'solution': {'answer': ['8']},
    'gold': 'PRIVATE_GOLD',
}
DIAGNOSIS = {'diagnosis': 'Recall Failure', 'evidence': [21], 'reason': 'seen before'}
TEACHING = {'action': 'C', 'feedback': 'Recall the rule.'}


def test_prompts_share_context_and_only_team_receives_prior_decisions():
    done = {'diagnoser': {'raw': json.dumps(DIAGNOSIS)}, 'pedagogue': {'raw': json.dumps(TEACHING)}}
    baseline = compare.messages(ROW, 'single', done)
    shared = baseline[1]['content'].removesuffix('\n\n' + compare.INSTRUCTIONS['single'])
    for stage in compare.STAGES:
        system, user = compare.messages(ROW, stage, done)
        assert system == baseline[0]
        assert user['content'].startswith(shared)
        assert 'PRIVATE_GOLD' not in user['content']
        assert ('seen before' in user['content']) == (stage in ('pedagogue', 'verifier'))
        assert ('Recall the rule.' in user['content']) == (stage == 'verifier')


@pytest.mark.parametrize('citation', [[20], [True], [21, 21], '21'])
def test_parse_rejects_hidden_or_invalid_evidence(citation):
    result = compare.parse(json.dumps({**DIAGNOSIS, 'evidence': citation}), 'single', ROW)
    assert result['evidence'] == []
    assert result['evidence_list_valid'] is False
    assert result['diagnosis'] == 'Recall Failure'


def model_response(*args):
    stage = next(stage for stage, instruction in compare.INSTRUCTIONS.items()
                 if args[3][1]['content'].endswith(instruction))
    payload = {
        'single': {**DIAGNOSIS, **TEACHING, 'diagnosis': 'Conceptual Gap'},
        'diagnoser': DIAGNOSIS,
        'pedagogue': TEACHING,
        'verifier': {'final': {**DIAGNOSIS, **TEACHING}, 'checks': {'math_correct': 'pass'}},
    }[stage]
    return {'raw': json.dumps(payload), 'finish': 'stop', 'prompt_tokens': 10,
            'completion_tokens': 20, 'seconds': 1}


def test_manifest_rejects_nonfrozen_data(tmp_path: Path):
    compare.gold.write_json(tmp_path / 'manifest.json', {
        'scope': 'validation', 'protocol': compare.PROTOCOL, 'source_manifest_sha256': compare.SOURCE_MANIFEST,
        'ids': ['s1'], 'files': compare.BUNDLE_FILES,
    })
    with pytest.raises(ValueError, match='frozen Gold 32'):
        compare.read_manifest(tmp_path)


def test_checkpoint_identity_rejects_adapter_on_different_base(tmp_path: Path):
    args = SimpleNamespace(port=8020, model='sft', service_bundle=tmp_path)
    models = {'data': [
        {'id': 'sft', 'parent': 'other model'},
        {'id': compare.service.BASE_NAME, 'max_model_len': 32768},
    ]}
    with (patch.object(compare.service, 'verify_bundle', return_value={'base_revision': compare.BASE_REVISION}),
          patch.object(compare, 'model_request', side_effect=[{'version': compare.service.VLLM_VERSION}, models]),
          pytest.raises(ValueError, match='model registration')):
        compare.checkpoint_identity(args)


def test_resume_preserves_completed_requests_and_scores_paired_results(tmp_path: Path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    gold = compare.gold
    gold.write_rows(bundle / 'inputs.jsonl', [ROW])
    gold.write_rows(bundle / 'references.jsonl', [{
        'sample_id': 's1', 'diagnosis': 'Recall Failure', 'strategy': 'Retrieval Practice',
        'feedback': 'gold feedback', 'related_positions': [21],
    }])
    manifest = {
        'scope': 'validation', 'ids': ['s1'],
        'files': {f'{name}.jsonl': gold.digest(bundle / f'{name}.jsonl') for name in ('inputs', 'references')},
    }
    gold.write_json(bundle / 'manifest.json', manifest)
    args = SimpleNamespace(bundle=bundle, output=tmp_path / 'run', port=8018, model='mock',
                           max_tokens=384, batch=None, partial=False)
    single = model_response(8018, 'mock', 384, compare.messages(ROW, 'single', {}))
    with (patch.object(compare, 'read_manifest', return_value=manifest),
          patch.object(compare, 'checkpoint_identity', side_effect=lambda args: {'port': args.port})):
        with (patch.object(compare, 'generate', side_effect=[single, OSError('offline')]),
              pytest.raises(OSError, match='offline')):
            compare.infer(args)
        with pytest.raises(ValueError, match='Missing complete pairs'):
            compare.score(args)
        args.port = 8020
        with pytest.raises(ValueError, match='another run'):
            compare.infer(args)
        args.port = 8018
        with patch.object(compare, 'generate', side_effect=model_response) as generate:
            compare.infer(args)
            assert generate.call_count == 3
            compare.infer(args)
            assert generate.call_count == 3
        compare.score(args)
    report = gold.read_json(args.output / 'score.json')
    assert report['single']['diagnosis_correct'] == 0
    assert report['team']['diagnosis_correct'] == 1
    assert report['team']['completion_tokens'] == 60
    assert report['single_vs_team']['teacher_only_matching'] == 1
    assert report['related_history_visible_rows'] == 1
    assert not report['partial']
