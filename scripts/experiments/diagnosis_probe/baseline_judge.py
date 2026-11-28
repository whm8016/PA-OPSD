"""Run the unchanged LongTutor scorer on the archived weighted SFT predictions."""

import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import httpx
from dotenv import dotenv_values
from openai import APIError, OpenAI

from scripts.build_pa_opsd_gold_split import question_group
from scripts.experiments.action_distillation.analyze import DIMENSIONS, pair
from scripts.experiments.action_distillation.judge import CacheAwareMeter
from scripts.experiments.diagnosis_probe import baseline, run
from scripts.experiments.longtutor_reeval import run as official
from scripts.experiments.training_free_memory import judge_retrieval as billing

ARCHIVE = baseline.ROOT / 'outputs/pa_opsd/sft_diagnosis_baseline_20261128_run2/autodl_4090_38789'
RUN = ARCHIVE / 'official_dk'
BUDGET = 2_000_000
PREVIOUS_SCORES = '986ed59260b76232aad2b9013980468c19c872edb11db1e471db4378a805ea2f'


class BaselineMeter(CacheAwareMeter):
    def request(self, request):
        data = json.loads(request.content)
        maximum = data.get('max_tokens')
        if (request.url.host != 'api.deepseek.com' or request.url.path != '/chat/completions'
                or data.get('model') != billing.MODEL or type(maximum) is not int or not 1 <= maximum <= 2500
                or any(key in data for key in ('max_completion_tokens', 'thinking', 'reasoning_effort', 'tools'))):
            billing.stop('Unexpected judge endpoint, model or token limit')
        reserve = 2 * (len(request.content) + 512) + 8 * maximum
        with self.lock:
            if self.spent + sum(self.pending.values()) + reserve > BUDGET:
                billing.stop('The authorized 2-yuan ceiling would be exceeded')
            key = time.time_ns()
            request.extensions['billing_id'] = key
            self.log({'type': 'reserve', 'id': key, 'microyuan': reserve})


def paired_teaching(previous, current, inputs, manifest, ledger):
    ids = manifest['ids']
    new = official.prediction_rows(current, ids)
    old = official.prediction_rows(previous, ids)
    qids = {key: question_group(value['question_info'])[0] for key, value in inputs.items()}
    if set(ids) != qids.keys():
        raise ValueError('Evaluation and input samples differ')

    def fields(row):
        values = row['content_scores'] or {}
        return {'diagnosis_correct': int(row['diagnosis_correct']),
                **{f'Q{i}': int(v) for i, v in enumerate(row['memory_correct'], 1)},
                **values, 'teaching_four': mean(values[d] for d in DIMENSIONS) if values else None}

    scored = {key: (fields(old[key]), fields(new[key])) for key in ids}
    metrics = {}
    for field in ('teaching_four', *DIMENSIONS, 'overall_score', 'diagnosis_correct', 'Q1', 'Q2', 'Q3'):
        keys = [key for key in ids if all(row.get(field) is not None for row in scored[key])]
        if not keys:
            metrics[field] = {'paired_n': 0}
            continue
        left, right = ([scored[key][arm][field] for key in keys] for arm in (0, 1))
        metrics[field] = {'paired_n': len(keys), 'ordinary_sft': mean(left), 'weighted_sft': mean(right),
                          **pair(left, right, [qids[key] for key in keys])}
    official.write_rows(RUN / 'paired_by_id.jsonl', (
        {'sample_id': key, 'qid': qids[key], 'ordinary_sft': scored[key][0], 'weighted_sft': scored[key][1]}
        for key in ids
    ))
    events = billing.rows(ledger) if ledger.exists() else []
    meter = CacheAwareMeter(ledger)
    usage = Counter(e['model'] for e in events if e['type'] == 'settle')
    report = {'rows': len(ids), 'upstream_commit': manifest['upstream_commit'], 'metrics': metrics,
              'previous_scoring_sha256': run.digest(previous), 'weighted_scoring_sha256': run.digest(current),
              'estimated_yuan': sum(e.get('estimate_microyuan', 0) for e in events) / 1e6,
              'upper_yuan': (meter.spent + sum(meter.pending.values())) / 1e6,
              'model_response_versions': dict(usage), 'unknown_calls': len(meter.pending)}
    run.write_json(RUN / 'summary.json', report)
    return report


def main():
    bundle = baseline.OLD_OFFICIAL / 'bundle'
    manifest = official.verify_bundle(bundle)
    paths = {'pred': ARCHIVE / 'results/official_140.jsonl', 'gold': bundle / 'gold.jsonl',
             'scorer': bundle / 'upstream/compute_ai_tutor_eval_metrics.py',
             'helper': bundle / 'upstream/openai_helper.py'}
    receipt = run.read_json(ARCHIVE / 'backup_receipt.json')
    binding = {name: run.digest(path) for name, path in paths.items()}
    if (binding['pred'] != receipt['official_predictions_sha256'] or receipt['driver_exit'] != 0
            or receipt['official_rows'] != len(manifest['ids']) or len(manifest['ids']) != 140
            or binding['scorer'] != billing.SCORER_SHA
            or manifest['upstream_commit'] != 'ba40af3b9e976f5960c42a15a645c0e9c3a7d718'):
        raise ValueError('Missing or mismatched verified archive or official protocol')
    predictions = official.prediction_rows(paths['pred'], manifest['ids'])
    if any(official.inference_failure(row) for row in predictions.values()):
        raise ValueError('Official generation has unresolved infrastructure failures')
    previous = baseline.OLD_OFFICIAL / 'sft.metrics.jsonl'
    if run.digest(previous) != PREVIOUS_SCORES:
        raise ValueError('Archived ordinary SFT scores changed')
    official.prediction_rows(previous, manifest['ids'])
    grader = official.scorer_module(bundle)
    import openai_helper

    if Path(openai_helper.__file__).resolve() != paths['helper'].resolve():
        raise ValueError('Unexpected LongTutor helper path')
    inputs = official.index(run.read_rows(bundle / 'inputs.jsonl'), '_key')
    # Like the original scorer run, leave the provider's thinking option unspecified.
    binding.update({'model': billing.MODEL, 'temperature': 0, 'thinking': 'provider_default',
                    'workers': 4, 'wire_token_cap_key': 'max_tokens', 'previous_scores': PREVIOUS_SCORES,
                    'cap_microyuan': BUDGET})
    metrics, ledger, marker = (RUN / name for name in ('weighted_sft.metrics.jsonl', 'usage.jsonl', 'binding.json'))
    with official.lock_output(RUN):
        if marker.exists():
            if run.read_json(marker) != binding or not ledger.exists():
                raise ValueError('Mismatched binding or missing usage ledger')
        else:
            if metrics.exists() or (ledger.exists() and ledger.stat().st_size):
                raise ValueError('Unbound results or usage ledger')
            ledger.touch()
            run.write_json(marker, binding)
        completed = {}
        if metrics.exists():
            official.recover_prediction_tail(metrics)
            completed = official.prediction_rows(metrics, manifest['ids'], complete=False)
            if metrics.stat().st_size and not ledger.stat().st_size:
                raise ValueError('Recorded grades without usage ledger')
        meter = BaselineMeter(ledger)
        if len(completed) != len(manifest['ids']):
            with OpenAI(
                base_url='https://api.deepseek.com', api_key=dotenv_values(baseline.ROOT / '.env')['DEEPSEEK_API_KEY'],
                max_retries=0, http_client=httpx.Client(timeout=90, event_hooks={
                    'request': [meter.request], 'response': [meter.response],
                }),
            ) as client:
                create = client.chat.completions.create

                def call(**kwargs):
                    try:
                        kwargs['max_tokens'] = kwargs.pop('max_completion_tokens')
                        response = create(**kwargs)
                    except (APIError, httpx.HTTPError) as error:
                        billing.stop(f'DK transport failed: {type(error).__name__}')
                    if not response.choices or response.choices[0].message is None:
                        billing.stop('DeepSeek returned no completion choice')
                    reason = response.choices[0].finish_reason
                    if reason in ('insufficient_system_resource', 'aborted'):
                        billing.stop(f'DeepSeek could not complete the request ({reason})')
                    if reason == 'length':
                        print('DeepSeek reached the upstream completion limit', flush=True)
                    return response

                client.chat.completions.create = call
                openai_helper._get_openai_client = lambda *args: client
                grader.main(['--pred', str(paths['pred']), '--gold', str(paths['gold']), '--model', billing.MODEL,
                             '--temperature', '0', '--limit', '140', '--workers', '4', '--out', str(metrics)])
        print(json.dumps(paired_teaching(previous, metrics, inputs, manifest, ledger), ensure_ascii=False))


if __name__ == '__main__':
    main()
