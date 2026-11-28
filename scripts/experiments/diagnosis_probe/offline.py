"""Prepare fixed diagnosis cases and score paired predictions locally."""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from itertools import takewhile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.diagnoser_evidence.evidence import fields, recency
from scripts.experiments.diagnosis_probe import run
from scripts.experiments.gold_evidence_first import data
from scripts.experiments.longtutor_reeval.run import (
    GENERATION_CONFIG_SHA256,
    WEIGHT_INDEX_SHA256,
)
from scripts.pa_opsd_contract import XES, official_module

SOURCE = ROOT / 'outputs/pa_opsd/gold_evidence_first/inputs_v1'
ANCHOR = (SOURCE.parent / 'autodl_5090_28031/sft_base_restart/runs/memory_first')
BASE_PIN = data.FROZEN / 'pa_opsd_gold_20261120/base_model.json'
TAUS = (0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2)


def history_statistics(history, current):
    current = fields(current)
    records = [fields(text) for text in history]
    dates = [recency(current['date'], record['date']) for record in records]
    if any(gap is not None and gap < 0 for gap in dates):
        raise ValueError('History contains a record after the current question')
    tags = set(current['concepts'])
    relevant = [(record, gap) for record, gap in zip(records, dates) if tags.intersection(record['concepts'])]
    known = [(record['result'], gap) for record, gap in relevant if record['result'] in ('正确', '错误')]
    correct = [gap for result, gap in known if result == '正确']
    return {
        '当前题之前的记录数': len(records), '当前题的知识点数': len(tags),
        '共享至少一个知识点的记录数': len(relevant), '其中有正误的记录数': len(known),
        '其中答对次数': len(correct), '其中正确率': len(correct) / len(known) if known else None,
        '距最近一次同知识点作答天数': relevant[-1][1] if relevant else None,
        '距最近一次同知识点答对天数': correct[-1] if correct else None,
        '从最近一次往前同知识点连续答对数': len(list(takewhile(
            lambda item: item[0]['result'] == '正确', reversed(relevant)))),
        '当前题此前作答次数': sum(current['qid'] is not None and record['qid'] == current['qid']
                                   for record in records),
    }


def prepare(out, source=SOURCE, anchor=ANCHOR, base_pin=BASE_PIN):
    source_manifest = data.verify(source)
    receipt = run.read_json(anchor / 'run_receipt.json')
    if (receipt['arm'] != 'memory_first' or receipt['rows'] != 420 or receipt['smoke']
            or receipt['screen'] or receipt['initial_adapter'] or receipt['effective_steps'] != 105
            or receipt.get('diagnosis_class_balance')
            or receipt['data_sha256'] != source_manifest['files'][data.TRAIN_FILES['memory_first'][0]]
            or receipt['manifest_sha256'] != run.digest(source / data.MANIFEST)
            or receipt['base_pin_sha256'] != run.digest(base_pin) or not receipt['budget_complete']):
        raise ValueError('Expected the full ordinary 420-example memory-first SFT checkpoint')
    adapter = {name: receipt['adapter'][name] for name in ('adapter_config.json', 'adapter_model.safetensors')}
    run.verify_model_files(anchor, adapter)

    stage2 = run.read_rows(source / data.TRAIN_FILES['stage2'][0])
    dev = run.read_rows(source / data.DEV_PROMPTS)
    refs = {row['sample_id']: row['target']['diagnosis'] for row in run.read_rows(source / data.DEV_REFERENCES)}
    refs.update({row['sample_id']: data.assistant_target(row)['diagnosis'] for row in stage2})
    official = official_module()
    from gpt_memory_diagnose import _sample_key

    history_path = XES / 'history_features_lastq.jsonl'
    histories = {}
    with history_path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = _sample_key(row)
            if key in refs:
                if key in histories:
                    raise ValueError('Duplicate official history key')
                histories[key] = row
    if histories.keys() != refs.keys() or set(refs.values()) != set(run.LABELS):
        raise ValueError('Official histories or diagnosis labels differ from the fixed data')

    cases, references = [], []
    for split, rows in (('stage2', stage2), ('dev', dev)):
        for row in rows:
            key = row['sample_id']
            original = row['messages'][:2]
            sample = histories[key]
            history, current = sample['history_info'][-200:], sample['question_info']
            user = original[1]['content']
            queries = user.partition('\n\n### MEMORY QUERIES\n')[2].split('\n\n', 1)[0]
            test_obj = {'memory': [{'query': line[6:]} for line in queries.splitlines()
                                   if re.match(r'^- Q[123]: ', line)]}
            _, expected_user = official._build_prompts(sample, test_obj)
            if len(test_obj['memory']) != 3 or user != expected_user:
                raise ValueError(f'Official visible history and fixed user prompt differ for {key}')
            stats = history_statistics(history, current)
            suffix = '\n\nAnalyze the data and generate the JSON response.'
            added = ('\n\n### 当前题之前的历史统计\n'
                     + json.dumps(stats, ensure_ascii=False, separators=(',', ':')))
            stats_user = user.removesuffix(suffix) + added + suffix
            cases.append({'sample_id': key, 'split': split, 'original': original,
                          'history_stats': [original[0], {**original[1], 'content': stats_user}]})
            references.append({'sample_id': key, 'split': split, 'gold': refs[key], 'statistics': stats})

    rng = random.Random(42)
    smoke_ids = [key for label in run.LABELS for key in rng.sample(
        sorted(row['sample_id'] for row in references if row['split'] == 'stage2' and row['gold'] == label), 4)]
    rng.shuffle(smoke_ids)
    out.mkdir(parents=True, exist_ok=False)
    cases_path = out / 'cases.jsonl'
    data.write_rows(cases_path, cases)
    data.write_rows(out / 'references.jsonl', references)
    gpu_manifest = out / 'gpu_manifest.json'
    pin = {**run.read_json(base_pin), 'generation_config_sha256': GENERATION_CONFIG_SHA256,
           'weight_index_sha256': WEIGHT_INDEX_SHA256}
    run.write_json(gpu_manifest, {'base': pin, 'adapter': adapter,
                                  'cases_sha256': run.digest(cases_path), 'smoke_ids': smoke_ids})
    run.write_json(out / 'local_manifest.json', {
        'references_sha256': run.digest(out / 'references.jsonl'), 'gpu_manifest_sha256': run.digest(gpu_manifest),
        'prepared_gold_manifest_sha256': run.digest(source / data.MANIFEST),
        'official_history_sha256': run.digest(history_path), 'adapter_receipt_sha256': run.digest(anchor / 'run_receipt.json'),
        'sft_prior_counts': source_manifest['counts']['sft'], 'rows': dict(Counter(row['split'] for row in cases)),
    })
    with ZipFile(out / 'autodl.zip', 'x', compression=ZIP_DEFLATED, compresslevel=3) as archive:
        for path in (cases_path, gpu_manifest, Path(run.__file__)):
            archive.write(path, path.name)
        for name in adapter:
            archive.write(anchor / name, f'adapter/{name}')


def summary(gold, predictions):
    table = {label: dict.fromkeys((*run.LABELS, 'missing'), 0) for label in run.LABELS}
    for actual, predicted in zip(gold, predictions, strict=True):
        table[actual][predicted or 'missing'] += 1
    classes = {}
    for label, row in table.items():
        tp, support = row[label], sum(row.values())
        selected = sum(counts[label] for counts in table.values())
        classes[label] = {'correct': tp, 'support': support, 'predicted': selected,
                          'f1': 2 * tp / (support + selected) if support + selected else 0}
    correct = sum(entry['correct'] for entry in classes.values())
    return {'correct': correct, 'total': len(gold), 'accuracy': correct / len(gold),
            'macro_f1': sum(entry['f1'] for entry in classes.values()) / len(run.LABELS),
            'classes': classes, 'confusion': table, 'missing': predictions.count(None)}


def ranked(scores, prior_penalty, tau):
    if scores is None:
        return []
    if scores.keys() != set(run.LABELS) or not all(math.isfinite(value) for value in scores.values()):
        raise ValueError('Expected four finite candidate log probabilities')
    return sorted(run.LABELS, key=lambda label: scores[label] - tau * prior_penalty[label], reverse=True)


def top_label(scores, prior, tau):
    return next(iter(ranked(scores, prior, tau)), None)


def verified_refs(bundle):
    local = run.read_json(bundle / 'local_manifest.json')
    run.verify_model_files(bundle, {'references.jsonl': local['references_sha256'],
                                    'gpu_manifest.json': local['gpu_manifest_sha256']})
    return run.read_rows(bundle / 'references.jsonl'), local


def collect(bundle, directory, phase):
    refs, local = verified_refs(bundle)
    settings = run.read_json(directory / 'settings.json')
    gpu = run.read_json(bundle / 'gpu_manifest.json')
    if settings['manifest_sha256'] != local['gpu_manifest_sha256'] or settings['phase'] != phase:
        raise ValueError('Predictions were made for a different bundle or phase')
    if (settings['base_revision'] != gpu['base']['revision'] or settings['arm'] not in ('base', 'sft')
            or settings['adapter'] != (gpu['adapter'] if settings['arm'] == 'sft' else None)):
        raise ValueError('Predictions used an unexpected checkpoint')
    if phase == 'smoke':
        index = {row['sample_id']: row for row in refs}
        refs = [index[key] for key in gpu['smoke_ids']]
    rows = run.read_rows(directory / 'predictions.jsonl')
    if [(r['sample_id'], r['split']) for r in rows] != [(r['sample_id'], r['split']) for r in refs]:
        raise ValueError('Prediction coverage or ordering differs from the fixed run inputs')
    for row in rows:
        expected = run.ARMS if phase == 'smoke' or row['split'] == 'dev' else run.ARMS[:1]
        if set(row['results']) != set(expected):
            raise ValueError('Prediction arms do not match the phase')
        for result in row['results'].values():
            if result['status'] == 'ok':
                if not ranked(result['scores'], dict.fromkeys(run.LABELS, 0), 0):
                    raise ValueError('Missing candidate scores for a successful result')
            elif result['status'] != 'missing_diagnosis_prefix' or result['scores'] is not None:
                raise ValueError('Unknown inference status')
    return refs, rows, local


def paired(gold, before, after):
    counts = {label: {'gained': 0, 'lost': 0} for label in run.LABELS}
    for actual, a, b in zip(gold, before, after, strict=True):
        counts[actual]['gained'] += a != actual and b == actual
        counts[actual]['lost'] += a == actual and b != actual
    return counts


def score(bundle, directory, out, phase):
    refs, rows, local = collect(bundle, directory, phase)
    out.mkdir(parents=True, exist_ok=False)
    counts = local['sft_prior_counts']
    prior = {label: math.log(counts[label] / sum(counts.values())) for label in run.LABELS}
    if phase == 'smoke':
        report = {arm: summary([ref['gold'] for ref in refs], [top_label(row['results'][arm]['scores'], prior, 0)
                                                              for row in rows]) for arm in run.ARMS}
        for arm in run.ARMS:
            report[arm]['retokenized_prefixes'] = sum(bool(row['results'][arm].get('retokenized_prefix')) for row in rows)
        run.write_json(out / 'summary.json', report)
        return report

    calibration = [(ref, row) for ref, row in zip(refs, rows) if ref['split'] == 'stage2']
    dev = [(ref, row) for ref, row in zip(refs, rows) if ref['split'] == 'dev']
    trials = {tau: summary([ref['gold'] for ref, _ in calibration], [
        top_label(row['results']['original']['scores'], prior, tau) for _, row in calibration]) for tau in TAUS}
    tau = max(TAUS, key=lambda weight: trials[weight]['macro_f1'])
    details = []
    for ref, row in dev:
        gold = ref['gold']
        scores = row['results']['original']['scores']
        order = ranked(scores, prior, 0)
        stats_scores = row['results']['history_stats']['scores']
        details.append({'sample_id': ref['sample_id'], 'gold': gold,
                        'original_scores': scores, 'history_stats_scores': stats_scores,
                        'gold_rank': order.index(gold) + 1 if order else None,
                        'gold_gap_from_top': scores[order[0]] - scores[gold] if order else None,
                        'original': order[0] if order else None,
                        'prior_corrected': top_label(scores, prior, tau),
                        'history_stats': top_label(stats_scores, prior, 0)})
    gold = [row['gold'] for row in details]
    predictions = {arm: [row[arm] for row in details] for arm in ('original', 'prior_corrected', 'history_stats')}
    report = {'selected_tau': tau, 'prior_counts': counts, 'calibration': trials,
              'dev': {arm: summary(gold, labels) for arm, labels in predictions.items()},
              'changes_from_original': {arm: paired(gold, predictions['original'], predictions[arm])
                                        for arm in ('prior_corrected', 'history_stats')},
              'rank_by_gold_label': {label: dict(Counter(str(row['gold_rank']) for row in details if row['gold'] == label))
                                     for label in run.LABELS},
              'dev_retokenized_prefixes': {arm: sum(bool(row['results'][arm].get('retokenized_prefix')) for _, row in dev)
                                           for arm in run.ARMS},
              'prediction_sha256': run.digest(directory / 'predictions.jsonl')}
    data.write_rows(out / 'dev_details.jsonl', details)
    run.write_json(out / 'summary.json', report)
    return report


def fit_logistic(features, labels):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                          StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0))
    return model.fit(features, labels)


def logistic(bundle, out, inference=None):
    from sklearn import __version__ as sklearn_version

    refs, local = verified_refs(bundle)
    feature_names = list(refs[0]['statistics'])
    values = [[row['statistics'][name] for name in feature_names] for row in refs]
    if inference:
        _, rows, _ = collect(bundle, inference, 'full')
        values = [row['results']['original'].get('hidden') for row in rows]
        feature_names = None
    train = [(value, ref['gold']) for value, ref in zip(values, refs) if ref['split'] == 'stage2' and value is not None]
    dev = [(value, ref) for value, ref in zip(values, refs) if ref['split'] == 'dev']
    features, gold = zip(*train)
    model = fit_logistic(features, gold)
    available = [value for value, _ in dev if value is not None]
    predicted = iter(model.predict(available).tolist()) if available else iter(())
    records = [{'sample_id': ref['sample_id'], 'gold': ref['gold'],
                'prediction': next(predicted) if value is not None else None} for value, ref in dev]
    report = {'feature_names': feature_names, 'reference_sha256': local['references_sha256'], 'training_rows': len(train),
              'scikit_learn': sklearn_version,
              'fit': {'class_weight': 'balanced', 'C': 1.0},
              'dev': summary([row['gold'] for row in records], [row['prediction'] for row in records])}
    if inference:
        report['vector_source'] = {'settings': run.read_json(inference / 'settings.json'),
                                   'prediction_sha256': run.digest(inference / 'predictions.jsonl'),
                                   'feature_count': len(features[0])}
    out.mkdir(parents=True, exist_ok=False)
    run.write_json(out / 'summary.json', report)
    data.write_rows(out / 'dev_details.jsonl', records)
    return report


def compare(bundle, base, sft, out):
    match = ('phase', 'max_new_tokens', 'manifest_sha256', 'runner_sha256', 'base_revision', 'dtype',
             'attention', 'torch', 'transformers')
    base_settings, sft_settings = (run.read_json(path / 'settings.json') for path in (base, sft))
    if (base_settings['arm'], sft_settings['arm']) != ('base', 'sft') or any(
            base_settings[key] != sft_settings[key] for key in match):
        raise ValueError('Both arms must use the same inputs, inference code and runtime settings')
    results = {arm: score(bundle, path, out / arm, 'full') for arm, path in (('base', base), ('sft', sft))}
    left, right = (run.read_rows(out / arm / 'dev_details.jsonl') for arm in ('base', 'sft'))
    report = {'runs': results, 'sft_changes_from_base': {
        arm: paired([row['gold'] for row in left], [row[arm] for row in left], [row[arm] for row in right])
        for arm in ('original', 'history_stats')}}
    run.write_json(out / 'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'score', 'statistics', 'linear-probe', 'compare'):
        command = commands.add_parser(name)
        command.add_argument('--output', type=Path, required=True)
        if name != 'prepare':
            command.add_argument('--bundle', type=Path, required=True)
        if name in ('score', 'linear-probe'):
            command.add_argument('--inference', type=Path, required=True)
        if name == 'score':
            command.add_argument('--phase', choices=('smoke', 'full'), required=True)
        if name == 'compare':
            command.add_argument('--base', type=Path, required=True)
            command.add_argument('--sft', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
        return
    if args.command == 'score':
        report = score(args.bundle, args.inference, args.output, args.phase)
    elif args.command == 'compare':
        report = compare(args.bundle, args.base, args.sft, args.output)
    else:
        report = logistic(args.bundle, args.output, getattr(args, 'inference', None))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
