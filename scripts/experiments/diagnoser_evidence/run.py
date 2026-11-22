"""Replay the fixed diagnoser and compare structured and compact history inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.diagnoser_evidence import evidence
from scripts.experiments.longtutor_reeval import run as service
from scripts.experiments.teacher_team import run as team

gold = team.gold
ARMS = ('full_history', 'full_with_summary', 'compact_with_summary')
BUDGET = team.LIMITS[team.STAGES.index('diagnoser')]


def build(source, out):
    if source.resolve() == out.resolve():
        raise ValueError('Source and output must be different directories')
    for scope in ('train', 'validation'):
        src, dst = source / scope, out / scope
        original = gold.read_json(src / 'manifest.json')
        if original['scope'] != scope or original['selection_tag'] != team.TAG:
            raise ValueError(f'Unexpected teacher comparison input: {src}')
        inputs = team.load(src, 'inputs.jsonl', original)
        refs = team.load(src, 'references.jsonl', original)
        features = [evidence.build_features(row) for row in inputs.values()]
        dst.mkdir(parents=True, exist_ok=True)
        gold.write_rows(dst / 'inputs.jsonl', inputs.values())
        gold.write_rows(dst / 'references.jsonl', refs.values())
        gold.write_rows(dst / 'features.jsonl', features)
        gold.write_json(dst / 'manifest.json', {**original,
            'source_manifest_sha256': gold.digest(src / 'manifest.json'),
            'feature_code_sha256': gold.digest(Path(evidence.__file__)),
            'feature_parameters': {'max_concepts': evidence.MAX_CONCEPTS, 'ranked_records': evidence.MATCHED_RECORDS,
                                   'recent_records': evidence.RECENT_RECORDS},
            'files': {name: gold.digest(dst / name) for name in ('inputs.jsonl', 'references.jsonl', 'features.jsonl')},
        })


def messages(row, features, arm):
    original = team.messages(row, 'diagnoser', {})
    if arm == ARMS[0]:
        return original
    content = original[1]['content']
    instruction = team.INSTRUCTIONS['diagnoser']
    context = content.removesuffix('\n\n' + instruction)
    if arm == ARMS[2]:
        _, marker, tail = context.partition('\n\n## 当前题\n')
        if not marker:
            raise ValueError('Cannot locate the current exercise in the original prompt')
        selected = '\n\n'.join(f'历史序号 {i}\n{row["history"][i - 1]}' for i in features['history_positions'])
        context = '## 学生历史\n' + selected + marker + tail
    summary = json.dumps(features['summary'], ensure_ascii=False)
    text = context + '\n\n## 当前题之前的历史统计\n' + summary + '\n\n' + instruction
    return [original[0], {'role': 'user', 'content': text}]


def parse(raw, row, features, arm):
    parsed = team.decision(raw, len(row['history']))
    if arm == ARMS[2] and not set(parsed['evidence']) <= set(features['history_positions']):
        parsed['evidence'] = []
    return parsed


def infer(args):
    from openai import APIError, OpenAI

    manifest = gold.read_json(args.bundle / 'manifest.json')
    inputs = team.load(args.bundle, 'inputs.jsonl', manifest)
    features = team.load(args.bundle, 'features.jsonl', manifest)
    pinned = service.verify_bundle(args.service_bundle)
    settings = {'model': args.model, 'temperature': 0, 'max_completion_tokens': BUDGET, 'arms': list(ARMS),
                'workers': args.workers,
                'service_manifest_sha256': gold.digest(args.service_bundle / 'manifest.json'),
                'base_revision': pinned['base_revision'], 'data': manifest['files'],
                'adapter': manifest['gold_sft_adapter_sha256'],
                'sources': {name: gold.digest(Path(module.__file__)) for name, module in (
                    ('run', sys.modules[__name__]), ('evidence', evidence), ('diagnoser', team), ('helpers', gold))}}
    out = args.output
    path, stamp = out / 'decisions.jsonl', out / 'settings.json'
    with team.lock_output(out), OpenAI(base_url=f'http://127.0.0.1:{args.port}/v1', api_key='EMPTY',
                                       timeout=240, max_retries=0) as client:
        checkpoint = gold.checkpoint_root(client, args.model, manifest)
        if checkpoint != (args.service_bundle / 'adapters' / args.model).resolve():
            raise ValueError('Served adapter belongs to another verified bundle')
        if stamp.exists():
            if gold.read_json(stamp) != settings:
                raise ValueError('Existing output belongs to another run')
        elif path.exists():
            raise ValueError('Found predictions without settings')
        else:
            gold.write_json(stamp, settings)
        saved = gold.indexed(gold.read_rows(path), 'sample_id') if path.exists() else {}
        if saved.keys() - inputs.keys():
            raise ValueError('Prediction IDs differ from the input')

        def request(key, arm):
            started = time.monotonic()
            completion = client.chat.completions.create(
                model=args.model, messages=messages(inputs[key], features[key], arm), temperature=0,
                max_completion_tokens=BUDGET, response_format={'type': 'json_object'},
            )
            usage = completion.usage
            choice = completion.choices[0]
            return {'raw': choice.message.content, 'finish': choice.finish_reason,
                    'prompt_tokens': usage.prompt_tokens if usage else None,
                    'completion_tokens': usage.completion_tokens if usage else None,
                    'request_seconds': round(time.monotonic() - started, 3)}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for arm in ARMS:
                tasks = {pool.submit(request, key, arm): key for key in inputs if arm not in saved.get(key, {})}
                failures = []
                for future in as_completed(tasks):
                    key = tasks[future]
                    try:
                        response = future.result()
                    except APIError as exc:
                        failures.append(f'{key}: {type(exc).__name__}: {exc}')
                        continue
                    saved.setdefault(key, {'sample_id': key})[arm] = response
                    gold.write_rows(path, (saved[k] for k in inputs if k in saved))
                    print(f'{arm} {sum(arm in row for row in saved.values())}/{len(inputs)}', flush=True)
                if failures:
                    raise RuntimeError('Completed requests saved. Failed requests:\n' + '\n'.join(failures))


def totals(rows, field):
    values = [row[field] for row in rows]
    return sum(values) if all(v is not None for v in values) else None


def score(args):
    manifest = gold.read_json(args.bundle / 'manifest.json')
    inputs = team.load(args.bundle, 'inputs.jsonl', manifest)
    features = team.load(args.bundle, 'features.jsonl', manifest)
    refs = team.load(args.bundle, 'references.jsonl', manifest)
    settings = gold.read_json(args.output / 'settings.json')
    if settings['data'] != manifest['files'] or settings['adapter'] != manifest['gold_sft_adapter_sha256']:
        raise ValueError('Prediction settings differ from the evaluation bundle')
    saved = gold.indexed(gold.read_rows(args.output / 'decisions.jsonl'), 'sample_id')
    if set(saved) != set(inputs) or any(not set(ARMS) <= row.keys() for row in saved.values()):
        raise ValueError('Incomplete pairings')
    related = {k: set(r['related_positions']) for k, r in refs.items() if r['related_positions']}
    found = {k: set(features[k]['history_positions']) & ref for k, ref in related.items()}
    denom = Counter(row['diagnosis'] for row in refs.values())
    parsed, results, hits, citations = {}, {}, {}, {}
    for arm in ARMS:
        decisions = {k: parse(saved[k][arm]['raw'], inputs[k], features[k], arm) for k in refs}
        parsed[arm] = decisions
        hits[arm] = {k for k in refs if decisions[k]['diagnosis'] == refs[k]['diagnosis']}
        citations[arm] = {k for k in refs if decisions[k]['evidence']}
        correct = Counter(refs[k]['diagnosis'] for k in hits[arm])
        guesses = Counter(row['diagnosis'] for row in decisions.values())
        responses = [saved[k][arm] for k in refs]
        results[arm] = {'diagnosis_correct': len(hits[arm]), 'missing_diagnosis': guesses[None],
                        'macro_f1': sum(2 * correct[c] / (n + guesses[c]) for c, n in denom.items()) / len(denom),
                        'by_class': {c: {'gold': n, 'correct': correct[c], 'predicted': guesses[c]}
                                     for c, n in denom.items()},
                        'valid_evidence_rows': len(citations[arm]),
                        'shared_concept_reference_overlap_rows': sum(bool(set(decisions[k]['evidence']) & r)
                                                                    for k, r in related.items()),
                        'prompt_tokens': totals(responses, 'prompt_tokens'),
                        'completion_tokens': totals(responses, 'completion_tokens'),
                        'median_request_seconds': median(r['request_seconds'] for r in responses),
                        'length_limit_hits': sum(r['finish'] == 'length' for r in responses)}
    paired = {}
    for name, left, right in (('summary_on_full_history', ARMS[0], ARMS[1]),
                              ('compression_with_same_summary', ARMS[1], ARMS[2]),
                              ('compact_vs_baseline', ARMS[0], ARMS[2])):
        paired[name] = {'reference': left, 'candidate': right,
                        'diagnosis': gold.paired_counts(hits[left], hits[right], len(refs)),
                        'valid_citation': gold.paired_counts(citations[left], citations[right], len(refs))}
    report = {'scope': manifest['scope'], 'rows': len(refs), 'settings': settings,
              'retrieval': {'rows_with_reference': len(related), 'rows_with_hit': sum(bool(v) for v in found.values()),
                            'matched_reference_positions': sum(map(len, found.values())),
                            'total_reference_positions': sum(map(len, related.values())),
                            'retained_history_records': sum(len(x['history_positions']) for x in features.values())},
              'arms': results, 'paired': paired,
              'score_code_sha256': gold.digest(Path(__file__)),
              'decisions_sha256': gold.digest(args.output / 'decisions.jsonl')}
    gold.write_json(args.output / 'score.json', report)
    for key, target in refs.items():
        target['current'] = inputs[key]['current']
        target['features'] = features[key]
        target['predictions'] = {arm: reports[key] for arm, reports in parsed.items()}
        positions = set(features[key]['history_positions']) | set(target['related_positions'])
        positions.update(i for record in target['predictions'].values() for i in record['evidence'])
        target['evidence_text'] = {i: inputs[key]['history'][i - 1] for i in sorted(positions)}
    gold.write_rows(args.output / 'review.jsonl', refs.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build_cmd = commands.add_parser('build')
    build_cmd.add_argument('--source', type=Path, required=True)
    infer_cmd = commands.add_parser('infer')
    infer_cmd.add_argument('--service-bundle', type=Path, required=True)
    infer_cmd.add_argument('--model', default='sft')
    infer_cmd.add_argument('--port', type=int, default=8020)
    infer_cmd.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    score_cmd = commands.add_parser('score')
    for command in (build_cmd, infer_cmd, score_cmd):
        command.add_argument('--output', type=Path, required=True)
    for command in (infer_cmd, score_cmd):
        command.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.source, args.output)
    else:
        {'infer': infer, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
