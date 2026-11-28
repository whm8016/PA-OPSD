"""Prepare the fixed Gold dev retrieval/statistics ablation and two structured memory arms."""

import argparse
import json
import os
import re
from collections import Counter
from datetime import date
from itertools import pairwise
from pathlib import Path
from statistics import median

from scripts.experiments.diagnoser_evidence.evidence import fields, recency
from scripts.experiments.longtutor_reeval.run import (
    digest,
    index,
    load_upstream,
    read_json,
    rows,
    write_rows,
)
from scripts.experiments.training_free_memory import prepare as prior

FIRST_ROUND = prior.ROOT / 'outputs/pa_opsd/training_free_memory/gold140_20261125_verified'
ARMS = ('rag', 'rag_stats', 'temporal', 'temporal_rag', 'event_ledger', 'learner_state')
REPEAT_ARMS = ('temporal_rag', 'temporal_rag_repeat')
MAX_LEDGER_ROWS = 24
MAX_PROMPT_TOKENS = 4096
FULL_DATE = re.compile(r'(?<!\d)(20\d{2})(?:年|-)(\d{1,2})(?:月|-)(\d{1,2})日?(?!\d)')
QUOTED = re.compile(r'“([^”]+)”|‘([^’]+)’|"([^"]+)"')
DATE_RANGE = re.compile(r'\s*(?:至|到|—|～|~)\s*')


def calendar_date(match):
    try:
        return date(*map(int, match.groups()))
    except ValueError:
        return None


def public_selectors(query, vocabulary):
    ids = {a or b for a, b in prior.EXPLICIT_ID.findall(query)}
    concepts = {a or b or c for a, b, c in QUOTED.findall(query)} & vocabulary
    date_matches = [(match, calendar_date(match)) for match in FULL_DATE.finditer(query)]
    days = {day for _, day in date_matches if day is not None}
    ranges = [(first, last) for ((left, first), (right, last)) in pairwise(date_matches)
              if first and last and first <= last and DATE_RANGE.fullmatch(query[left.end():right.start()])]
    return ids, concepts, days, ranges


def source_events(sample):
    current = fields(sample['question_info'])
    events = [(i, fields(record)) for i, record in enumerate(sample['history_info'], 1)]
    for _, event in events:
        gap = recency(current['date'], event['date'])
        if not event['qid'] or gap is None or gap < 0:
            raise ValueError(f"Invalid public event in {sample['_key']}")
    return current, sorted(events, key=lambda item: (item[1]['date'], item[0]))


def q2_candidates(events, current, query):
    vocabulary = set(current['concepts']) | {tag for _, event in events for tag in event['concepts']}
    ids, concepts, days, ranges = public_selectors(query, vocabulary)
    candidates = []
    for position, event in events:
        day = event['date'].date()
        if (event['qid'] in ids or concepts.intersection(event['concepts']) or day in days
                or any(start <= day <= end for start, end in ranges)):
            candidates.append((position, event))
    return bool(ids or concepts or days or ranges), candidates


def event_line(position, event):
    return [position, event['qid'], event['date'].isoformat(), event['result'] or None, list(event['concepts'])]


def learner_state(current, events):
    state = []
    for concept in current['concepts'][:6]:
        matches = [(pos, event) for pos, event in events if concept in event['concepts']]
        counts = Counter(event['result'] for _, event in matches)
        state.append({'知识点': concept, '历史作答次数': len(matches), '正确': counts['正确'], '错误': counts['错误'],
                      '最近三次记录': [event_line(pos, event)[:4] for pos, event in matches[-3:]]})
    return state


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def extended(temporal, section):
    history = temporal['history_info']
    return dict(temporal, history_info=[*history[:-1], f'{history[-1]}\n\n{section}'])


def followup_inputs(sample, queries, raw, temporal):
    current, events = source_events(sample)
    recognized, candidates = q2_candidates(events, current, queries['memory'][1]['query'])
    state = learner_state(current, events)
    statistic = temporal['history_info'][-1]
    if not statistic.startswith('### SUMMARY COMPUTED FROM THE FULL PAST INTERACTION HISTORY\n'):
        raise ValueError('First-round time-aware arm must end with the fixed full-history statistics')
    arms = {
        'rag': raw,
        'rag_stats': dict(raw, history_info=[*raw['history_info'], statistic]),
        'temporal': dict(temporal, history_info=temporal['history_info'][:-1]),
        'temporal_rag': temporal,
        'event_ledger': temporal,
        'learner_state': temporal,
    }
    if 0 < len(candidates) <= MAX_LEDGER_ROWS:
        entries = [event_line(pos, event) for pos, event in candidates]
        arms['event_ledger'] = extended(temporal, (
            '### 三道证据题中 Q2 的候选记录索引\n'
            '下列原始事实按时间排列，命中题问点名的题号、完整日期或带引号的已知知识点之一。'
            '请结合题问的全部条件筛选；每行依次是历史位置、题号、时间、正误和知识点。'
            '同题重做分别保留。\n' + json_text(entries)
        ))
    if state:
        arms['learner_state'] = extended(temporal, (
            '### 当前题各知识点的作答统计\n'
            '以下记录按时间排列，最近三次每行是历史位置、题号、时间和正误。作答统计来自完整可见历史。\n'
            + json_text(state)
        ))
    return arms, {'q2_recognized': recognized, 'q2_candidates': [pos for pos, _ in candidates],
                  'state_concepts': [row['知识点'] for row in state]}


def verified_rows(directory, manifest, name):
    path = directory / name
    if digest(path) != manifest['files_sha256'][name]:
        raise ValueError(f'First-round artifact checksum differs: {name}')
    return index(rows(path), '_key')


def prepare(output, first_round, tokenizer_dir, workers):
    from transformers import AutoTokenizer

    if os.environ.get('PYTHONHASHSEED') != '0' or output.exists():
        raise ValueError('Set PYTHONHASHSEED=0 and choose a new output directory')
    manifest = read_json(first_round / 'manifest.json')
    base = read_json(prior.BUNDLE / 'manifest.json')
    split = read_json(prior.SPLIT)['ids']
    pin = prior.verify_pins(prior.BUNDLE, base, tokenizer_dir)
    ids = manifest['ids']
    if (len(ids) != 140 or len(set(ids)) != 140 or ids != base['ids']
            or set(ids) != set(split['validation']) or set(ids) & (set(split['train']) | set(split['test']))
            or digest(prior.BUNDLE / 'manifest.json') != manifest['base_manifest_sha256']
            or digest(prior.SPLIT) != manifest['split_manifest_sha256']
            or manifest['upstream_commit'] != prior.COMMIT or manifest['base_revision'] != pin['revision']
            or manifest['inference'] != prior.EVAL or manifest['vllm_version'] != prior.VLLM_VERSION):
        raise ValueError('First-round experiment does not match the frozen development protocol')
    samples = index(rows(prior.BUNDLE / 'inputs.jsonl'), '_key')
    raw, temporal, queries = (verified_rows(first_round, manifest, name) for name in (
        'full.rag.jsonl', 'full.temporal_rag.jsonl', 'queries_only.jsonl'))
    if any(data.keys() != set(ids) for data in (samples, raw, temporal, queries)):
        raise ValueError('Cases, predictions inputs and public queries must match the fixed development IDs')
    prior.verify_queries(prior.BUNDLE, queries)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    upstream = load_upstream(prior.BUNDLE)
    prepared, audit = {}, {}
    for key in ids:
        controls, provenance = prior.sample_inputs(samples[key], queries[key])
        if controls['rag'] != raw[key] or controls['temporal_rag'] != temporal[key]:
            raise ValueError(f'First-round controls differ from the fixed construction: {key}')
        inputs, route = followup_inputs(samples[key], queries[key], raw[key], temporal[key])
        route.update(_key=key, prompt_tokens={}, q2_outside_temporal_count=len(
            set(route['q2_candidates']) - set(provenance['temporal_positions'])))
        for arm, sample in inputs.items():
            if upstream._sample_key(sample) != key:
                raise ValueError(f'Sample key changed: {key}')
            route['prompt_tokens'][arm] = prior.prompt_tokens(tokenizer, upstream, sample, queries[key])
        for arm in ('event_ledger', 'learner_state'):
            if route['prompt_tokens'][arm] > MAX_PROMPT_TOKENS:
                inputs[arm] = temporal[key]
                route['prompt_tokens'][arm] = route['prompt_tokens']['temporal_rag']
        route.update(ledger_applied=inputs['event_ledger'] != temporal[key],
                     state_applied=inputs['learner_state'] != temporal[key])
        prepared[key], audit[key] = inputs, route

    general = sorted(ids, key=lambda key: audit[key]['prompt_tokens']['temporal_rag'])
    repeat_ids = general[::9]
    smoke_ids = sorted(ids, key=lambda key: max(audit[key]['prompt_tokens'].values()), reverse=True)[:2]
    output.mkdir(parents=True)
    members = {f'upstream/{name}': prior.BUNDLE / 'upstream' / name for name in prior.UPSTREAM_FILES}
    for name in ('service.py', 'run_inference.sh'):
        members[name] = Path(__file__).with_name(name)
    members['base_model.json'] = prior.BUNDLE / 'base_model.json'
    for name, subset_ids, arms in (('full', ids, ARMS), ('smoke', smoke_ids, ARMS), ('repeat', repeat_ids, REPEAT_ARMS)):
        for arm in arms:
            path = output / f'{name}.{arm}.jsonl'
            src = 'temporal_rag' if arm == 'temporal_rag_repeat' else arm
            write_rows(path, (prepared[key][src] for key in subset_ids))
            members[path.name] = path
    query_path = output / 'queries_only.jsonl'
    write_rows(query_path, (queries[key] for key in ids))
    members[query_path.name] = query_path
    routing = output / 'routing.jsonl'
    write_rows(routing, audit.values())
    settings = {name: manifest[name] for name in ('base_manifest_sha256', 'split_manifest_sha256')}
    settings.update(status='prepared_on_cpu', base_model=prior.BASE_NAME, base_revision=pin['revision'],
                    inference=prior.EVAL, vllm_version=prior.VLLM_VERSION, upstream_commit=prior.COMMIT,
                    python_hash_seed=0, adapter_sha256={
                        name: base['files'][f'adapters/memory_first/{name}'] for name in prior.ADAPTER_FILES},
                    ids=ids, smoke_ids=smoke_ids, repeat_ids=repeat_ids, workers=workers,
                    arm_order=ARMS, subset_arms={'repeat': REPEAT_ARMS}, routing_sha256=digest(routing),
                    source_manifest_sha256=digest(first_round / 'manifest.json'),
                    files_sha256={name: digest(path) for name, path in members.items()},
                    code_sha256={path.name: digest(path) for path in (Path(__file__), Path(prior.__file__))},
                    max_ledger_rows=MAX_LEDGER_ROWS, max_structured_prompt_tokens=MAX_PROMPT_TOKENS,
                    coverage={name: sum(row[name] for row in audit.values())
                              for name in ('q2_recognized', 'ledger_applied', 'state_applied')},
                    ledger_candidate_rows_outside_temporal=sum(row['q2_outside_temporal_count']
                                                              for row in audit.values() if row['ledger_applied']),
                    ledger_adds_candidate_rows=sum(row['ledger_applied'] and row['q2_outside_temporal_count'] > 0
                                                   for row in audit.values()),
                    prompt_tokens={arm: {'median': median(row['prompt_tokens'][arm] for row in audit.values()),
                                         'max': max(row['prompt_tokens'][arm] for row in audit.values())} for arm in ARMS})
    prior.write_inference_bundle(output, members, settings)
    print(f'Prepared {len(ids)} dev cases on {len(ARMS)} arms; {settings["coverage"]}; {len(repeat_ids)} repeat cases')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--first-round', type=Path, default=FIRST_ROUND)
    parser.add_argument('--workers', type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.first_round.resolve(), args.tokenizer.resolve(), args.workers)


if __name__ == '__main__':
    main()
