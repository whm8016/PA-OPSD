"""Build a fixed-history, matched and mismatched teaching-card pilot on Gold dev."""

import argparse
import os
from collections import Counter
from pathlib import Path

from scripts.experiments.longtutor_reeval.run import (
    digest,
    index,
    load_upstream,
    read_json,
    rows,
    write_rows,
)
from scripts.experiments.teaching_skills.run import CARDS
from scripts.experiments.training_free_memory import prepare as prior
from scripts.experiments.training_free_memory.prepare_followup import (
    FIRST_ROUND,
    verified_rows,
)

FOLLOWUP = FIRST_ROUND.with_name('followup_gold140_20261125_verified')
ARMS = ('no_card', 'matched_card', 'swapped_card')
SWAP = {'recall': 'transfer', 'transfer': 'recall'}


def add_card(sample, card_id):
    card = CARDS[card_id]
    note = ('### OPTIONAL TEACHING GUIDANCE\n'
            '这是一条候选教学动作。先核对可见记录与适用条件，只在适用时参考。\n'
            f'{card["when"]}\n{card["do"]}\n{card["avoid"]}')
    history = sample['history_info']
    if len(history) >= 200:
        raise ValueError('A teaching card would displace visible evidence')
    return dict(sample, history_info=[*history, note])


def prepare(output, tokenizer_dir):
    from transformers import AutoTokenizer

    if os.environ.get('PYTHONHASHSEED') != '0' or output.exists():
        raise ValueError('Set PYTHONHASHSEED=0 and choose a new output directory')
    base = read_json(prior.BUNDLE / 'manifest.json')
    pin = prior.verify_pins(prior.BUNDLE, base, tokenizer_dir)
    first_path, follow_path = (parent / 'manifest.json' for parent in (FIRST_ROUND, FOLLOWUP))
    first, follow = read_json(first_path), read_json(follow_path)
    split = read_json(prior.SPLIT)['ids']
    expected = set(split['validation'])
    if (len(expected) != 140 or expected & (set(split['train']) | set(split['test']))
            or first['ids'] != base['ids'] or follow['ids'] != base['ids'] or set(base['ids']) != expected
            or follow['source_manifest_sha256'] != digest(first_path)):
        raise ValueError('Unexpected experiment or development split')
    for manifest in (first, follow):
        if (manifest['base_manifest_sha256'] != digest(prior.BUNDLE / 'manifest.json')
                or manifest['split_manifest_sha256'] != digest(prior.SPLIT)
                or manifest['upstream_commit'] != prior.COMMIT or manifest['base_revision'] != pin['revision']
                or manifest['inference'] != prior.EVAL or manifest['vllm_version'] != prior.VLLM_VERSION):
            raise ValueError('Experiment is inconsistent with the pinned protocol')
    common = verified_rows(FOLLOWUP, follow, 'full.temporal.jsonl')
    queries = verified_rows(FOLLOWUP, follow, 'queries_only.jsonl')
    routing_source = FIRST_ROUND / 'routing.jsonl'
    if digest(routing_source) != first['routing_sha256']:
        raise ValueError('Archived routing checksum differs')
    routes = index(rows(routing_source), '_key')
    old_temporal = verified_rows(FIRST_ROUND, first, 'full.temporal_rag.jsonl')
    if any(items.keys() != expected for items in (common, queries, routes, old_temporal)):
        raise ValueError('Inputs, routing and public questions must match the development split')
    prior.verify_queries(prior.BUNDLE, queries)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    upstream = load_upstream(prior.BUNDLE)
    cases, audit = {}, {}
    for key in first['ids']:
        route = routes[key]
        card_id = route['applied_card']
        if not card_id:
            continue
        sample = common[key]
        positions = route['card_support_positions']
        if (not positions or not set(positions).issubset(route['temporal_positions'])
                or old_temporal[key]['history_info'][:-1] != sample['history_info']):
            raise ValueError(f'Visible supporting evidence differs: {key}')
        cases[key] = dict(zip(ARMS, (sample, add_card(sample, card_id), add_card(sample, SWAP[card_id]))))
        audit[key] = {'_key': key, 'matched': card_id, 'swapped': SWAP[card_id],
                      'support_positions': positions, 'temporal_positions': route['temporal_positions'],
                      'prompt_tokens': {arm: prior.prompt_tokens(tokenizer, upstream, item, queries[key])
                                        for arm, item in cases[key].items()}}
        if any(upstream._sample_key(item) != key for item in cases[key].values()):
            raise ValueError(f'Official sample key differs: {key}')
    if list(cases) != first['card_ids'] or {row['matched'] for row in audit.values()} != set(SWAP):
        raise ValueError('Expected both candidate cards on the archived trigger set')
    smoke_ids = [max((key for key in cases if audit[key]['matched'] == card),
                     key=lambda key: max(audit[key]['prompt_tokens'].values())) for card in SWAP]
    output.mkdir(parents=True)
    members = {f'upstream/{name}': prior.BUNDLE / 'upstream' / name for name in prior.UPSTREAM_FILES}
    members.update({name: Path(__file__).with_name(name) for name in ('service.py', 'run_inference.sh')})
    members['base_model.json'] = prior.BUNDLE / 'base_model.json'
    for subset, ids in (('full', cases), ('smoke', smoke_ids)):
        for arm in ARMS:
            name = f'{subset}.{arm}.jsonl'
            members[name] = output / name
            write_rows(members[name], (cases[key][arm] for key in ids))
    members['queries_only.jsonl'] = output / 'queries_only.jsonl'
    write_rows(members['queries_only.jsonl'], (queries[key] for key in cases))
    routing = output / 'routing.jsonl'
    write_rows(routing, audit.values())
    settings = {name: follow[name] for name in ('base_manifest_sha256', 'split_manifest_sha256',
                'adapter_sha256', 'inference', 'base_model', 'base_revision', 'vllm_version', 'upstream_commit')}
    settings.update(status='prepared_on_cpu', scope='validation_card_triggers', ids=list(cases),
                    smoke_ids=smoke_ids, arm_order=ARMS, workers=4, python_hash_seed=0,
                    cards=dict(Counter(row['matched'] for row in audit.values())),
                    routing_sha256=digest(routing), followup_manifest_sha256=digest(follow_path),
                    first_manifest_sha256=digest(first_path),
                    files_sha256={name: digest(path) for name, path in members.items()},
                    code_sha256={path.name: digest(path) for path in
                                 (Path(__file__), Path(prior.__file__), Path(prior.skills.__file__))})
    prior.write_inference_bundle(output, members, settings)
    print(f'Prepared {len(cases)} triggered dev cases across {len(ARMS)} arms: {settings["cards"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.tokenizer.resolve())


if __name__ == '__main__':
    main()
