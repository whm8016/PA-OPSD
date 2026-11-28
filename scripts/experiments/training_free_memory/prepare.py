"""Prepare paired, training-free LongTutor inputs on the frozen Gold dev split."""

import argparse
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from statistics import median
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.experiments.diagnoser_evidence import evidence
from scripts.experiments.longtutor_reeval.run import (
    BASE_NAME,
    EVAL,
    VLLM_VERSION,
    digest,
    index,
    load_upstream,
    read_json,
    rows,
    write_json,
    write_rows,
)
from scripts.experiments.teaching_skills import run as skills

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / 'outputs/pa_opsd/longtutor_reeval_gold140/official_base_vs_memory_first_20261123/bundle'
SPLIT = ROOT / 'outputs/pa_opsd/gold_split_20261120/split_manifest.json'
QUERIES = BUNDLE.parent.parent / 'history_input_audit_20261124/checked/queries_only.jsonl'
UPSTREAM_REPO = ROOT / 'data/longtutor/repo'
COMMIT = 'ba40af3b9e976f5960c42a15a645c0e9c3a7d718'
ARMS = ('rag', 'temporal_rag', 'temporal_rag_card')
HISTORY_BUDGET = 12
EXPLICIT_ID = re.compile(r'\[\s*(\d+)\s*\]|(?:ID|题号|编号)\s*(?:为|[:：])?\s*(\d+)', re.IGNORECASE)
UPSTREAM_FILES = ('eval_ai_tutor.py', 'compute_ai_tutor_eval_metrics.py', 'gpt_memory_diagnose.py', 'openai_helper.py')
ADAPTER_FILES = ('adapter_config.json', 'adapter_model.safetensors')


def sample_inputs(sample, queries):
    current = sample['question_info']
    history = sample['history_info']
    # Extant retrieval ranks same-question and rare overlapping concepts, then adds recent hits.
    raw = evidence.build_features({'sample_id': sample['_key'], 'current': current, 'history': history})
    raw_positions = raw['history_positions']
    current_fields = evidence.fields(current)
    records = {position: evidence.fields(text) for position, text in enumerate(history, 1)}
    gaps = {position: evidence.recency(current_fields['date'], row['date']) for position, row in records.items()}
    if any(gap is None or gap < 0 for gap in gaps.values()):
        raise ValueError(f"{sample['_key']} has missing, incompatible or future timestamps")

    concepts = set(current_fields['concepts'])
    related = [position for position, row in records.items() if concepts.intersection(row['concepts'])]
    correct = [position for position in related if records[position]['result'] == '正确']
    wrong = [position for position in related if records[position]['result'] == '错误']
    latest_correct, latest_wrong = (
        min(positions, key=lambda pos: (gaps[pos], -pos), default=None) for positions in (correct, wrong)
    )
    anchors = {pos for pos in (latest_correct, latest_wrong) if pos is not None}
    named_ids = {first or second for item in queries['memory']
                 for first, second in EXPLICIT_ID.findall(item['query'])}
    named_positions = {pos for pos, row in records.items() if row['qid'] in named_ids}
    required = anchors | named_positions
    if len(required) > HISTORY_BUDGET:
        temporal_positions = list(records)
    else:
        remaining = max(0, len(raw_positions) - len(anchors))
        temporal = anchors | set(sorted(set(raw_positions) - anchors, reverse=True)[:remaining])
        remaining = max(0, len(raw_positions) - len(required))
        temporal_positions = sorted(required | set(sorted(temporal - required, reverse=True)[:remaining]))

    accuracy = f'{len(correct) / len(related):.2f}' if related else 'N/A'
    days = int(gaps[latest_correct]) if latest_correct is not None else 'N/A'
    summary = (
        '### SUMMARY COMPUTED FROM THE FULL PAST INTERACTION HISTORY\n'
        f'- Past exercises sharing any knowledge point with the current exercise: {len(related)}\n'
        f'- Correct answers among those past exercises: {len(correct)}\n'
        f'- Historical accuracy on those exercises: {accuracy}\n'
        f'- Whole days since the latest correct answer on those exercises: {days}'
    )
    common = {key: sample[key] for key in ('uid', 'question_info', '_key')}
    raw_input = dict(common, history_info=[history[pos - 1] for pos in raw_positions])
    temporal_input = dict(common, history_info=[history[pos - 1] for pos in temporal_positions] + [summary])
    decision = skills.route(current, history)
    card_id = decision['card_id']
    supporting_positions = [item['position'] for item in decision['evidence']] if card_id else []
    card_visible = bool(card_id and set(supporting_positions).issubset(temporal_positions))
    card_input = temporal_input
    if card_visible:
        card = skills.CARDS[card_id]
        # This is optional teaching guidance. The existing card's diagnosis and strategy labels stay local.
        guidance = (
            '### OPTIONAL TEACHING GUIDANCE\n'
            '以下是一条候选教学动作。请依据可见原始记录核对适用条件，用于选择下一步教学动作。\n'
            f'{card["do"]}\n{card["avoid"]}'
        )
        # The official prompt only retains the last 200 history items, including the summary slot.
        card_history = [*temporal_input['history_info'][:-1], f'{summary}\n\n{guidance}']
        card_input = dict(temporal_input, history_info=card_history)

    audit = {
        '_key': sample['_key'], 'rag_positions': raw_positions, 'temporal_positions': temporal_positions,
        'latest_related_correct': latest_correct, 'latest_related_wrong': latest_wrong,
        'named_positions': sorted(named_positions), 'full_history_fallback': len(required) > HISTORY_BUDGET,
        'candidate_card': card_id, 'applied_card': card_id if card_visible else None,
        'card_support_positions': supporting_positions, 'route_reason': decision['why'],
    }
    return dict(zip(ARMS, (raw_input, temporal_input, card_input))), audit


def verify_pins(bundle, manifest, tokenizer_dir):
    if (manifest['upstream_commit'] != COMMIT or manifest['inference'] != EVAL
            or manifest['vllm_version'] != VLLM_VERSION or manifest['scope'] != 'validation'):
        raise ValueError('Unexpected evaluation protocol')
    filenames = ['inputs.jsonl', 'gold.jsonl', 'base_model.json', *(f'upstream/{name}' for name in UPSTREAM_FILES)]
    for name in filenames:
        if digest(bundle / name) != manifest['files'][name]:
            raise ValueError(f'Frozen file hash mismatch: {name}')
    for name in UPSTREAM_FILES:
        pinned = subprocess.check_output(['git', '-C', str(UPSTREAM_REPO), 'cat-file', 'blob',
                                          f'{COMMIT}:scripts/{name}'])
        if (bundle / 'upstream' / name).read_bytes() != pinned:
            raise ValueError(f'Upstream source differs from pinned commit: {name}')
    pin = read_json(bundle / 'base_model.json')
    if any((tokenizer_dir / name).exists() for name in ('chat_template.jinja', 'chat_template.json', 'chat_templates')):
        raise ValueError('Use the embedded chat template from the pinned tokenizer')
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        if digest(tokenizer_dir / name) != pin['config_and_tokenizer_sha256'][name]:
            raise ValueError(f'Pinned tokenizer file hash mismatch: {name}')
    return pin


def prompt_tokens(tokenizer, upstream, sample, queries):
    system, user = upstream._build_prompts(sample, queries)
    encoded = tokenizer.apply_chat_template(upstream.build_messages(system, user), tokenize=True,
                                            add_generation_prompt=True, return_dict=False)
    count = len(encoded)
    if count + EVAL['max_tokens'] > EVAL['max_model_len']:
        raise ValueError(f"{sample['_key']} exceeds the pinned context budget")
    return count


def verify_queries(bundle, queries):
    for query in queries.values():
        if (set(query) != {'_key', 'memory'} or len(query['memory']) != 3
                or any(set(item) != {'query'} or not item['query'].strip() for item in query['memory'])):
            raise ValueError('Only the three published memory questions may enter the inference package')
    published_queries = {
        row['_key']: [{'query': item['query']} for item in row['memory']]
        for row in rows(bundle / 'gold.jsonl')
    }
    if queries.keys() != published_queries.keys() or any(
            queries[key]['memory'] != published_queries[key] for key in queries):
        raise ValueError('Query text differs from the frozen public memory questions')


def prepare(output, bundle, query_path, tokenizer_dir):
    from transformers import AutoTokenizer

    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Set PYTHONHASHSEED=0 before launching Python to match LongTutor prompt ordering')
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    manifest = read_json(bundle / 'manifest.json')
    pin = verify_pins(bundle, manifest, tokenizer_dir)
    split = read_json(SPLIT)['ids']
    ids = manifest['ids']
    dev_ids = set(ids)
    samples = index(rows(bundle / 'inputs.jsonl'), '_key')
    queries = index(rows(query_path), '_key')
    if not (len(ids) == 140 and len(dev_ids) == 140 and dev_ids == samples.keys() == queries.keys()
            and dev_ids == set(split['validation']) and dev_ids.isdisjoint(split['test'])
            and dev_ids.isdisjoint(split['train'])):
        raise ValueError('Inputs must match the frozen 140 validation IDs')
    # Cross-check the public text against the pinned annotation file, then discard the annotation rows.
    verify_queries(bundle, queries)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    upstream = load_upstream(bundle)
    prepared, audit = {arm: {} for arm in ARMS}, {}
    for key in ids:
        sample = samples[key]
        if len(sample['history_info']) != 199 or upstream._sample_key(sample) != key:
            raise ValueError(f'Unexpected original input: {key}')
        arm_inputs, detail = sample_inputs(sample, queries[key])
        detail['prompt_tokens'] = {}
        for arm, arm_input in arm_inputs.items():
            if upstream._sample_key(arm_input) != key:
                raise ValueError(f'Unexpected sample key in {arm}: {key}')
            prepared[arm][key] = arm_input
            detail['prompt_tokens'][arm] = prompt_tokens(tokenizer, upstream, arm_input, queries[key])
        audit[key] = detail

    triggered = [key for key in ids if audit[key]['applied_card']]
    if not triggered:
        raise ValueError('No supported card triggers in the dev split')
    longest = sorted(ids, key=lambda key: max(audit[key]['prompt_tokens'].values()), reverse=True)
    longest_card = next(key for key in longest if audit[key]['applied_card'])
    longest_other = next(key for key in longest if key != longest_card)
    smoke_ids = [longest_card, longest_other]

    output.mkdir(parents=True, exist_ok=False)
    members = {
        'run_inference.sh': Path(__file__).with_name('run_inference.sh'),
        'service.py': Path(__file__).with_name('service.py'),
        'base_model.json': bundle / 'base_model.json',
        **{f'upstream/{name}': bundle / 'upstream' / name for name in UPSTREAM_FILES},
    }
    for subset, keys in (('full', ids), ('smoke', smoke_ids)):
        for arm in ARMS:
            path = output / f'{subset}.{arm}.jsonl'
            write_rows(path, (prepared[arm][key] for key in keys))
            members[path.name] = path
    query_output = output / 'queries_only.jsonl'
    write_rows(query_output, (queries[key] for key in ids))
    members[query_output.name] = query_output
    routing_output = output / 'routing.jsonl'
    write_rows(routing_output, audit.values())

    run_manifest = {
        'status': 'prepared_on_cpu', 'ids': ids, 'smoke_ids': smoke_ids, 'card_ids': triggered,
        'cards': dict(Counter(audit[key]['applied_card'] for key in triggered)),
        'candidate_cards': dict(Counter(row['candidate_card'] for row in audit.values() if row['candidate_card'])),
        'upstream_commit': COMMIT, 'base_model': BASE_NAME, 'base_revision': pin['revision'],
        'inference': EVAL, 'vllm_version': VLLM_VERSION, 'python_hash_seed': 0,
        'adapter_sha256': {name: manifest['files'][f'adapters/memory_first/{name}'] for name in ADAPTER_FILES},
        'base_manifest_sha256': digest(bundle / 'manifest.json'), 'split_manifest_sha256': digest(SPLIT),
        'queries_source_sha256': digest(query_path), 'arm_order': ARMS,
        'routing_sha256': digest(routing_output),
        'code_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in (
            Path(__file__), Path(evidence.__file__), Path(skills.__file__),
        )},
        'prompt_tokens': {arm: {
            'median': median(row['prompt_tokens'][arm] for row in audit.values()),
            'max': max(row['prompt_tokens'][arm] for row in audit.values()),
        } for arm in ARMS},
        'full_history_fallbacks': sum(row['full_history_fallback'] for row in audit.values()),
        'files_sha256': {name: digest(path) for name, path in members.items()},
    }
    write_inference_bundle(output, members, run_manifest)
    print(f'Prepared {len(ids)} dev samples, {len(triggered)} cards, and 2 paired smoke samples: {output}')


def write_inference_bundle(output, members, manifest):
    config = output / 'manifest.json'
    write_json(config, manifest)
    members[config.name] = config
    checksums = output / 'SHA256SUMS'
    checksums.write_text(''.join(f'{digest(path)}  {name}\n' for name, path in members.items()),
                         encoding='ascii', newline='\n')
    members[checksums.name] = checksums
    with ZipFile(output / 'inference.zip', 'x', compression=ZIP_DEFLATED) as archive:
        for name, path in members.items():
            archive.write(path, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='A new result directory')
    parser.add_argument('--tokenizer', type=Path, required=True, help='Local pinned Qwen tokenizer directory')
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--queries', type=Path, default=QUERIES)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.bundle.resolve(), args.queries.resolve(), args.tokenizer.resolve())


if __name__ == '__main__':
    main()
