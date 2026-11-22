"""Compare a single tutor with a three-role team on frozen LongTutor development data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.action_pilot import run as gold
from scripts.experiments.longtutor_reeval.run import lock_output

STAGES = ('single', 'diagnoser', 'pedagogue', 'verifier')
LIMITS = (256, 128, 256, 256)
TAG = 'teacher-team-gold32-v1|'
ACTION_NAMES = '\n'.join(f'{k} {v[0]}。{v[1]}' for k, v in gold.ACTIONS.items())
SYSTEM = (
    '你是小学数学导师。依据学生历史、当前题和答题结果判断。历史只有正误时，勿臆造学生的具体解法。'
    '标准解仅供判断，请给下一步提示，并给学生留出思考空间。只输出 JSON。\n'
    f'诊断标签使用英文原名，选择：{", ".join(gold.LABELS)}。\n教学动作：\n{ACTION_NAMES}'
)
DECISION = ('输出 {"diagnosis": 英文标签, "evidence": [1至3个历史序号], "action": "A到H的一个字母", '
            '"reason": "40字内的证据理由", "feedback": "60字内、直接对学生说的话"}。'
            'evidence 使用历史条目左边的序号，当前题没有历史序号。')
INSTRUCTIONS = {
    'single': '独立选证据、给诊断并提出下一步教学动作。' + DECISION,
    'diagnoser': ('定位最相关的证据，对比遗忘、概念缺口、程序错误、迁移不足。'
                  '输出 {"diagnosis": 英文标签, "evidence": [1至3个历史序号], "reason": "40字内的证据理由"}。'),
    'pedagogue': '根据诊断建议选下一步的教学动作并拟一句反馈。可根据原记录修订建议。' + DECISION,
    'verifier': '核验诊断、历史证据和教学建议。修正无依据的推断、过早泄题或与诊断不符的动作。给出最终决定。' + DECISION,
}


def select(targets, histories, bank, quota):
    groups = {label: [] for label in gold.LABELS}
    for key, target in targets.items():
        h = histories[key]
        current = h['question_info']
        qid = gold.QUESTION_ID.match(current).group(1)
        item = bank[qid]
        if item.get('answer') and isinstance(item.get('analysis'), str) and item['analysis'].strip():
            groups[target['diagnosis']].append((key, current, item))
    picked = []
    for label, group in groups.items():
        if len(group) < quota:
            raise ValueError(f'Insufficient rows with a worked solution: {label}')
        # Prefer readable current exercises. Hash order keeps selection independent of model outputs.
        group.sort(key=lambda row: (bool(gold.IMAGE.search(row[1])), sha256((TAG + row[0]).encode()).digest()))
        picked.extend(group[:quota])
    return sorted(picked, key=lambda row: sha256((TAG + 'order|' + row[0]).encode()).digest())


def build(out):
    split = gold.read_json(gold.SPLIT / 'split_manifest.json')
    history = gold.REPRO / 'gold_inputs.jsonl'
    if gold.digest(history) != gold.read_json(gold.REPRO / 'preflight.json')['gold_input_sha256']:
        raise ValueError('Official history snapshot changed')
    histories = gold.indexed((dict(row, sample_id=gold.history_key(row)) for row in gold.read_rows(history)), 'sample_id')
    bank = {str(row['id']): row['info'] for row in gold.read_rows(gold.BANK)}
    out.mkdir(parents=True, exist_ok=True)
    for scope, file, quota in (('train', 'train.jsonl', 1), ('validation', 'references.jsonl', 8)):
        source = gold.SPLIT / 'development' / file
        if gold.digest(source) != split['files'][f'development/{file}']:
            raise ValueError(f'Frozen development split changed: {file}')
        rows = gold.indexed(gold.read_rows(source), 'sample_id')
        if list(rows) != split['ids'][scope]:
            raise ValueError(f'Frozen development IDs differ: {scope}')
        targets = {k: (row['target'] if scope == 'validation' else row['teacher_privileged']['gold_reference'])
                   for k, row in rows.items()}
        picked = select(targets, histories, bank, quota)
        inputs, refs = [], []
        for key, current, item in picked:
            h = histories[key]
            related = set(h['related_history'])
            inputs.append({'sample_id': key, 'history': h['history_info'], 'current': current,
                           'solution': {'answer': item['answer'], 'analysis': item['analysis']}})
            refs.append({'sample_id': key, 'diagnosis': targets[key]['diagnosis'], 'strategy': targets[key]['strategy'],
                         'feedback': targets[key]['content'],
                         'related_positions': [i for i, row in enumerate(h['history_info'], 1) if row in related]})
        folder = out / scope
        folder.mkdir(exist_ok=True)
        gold.write_rows(folder / 'inputs.jsonl', inputs)
        gold.write_rows(folder / 'references.jsonl', refs)
        gold.write_json(folder / 'manifest.json', {
            'scope': scope, 'ids': [row['sample_id'] for row in inputs], 'selection_tag': TAG,
            'class_counts': dict(Counter(row['diagnosis'] for row in refs)),
            'current_images': sum(bool(gold.IMAGE.search(row['current'])) for row in inputs),
            'question_bank_sha256': gold.digest(gold.BANK),
            'files': {name: gold.digest(folder / name) for name in ('inputs.jsonl', 'references.jsonl')},
            'gold_sft_adapter_sha256': {name: gold.digest(gold.GOLD_SFT / name) for name in gold.ADAPTER_FILES},
        })


def load(folder, name, manifest):
    path = folder / name
    if gold.digest(path) != manifest['files'][name]:
        raise ValueError(f'Pilot data differs: {path}')
    result = gold.indexed(gold.read_rows(path), 'sample_id')
    if list(result) != manifest['ids']:
        raise ValueError(f'Pilot IDs differ: {path}')
    return result


def decision(raw, history_length):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    label = value.get('diagnosis')
    action = value.get('action')
    evidence = value.get('evidence')
    if (not isinstance(evidence, list) or not 1 <= len(evidence) <= 3
            or not all(type(i) is int and 1 <= i <= history_length for i in evidence)
            or len(set(evidence)) != len(evidence)):
        evidence = []
    return {'diagnosis': label if label in gold.LABELS else None,
            'action': action if isinstance(action, str) and action in gold.ACTIONS else None,
            'evidence': evidence,
            'reason': value.get('reason') if isinstance(value.get('reason'), str) else '',
            'feedback': value.get('feedback') if isinstance(value.get('feedback'), str) else ''}


def messages(row, stage, done):
    history = '\n\n'.join(f'历史序号 {i}\n{line}' for i, line in enumerate(row['history'], 1))
    context = f'## 学生历史\n{history}\n\n## 当前题\n{row["current"]}'
    context += '\n\n## 教师依据\n' + json.dumps({'solution': row['solution'], 'rubric': gold.RUBRIC}, ensure_ascii=False)
    for earlier in ('diagnoser', 'pedagogue'):
        if earlier in done and stage in ('pedagogue', 'verifier'):
            result = decision(done[earlier]['raw'], len(row['history']))
            context += f'\n\n## {earlier}建议\n' + json.dumps(result, ensure_ascii=False)
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': context + '\n\n' + INSTRUCTIONS[stage]}]


def infer(args):
    from openai import APIError, OpenAI

    manifest = gold.read_json(args.bundle / 'manifest.json')
    inputs = load(args.bundle, 'inputs.jsonl', manifest)
    path = args.output / 'decisions.jsonl'
    stamp = args.output / 'settings.json'
    limits = (args.max_new_tokens,) * len(STAGES) if args.max_new_tokens is not None else LIMITS
    settings = {'model': args.model, 'input_sha256': manifest['files']['inputs.jsonl'],
                'runner_sha256': gold.digest(Path(__file__)), 'contract_sha256': gold.digest(Path(gold.__file__)),
                'adapter': manifest['gold_sft_adapter_sha256'], 'limits': list(limits), 'temperature': 0}
    with lock_output(args.output), OpenAI(base_url=f'http://127.0.0.1:{args.port}/v1', api_key='EMPTY',
                                          timeout=240, max_retries=0) as client:
        gold.checkpoint_root(client, args.model, manifest)
        if stamp.exists():
            if gold.read_json(stamp) != settings:
                raise ValueError('Output belongs to another run')
        elif path.exists():
            raise ValueError('Find predictions without settings')
        else:
            gold.write_json(stamp, settings)
        saved = gold.indexed(gold.read_rows(path), 'sample_id') if path.exists() else {}
        if saved.keys() - inputs.keys():
            raise ValueError('Saved results contain unexpected samples')

        def request(key, stage, tokens):
            row = inputs[key]
            prompt = messages(row, stage, saved.get(key, {}))
            started = time.monotonic()
            completion = client.chat.completions.create(
                model=args.model, messages=prompt, temperature=0, max_completion_tokens=tokens,
                response_format={'type': 'json_object'},
            )
            choice = completion.choices[0]
            return {'raw': choice.message.content, 'finish': choice.finish_reason,
                    'completion_tokens': completion.usage.completion_tokens if completion.usage else None,
                    'seconds': round(time.monotonic() - started, 3)}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for stage, tokens in zip(STAGES, limits, strict=True):
                tasks = {pool.submit(request, key, stage, tokens): key for key in inputs if stage not in saved.get(key, {})}
                failures = []
                for future in as_completed(tasks):
                    key = tasks[future]
                    try:
                        result = future.result()
                    except APIError as exc:
                        failures.append(f'{key}: {type(exc).__name__}: {exc}')
                        continue
                    saved.setdefault(key, {'sample_id': key})[stage] = result
                    gold.write_rows(path, (saved[k] for k in inputs if k in saved))
                    print(f'{stage} {sum(stage in row for row in saved.values())}/{len(inputs)}', flush=True)
                if failures:
                    raise RuntimeError('Requests failed, saved completed responses:\n' + '\n'.join(failures))


def score(args):
    manifest = gold.read_json(args.bundle / 'manifest.json')
    inputs = load(args.bundle, 'inputs.jsonl', manifest)
    refs = load(args.bundle, 'references.jsonl', manifest)
    settings = gold.read_json(args.output / 'settings.json')
    if settings['input_sha256'] != manifest['files']['inputs.jsonl'] or settings['adapter'] != manifest['gold_sft_adapter_sha256']:
        raise ValueError('Saved run belongs to a different pilot')
    saved = gold.indexed(gold.read_rows(args.output / 'decisions.jsonl'), 'sample_id')
    if set(saved) != set(refs) or any(not set(STAGES) <= row.keys() for row in saved.values()):
        raise ValueError('Missing pilot results')
    results, correct = {}, {}
    reports = {stage: {} for stage in STAGES}
    denom = Counter(row['diagnosis'] for row in refs.values())
    for stage in STAGES:
        parsed = {k: decision(r[stage]['raw'], len(inputs[k]['history'])) for k, r in saved.items()}
        reports[stage] = parsed
        hits = {k for k in refs if parsed[k]['diagnosis'] == refs[k]['diagnosis']}
        correct[stage] = hits
        hits_by_class = Counter(refs[k]['diagnosis'] for k in hits)
        guessed = Counter(row['diagnosis'] for row in parsed.values())
        has_ref = [k for k in refs if refs[k]['related_positions']]
        results[stage] = {
            'diagnosis_correct': len(hits), 'diagnosis_missing': guessed[None],
            'diagnosis_macro_f1': sum(2 * hits_by_class[label] / (denom[label] + guessed[label]) for label in denom) / len(denom),
            'diagnosis_by_class': {label: {'gold': denom[label], 'correct': hits_by_class[label], 'predicted': guessed[label]}
                                   for label in gold.LABELS},
            'valid_evidence': sum(bool(row['evidence']) for row in parsed.values()),
            'benchmark_related_overlap': sum(bool(set(parsed[k]['evidence']) & set(refs[k]['related_positions'])) for k in has_ref),
            'benchmark_related_rows': len(has_ref),
            'valid_action': sum(row['action'] is not None for row in parsed.values()),
            'strategy_candidate_matches': sum(parsed[k]['action'] in gold.STRATEGY_ACTIONS[refs[k]['strategy']] for k in refs),
            'actions': dict(Counter(row['action'] for row in parsed.values() if row['action'])),
            'completion_tokens': sum(row[stage]['completion_tokens'] or 0 for row in saved.values()),
            'length_limit_hits': sum(row[stage]['finish'] == 'length' for row in saved.values()),
        }
    report = {'scope': manifest['scope'], 'rows': len(refs), 'class_counts': manifest['class_counts'],
              'current_images': manifest['current_images'], 'stages': results,
              'single_vs_team': gold.paired_counts(correct['single'], correct['verifier'], len(refs)),
              'settings': settings, 'references_sha256': manifest['files']['references.jsonl']}
    gold.write_json(args.output / 'score.json', report)
    for k, row in refs.items():
        row['current'] = inputs[k]['current']
        row['decisions'] = {stage: reports[stage][k] for stage in STAGES}
        indices = set(row['related_positions']) | {i for choices in row['decisions'].values() for i in choices['evidence']}
        row['history_evidence'] = {i: inputs[k]['history'][i - 1] for i in sorted(indices)}
    gold.write_rows(args.output / 'review.jsonl', refs.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    b = commands.add_parser('build')
    i = commands.add_parser('infer')
    s = commands.add_parser('score')
    for tool in (b, i, s):
        tool.add_argument('--output', type=Path, required=True)
    for tool in (i, s):
        tool.add_argument('--bundle', type=Path, required=True)
    i.add_argument('--port', type=int, default=8018)
    i.add_argument('--model', default='sft')
    i.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    i.add_argument('--max-new-tokens', type=int, help='Set the same completion budget for all four roles')
    args = parser.parse_args()
    if args.command == 'infer' and args.max_new_tokens is not None and args.max_new_tokens < 1:
        i.error('--max-new-tokens must be positive')
    if args.command == 'build':
        build(args.output)
    else:
        {'infer': infer, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
