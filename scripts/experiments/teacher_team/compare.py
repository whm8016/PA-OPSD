"""Compare a single teacher with the staged teacher team on the frozen Gold 32."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.diagnoser_evidence import evidence
from scripts.experiments.longtutor_reeval import run as service
from scripts.experiments.teacher_team import run as previous

gold = previous.gold
STAGES = ('single', 'diagnoser', 'pedagogue', 'verifier')
PROTOCOL = 'teacher-team-shared-evidence-v2'
SOURCE_MANIFEST = 'e92ad5d2870cb10007431b7da55c65c6bf8637ab8868b4727c14b31419eaf6cd'
BASE_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
BUNDLE_FILES = {
    'inputs.jsonl': '2c23372c6f7e1a666a166941c96d19437419e9a99453618fef118f5adca9706f',
    'references.jsonl': '13cda8eb59496eec8c0a6e3200fe0a3da507d32a77faa2dc6bb0a3d426f4b036',
}
ADAPTER_FILES = {
    'adapter_config.json': '803dd29dd1b7c00af4dfa36cccbdb7f578ffa8ff846f4459063b1775d1b687da',
    'adapter_model.safetensors': '40be07306e9070de6b47d9dfdd372ae7a6183b81aad1d728ae27141f4f37313b',
}
CHECKS = ('evidence_grounded', 'diagnosis_action_compatible', 'math_correct', 'no_answer_leak')
CHECK_VALUES = ('pass', 'fail', 'uncertain')
DIAGNOSIS = ('"diagnosis": 英文标签, "evidence": [最多3个可见历史序号], '
             '"reason": "基于记录的诊断理由"')
TEACHING = '"action": "A到H的一个字母", "feedback": "只给学生下一步提示"'
FINAL = '{' + DIAGNOSIS + ', ' + TEACHING + '}'
GUIDE = previous.SYSTEM + (
    '\nRecall Failure：曾掌握相同知识，时隔较久后提取困难。'
    'Conceptual Gap：相关基础概念还未建立，历史持续有困难或没有掌握证据。'
    'Procedural Error：错误更像出在计算、操作或多步程序。'
    'Transfer Deficit：基础情境会做，换到新表达或新情境时难以应用。'
    '\n只引用可见历史的原始序号，证据不足时用空列表或承认证据有限。'
    '同知识点记录不等于同题；比较题型、时间和历史正误。'
    '当前作答只有正误时，不能声称看到了错误步骤、学生选项或学生使用的公式。'
    '如题干图片不可见，不猜图中内容。用标准解核对数学，只给一个最小可执行提示，不直接报当前题最终答案。'
)
INSTRUCTIONS = {
    'single': f'独立完成证据定位、四类诊断、教学动作和反馈，并检查事实、数学、动作和是否泄题。输出 {FINAL}。',
    'diagnoser': '比较四种解释。先定位证据，再选择最有依据的诊断。输出 {' + DIAGNOSIS + '}。',
    'pedagogue': '按诊断建议选择最小教学动作和下一步提示。'
                 '如果诊断有疑点，给验证者一句异议；无异议时填空字符串。'
                 '输出 {' + TEACHING + ', "objection": "对诊断的异议"}。',
    'verifier': f'用原记录和标准解逐项核对前两步，必要时修正。'
                f'输出 {{"final": {FINAL}, "checks": {{'
                + ', '.join(f'"{key}": "pass、fail 或 uncertain"' for key in CHECKS)
                + '}, "revision": "修改理由，无修改时填空字符串"}。checks 评价修订后的 final。',
}


def build(source, output):
    source_manifest = source / 'manifest.json'
    if gold.digest(source_manifest) != SOURCE_MANIFEST:
        raise ValueError('Source manifest differs from the frozen Gold 32 bundle')
    manifest = gold.read_json(source_manifest)
    if manifest['scope'] != 'validation' or manifest['selection_tag'] != previous.TAG:
        raise ValueError('Expected the frozen teacher Gold 32 validation bundle')
    inputs = previous.load(source, 'inputs.jsonl', manifest)
    refs = previous.load(source, 'references.jsonl', manifest)
    split_path = gold.SPLIT / 'split_manifest.json'
    split = gold.read_json(split_path)
    reference_path = gold.SPLIT / 'development/references.jsonl'
    if gold.digest(reference_path) != split['files']['development/references.jsonl']:
        raise ValueError('Frozen development references changed')
    dev = gold.indexed(gold.read_rows(reference_path), 'sample_id')
    if list(dev) != split['ids']['validation'] or len(inputs) != 32 or inputs.keys() - dev.keys():
        raise ValueError('Expected exactly 32 frozen development samples')
    for key, ref in refs.items():
        target = dev[key]['target']
        if (ref['diagnosis'], ref['strategy'], ref['feedback']) != (
                target['diagnosis'], target['strategy'], target['content']):
            raise ValueError(f'Reference does not match frozen development sample: {key}')
    first = [next(key for key, ref in refs.items() if ref['diagnosis'] == label) for label in gold.LABELS]
    order = first + [key for key in inputs if key not in first]
    rows = []
    for key in order:
        row = inputs[key]
        features = evidence.build_features(row)
        rows.append({
            'sample_id': key, 'current': row['current'], 'solution': row['solution'],
            'history': [{'position': i, 'text': row['history'][i - 1]} for i in features['history_positions']],
            'history_summary': features['summary'],
        })
    with previous.lock_output(output):
        if (output / 'manifest.json').exists():
            raise ValueError('Choose a new bundle directory')
        gold.write_rows(output / 'inputs.jsonl', rows)
        gold.write_rows(output / 'references.jsonl', (refs[k] for k in order))
        if any(gold.digest(output / name) != expected for name, expected in BUNDLE_FILES.items()):
            raise ValueError('Built files differ from the frozen comparison bundle')
        gold.write_json(output / 'manifest.json', {
            'scope': 'validation', 'protocol': PROTOCOL, 'ids': order,
            'source_manifest_sha256': SOURCE_MANIFEST,
            'split_sha256': gold.digest(split_path),
            'evidence_code_sha256': gold.digest(Path(evidence.__file__)),
            'files': BUNDLE_FILES,
        })


def read_manifest(bundle):
    manifest = gold.read_json(bundle / 'manifest.json')
    if (manifest.get('protocol') != PROTOCOL or manifest.get('scope') != 'validation'
            or manifest.get('source_manifest_sha256') != SOURCE_MANIFEST or manifest.get('files') != BUNDLE_FILES
            or len(set(manifest.get('ids', ()))) != 32):
        raise ValueError('Expected the frozen Gold 32 comparison bundle')
    return manifest


def object_from(raw):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def string_field(value, key):
    field = value.get(key)
    return field.strip() if isinstance(field, str) else ''


def parse(raw, stage, row):
    value = object_from(raw)
    final = value.get('final') if stage == 'verifier' else value
    final = final if isinstance(final, dict) else {}
    positions = {record['position'] for record in row['history']}
    label = final.get('diagnosis')
    citations = final.get('evidence')
    action = string_field(final, 'action')
    citations_valid = (isinstance(citations, list) and len(citations) <= 3
                       and all(type(i) is int and i in positions for i in citations)
                       and len(set(citations)) == len(citations))
    result = {}
    if stage != 'pedagogue':
        result.update(diagnosis=label if label in gold.LABELS else None,
                      evidence=citations if citations_valid else [], evidence_list_valid=citations_valid,
                      reason=string_field(final, 'reason'))
    if stage != 'diagnoser':
        result.update(action=action if action in gold.ACTIONS else None, feedback=string_field(final, 'feedback'))
    if stage == 'pedagogue':
        result['objection'] = string_field(value, 'objection')
    if stage == 'verifier':
        reported = value.get('checks')
        reported = reported if isinstance(reported, dict) else {}
        result['checks'] = {k: reported.get(k) if reported.get(k) in CHECK_VALUES else None for k in CHECKS}
        result['revision'] = string_field(value, 'revision')
    return result


def messages(row, stage, saved):
    history = '\n\n'.join(f'历史序号 {r["position"]}\n{r["text"]}' for r in row['history'])
    common = f'## 学生历史\n{history}\n\n## 当前题\n{row["current"]}'
    common += '\n\n## 历史统计与教师依据\n' + json.dumps(
        {'history_summary': row['history_summary'], 'solution': row['solution'], 'rubric': gold.RUBRIC},
        ensure_ascii=False,
    )
    prior = []
    if stage in ('pedagogue', 'verifier'):
        prior.append(('诊断建议', 'diagnoser'))
    if stage == 'verifier':
        prior.append(('教学建议', 'pedagogue'))
    for heading, name in prior:
        proposal = parse(saved[name]['raw'], name, row)
        common += f'\n\n## {heading}\n' + json.dumps(proposal, ensure_ascii=False)
    return [{'role': 'system', 'content': GUIDE},
            {'role': 'user', 'content': common + '\n\n' + INSTRUCTIONS[stage]}]


def model_request(port, path, data=None, timeout=30):
    request = Request(f'http://127.0.0.1:{port}{path}', data=data, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as reply:
        return json.load(reply)


def generate(port, model, cap, prompt):
    body = json.dumps({
        'model': model, 'messages': prompt, 'temperature': 0, 'max_completion_tokens': cap,
        'response_format': {'type': 'json_object'},
    }, ensure_ascii=False).encode()
    started = time.monotonic()
    response = model_request(port, '/v1/chat/completions', data=body, timeout=240)
    if response['model'] != model:
        raise ValueError(f'Expected served model {model}, received {response["model"]}')
    choice = response['choices'][0]
    usage = response.get('usage') or {}
    return {'raw': choice['message']['content'], 'finish': choice['finish_reason'],
            'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
            'seconds': round(time.monotonic() - started, 3)}


def checkpoint_identity(args):
    pinned = service.verify_bundle(args.service_bundle)
    if pinned['base_revision'] != BASE_REVISION:
        raise ValueError('Expected the previous teacher comparison base revision')
    version = model_request(args.port, '/version')['version']
    if version != service.VLLM_VERSION:
        raise ValueError(f'Expected vLLM {service.VLLM_VERSION}, got {version}')
    models = gold.indexed(model_request(args.port, '/v1/models')['data'], 'id')
    adapter = models[args.model]
    base = models[service.BASE_NAME]
    if adapter.get('parent') != service.BASE_NAME or base.get('max_model_len') != service.EVAL['max_model_len']:
        raise ValueError('Expected the previous teacher comparison model registration')
    path = Path(adapter['root']).resolve(strict=True)
    if path != (args.service_bundle / 'adapters' / args.model).resolve():
        raise ValueError('Server is not using the specified Gold SFT bundle')
    for name, expected in ADAPTER_FILES.items():
        if gold.digest(path / name) != expected:
            raise ValueError(f'Gold SFT file hash mismatch: {name}')
    base_path = Path(base['root']).resolve(strict=True)
    pin = gold.read_json(args.service_bundle / 'base_model.json')
    hashes = {**pin['config_and_tokenizer_sha256'],
              **{name: info['sha256'] for name, info in pin['weights'].items()},
              'generation_config.json': pin['generation_config_sha256'],
              'model.safetensors.index.json': pin['weight_index_sha256']}
    for name, expected in hashes.items():
        if gold.digest(base_path / name) != expected:
            raise ValueError(f'Served base file hash mismatch: {name}')
    return {'port': args.port, 'base_revision': pinned['base_revision'], 'vllm_version': version,
            'service_manifest_sha256': gold.digest(args.service_bundle / 'manifest.json'),
            'adapter_sha256': ADAPTER_FILES, 'checkpoint_root': str(path), 'base_root': str(base_path)}


def infer(args):
    manifest = read_manifest(args.bundle)
    inputs = previous.load(args.bundle, 'inputs.jsonl', manifest)
    settings = {
        'protocol': PROTOCOL, 'model': args.model, 'service': checkpoint_identity(args),
        'temperature': 0, 'max_tokens_per_call': args.max_tokens,
        'bundle_sha256': gold.digest(args.bundle / 'manifest.json'),
        'sources': {name: gold.digest(Path(path)) for name, path in (
            ('compare', __file__), ('previous', previous.__file__), ('action_contract', gold.__file__))},
    }
    path = args.output / 'decisions.jsonl'
    stamp = args.output / 'settings.json'
    with previous.lock_output(args.output):
        if stamp.exists():
            if gold.read_json(stamp) != settings:
                raise ValueError('Existing output belongs to another run')
        elif path.exists():
            raise ValueError('Found predictions without settings')
        else:
            gold.write_json(stamp, settings)
        saved = gold.indexed(gold.read_rows(path), 'sample_id') if path.exists() else {}
        if saved.keys() - inputs.keys():
            raise ValueError('Saved predictions contain unexpected IDs')
        pending = [key for key in inputs if not set(STAGES) <= saved.get(key, {}).keys()]
        for key in pending[:args.batch]:
            record = saved.setdefault(key, {'sample_id': key})
            for stage in STAGES:
                if stage not in record:
                    result = generate(args.port, args.model, args.max_tokens, messages(inputs[key], stage, record))
                    record[stage] = result
                    gold.write_rows(path, (saved[k] for k in inputs if k in saved))
            complete = sum(set(STAGES) <= saved.get(k, {}).keys() for k in inputs)
            print(f'Paired rows: {complete}/{len(inputs)}', flush=True)


def metrics(parsed, saved, refs, stages):
    parsed = parsed[stages[-1]]
    correct = {k for k, p in parsed.items() if p['diagnosis'] == refs[k]['diagnosis']}
    actual = Counter(row['diagnosis'] for row in refs.values())
    hit = Counter(refs[k]['diagnosis'] for k in correct)
    predicted = Counter(p['diagnosis'] for p in parsed.values())
    has_related = {k for k, r in refs.items() if r['related_positions']}
    usage = [saved[k][stage] for k in refs for stage in stages]
    costs = {}
    for name in ('prompt_tokens', 'completion_tokens', 'seconds'):
        values = [response[name] for response in usage]
        costs[name] = sum(values) if all(v is not None for v in values) else None
    return {
        'diagnosis_correct': len(correct), 'diagnosis_missing': predicted[None],
        'four_class_macro_f1': sum(
            2 * hit[label] / (actual[label] + predicted[label])
            for label in gold.LABELS if actual[label] + predicted[label]
        ) / len(gold.LABELS),
        'by_class': {label: {'gold': actual[label], 'correct': hit[label], 'predicted': predicted[label]}
                     for label in gold.LABELS},
        'valid_nonempty_evidence': sum(bool(p['evidence']) for p in parsed.values()),
        'related_history_hits': sum(bool(set(parsed[k]['evidence']) & set(refs[k]['related_positions']))
                                    for k in has_related),
        'related_history_rows': len(has_related),
        'valid_action': sum(p['action'] is not None for p in parsed.values()),
        'strategy_candidate_matches': sum(p['action'] in gold.STRATEGY_ACTIONS[refs[k]['strategy']]
                                          for k, p in parsed.items()),
        'feedback_present': sum(bool(p['feedback']) for p in parsed.values()),
        'calls': len(usage), 'length_limit_hits': sum(response['finish'] == 'length' for response in usage),
        **costs,
    }, correct


def score(args):
    manifest = read_manifest(args.bundle)
    inputs = previous.load(args.bundle, 'inputs.jsonl', manifest)
    reference = previous.load(args.bundle, 'references.jsonl', manifest)
    with previous.lock_output(args.output):
        settings = gold.read_json(args.output / 'settings.json')
        if settings['protocol'] != PROTOCOL or settings['bundle_sha256'] != gold.digest(args.bundle / 'manifest.json'):
            raise ValueError('Predictions belong to a different protocol or sample bundle')
        saved = gold.indexed(gold.read_rows(args.output / 'decisions.jsonl'), 'sample_id')
        if saved.keys() - inputs.keys():
            raise ValueError('Saved predictions contain unexpected IDs')
        refs = {k: r for k, r in reference.items() if set(STAGES) <= saved.get(k, {}).keys()}
        if not refs or (len(refs) != len(inputs) and not args.partial):
            raise ValueError('Missing complete pairs. Use --partial only for a smoke comparison.')
        parsed = {s: {k: parse(saved[k][s]['raw'], s, inputs[k]) for k in refs} for s in STAGES}
        base, base_hits = metrics(parsed, saved, refs, STAGES[:1])
        team, team_hits = metrics(parsed, saved, refs, STAGES[1:])
        diagnoser_hits = {k for k in refs if parsed['diagnoser'][k]['diagnosis'] == refs[k]['diagnosis']}
        related_visible = sum(bool(set(ref['related_positions']) & {r['position'] for r in inputs[k]['history']})
                              for k, ref in refs.items())
        report = {
            'scope': manifest['scope'], 'protocol': PROTOCOL, 'rows': len(refs), 'expected_rows': len(inputs),
            'partial': len(refs) != len(inputs), 'related_history_visible_rows': related_visible,
            'single': base, 'team': team,
            'single_vs_team': gold.paired_counts(base_hits, team_hits, len(refs)),
            'diagnoser_vs_verifier': gold.paired_counts(diagnoser_hits, team_hits, len(refs)),
            'verifier_diagnosis_changes': sum(parsed['verifier'][k]['diagnosis'] != parsed['diagnoser'][k]['diagnosis']
                                              for k in refs),
            'verifier_action_changes': sum(parsed['verifier'][k]['action'] != parsed['pedagogue'][k]['action']
                                           for k in refs),
            'settings': settings, 'decisions_sha256': gold.digest(args.output / 'decisions.jsonl'),
        }
        gold.write_json(args.output / 'score.json', report)
        gold.write_rows(args.output / 'review.jsonl', (
            {**ref, **inputs[k], 'current_image_placeholder': bool(gold.IMAGE.search(inputs[k]['current'])),
             'decisions': {s: parsed[s][k] for s in STAGES}} for k, ref in refs.items()
        ))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build_cmd = sub.add_parser('build')
    build_cmd.add_argument('--source', type=Path, required=True)
    infer_cmd = sub.add_parser('infer')
    infer_cmd.add_argument('--port', required=True, type=positive_int)
    infer_cmd.add_argument('--model', choices=('sft',), default='sft')
    infer_cmd.add_argument('--service-bundle', type=Path, required=True)
    infer_cmd.add_argument('--max-tokens', type=positive_int, default=1024)
    infer_cmd.add_argument('--batch', type=positive_int, help='Limit newly processed sample pairs')
    score_cmd = sub.add_parser('score')
    score_cmd.add_argument('--partial', action='store_true')
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
