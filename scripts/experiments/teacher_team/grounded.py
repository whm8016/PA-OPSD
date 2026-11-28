"""Run a paired teacher experiment with evidence gates and blind diagnosis review."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.experiments.diagnoser_evidence import evidence
from scripts.experiments.teacher_team import compare, grounding

gold = compare.gold
previous = compare.previous
PROTOCOL = 'teacher-team-grounded-v3'
SPLIT_SHA256 = 'd1497e692514c01ed09a26c69ac555272621f07d3877d0a2f86d11389be6bf5f'
COHORTS = {
    'regression32': (32, compare.SOURCE_MANIFEST),
    'holdout24': (24, '718bb0dc83a1cf5cf017363cbe87c83c49b2cb01c3bae629cc2fe50fff89e1be'),
}
BUNDLE_FILES = {
    'regression32': {
        'inputs.jsonl': '170b66470978075d66d1d3888b0fbd36281db7cd48d56d97efcb3e9ef5a33fa6',
        'references.jsonl': '5c2017e21af89b5a17b454512c223ba1b92a91d31880c82af04b0f5413b42684',
    },
    'holdout24': {
        'inputs.jsonl': 'c865ede13a57fab8b270d1f8833ee07f979f4b4ff1ae79fad8fdd2513589d925',
        'references.jsonl': 'a7a1e13d723acf17cfc9407e201f3bf6b7313d3246db42dbdf164c43e29b4fda',
    },
}
REGRESSION_SMOKE = {'409', '4086', '1267', '2379'}
DIAGNOSIS = (
    '\n先检查题干和真实作答记录，再比较最可能的解释和一个备选。只按给定记录引用历史，'
    '原样复制历史卡片的 id、result、interval，自行判断题型关系和它支持还是反驳首选。'
    '同标签仅供检索，需读题确认方法是否一致。依据不足时填 uncertain，同时保留用于评测的首选标签。'
    '\nRecall Failure 需要同一技能曾做对，且相隔较久。4 天只是本轮复核阈值。'
    '\nConceptual Gap 看同一基础方法的独立错误。'
    '\nProcedural Error 看同一方法近期做对，但本题计算或多步骤执行可能受阻。正误无法证明具体错步。'
    '\nTransfer Deficit 看基础方法有成功记录，当前题需要把它用到不同表达或情境。'
    '\n诊断字段示例（值仅示意）'
    '{"diagnosis":"Recall Failure","alternative":"Procedural Error","confidence":"supported",'
    '"evidence":[{"id":"H03","result":"correct","interval":"four_days_or_more","relation":"same_skill",'
    '"role":"support","method":"具体共同方法或关键差异"}],"why_over_alternative":"用一句话比较两种解释"}。'
    'confidence 取 supported/uncertain。relation 取 same_skill/foundation/other。'
    'role 取 support/counter，无可引用记录时 evidence 为空。'
)
TEACHING = (
    '\n根据标准解检查数学。给一个具体且学生做得动的下一步。不要给当前题最终答案，'
    '不要声称看到了学生未提供的做法。诊断未确认时，动作只选 C 或 G，优先问一个能区分错因的问题。'
    '当前题图缺失时，动作选 C，询问图或缺失信息，勿猜图。'
    '输出教学字段 {"action":"A到H的一个字母","feedback":"直接对学生说的一步提示或一个问题"}。'
)
NON_OBSERVABLE = {'current_image_unavailable', 'current_error_unobserved'}
BLIND_FLAGS = grounding.BLIND_FLAGS


def build(cohort, source, output):
    expected_count, source_hash = COHORTS[cohort]
    source_manifest = source / 'manifest.json'
    if gold.digest(source_manifest) != source_hash:
        raise ValueError(f'Source manifest differs from the fixed cohort {cohort}')
    manifest = gold.read_json(source_manifest)
    inputs = previous.load(source, 'inputs.jsonl', manifest)
    references = previous.load(source, 'references.jsonl', manifest)
    split_path = gold.SPLIT / 'split_manifest.json'
    if gold.digest(split_path) != SPLIT_SHA256:
        raise ValueError('Development split manifest has changed')
    split = gold.read_json(split_path)
    if manifest['scope'] != 'validation' or len(inputs) != expected_count or inputs.keys() - set(split['ids']['validation']):
        raise ValueError('Expected the fixed development samples only')
    order = list(inputs)
    if cohort == 'regression32':
        smoke = [k for k, row in inputs.items() if evidence.fields(row['current'])['qid'] in REGRESSION_SMOKE]
        if len(smoke) != len(REGRESSION_SMOKE):
            raise ValueError('Unexpected regression smoke sample list')
        order = smoke + [k for k in inputs if k not in smoke]
    with previous.lock_output(output):
        if (output / 'manifest.json').exists():
            raise ValueError('Choose a new bundle directory')
        gold.write_rows(output / 'inputs.jsonl', (grounding.cards(inputs[k]) for k in order))
        gold.write_rows(output / 'references.jsonl', (references[k] for k in order))
        if any(gold.digest(output / name) != sha for name, sha in BUNDLE_FILES[cohort].items()):
            raise ValueError('Built files differ from the frozen bundle')
        gold.write_json(output / 'manifest.json', {
            'scope': 'validation', 'protocol': PROTOCOL, 'cohort': cohort, 'ids': order,
            'source_manifest_sha256': source_hash, 'split_sha256': SPLIT_SHA256,
            'grounding_code_sha256': gold.digest(Path(grounding.__file__)),
            'files': BUNDLE_FILES[cohort],
        })


def manifest_for(bundle):
    manifest = gold.read_json(bundle / 'manifest.json')
    count, origin = COHORTS[manifest['cohort']]
    if (manifest['scope'] != 'validation' or manifest['protocol'] != PROTOCOL
            or len(manifest['ids']) != count or len(set(manifest['ids'])) != count
            or manifest['source_manifest_sha256'] != origin or manifest['split_sha256'] != SPLIT_SHA256
            or manifest['files'] != BUNDLE_FILES[manifest['cohort']]
            or manifest['grounding_code_sha256'] != gold.digest(Path(grounding.__file__))):
        raise ValueError('Invalid development bundle')
    return manifest


def prompt(row, stage, decision=None):
    data = {'observations': grounding.visible(row), 'standard_solution': row['solution'], 'rubric': gold.RUBRIC}
    task = {
        'single': '一次完成诊断和辅导，所有输出字段合并在同一个 JSON 对象。' + DIAGNOSIS + TEACHING,
        'diagnoser': '完成初次诊断，任务仅包括证据和诊断。' + DIAGNOSIS,
        'verifier': '你独立复核此病例，请主动用原始记录排除看似合理却没有证据的解释。' + DIAGNOSIS,
        'pedagogue': '只完成辅导。以下诊断状态由程序给出，未确认时只按观察到的事实提供帮助。' + TEACHING,
    }[stage]
    if stage == 'pedagogue':
        data['decision'] = {
            'status': 'provisional' if decision['status'] == 'eligible' else 'unresolved',
            'diagnosis': decision['label'],
            'evidence_ids': [card['id'] for card in decision['evidence']],
        }
    return [
        {'role': 'system', 'content': previous.SYSTEM},
        {'role': 'user', 'content': task + '\n\n个案数据\n' + json.dumps(data, ensure_ascii=False)},
    ]


def requires_review(first):
    return bool(first['flags']) and not (NON_OBSERVABLE & set(first['flags']))


def reconcile(first, review):
    if first['status'] == 'eligible':
        return {**first, 'route': 'initial'}
    if review:
        first_flags = set(first['flags'])
        review_flags = set(review['flags'])
        same_label = first['label'] == review['label']
        plausible_first = first_flags <= {'missing_comparison', 'model_uncertain'} | BLIND_FLAGS
        if (review['status'] == 'eligible' and (not plausible_first or same_label)
                and not (first_flags & BLIND_FLAGS)):
            return {**review, 'route': 'blind_review'}
        if first_flags <= BLIND_FLAGS and review_flags <= BLIND_FLAGS and same_label:
            first_support = {card['id']: card for card in grounding.supporting_cards(first['evidence'])}
            shared_support = [
                card for card in grounding.supporting_cards(review['evidence'])
                if card['id'] in first_support and card['relation'] == first_support[card['id']]['relation']
            ]
            if grounding.has_class_support(review['label'], shared_support):
                counter = [card for card in review['evidence'] if card['role'] == 'counter']
                return {**review, 'evidence': shared_support + counter, 'status': 'eligible', 'route': 'blind_agreement'}
    return {'status': 'unresolved', 'label': None, 'route': 'unresolved', 'evidence': [],
            'flags': list(first['flags']), 'alternative': None, 'comparison': ''}


def assess(row, record):
    first = grounding.diagnosis(record['diagnoser'], row)
    if requires_review(first) != ('verifier' in record):
        raise ValueError('Saved verification does not match the routing rule')
    review = grounding.diagnosis(record['verifier'], row) if 'verifier' in record else None
    team = reconcile(first, review)
    single = grounding.diagnosis(record['single'], row)
    return {
        'single': single, 'first': first, 'review': review, 'team': team,
        'single_lesson': grounding.lesson(record['single'], row, single),
        'team_lesson': grounding.lesson(record['pedagogue'], row, team),
    }


def source_digests():
    paths = {
        'runner': __file__, 'grounding': grounding.__file__, 'compare': compare.__file__,
        'evidence': evidence.__file__, 'teacher_v1': previous.__file__, 'action_contract': gold.__file__,
        'service': compare.service.__file__,
    }
    return {name: gold.digest(Path(path)) for name, path in paths.items()}


def input_support(inputs, refs, keys):
    """Count rows that could meet each gate if the model chose the reference label."""
    totals = {label: Counter() for label in gold.LABELS}
    related_visible = 0
    for key in keys:
        row, ref = inputs[key], refs[key]
        counts = totals[ref['diagnosis']]
        counts['rows'] += 1
        positions = {i for card in row['cards'] for i in card['positions']}
        related_visible += bool(positions.intersection(ref['related_positions']))
        if row['current']['image_missing'] or row['current']['result'] != 'incorrect':
            counts['unobservable_current'] += 1
            continue
        candidates = [{**c, 'relation': 'same_skill'} for c in row['cards']
                      if not c['image_missing'] and c['tag_match'] != 'background']
        specific = [c for c in candidates if c['tag_match'] != 'shared_tags']
        counts['with_initial_gate_inputs'] += grounding.has_class_support(ref['diagnosis'], specific)
        counts['with_initial_or_blind_agreement_inputs'] += grounding.has_class_support(ref['diagnosis'], candidates)
    names = ('rows', 'unobservable_current', 'with_initial_gate_inputs', 'with_initial_or_blind_agreement_inputs')
    return {
        'by_gold_class': {label: {name: counts[name] for name in names} for label, counts in totals.items()},
        'rows_with_visible_reference_history': related_visible,
        'rows_with_reference_history': sum(bool(refs[key]['related_positions']) for key in keys),
    }


def infer(args):
    manifest = manifest_for(args.bundle)
    inputs = previous.load(args.bundle, 'inputs.jsonl', manifest)
    settings = {
        'protocol': PROTOCOL, 'cohort': manifest['cohort'], 'model': 'sft',
        'service': compare.checkpoint_identity(args), 'temperature': 0, 'max_tokens_per_call': args.max_tokens,
        'bundle_sha256': gold.digest(args.bundle / 'manifest.json'),
        'sources': source_digests(),
        'team_max_calls_per_row': 3, 'recall_days': grounding.RECALL_DAYS,
    }
    path = args.output / 'decisions.jsonl'
    stamp = args.output / 'settings.json'
    with previous.lock_output(args.output):
        if stamp.exists():
            if gold.read_json(stamp) != settings:
                raise ValueError('Existing output belongs to another run')
        elif path.exists():
            raise ValueError('Found decisions without settings')
        else:
            gold.write_json(stamp, settings)
        saved = gold.indexed(gold.read_rows(path), 'sample_id') if path.exists() else {}
        if saved.keys() - inputs.keys():
            raise ValueError('Saved decisions contain unexpected IDs')
        pending = [k for k in inputs if 'pedagogue' not in saved.get(k, {})]
        for key in pending[:args.batch]:
            row = inputs[key]
            record = saved.setdefault(key, {'sample_id': key})
            for stage in ('single', 'diagnoser', 'verifier', 'pedagogue'):
                if stage in record:
                    continue
                decision = None
                if stage in ('verifier', 'pedagogue'):
                    first = grounding.diagnosis(record['diagnoser'], row)
                    if stage == 'verifier' and not requires_review(first):
                        continue
                    review = grounding.diagnosis(record['verifier'], row) if 'verifier' in record else None
                    decision = reconcile(first, review)
                record[stage] = compare.generate(args.port, 'sft', args.max_tokens, prompt(row, stage, decision))
                gold.write_rows(path, (saved[k] for k in inputs if k in saved))
            print(f'Paired rows: {sum("pedagogue" in row for row in saved.values())}/{len(inputs)}', flush=True)


def metrics(keys, parsed, saved, refs, arm):
    decisions = {key: parsed[key][arm] for key in keys}
    proposed = {key: parsed[key]['first' if arm == 'team' else arm]['label'] for key in keys}
    eligible = {key: row for key, row in decisions.items() if row['status'] == 'eligible'}
    shared_candidate = {key: row for key, row in decisions.items()
                        if row['label'] and set(row['flags']) <= BLIND_FLAGS}
    actual = Counter(refs[key]['diagnosis'] for key in keys)
    proposed_counts = Counter(proposed.values())
    proposed_hits = Counter(refs[k]['diagnosis'] for k, label in proposed.items() if label == refs[k]['diagnosis'])
    predicted = Counter(row['label'] for row in eligible.values())
    correct = {key for key, row in eligible.items() if row['label'] == refs[key]['diagnosis']}
    hits = Counter(refs[key]['diagnosis'] for key in correct)
    related = {key for key in keys if refs[key]['related_positions']}
    stages = ('single',) if arm == 'single' else ('diagnoser', 'verifier', 'pedagogue')
    usage = [saved[k][stage] for k in keys for stage in stages if stage in saved[k]]
    lessons = {key: parsed[key][f'{arm}_lesson'] for key in keys}
    usage_totals = {}
    for name in ('prompt_tokens', 'completion_tokens', 'seconds'):
        values = [reply[name] for reply in usage]
        usage_totals[name] = sum(values) if all(value is not None for value in values) else None
    return {
        'raw_top1_correct': sum(proposed_hits.values()),
        'raw_top1_four_class_macro_f1': sum(
            2 * proposed_hits[label] / (actual[label] + proposed_counts[label])
            for label in gold.LABELS if actual[label] + proposed_counts[label]
        ) / len(gold.LABELS),
        'shared_evidence_candidate_rows': len(shared_candidate),
        'shared_evidence_candidate_correct': sum(row['label'] == refs[key]['diagnosis']
                                                 for key, row in shared_candidate.items()),
        'eligible': len(eligible), 'unresolved': len(keys) - len(eligible), 'accepted_correct': len(correct),
        'accepted_accuracy_all_rows': len(correct) / len(keys),
        'accepted_accuracy_on_eligible': len(correct) / len(eligible) if eligible else None,
        'four_class_macro_f1_all_rows': sum(
            2 * hits[label] / (actual[label] + predicted[label])
            for label in gold.LABELS if actual[label] + predicted[label]
        ) / len(gold.LABELS),
        'by_class': {label: {'gold': actual[label], 'raw_prediction': proposed_counts[label],
                             'raw_top1_correct': proposed_hits[label], 'eligible_prediction': predicted[label],
                             'eligible_correct': hits[label]}
                     for label in gold.LABELS},
        'related_history_hits': sum(any(set(card['positions']) & set(refs[k]['related_positions'])
                                        for card in eligible[k]['evidence']) for k in related & eligible.keys()),
        'rows_with_reference_history': len(related),
        'shared_checks_candidate_lessons': sum(row['shared_checks_candidate'] for row in lessons.values()),
        'candidate_lessons': sum(row['status'] == 'candidate' for row in lessons.values()),
        'candidate_strategy_matches': sum(lesson['status'] == 'candidate'
                                          and lesson['action'] in gold.STRATEGY_ACTIONS[refs[key]['strategy']]
                                          for key, lesson in lessons.items()),
        'teaching_flags': dict(Counter(flag for lesson in lessons.values() for flag in lesson['flags'])),
        'calls': len(usage), 'length_limit_hits': sum(reply['finish'] == 'length' for reply in usage),
        **usage_totals,
    }, correct


def score(args):
    manifest = manifest_for(args.bundle)
    inputs = previous.load(args.bundle, 'inputs.jsonl', manifest)
    refs = previous.load(args.bundle, 'references.jsonl', manifest)
    with previous.lock_output(args.output):
        settings = gold.read_json(args.output / 'settings.json')
        if (settings['protocol'] != PROTOCOL
                or settings['bundle_sha256'] != gold.digest(args.bundle / 'manifest.json')
                or settings['sources'] != source_digests()):
            raise ValueError('Predictions belong to a different protocol, sample bundle, or grading code')
        path = args.output / 'decisions.jsonl'
        saved = gold.indexed(gold.read_rows(path), 'sample_id')
        if saved.keys() - inputs.keys():
            raise ValueError('Saved decisions contain unexpected IDs')
        completed = [k for k in inputs if {'single', 'diagnoser', 'pedagogue'} <= saved.get(k, {}).keys()]
        if not completed or (len(completed) != len(inputs) and not args.partial):
            raise ValueError('Missing complete pairs. Use --partial for a smoke run.')
        parsed = {k: assess(inputs[k], saved[k]) for k in completed}
        subsets = {'all': completed, 'current_text_visible': [k for k in completed if not inputs[k]['current']['image_missing']],
                   'current_image_missing': [k for k in completed if inputs[k]['current']['image_missing']]}
        summaries = {}
        for subset, keys in subsets.items():
            if keys:
                single, single_correct = metrics(keys, parsed, saved, refs, 'single')
                team, team_correct = metrics(keys, parsed, saved, refs, 'team')
                summaries[subset] = {'rows': len(keys), 'single': single, 'team': team,
                                     'input_support': input_support(inputs, refs, keys),
                                     'paired_accepted_correct': gold.paired_counts(single_correct, team_correct, len(keys))}
        report = {
            'scope': manifest['scope'], 'protocol': PROTOCOL, 'cohort': manifest['cohort'],
            'rows': len(completed), 'expected_rows': len(inputs), 'partial': len(completed) != len(inputs),
            'routing': dict(Counter(row['team']['route'] for row in parsed.values())),
            'reviews_triggered': sum(row['review'] is not None for row in parsed.values()),
            'initial_flags': dict(Counter(flag for row in parsed.values() for flag in row['first']['flags'])),
            'settings': settings, 'decisions_sha256': gold.digest(path), 'subsets': summaries,
        }
        gold.write_json(args.output / 'score.json', report)
        gold.write_rows(args.output / 'review.jsonl', (
            {**inputs[k], 'reference': refs[k], 'decisions': parsed[k]} for k in completed
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build_cmd = sub.add_parser('build')
    build_cmd.add_argument('--cohort', choices=COHORTS, required=True)
    build_cmd.add_argument('--source', type=Path, required=True)
    infer_cmd = sub.add_parser('infer')
    infer_cmd.add_argument('--port', required=True, type=compare.positive_int)
    infer_cmd.add_argument('--service-bundle', type=Path, required=True)
    infer_cmd.add_argument('--max-tokens', type=compare.positive_int, default=1024)
    infer_cmd.add_argument('--batch', type=compare.positive_int, help='Limit newly processed sample pairs')
    infer_cmd.set_defaults(model='sft')
    score_cmd = sub.add_parser('score')
    score_cmd.add_argument('--partial', action='store_true')
    for command in (build_cmd, infer_cmd, score_cmd):
        command.add_argument('--output', type=Path, required=True)
    for command in (infer_cmd, score_cmd):
        command.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.cohort, args.source, args.output)
    else:
        {'infer': infer, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
