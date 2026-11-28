"""Route candidate teaching skills from visible LongTutor history and build offline probes."""

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

from scripts.experiments.diagnoser_evidence.evidence import (
    build_features,
    fields,
    recency,
)

GAP_DAYS = 4
CARDS = {
    'recall': {
        'title': '间隔后先提示回忆', 'diagnosis': 'Recall Failure',
        'strategy': 'Retrieval Practice', 'action': 'C 追问',
        'when': '当前答错，匹配题最近一次答对已超过四天，匹配题已知正确率超过 80%。',
        'do': '确认两题使用同一条规则，提示学生回想当时用的规则，再请学生说出关键一步。',
        'avoid': '当前已明确是某一步算错、两题所用规则不一致或历史不足时，放弃这张卡。不要据此断言学生遗忘。',
    },
    'transfer': {
        'title': '用已做对的题搭桥', 'diagnosis': 'Transfer Deficit',
        'strategy': 'Analogical Transfer', 'action': 'F 举例',
        'when': '当前答错，历史无同题作答；知识点完全一致的题目中，至少做对两道不同的题、正确率超过 50%，且最近一条在四天内答对。',
        'do': '从已做对的旧题指出一项可以复用的方法和当前题一处发生变化的条件，让学生尝试对应。',
        'avoid': '找不到可说明的共同方法和条件变化，或现有证据只支持步骤失误，就放弃这张卡。不要直接给出当前题的解法和答案。',
    },
}
COHORTS = {
    'train64': ('probe_inputs.jsonl', 'review_targets.jsonl', 'files_sha256', 'train'),
    'dev24': ('inputs.jsonl', 'references.jsonl', 'files', 'validation'),
}


def route(current_text, history):
    current = fields(current_text)
    if current['result'] != '错误' or not current['qid'] or not current['date']:
        return {'card_id': None, 'why': '缺少可核对的当前错题或时间', 'evidence': []}

    tags = set(current['concepts'])
    matches = []
    unknown_time = False
    for position, text in enumerate(history, 1):
        old = fields(text)
        gap = recency(current['date'], old['date'])
        if gap is not None and gap < 0:
            raise ValueError(f'History position {position} occurs after the current exercise')
        same_question = current['qid'] == old['qid']
        if not old['qid'] or not (same_question or (tags and tags == set(old['concepts']))):
            continue
        if gap is None:
            unknown_time = True
        elif not (same_question and gap == 0):
            matches.append({'position': position, 'question_id': old['qid'], 'result': old['result'],
                            'gap_days': gap, 'same_question': same_question})

    # Records for distinct questions can share a timestamp. Preserve their given history order.
    matches.sort(key=lambda item: (item['gap_days'], -item['position']))
    counts = Counter(item['result'] for item in matches)
    wins, losses = counts['正确'], counts['错误']
    latest = matches[0] if matches else None
    decision = {'card_id': None, 'matched_correct': wins, 'matched_wrong': losses,
                'matched_positions': [item['position'] for item in matches], 'evidence': matches[:2]}

    if unknown_time:
        return dict(decision, why='存在无法核对时间的匹配记录')
    if latest is None:
        return dict(decision, why='没有同题或知识点组完全一致的可用历史')
    if latest['result'] != '正确':
        return dict(decision, why='最近一条匹配记录没有明确答对')
    if len(matches) != wins + losses:
        return dict(decision, why='匹配历史含缺失正误的记录')

    accuracy = wins / (wins + losses)
    if latest['gap_days'] > GAP_DAYS and accuracy > .8:
        return dict(decision, card_id='recall', why='此前答对，间隔超过四天，适合尝试回忆提示')
    if latest['gap_days'] < GAP_DAYS and wins >= 2 and accuracy > .5 and not any(item['same_question'] for item in matches):
        second_success = next((item for item in matches if item['result'] == '正确'
                               and item['question_id'] != latest['question_id']), None)
        if second_success:
            last_error = next((item for item in matches if item['result'] == '错误'), None)
            cited = [latest, second_success] + ([last_error] if last_error else [])
            return dict(decision, card_id='transfer', evidence=cited,
                        why='做对过多道同知识点异题，最近一次答对在四天内，可请教师核对是否可类比')
    return dict(decision, why='时间间隔或匹配答题记录未达到两张卡的条件')


def teacher_pair(row, decision):
    card = CARDS[decision['card_id']]
    system = ('你是数学辅导老师。根据当前错题和给出的同一学生的历史原文判断。'
              '区分历史中可核对的事实和推测。根据证据独立选诊断、策略和首个教学动作；'
              '可参考附加建议，同时核对适用条件。图像占位符不能证明图中内容。不要直接给当前题答案。'
              '返回 JSON 字段 diagnosis、strategy、action、evidence_positions、reason、feedback。'
              '诊断用 Recall Failure、Conceptual Gap、Procedural Error、Transfer Deficit 四类之一。'
              '策略用 Retrieval Practice、Conceptual Explanation、Stepwise Scaffolding、Analogical Transfer 四类之一。'
              '动作填写首个具体教学行为，例如追问、举例或概念提示。')
    positions = set(build_features(row)['history_positions'])
    positions.update(item['position'] for item in decision['evidence'])
    history = '\n\n'.join(f"历史第 {pos} 条\n{row['history'][pos - 1]}" for pos in sorted(positions))
    hint = '；'.join(f"第 {item['position']} 条距当前 {item['gap_days']:.2f} 天" for item in decision['evidence'])
    visible = f"当前题\n{row['current']}\n\n历史片段\n{history}\n\n请核对 {hint}。"
    guidance = f"\n\n候选教学卡「{card['title']}」\n{card['do']}\n{card['avoid']}"
    return {'sample_id': row['sample_id'], 'card_id': decision['card_id'], 'history_positions': sorted(positions),
            'without_card': [{'role': 'system', 'content': system}, {'role': 'user', 'content': visible}],
            'with_card': [{'role': 'system', 'content': system}, {'role': 'user', 'content': visible + guidance}]}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def load_cohort(source, group, split_ids):
    input_file, refs_file, hash_field, split_name = COHORTS[group]
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    declared = manifest[hash_field]
    if digest(source / input_file) != declared[input_file]:
        raise ValueError(f'{group} input differs from its frozen manifest')
    rows = read_rows(source / input_file)
    ids = [row['sample_id'] for row in rows]
    count = 64 if group == 'train64' else 24
    if ids != manifest['ids'] or len(set(ids)) != count or not set(ids) <= set(split_ids[split_name]):
        raise ValueError(f'{group} does not match its frozen Gold IDs')
    if group == 'train64':
        return [{'sample_id': row['sample_id'], 'current': row['x']['current_question'],
                 'history': row['x']['history']} for row in rows], source / refs_file, declared[refs_file]
    return [{key: row[key] for key in ('sample_id', 'current', 'history')} for row in rows], source / refs_file, declared[refs_file]


def render_report(groups, examples):
    lines = ['# 教学技能卡试跑', '', ('候选卡按可见历史触发。Gold 诊断在路由完成后才对照。'
             '卡片命中表示发现了可检查的线索，教学是否改善留待教师配对验证。'), '']
    for card in CARDS.values():
        lines += [f"## {card['title']}", '', f"- 触发：{card['when']}", f"- 动作：{card['action']}。{card['do']}",
                  f"- 停用：{card['avoid']}", '']
    for group, rows in groups.items():
        counts = Counter(row['card_id'] or 'none' for row in rows)
        recall_agree = sum(row['card_id'] == 'recall' and row['gold_diagnosis'] == CARDS['recall']['diagnosis'] for row in rows)
        transfer_agree = sum(row['card_id'] == 'transfer' and row['gold_diagnosis'] == CARDS['transfer']['diagnosis'] for row in rows)
        lines += [f'## {group}（{len(rows)} 道）', '',
                  f"回忆卡 {counts['recall']}，类比卡 {counts['transfer']}，未触发 {counts['none']}。", '',
                  (f"候选卡对应诊断与 Gold 的重合：回忆 {recall_agree}/{counts['recall']}，"
                   f"类比 {transfer_agree}/{counts['transfer']}。"), '',
                  '| 序号 | 当前题号 | Gold 诊断 | 候选卡 | 原历史位置／间隔 | 依据 |',
                  '|---:|---:|---|---|---|---|']
        for i, row in enumerate(rows, 1):
            title = CARDS[row['card_id']]['title'] if row['card_id'] else '—'
            ev = '、'.join(f"{item['position']}（{item['gap_days']:.2f} 天，{item['result'] or '无结果'}）"
                          for item in row['evidence']) or '—'
            lines.append(f"| {i} | {row['question_id']} | {row['gold_diagnosis']} | {title} | {ev} | {row['why']} |")
        lines.append('')
    lines += ['## 开发样本原文示例', '']
    for card_id, (i, row, decision) in examples.items():
        lines += [f"### 开发样本 {i} · {CARDS[card_id]['title']}", '', '当前题', '', '```text', row['current'], '```', '']
        for item in decision['evidence']:
            pos = item['position']
            lines += [f'历史第 {pos} 条', '', '```text', row['history'][pos - 1], '```', '']
    return '\n'.join(lines) + '\n'


def build(split, train, dev, out):
    split_ids = json.loads(split.read_text(encoding='utf-8'))['ids']
    groups, pairs, examples, source_hashes = {}, [], {}, {}
    for group, source in (('train64', train), ('dev24', dev)):
        rows, refs_path, refs_hash = load_cohort(source, group, split_ids)
        decisions = [route(row['current'], row['history']) for row in rows]
        if group == 'dev24':
            for i, (row, decision) in enumerate(zip(rows, decisions), 1):
                if decision['card_id']:
                    pairs.append(teacher_pair(row, decision))
                    examples.setdefault(decision['card_id'], (i, row, decision))

        if digest(refs_path) != refs_hash:
            raise ValueError(f'{group} reference differs from its frozen manifest')
        refs = read_rows(refs_path)
        if [ref['sample_id'] for ref in refs] != [row['sample_id'] for row in rows]:
            raise ValueError(f'{group} references do not align with routed rows')
        groups[group] = [dict(decision, sample_id=row['sample_id'], question_id=fields(row['current'])['qid'],
                              gold_diagnosis=ref['diagnosis'])
                         for row, decision, ref in zip(rows, decisions, refs)]
        source_hashes[group] = digest(source / 'manifest.json')

    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'cards.json', CARDS)
    for group, rows in groups.items():
        write_rows(out / f'{group}.audit.jsonl', rows)
    write_rows(out / 'dev24.teacher_pairs.jsonl', pairs)
    (out / '逐题核对.md').write_text(render_report(groups, examples), encoding='utf-8')
    write_json(out / 'run.json', {
        'scope': 'local skill routing and teacher-pair preparation', 'gap_days': GAP_DAYS,
        'split_sha256': digest(split), 'source_manifest_sha256': source_hashes,
        'code_sha256': {'runner': digest(Path(__file__)),
                        'evidence': digest(Path(__file__).parents[1] / 'diagnoser_evidence/evidence.py')},
        'counts': {group: dict(Counter(row['card_id'] or 'none' for row in rows)) for group, rows in groups.items()},
        'teacher_pairs': len(pairs), 'model_calls': 0,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, required=True, help='Frozen Gold split manifest')
    parser.add_argument('--train', type=Path, required=True, help='Existing training-only action pilot directory')
    parser.add_argument('--dev', type=Path, required=True, help='Existing Gold24 diagnosis input directory')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    build(args.split, args.train, args.dev, args.output)


if __name__ == '__main__':
    main()
