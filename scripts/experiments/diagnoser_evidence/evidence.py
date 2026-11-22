"""Derive citable records and knowledge-point statistics from public history."""

import re
from collections import Counter
from datetime import datetime
from math import log1p

MAX_CONCEPTS = 6
MATCHED_RECORDS = 8
RECENT_RECORDS = 4


def fields(text):
    parts = {}
    for line in reversed(text.splitlines()):
        key, separator, value = line.partition(':')
        if separator and key in ('知识点', '时间', '结果'):
            parts.setdefault(key, value.strip())
    match = re.match(r'^\[(\d+)\]', text)
    concepts = tuple(dict.fromkeys(s.strip() for s in parts.get('知识点', '').split(',') if s.strip()))
    date = parts.get('时间')
    try:
        date = datetime.fromisoformat(date) if date else None
    except ValueError:
        date = None
    return {'qid': match[1] if match else None, 'concepts': concepts, 'date': date,
            'result': parts.get('结果', '')}


def recency(current, previous):
    if current is None or previous is None or (current.utcoffset() is None) != (previous.utcoffset() is None):
        return None
    return (current - previous).total_seconds() / 86400


def build_features(row):
    current = fields(row['current'])
    records = {}
    for i, text in enumerate(row['history'], 1):
        record = fields(text)
        gap = recency(current['date'], record['date'])
        if gap is not None and gap < 0:
            raise ValueError(f'History position {i} occurs after the current exercise')
        records[i] = record
    counts = Counter(concept for record in records.values() for concept in record['concepts'])
    tags = [tag for tag in current['concepts'] if counts[tag]]
    tags.sort(key=lambda tag: counts[tag])
    summary = []
    for tag in tags[:MAX_CONCEPTS]:
        relevant = [record for record in records.values() if tag in record['concepts']]
        known = Counter(record['result'] for record in relevant)
        last = relevant[-1]['date']
        gap = recency(current['date'], last)
        summary.append({'知识点': tag, '历史作答数': len(relevant), '正确数': known['正确'], '错误数': known['错误'],
                        '最近作答': last.isoformat() if last else None, '距当前天数': round(gap, 3) if gap is not None else None,
                        '近三次结果': [record['result'] or '未记录' for record in relevant[-3:]]})

    weights = {tag: log1p(len(records) / counts[tag]) for tag in tags}
    related = []
    for index, record in records.items():
        exact = current['qid'] is not None and record['qid'] == current['qid']
        weight = sum(weights.get(tag, 0) for tag in record['concepts'])
        if exact or weight:
            related.append((index, exact, weight))
    related.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    selected = {index for index, _, _ in related[:MATCHED_RECORDS]}
    recent = {index for index, _, _ in related} if related else records.keys()
    selected.update(sorted(recent - selected, reverse=True)[:RECENT_RECORDS])
    return {'sample_id': row['sample_id'], 'history_positions': sorted(selected),
            'summary': {'历史作答总数': len(records), '共享知识点': summary,
                        '当前题的同题历史作答数': sum(exact for _, exact, _ in related)},
            'related_record_count': len(related)}
