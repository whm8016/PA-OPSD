"""Build evidence cards and check which teacher claims the records can support."""

import re
from collections import Counter, defaultdict

from scripts.experiments.diagnoser_evidence import evidence
from scripts.experiments.teacher_team import compare

gold = compare.gold
DISPLAY_LIMIT = 12
RECALL_DAYS = 4
RESULTS = {'正确': 'correct', '错误': 'incorrect'}
RELATIONS = ('same_skill', 'foundation', 'other')
BLIND_FLAGS = frozenset(('needs_semantic_confirmation', 'ambiguous_support'))
METADATA = re.compile(r'\n知识点: [^\n]*\n时间: [^\n]*\n结果: [^\n]*\s*$')
DIRECT_OPTION = re.compile(
    r'(?:^|[，,。！!；;\n])\s*(?:所以|因此|故)?\s*'
    r'(?:应选|选|选择|(?:正确)?(?:答案|选项)\s*(?:是|为|[:：]))\s*[（(]?([A-D])(?![A-Za-z])',
    re.IGNORECASE,
)
DIRECT_ANSWER = re.compile(
    r'(?:^|[，,。！!；;\n])\s*(?:所以|因此|故)?\s*'
    r'(?:(?:最终答案|最后结果|正确答案|答案|结果)\s*(?:是|为|等于|[:：])|'
    r'(?:空格|空白)(?:里|中)?\s*(?:应填|填入|填写|填))\s*\$*\s*(-?\d+(?:\.\d+)?)(?![\d.])',
)
NUMBER = re.compile(r'-?\d+(?:\.\d+)?')


def body(text):
    return METADATA.sub('', gold.QUESTION_ID.sub('', text, count=1)).strip()


def interval(gap_days):
    if gap_days is None:
        return 'unknown'
    if gap_days == 0:
        return 'same_timestamp'
    if gap_days < 1:
        return 'under_day'
    if gap_days < RECALL_DAYS:
        return 'one_to_four_days'
    return 'four_days_or_more'


def cards(row):
    current = evidence.fields(row['current'])
    parsed = [evidence.fields(text) for text in row['history']]
    frequencies = Counter(tag for item in parsed for tag in item['concepts'])
    attempts = {}
    conflicting = defaultdict(set)
    for position, (item, text) in enumerate(zip(parsed, row['history']), 1):
        qid, date = item['qid'], item['date']
        gap = evidence.recency(current['date'], date)
        if gap is not None and gap < 0:
            raise ValueError(f'History position {position} is after the current exercise')
        if qid and qid == current['qid'] and gap == 0:
            continue
        attempt = (qid, date) if qid and date else (None, position)
        result = RESULTS.get(item['result'], 'unknown')
        conflicting[attempt].add(result)
        key = (attempt, result)
        if key in attempts:
            attempts[key]['positions'].append(position)
            continue
        common = set(current['concepts']) & set(item['concepts'])
        same_tags = len(common) >= 2 and common == set(current['concepts']) == set(item['concepts'])
        match = ('same_item' if qid and qid == current['qid'] else 'same_tags' if same_tags else
                 'shared_tags' if common else 'background')
        attempts[key] = {
            'positions': [position], 'qid': qid, 'result': result, 'interval': interval(gap),
            'gap_minutes': round(gap * 1440, 2) if gap is not None else None,
            'tag_match': match, 'common_tags': sorted(common), 'concepts': item['concepts'],
            'question': body(text), 'image_missing': bool(gold.IMAGE.search(text)),
        }

    candidates = [card for (attempt, _), card in attempts.items() if len(conflicting[attempt]) == 1]
    tier = {'same_item': 3, 'same_tags': 2, 'shared_tags': 1, 'background': 0}
    candidates.sort(key=lambda c: (
        tier[c['tag_match']], sum(1 / frequencies[tag] for tag in c['common_tags']), c['positions'][-1],
    ), reverse=True)
    # Preserve correct and incorrect attempts and an older-success contrast within the card budget.
    relevant = [c for c in candidates if c['tag_match'] != 'background']
    kept = (
        [c for c in relevant if c['result'] == 'correct'][:2]
        + [c for c in relevant if c['result'] == 'incorrect'][:2]
        + [c for c in relevant if c['result'] == 'correct' and c['interval'] == 'four_days_or_more'][:2]
        + relevant
        + sorted((c for c in candidates if c['tag_match'] == 'background'),
                 key=lambda c: c['positions'][-1], reverse=True)[:3]
    )
    unique = {}
    for card in kept:
        unique.setdefault(card['positions'][0], card)
    selected = list(unique.values())[:DISPLAY_LIMIT]
    selected.sort(key=lambda c: c['positions'][0])
    return {
        'sample_id': row['sample_id'],
        'current': {
            'question': body(row['current']), 'concepts': list(current['concepts']),
            'result': RESULTS.get(current['result'], 'unknown'),
            'image_missing': bool(gold.IMAGE.search(row['current'])),
        },
        'solution': row['solution'],
        'cards': [dict(card, id=f'H{i:02}') for i, card in enumerate(selected, 1)],
    }


def visible(row):
    return {**row['current'], 'history': [
        {field: value for field, value in card.items() if field not in ('qid', 'positions')}
        for card in row['cards']
    ]}


def has_class_support(label, cards):
    same_skill = [c for c in cards if c['relation'] == 'same_skill']
    if label == 'Recall Failure':
        return any(c['result'] == 'correct' and c['interval'] == 'four_days_or_more' for c in same_skill)
    if label == 'Conceptual Gap':
        return sum(c['result'] == 'incorrect' for c in same_skill) >= 2
    if label == 'Procedural Error':
        return any(c['result'] == 'correct' and c['interval'] in ('under_day', 'one_to_four_days') for c in same_skill)
    if label == 'Transfer Deficit':
        return any(c['result'] == 'correct' and c['relation'] in ('same_skill', 'foundation')
                   and c['tag_match'] != 'same_item' for c in cards)
    return False


def supporting_cards(cards):
    return [c for c in cards if c['role'] == 'support' and not c['image_missing'] and c['tag_match'] != 'background']


def diagnosis(reply, row):
    raw = compare.object_from(reply['raw'])
    label = raw.get('diagnosis')
    alternative = raw.get('alternative')
    reason = compare.string_field(raw, 'why_over_alternative')
    claims = raw.get('evidence')
    cards_by_id = {card['id']: card for card in row['cards']}
    errors = []
    if reply['finish'] != 'stop' or label not in gold.LABELS:
        errors.append('invalid_output')
    if alternative not in gold.LABELS or alternative == label or not reason:
        errors.append('missing_comparison')
    if raw.get('confidence') != 'supported':
        errors.append('model_uncertain')
    if row['current']['image_missing']:
        errors.append('current_image_unavailable')
    if row['current']['result'] != 'incorrect':
        errors.append('current_error_unobserved')
    valid = []
    used = set()
    if not isinstance(claims, list) or not 1 <= len(claims) <= 3:
        errors.append('missing_or_invalid_evidence')
    else:
        for item in claims:
            claim_id = item.get('id') if isinstance(item, dict) else None
            card = cards_by_id.get(claim_id) if isinstance(claim_id, str) else None
            if card is None or claim_id in used:
                errors.append('invalid_evidence_id')
                continue
            used.add(claim_id)
            if item.get('result') != card['result'] or item.get('interval') != card['interval']:
                errors.append('evidence_fact_mismatch')
                continue
            relation = item.get('relation')
            role = item.get('role')
            method = compare.string_field(item, 'method')
            if relation not in RELATIONS or role not in ('support', 'counter') or not method:
                errors.append('invalid_claim')
                continue
            valid.append({**card, 'relation': relation, 'role': role, 'method': method})

    support = supporting_cards(valid)
    if not has_class_support(label, support):
        errors.append('weak_class_support')
    else:
        if not has_class_support(label, [c for c in support if c['tag_match'] != 'shared_tags']):
            errors.append('needs_semantic_confirmation')
        if any(has_class_support(other, support) for other in gold.LABELS if other != label):
            errors.append('ambiguous_support')
    return {
        'label': label if label in gold.LABELS else None,
        'alternative': alternative if alternative in gold.LABELS else None,
        'comparison': reason, 'evidence': valid, 'flags': sorted(set(errors)),
        'status': 'needs_review' if errors else 'eligible',
    }


def lesson(reply, row, decision):
    raw = compare.object_from(reply['raw'])
    action = compare.string_field(raw, 'action')
    feedback = compare.string_field(raw, 'feedback')
    errors = []
    if reply['finish'] != 'stop' or action not in gold.ACTIONS or not feedback:
        errors.append('invalid_lesson')
    if action in ('B', 'E'):
        errors.append('student_steps_unavailable')
    if row['current']['image_missing'] and action != 'C':
        errors.append('needs_visible_figure')
    known = {str(answer).strip().strip('$').strip().upper() for answer in row['solution']['answer']}
    # These checks quarantine explicit disclosures. Human review still checks the math and indirect disclosures.
    if (any(option.upper() in known for option in DIRECT_OPTION.findall(feedback))
            or any(number in known for number in DIRECT_ANSWER.findall(feedback))
            or (NUMBER.fullmatch(feedback.strip('$')) and feedback.strip('$') in known)):
        errors.append('explicit_reference_answer')
    raw_candidate = not errors
    if decision['status'] != 'eligible' and action not in ('C', 'G'):
        errors.append('extension_beyond_available_evidence')
    return {'action': action if action in gold.ACTIONS else None, 'feedback': feedback, 'flags': errors,
            'shared_checks_candidate': raw_candidate, 'status': 'needs_review' if errors else 'candidate'}
