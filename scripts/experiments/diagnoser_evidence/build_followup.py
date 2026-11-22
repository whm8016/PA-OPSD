"""Freeze 24 new Gold development rows for the three-arm diagnosis history comparison."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.diagnoser_evidence import evidence, run

gold, team = run.gold, run.team
QUOTA = 6
SELECTION_TAG = 'diagnoser-evidence-new-gold24-v1'


def build(previous, out):
    previous_manifest = gold.read_json(previous / 'manifest.json')
    if previous_manifest['scope'] != 'validation' or previous_manifest['selection_tag'] != team.TAG:
        raise ValueError('Expected the original Gold32 validation input')
    previous_ids = set(team.load(previous, 'inputs.jsonl', previous_manifest))

    split_path = gold.SPLIT / 'split_manifest.json'
    refs_path = gold.SPLIT / 'development/references.jsonl'
    split = gold.read_json(split_path)
    refs_hash = gold.digest(refs_path)
    if refs_hash != split['files']['development/references.jsonl']:
        raise ValueError('Frozen development references changed')
    dev = gold.indexed(gold.read_rows(refs_path), 'sample_id')
    if list(dev) != split['ids']['validation'] or len(previous_ids) != 32 or not previous_ids <= dev.keys():
        raise ValueError('Gold32 or frozen development IDs changed')
    targets = {key: row['target'] for key, row in dev.items() if key not in previous_ids}

    history_path = gold.REPRO / 'gold_inputs.jsonl'
    history_hash = gold.digest(history_path)
    if history_hash != gold.read_json(gold.REPRO / 'preflight.json')['gold_input_sha256']:
        raise ValueError('Official history snapshot changed')
    histories = gold.indexed(
        (dict(row, sample_id=key) for row in gold.read_rows(history_path) if (key := gold.history_key(row)) in targets),
        'sample_id',
    )
    if histories.keys() != targets.keys():
        raise ValueError('Public histories do not cover the candidate development IDs')

    bank_hash = gold.digest(gold.BANK)
    adapter_hashes = {name: gold.digest(gold.GOLD_SFT / name) for name in gold.ADAPTER_FILES}
    if bank_hash != previous_manifest['question_bank_sha256']:
        raise ValueError('Question bank differs from Gold32')
    if adapter_hashes != previous_manifest['gold_sft_adapter_sha256']:
        raise ValueError('Gold SFT adapter differs from Gold32')
    bank = {str(row['id']): row['info'] for row in gold.read_rows(gold.BANK)}

    inputs, refs, features = [], [], []
    for key, current, item in team.select(targets, histories, bank, QUOTA):
        h, target = histories[key], targets[key]
        row = {'sample_id': key, 'history': h['history_info'], 'current': current,
               'solution': {'answer': item['answer'], 'analysis': item['analysis']}}
        related = set(h['related_history'])
        refs.append({'sample_id': key, 'diagnosis': target['diagnosis'], 'strategy': target['strategy'],
                     'feedback': target['content'],
                     'related_positions': [i for i, record in enumerate(row['history'], 1) if record in related]})
        inputs.append(row)
        features.append(evidence.build_features(row))

    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (('inputs.jsonl', inputs), ('references.jsonl', refs), ('features.jsonl', features)):
        gold.write_rows(out / name, rows)
    gold.write_json(out / 'manifest.json', {
        'scope': 'validation', 'ids': [row['sample_id'] for row in inputs], 'selection_tag': SELECTION_TAG,
        'ranking_tag': team.TAG, 'candidate_count': len(targets), 'excluded_gold32_rows': len(previous_ids),
        'class_counts': dict(Counter(row['diagnosis'] for row in refs)),
        'current_images': sum(bool(gold.IMAGE.search(row['current'])) for row in inputs),
        'question_bank_sha256': bank_hash, 'gold_sft_adapter_sha256': adapter_hashes,
        'source_sha256': {'split_manifest': gold.digest(split_path), 'development_references': refs_hash,
                          'gold_history_snapshot': history_hash, 'gold32_manifest': gold.digest(previous / 'manifest.json'),
                          'builder': gold.digest(Path(__file__)), 'ranking': gold.digest(Path(team.__file__)),
                          'features': gold.digest(Path(evidence.__file__))},
        'feature_parameters': {'max_concepts': evidence.MAX_CONCEPTS, 'ranked_records': evidence.MATCHED_RECORDS,
                               'recent_records': evidence.RECENT_RECORDS},
        'files': {name: gold.digest(out / name) for name in ('inputs.jsonl', 'references.jsonl', 'features.jsonl')},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True, help='Packed Gold32 validation input directory')
    parser.add_argument('--output', type=Path, required=True, help='New Gold24 validation input directory')
    args = parser.parse_args()
    build(args.previous, args.output)


if __name__ == '__main__':
    main()
