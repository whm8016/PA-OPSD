"""Paired offline diagnosis summary for the SFT baseline and gold-memory study."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_pa_opsd_gold_split import question_group
from scripts.experiments.diagnosis_probe import baseline, offline, run
from scripts.experiments.longtutor_reeval import run as official

DRAW_COUNT = 10_000
SEED = 20261128


def paired_delta(qids, labels, left, right):
    """Resample current-question IDs, applying the same draws to both predictions."""
    groups = {qid: i for i, qid in enumerate(sorted(set(qids)))}
    counts = np.zeros((2, len(groups), len(run.LABELS), 3), dtype=int)
    label_ids = {label: i for i, label in enumerate(run.LABELS)}
    for qid, gold, first, second in zip(qids, labels, left, right, strict=True):
        group = groups[qid]
        gold_idx = label_ids[gold]
        for index, prediction in enumerate((first, second)):
            counts[index, group, gold_idx, 0] += 1
            if prediction in label_ids:
                counts[index, group, label_ids[prediction], 1] += 1
                counts[index, group, gold_idx, 2] += prediction == gold

    def metrics(table):
        gold, pred, correct = np.moveaxis(table, -1, 0)
        denom = gold + pred
        f1 = np.divide(2 * correct, denom, out=np.zeros_like(correct, dtype=float), where=denom != 0)
        return {'accuracy': correct.sum(axis=-1) / gold.sum(axis=-1), 'macro_f1': f1.mean(axis=-1)}

    observed = metrics(counts.sum(axis=1))
    draws = np.random.default_rng(SEED).integers(len(groups), size=(DRAW_COUNT, len(groups)))
    sampled = metrics(counts[:, draws].sum(axis=2))
    return {name: {'delta': float(values[1] - values[0]),
                   'ci95': np.quantile(sampled[name][1] - sampled[name][0], [0.025, 0.975]).tolist()}
            for name, values in observed.items()} | {'groups': len(groups), 'draws': DRAW_COUNT, 'seed': SEED}


def paired_probe(path, arms, gold, qids, common=False):
    rows = run.read_rows(path)
    ref = {row['sample_id']: row['gold'] for row in rows}
    if len(rows) != len(ref) or ref != gold:
        raise ValueError('Probe samples or labels differ from the full fixed development set')
    used = [row for row in rows if not common or all(row['prediction'][arm] is not None for arm in arms)]
    if not used:
        raise ValueError('No samples are eligible for paired analysis')
    labels = [row['gold'] for row in used]
    clusters = [qids[row['sample_id']] for row in used]
    predictions = [[row['prediction'][arm] for row in used] for arm in arms]
    return {arm: offline.summary(labels, pred) for arm, pred in zip(arms, predictions, strict=True)} | {
        'paired_delta': paired_delta(clusters, labels, *predictions),
        'class_changes': offline.paired(labels, *predictions),
    }


def generated_predictions(path, ids, grader):
    indexed = official.prediction_rows(path, ids)
    if any(official.inference_failure(row) for row in indexed.values()):
        raise ValueError(f'Unresolved generation failure in {path}')
    labels = [grader._norm_text(indexed[key].get('diagnosis')) for key in ids]
    return [label if label in run.LABELS else None for label in labels], sum(not label for label in labels)


def generated_summary(new_path, by_id_path, source, manifest, gold, qids):
    ids = manifest['ids']
    grader = official.scorer_module(source)
    labels = [gold[key] for key in ids]
    base, base_empty = generated_predictions(source / 'base.jsonl', ids, grader)
    left, left_empty = generated_predictions(baseline.OLD_OFFICIAL / 'sft.jsonl', ids, grader)
    right, right_empty = generated_predictions(new_path, ids, grader)
    with by_id_path.open('x', encoding='utf-8') as stream:
        for key, label, foundation, ordinary, weighted in zip(ids, labels, base, left, right, strict=True):
            stream.write(json.dumps({'sample_id': key, 'qid': qids[key], 'gold': label, 'base': foundation,
                                     'original_sft': ordinary, 'weighted_sft': weighted}) + '\n')
    return {'base': offline.summary(labels, base), 'original_sft': offline.summary(labels, left),
            'weighted_sft': offline.summary(labels, right),
            'paired_delta': paired_delta([qids[key] for key in ids], labels, left, right),
            'class_changes': offline.paired(labels, left, right),
            'empty_diagnosis': {'base': base_empty, 'original_sft': left_empty, 'weighted_sft': right_empty},
            'upstream_commit': manifest['upstream_commit'], 'new_predictions_sha256': official.digest(new_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, required=True, help='oracle paired_by_id.jsonl')
    parser.add_argument('--weighted', type=Path, required=True, help='weighted SFT paired_by_id.jsonl')
    parser.add_argument('--generated', type=Path, required=True, help='unmodified official predictions JSONL')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = baseline.OLD_OFFICIAL / 'bundle'
    manifest = official.verify_bundle(source)
    gold = {key: value['diagnosis'] for key, value in official.index(official.rows(source / 'gold.jsonl'), '_key').items()}
    qids = {key: question_group(value['question_info'])[0]
            for key, value in official.index(official.rows(source / 'inputs.jsonl'), '_key').items()}
    if gold.keys() != qids.keys() or set(manifest['ids']) != gold.keys():
        raise ValueError('Frozen development keys differ across inputs, labels, and manifest')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'oracle': paired_probe(args.oracle, ('model_memory', 'gold_memory'), gold, qids, common=True),
              'weighted_label_rank': paired_probe(args.weighted, ('original', 'weighted'), gold, qids),
              'generated_exact_diagnosis': generated_summary(args.generated, args.output / 'generated_by_id.jsonl',
                                                             source, manifest, gold, qids)}
    run.write_json(args.output / 'summary.json', report)


if __name__ == '__main__':
    main()
