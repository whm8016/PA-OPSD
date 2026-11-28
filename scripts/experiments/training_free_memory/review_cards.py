"""Verify the GPU backup and export a blinded packet for reviewing first teaching actions."""

import argparse
import random
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.experiments.longtutor_reeval.run import (
    digest,
    index,
    inference_failure,
    prediction_rows,
    read_json,
    rows,
    write_json,
    write_rows,
)


def packet(prepared, destination, checksum, seed):
    manifest_path = prepared / 'manifest.json'
    manifest = read_json(manifest_path)
    archive = destination / 'gpu_results.tar.gz'
    if digest(archive) != checksum:
        raise ValueError('GPU archive differs from remote checksum')
    with TemporaryDirectory(dir=destination) as staging:
        extracted = Path(staging) / 'archive'
        extracted.mkdir()
        extract(archive, extracted)
        receipt, predictions = verify_outputs(prepared, extracted, manifest, checksum, digest(manifest_path))
        extracted.rename(destination / 'verified_archive')
    write_json(destination / 'local_backup_verified.json', receipt)
    write_blind_packet(prepared, destination, manifest, predictions, seed)


def extract(archive, destination):
    with tarfile.open(archive) as source:
        members = source.getmembers()
        targets = [(destination / member.name).resolve() for member in members]
        if len(set(targets)) != len(targets) or any(
            not (member.isfile() or member.isdir()) or not target.is_relative_to(destination) or target.exists()
            for member, target in zip(members, targets)
        ):
            raise ValueError('Unexpected or duplicate archive entries, or destination already extracted')
        source.extractall(destination, members=members, filter='data')


def verify_outputs(prepared, destination, manifest, checksum, manifest_hash):
    queue_exit = (destination / 'queue.exit').read_text().strip()
    if queue_exit != '0':
        raise ValueError(f'GPU run did not complete: {queue_exit}. See the original GPU archive')
    receipt = {'archive_sha256': checksum, 'manifest_sha256': manifest_hash, 'stages': {}}
    by_phase = {}
    for phase, keys in (('smoke', manifest['smoke_ids']), ('full', manifest['ids'])):
        directory = destination / 'runs' / phase
        settings = dict(line.split('\t', 1) for line in (directory / 'run.tsv').read_text().splitlines())
        required = {'subset': phase, 'model': manifest['base_model'], 'temperature': '0',
                    'max_tokens': '3000', 'workers': str(manifest['workers']), 'python_hash_seed': '0',
                    'manifest_sha256': receipt['manifest_sha256']}
        if any(settings.get(key) != value for key, value in required.items()):
            raise ValueError(f'GPU stage settings differ: {phase}')
        verified_names = set()
        for line in (directory / 'predictions.sha256').read_text().splitlines():
            hashed, name = line.split('  ', 1)
            if Path(name).name != name or name in verified_names or digest(directory / name) != hashed:
                raise ValueError(f'Invalid GPU stage member: {phase}/{name}')
            verified_names.add(name)
        predictions = {}
        for arm in manifest['arm_order']:
            name = f'{arm}.jsonl'
            input_name = f'{phase}.{name}'
            if name not in verified_names or digest(prepared / input_name) != manifest['files_sha256'][input_name]:
                raise ValueError(f'GPU output or input is missing from the pinned manifest: {phase}/{arm}')
            if [row['_key'] for row in rows(prepared / input_name)] != keys:
                raise ValueError(f'GPU input keys differ: {phase}/{arm}')
            predictions[arm] = prediction_rows(directory / name, keys)
            if any(inference_failure(row) for row in predictions[arm].values()):
                raise ValueError(f'GPU infrastructure failure: {phase}/{arm}')
        by_phase[phase] = predictions
        receipt['stages'][phase] = {
            arm: {'rows': len(records), 'format_failures': sum(not row['diagnosis'] for row in records.values())}
            for arm, records in predictions.items()
        }
    return receipt, by_phase['full']


def write_blind_packet(prepared, destination, manifest, predictions, seed):
    if digest(prepared / 'routing.jsonl') != manifest['routing_sha256']:
        raise ValueError('Routing checksum differs')
    routes = index(rows(prepared / 'routing.jsonl'), '_key')
    inputs = index(rows(prepared / 'full.no_card.jsonl'), '_key')
    rng = random.SystemRandom() if seed is None else random.Random(seed)
    blind, key_map = [], []
    for number, key in enumerate(manifest['ids'], 1):
        arms = list(manifest['arm_order'])
        rng.shuffle(arms)
        options = {}
        for letter, arm in zip('ABC', arms):
            pred = predictions[arm][key]
            options[letter] = pred['content'] if pred['diagnosis'] else None
            key_map.append({'case': number, 'option': letter, 'arm': arm, '_key': key,
                            'route': routes[key]['matched']})
        item = inputs[key]
        blind.append({'case': number, 'question': item['question_info'], 'history': item['history_info'],
                      'feedback': options})
    write_rows(destination / 'blind_review.jsonl', blind)
    private = destination / 'internal'
    private.mkdir(exist_ok=True)
    write_rows(private / 'blind_key.jsonl', key_map)
    print(f'Verified GPU backup and prepared {len(blind)} blinded cases')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--seed', type=int, help='Supply only for a controlled rerun')
    args = parser.parse_args()
    packet(args.prepared.resolve(), args.destination.resolve(), args.sha256, args.seed)


if __name__ == '__main__':
    main()
