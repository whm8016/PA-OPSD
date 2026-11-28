"""Gold-memory intervention and diagnosis evaluation for the weighted SFT baseline."""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.experiments.diagnosis_probe import run

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / 'outputs/pa_opsd/diagnosis_probe_20261125'
OLD = PROBE / 'autodl_4090d_22532/runs/sft/full'
OLD_OFFICIAL = ROOT / 'outputs/pa_opsd/longtutor_reeval_gold140/official_base_vs_memory_first_20261123'
ADAPTER_FILES = ('adapter_config.json', 'adapter_model.safetensors')
MAX_NEW_TOKENS = 640
DIAGNOSIS_MULTIPLIER = 8.0


def memory_prefix(memory):
    clean = [{key: entry[key] for key in ('id', 'answer')} for entry in memory]
    return json.dumps({'memory': clean}, ensure_ascii=False)[:-1] + ', "diagnosis": "'


def prepare(output, tokenizer_path):
    from scripts.experiments.diagnosis_probe import offline
    from scripts.experiments.gold_evidence_first import data
    from scripts.experiments.gold_evidence_first import run as training
    from scripts.train_pa_opsd_sft import load_tokenizer, prepare_rows

    old_bundle = PROBE / 'deploy'
    refs, predictions, profile = offline.collect(old_bundle, OLD, 'full')
    pin = run.read_json(old_bundle / 'gpu_manifest.json')
    settings = run.read_json(OLD / 'settings.json')
    if (settings['max_new_tokens'] != MAX_NEW_TOKENS or settings['runner_sha256'] != run.digest(Path(run.__file__))
            or run.digest(old_bundle / 'cases.jsonl') != pin['cases_sha256']):
        raise ValueError('Archived diagnosis controls use a different protocol')
    gold_manifest = data.verify(offline.SOURCE)
    gold = {row['sample_id']: row['target'] for row in run.read_rows(offline.SOURCE / data.DEV_REFERENCES)}
    public = {row['sample_id']: row['original'] for row in run.read_rows(old_bundle / 'cases.jsonl')
              if row['split'] == 'dev'}

    tokenizer = load_tokenizer(str(tokenizer_path), local_files_only=True)
    train = run.read_rows(offline.SOURCE / data.TRAIN_FILES['memory_first'][0])
    balance = training.diagnosis_class_balance(train)
    encoded, audit = prepare_rows(train, tokenizer, training.MAX_LENGTH, class_weights=balance['weights'],
                                  diagnosis_token_multiplier=DIAGNOSIS_MULTIPLIER)
    audit.update({'class_balance': balance, 'diagnosis_token_multiplier': DIAGNOSIS_MULTIPLIER,
                  'total_token_weight': sum(sum(weight for label, weight in zip(row['labels'], row['token_weights'],
                                                                                 strict=True) if label != -100)
                                            for row in encoded)})
    if len(encoded) != 420 or audit['reweighted_tokens'] < 420:
        raise ValueError('The baseline training audit did not cover all 420 examples')

    cases, controls = [], []
    for ref, prediction in zip(refs, predictions, strict=True):
        if ref['split'] != 'dev':
            continue
        key = ref['sample_id']
        original = prediction['results']['original']
        target = gold[key]
        if ref['gold'] != target['diagnosis']:
            raise ValueError(f'Diagnosis reference changed for {key}')
        predicted = None
        reference_ids = [entry['id'] for entry in target['memory']]
        if original['status'] == 'ok':
            value, _ = json.JSONDecoder().raw_decode(original['prefix'], run.MEMORY_START.match(original['prefix']).end())
            if all(isinstance(entry, dict) and isinstance(entry.get('id'), str)
                   and isinstance(entry.get('answer'), str) for entry in value) and [entry['id'] for entry in value] == reference_ids:
                predicted = memory_prefix(value)
        reference = memory_prefix(target['memory'])
        if any(label in reference for label in run.LABELS):
            raise ValueError(f'Reference memory contains a diagnosis label for {key}')
        messages = public[key]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
        for prefix in filter(None, (predicted, reference)):
            prefix_ids, labels = run.candidate_tokens(tokenizer, prefix)
            if (run.diagnosis_prefix(prefix) != prefix or tokenizer.decode(prefix_ids, skip_special_tokens=True,
                                                                          clean_up_tokenization_spaces=False) != prefix
                    or prompt_length + len(prefix_ids) + max(map(len, labels.values())) > training.MAX_LENGTH):
                raise ValueError(f'Invalid evidence prefix for {key}')
        cases.append({'sample_id': key, 'messages': messages, 'model_memory': predicted, 'gold_memory': reference})
        controls.append({'sample_id': key, 'gold': ref['gold'], 'original_scores': original['scores'],
                         'history_stats_scores': prediction['results']['history_stats']['scores']})

    official_root = OLD_OFFICIAL / 'bundle'
    official = run.read_json(official_root / 'manifest.json')
    ids = [case['sample_id'] for case in cases]
    if len(ids) != 140 or ids != gold_manifest['ids']['dev'] or set(ids) != set(official['ids']):
        raise ValueError('Expected the same fixed LongTutor development split everywhere')
    smoke_ids = [next(row['sample_id'] for row, case in zip(controls, cases, strict=True)
                      if row['gold'] == label and case['model_memory']) for label in run.LABELS]
    output.mkdir(parents=True, exist_ok=False)
    data.write_rows(output / 'cases.jsonl', cases)
    data.write_rows(output / 'controls.jsonl', controls)
    audit.update({'dev_rows': len(cases), 'paired_memory_rows': sum(case['model_memory'] is not None for case in cases),
                  'official_commit': official['upstream_commit'], 'old_sft_adapter': pin['adapter']})
    run.write_json(output / 'local_preflight.json', audit)
    run.write_json(output / 'protocol.json', {
        'cases_sha256': run.digest(output / 'cases.jsonl'), 'controls_sha256': run.digest(output / 'controls.jsonl'),
        'training_data_sha256': gold_manifest['files'][data.TRAIN_FILES['memory_first'][0]],
        'training_manifest_sha256': profile['prepared_gold_manifest_sha256'],
        'adapter': pin['adapter'], 'base': pin['base'], 'ids': ids, 'smoke_ids': smoke_ids,
        'old_settings': settings, 'diagnosis_multiplier': DIAGNOSIS_MULTIPLIER,
    })
    training.pack(offline.SOURCE, output / 'training.zip', offline.BASE_PIN)
    with ZipFile(output / 'evaluation.zip', 'x', ZIP_DEFLATED) as archive:
        for name in ('cases.jsonl', 'protocol.json'):
            archive.write(output / name, name)
        for path in (Path(__file__), Path(run.__file__), Path(__file__).with_name('run_baseline_gpu.sh')):
            archive.write(path, path.relative_to(ROOT))
        for name in ADAPTER_FILES:
            path = offline.ANCHOR / name
            if run.digest(path) != pin['adapter'][name]:
                raise ValueError(f'Baseline adapter changed: {name}')
            archive.write(path, f'old_sft/{name}')
        official_files = [name for name in official['files'] if name.startswith('upstream/')]
        official_files += ['inputs.jsonl', 'gold.jsonl']
        for name in official_files:
            path = official_root / name
            if run.digest(path) != official['files'][name]:
                raise ValueError(f'Official bundle file changed: {name}')
            archive.write(path, f'official/{name}')
        archive.write(official_root / 'manifest.json', 'official/manifest.json')
    print(json.dumps(audit, indent=2))


def verified_protocol(bundle):
    protocol = run.read_json(bundle / 'protocol.json')
    run.verify_model_files(bundle, {'cases.jsonl': protocol['cases_sha256']})
    cases = run.read_rows(bundle / 'cases.jsonl')
    if [row['sample_id'] for row in cases] != protocol['ids']:
        raise ValueError('Cases do not match the frozen ID order')
    return protocol, cases


def forced_scores(model, tokenizer, prompt_ids, prefix):
    if prefix is None:
        return None
    prefix_ids, candidates = run.candidate_tokens(tokenizer, prefix)
    scores, _ = run.score_tokens(model, prompt_ids + prefix_ids, candidates, model.device, keep_hidden=False)
    return scores


def verify_adapter(bundle, model_dir, variant, adapter):
    protocol, _ = verified_protocol(bundle)
    pin = protocol['base']
    run.verify_model_files(model_dir, {**pin['config_and_tokenizer_sha256'],
                           'generation_config.json': pin['generation_config_sha256'],
                           'model.safetensors.index.json': pin['weight_index_sha256'],
                           **{name: entry['sha256'] for name, entry in pin['weights'].items()}})
    if any((model_dir / name).exists() for name in ('chat_template.jinja', 'chat_template.json', 'chat_templates')):
        raise ValueError('Unexpected chat template override')
    if variant == 'oracle':
        adapter_files = protocol['adapter']
    else:
        receipt = run.read_json(adapter / 'run_receipt.json')
        if (receipt['diagnosis_token_multiplier'] != protocol['diagnosis_multiplier'] or receipt['smoke']
                or receipt['arm'] != 'memory_first' or receipt['data_sha256'] != protocol['training_data_sha256']
                or receipt['manifest_sha256'] != protocol['training_manifest_sha256']
                or receipt['rows'] != 420 or receipt['effective_steps'] != 105 or not receipt['budget_complete']):
            raise ValueError('Expected the full 420/105 diagnosis-token SFT checkpoint')
        adapter_files = {name: receipt['adapter'][name] for name in ADAPTER_FILES}
    run.verify_model_files(adapter, adapter_files)
    return adapter_files


def infer(bundle, model_dir, adapter, output, variant, smoke):
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol, cases = verified_protocol(bundle)
    if not torch.cuda.is_available():
        raise RuntimeError('This baseline experiment requires a CUDA GPU')
    if variant == 'oracle':
        adapter = bundle / 'old_sft'
    adapter_files = verify_adapter(bundle, model_dir, variant, adapter)
    if smoke:
        by_id = {row['sample_id']: row for row in cases}
        cases = [by_id[key] for key in protocol['smoke_ids']]
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'protocol_sha256': run.digest(bundle / 'protocol.json'), 'variant': variant,
                'smoke': smoke, 'adapter': adapter_files, 'torch': torch.__version__,
                'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
                'max_new_tokens': MAX_NEW_TOKENS, 'dtype': 'bfloat16', 'attention': 'sdpa',
                'runner_sha256': run.digest(Path(__file__)), 'core_sha256': run.digest(Path(run.__file__))}
    for key in ('torch', 'transformers', 'max_new_tokens', 'dtype', 'attention'):
        if metadata[key] != protocol['old_settings'][key]:
            raise ValueError(f'{key} must match archived SFT diagnosis settings: {protocol["old_settings"][key]}')
    records = output / 'predictions.jsonl'
    settings = output / 'settings.json'
    lock = output / 'inference.lock'
    lock.touch(exist_ok=False)
    try:
        if (settings.exists() and run.read_json(settings) != metadata) or (records.exists() and not settings.exists()):
            raise ValueError('Cannot resume a different or untracked run')
        if not settings.exists():
            run.write_json(settings, metadata)
        saved = run.resume_rows(records)
        if [row['sample_id'] for row in saved] != [row['sample_id'] for row in cases[:len(saved)]]:
            raise ValueError('Saved IDs differ from the planned order')
        if len(saved) == len(cases):
            return
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                                    torch_dtype=torch.bfloat16, attn_implementation='sdpa')
        model = PeftModel.from_pretrained(base, adapter).get_base_model().to('cuda').eval()
        with torch.inference_mode(), records.open('a', encoding='utf-8', newline='\n') as stream:
            for number, case in enumerate(cases[len(saved):], len(saved) + 1):
                if variant == 'oracle':
                    prompt = tokenizer.apply_chat_template(case['messages'], tokenize=False, add_generation_prompt=True)
                    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    results = {arm: forced_scores(model, tokenizer, prompt_ids, case[arm])
                               for arm in ('model_memory', 'gold_memory')}
                else:
                    results = run.infer_case(model, tokenizer, case['messages'], MAX_NEW_TOKENS, keep_hidden=False)
                    if smoke and results['status'] != 'ok':
                        print(f"Smoke format miss for {case['sample_id']}", flush=True)
                stream.write(json.dumps({'sample_id': case['sample_id'], 'results': results}, ensure_ascii=False,
                                        allow_nan=False) + '\n')
                stream.flush()
                print(f'{variant} {number}/{len(cases)}', flush=True)
    finally:
        lock.unlink()


def score(bundle, source, output):
    from scripts.experiments.diagnosis_probe import offline

    protocol, cases = verified_protocol(bundle)
    run.verify_model_files(bundle, {'controls.jsonl': protocol['controls_sha256']})
    controls = run.read_rows(bundle / 'controls.jsonl')
    settings = run.read_json(source / 'settings.json')
    rows = run.read_rows(source / 'predictions.jsonl')
    variant = settings['variant']
    if (variant not in ('oracle', 'weighted') or settings['protocol_sha256'] != run.digest(bundle / 'protocol.json')
            or settings['core_sha256'] != protocol['old_settings']['runner_sha256'] or settings['smoke']
            or not [r['sample_id'] for r in rows] == [r['sample_id'] for r in controls] == protocol['ids']
            or (variant == 'oracle' and settings['adapter'] != protocol['adapter'])):
        raise ValueError('Expected a matched full evaluation')
    old_settings = protocol['old_settings']
    if any(settings[key] != old_settings[key] for key in ('torch', 'transformers', 'max_new_tokens', 'dtype', 'attention')):
        raise ValueError('Scoring settings differ from the archived SFT')

    no_prior = dict.fromkeys(run.LABELS, 0)
    by_id = []
    for baseline, row, case in zip(controls, rows, cases, strict=True):
        scores = {'original': baseline['original_scores'], 'history_stats': baseline['history_stats_scores']}
        result = row['results']
        if variant == 'oracle':
            if result['gold_memory'] is None or (result['model_memory'] is None) != (case['model_memory'] is None):
                raise ValueError('Evidence results do not match the prepared arms')
            scores.update(result)
        else:
            scores['weighted'] = result['scores']
        ranks = {arm: offline.ranked(values, no_prior, 0) for arm, values in scores.items()}
        label = baseline['gold']
        by_id.append({'sample_id': row['sample_id'], 'gold': label, 'scores': scores,
                      'prediction': {arm: order[0] if order else None for arm, order in ranks.items()},
                      'gold_rank': {arm: order.index(label) + 1 if order else None for arm, order in ranks.items()}})

    def compare(records, left, right):
        gold = [record['gold'] for record in records]
        a, b = ([record['prediction'][name] for record in records] for name in (left, right))
        return {left: offline.summary(gold, a), right: offline.summary(gold, b),
                'changes': offline.paired(gold, a, b)}

    if variant == 'oracle':
        common = [row for row in by_id if row['prediction']['model_memory'] is not None]
        report = {'paired_prefix': compare(common, 'model_memory', 'gold_memory'),
                  'canonical_prefix_check': compare(common, 'original', 'model_memory'),
                  'all_rows_with_oracle_format': compare(by_id, 'original', 'gold_memory')}
    else:
        report = compare(by_id, 'original', 'weighted')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'paired_by_id.jsonl').open('w', encoding='utf-8') as stream:
        for row in by_id:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    run.write_json(output / 'summary.json', {'comparison': report, 'settings': settings, 'old_settings': old_settings,
                                           'same_gpu_model': settings['gpu'] == old_settings['gpu'],
                                           'predictions_sha256': run.digest(source / 'predictions.jsonl')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--tokenizer', type=Path, required=True)
    gpu = commands.add_parser('infer')
    gpu.add_argument('--bundle', type=Path, required=True)
    gpu.add_argument('--model', type=Path, required=True)
    gpu.add_argument('--adapter', type=Path)
    gpu.add_argument('--output', type=Path, required=True)
    gpu.add_argument('--variant', choices=('oracle', 'weighted'), required=True)
    gpu.add_argument('--smoke', action='store_true')
    scoring = commands.add_parser('score')
    scoring.add_argument('--bundle', type=Path, required=True)
    scoring.add_argument('--source', type=Path, required=True)
    scoring.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output, args.tokenizer)
    elif args.command == 'score':
        score(args.bundle, args.source, args.output)
    else:
        if (args.variant == 'weighted') != (args.adapter is not None):
            parser.error('--adapter is required for weighted and supplied by the bundle for oracle')
        infer(args.bundle, args.model, args.adapter, args.output, args.variant, args.smoke)


if __name__ == '__main__':
    main()
