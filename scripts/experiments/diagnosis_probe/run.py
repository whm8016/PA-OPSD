"""Generate to the diagnosis field, then score all four complete label strings."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

LABELS = ('Recall Failure', 'Conceptual Gap', 'Procedural Error', 'Transfer Deficit')
ARMS = ('original', 'history_stats')
MEMORY_START = re.compile(r'\s*(?:```(?:json)?\s*)?\{\s*"memory"\s*:\s*', re.IGNORECASE)
DIAGNOSIS_START = re.compile(r'\s*,\s*"diagnosis"\s*:\s*"')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def diagnosis_prefix(text):
    start = MEMORY_START.match(text)
    if not start:
        return None
    try:
        memory, end = json.JSONDecoder().raw_decode(text, start.end())
    except ValueError:
        return None
    field = DIAGNOSIS_START.match(text, end)
    return text[:field.end()] if isinstance(memory, list) and field else None


def candidate_tokens(tokenizer, prefix):
    shared = tokenizer.encode(prefix, add_special_tokens=False)
    candidates = {}
    for label in LABELS:
        complete = tokenizer.encode(prefix + label, add_special_tokens=False)
        if complete[:len(shared)] != shared or len(complete) <= len(shared):
            raise ValueError(f'Unstable token boundary at diagnosis label: {label}')
        candidates[label] = complete[len(shared):]
    return shared, candidates


def score_tokens(model, prefix, candidates, device, keep_hidden):
    import torch

    prefix = torch.tensor([prefix], dtype=torch.long, device=device)
    output = model.model(input_ids=prefix, use_cache=True)
    cache = output.past_key_values
    last_hidden = output.last_hidden_state[:, -1:, :]
    first = model.lm_head(last_hidden)[0, 0].float().log_softmax(dim=-1)
    hidden = last_hidden[0, 0].float().tolist() if keep_hidden else None
    scores = {}
    for label, ids in candidates.items():
        value = first[ids[0]]
        if len(ids) > 1:
            tail = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            branch = model.model(input_ids=tail, past_key_values=cache, use_cache=True)
            log_probs = model.lm_head(branch.last_hidden_state)[0].float().log_softmax(dim=-1)
            value = value + log_probs.gather(1, torch.tensor(ids[1:], device=device)[:, None]).sum()
            cache.crop(prefix.shape[-1])
        scores[label] = value.item()
    return scores, hidden


def infer_case(model, tokenizer, messages, max_new_tokens, keep_hidden):
    import torch
    from transformers import StoppingCriteria

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) + max_new_tokens + 16 > model.config.max_position_embeddings:
        raise ValueError('Prompt and generation budget exceed the checkpoint context window')

    def decode(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    class StopAtDiagnosis(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            new_count = input_ids.shape[-1] - len(prompt_ids)
            if new_count % 8:
                return False
            return diagnosis_prefix(decode(input_ids[0, len(prompt_ids):].tolist())) is not None

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                            temperature=None, top_p=None, top_k=None, max_new_tokens=max_new_tokens,
                            stopping_criteria=[StopAtDiagnosis()], use_cache=True)
    generated = output[0, len(prompt_ids):].tolist()
    raw = decode(generated)
    prefix = diagnosis_prefix(raw)
    result = {'prompt_tokens': len(prompt_ids), 'generated_tokens': len(generated)}
    if prefix is None:
        return {**result, 'status': 'missing_diagnosis_prefix', 'raw': raw, 'scores': None}
    prefix_ids, targets = candidate_tokens(tokenizer, prefix)
    if decode(prefix_ids) != prefix:
        raise ValueError('Diagnosis prefix does not round-trip through the tokenizer')
    retokenized = generated[:len(prefix_ids)] != prefix_ids
    if retokenized:
        # Preserve the generated token state when the opening quotation mark lands on a boundary.
        for end in range(len(generated), 0, -1):
            actual = decode(generated[:end])
            if len(actual) < len(prefix):
                break
            if actual == prefix and all(decode(generated[:end] + tokens) == prefix + label
                                        for label, tokens in targets.items()):
                prefix_ids, retokenized = generated[:end], False
                break
    scores, hidden = score_tokens(model, prompt_ids + prefix_ids, targets, model.device, keep_hidden)
    return {**result, 'status': 'ok', 'raw': raw, 'prefix': prefix, 'scores': scores, 'hidden': hidden,
            'retokenized_prefix': retokenized}


def verify_model_files(directory, files):
    for name, expected in files.items():
        if digest(directory / name) != expected:
            raise ValueError(f'Checkpoint file hash mismatch: {name}')


def resume_rows(path):
    if not path.exists():
        return []
    with path.open('rb+') as stream:
        contents = stream.read()
        if contents and not contents.endswith(b'\n'):
            end = contents.rfind(b'\n') + 1
            stream.truncate(end)
            contents = contents[:end]
    return [json.loads(line) for line in contents.splitlines()]


def run(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('Run model inference on a CUDA instance')
    manifest = read_json(args.bundle / 'gpu_manifest.json')
    cases_path = args.bundle / 'cases.jsonl'
    if digest(cases_path) != manifest['cases_sha256']:
        raise ValueError('Inference inputs differ from the prepared bundle')
    cases = read_rows(cases_path)
    index = {case['sample_id']: case for case in cases}
    if len(index) != len(cases) or len(cases) != 280:
        raise ValueError('Expected 280 distinct inference inputs')
    if args.phase == 'smoke':
        cases = [index[key] for key in manifest['smoke_ids']]
    pin = manifest['base']
    verify_model_files(args.model_dir, {**pin['config_and_tokenizer_sha256'],
                       'generation_config.json': pin['generation_config_sha256'],
                       'model.safetensors.index.json': pin['weight_index_sha256'],
                       **{name: entry['sha256'] for name, entry in pin['weights'].items()}})
    if any((args.model_dir / name).exists() for name in ('chat_template.jinja', 'chat_template.json', 'chat_templates')):
        raise ValueError('Extra chat template would override the fixed tokenizer config')
    adapter = manifest['adapter'] if args.arm == 'sft' else None
    if adapter:
        verify_model_files(args.bundle / 'adapter', adapter)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    settings = {'arm': args.arm, 'phase': args.phase, 'max_new_tokens': args.max_new_tokens,
                'manifest_sha256': digest(args.bundle / 'gpu_manifest.json'),
                'runner_sha256': digest(Path(__file__)), 'base_revision': pin['revision'],
                'adapter': adapter, 'torch': torch.__version__, 'transformers': transformers.__version__,
                'gpu': torch.cuda.get_device_name(), 'dtype': 'bfloat16', 'attention': 'sdpa'}
    records = out / 'predictions.jsonl'
    state = out / 'settings.json'
    # Both instances use independent directories. Prevent two writers to one result file.
    lock = out / 'inference.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write('Inference is active. Remove this file only after the process has ended.\n')
    try:
        if state.exists() and read_json(state) != settings:
            raise ValueError('The output directory belongs to a different inference run')
        if records.exists() and not state.exists():
            raise ValueError('Predictions without run settings')
        if not state.exists():
            write_json(state, settings)
        saved = resume_rows(records)
        if [row['sample_id'] for row in saved] != [row['sample_id'] for row in cases[:len(saved)]]:
            raise ValueError('Saved prediction IDs do not match the run order')

        if len(saved) < len(cases):
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                                        torch_dtype=torch.bfloat16, attn_implementation='sdpa')
            if adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, args.bundle / 'adapter').get_base_model()
            model = model.to('cuda').eval()
            with torch.inference_mode(), records.open('a', encoding='utf-8', newline='\n') as stream:
                for number, case in enumerate(cases[len(saved):], len(saved) + 1):
                    start = time.monotonic()
                    arms = ARMS if args.phase == 'smoke' or case['split'] == 'dev' else ARMS[:1]
                    results = {arm: infer_case(model, tokenizer, case[arm], args.max_new_tokens,
                                               keep_hidden=arm == 'original') for arm in arms}
                    row = {'sample_id': case['sample_id'], 'split': case['split'], 'results': results,
                           'seconds': round(time.monotonic() - start, 3)}
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    print(number, len(cases), {arm: result['status'] for arm, result in results.items()}, flush=True)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('base', 'sft'))
    parser.add_argument('--phase', required=True, choices=('smoke', 'full'))
    parser.add_argument('--max-new-tokens', type=int, default=640)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error('--max-new-tokens must be positive')
    run(args)


if __name__ == '__main__':
    main()
