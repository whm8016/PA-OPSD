"""Check saved OPSD rollout completeness against the original generation log."""

import argparse
import json
import re
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.action_pilot import run as files
from scripts.train_pa_opsd_sft import validate_native_output

TOKENS = re.compile(r'generation done - elapsed time: [^,]+, prompts: 1, total tokens: (\d+),')
STOP_SUFFIX = re.compile(r'(?:\s*<\|(?:im_end|endoftext)\|>)+\s*$')


def json_status(completion):
    text, stripped = STOP_SUFFIX.subn('', completion)
    try:
        raw = json.loads(text)
    except ValueError:
        return 'invalid_json', bool(stripped)
    if not isinstance(raw, dict):
        return 'not_json_object', bool(stripped)
    try:
        validate_native_output(text)
    except ValueError:
        return 'invalid_native_schema', bool(stripped)
    return 'valid_native_json', bool(stripped)


def audit(run, out, accumulation):
    summary = files.read_json(run / 'training_summary.json')
    cap = summary['max_completion_length']
    lengths = [int(match) for match in TOKENS.findall((run / 'train.log').read_text(encoding='utf-8'))]
    steps = summary['teacher_context']['optimizer_steps']
    if len(lengths) != steps * accumulation or not all(0 < n <= cap for n in lengths):
        raise ValueError('Unexpected rollout counts or lengths in the training log')
    paths = sorted((run / 'generations').glob('generations_step_*.json'), key=lambda p: int(p.stem.split('_')[-1]))
    if not paths:
        raise ValueError('No stored trajectory chunks')
    audited = []
    for path in paths:
        chunk = files.read_json(path)
        if chunk['num_samples'] != len(chunk['generations']):
            raise ValueError(f'Chunk sample count changed: {path}')
        for row in chunk['generations']:
            i = len(audited)
            if row['step'] != i // accumulation or i >= len(lengths):
                raise ValueError('Cannot align saved trajectories with the logged single-example batch order')
            status, eos = json_status(row['completion'])
            audited.append({'rollout': i + 1, 'step_before_update': row['step'], 'tokens': lengths[i],
                            'at_cap': lengths[i] == cap, 'json_status': status, 'decoded_eos': eos,
                            'prompt_sha256': sha256(row['prompt'].encode()).hexdigest(),
                            'completion_sha256': sha256(row['completion'].encode()).hexdigest()})
    groups = {'at_cap': Counter(), 'below_cap': Counter()}
    for row in audited:
        groups['at_cap' if row['at_cap'] else 'below_cap'][row['json_status']] += 1
    out.mkdir(parents=True, exist_ok=True)
    files.write_rows(out / 'rollout_checks.jsonl', audited)
    files.write_json(out / 'audit.json', {'budget': cap, 'rollouts': len(lengths), 'at_cap': lengths.count(cap),
        'saved_rollouts': len(audited), 'unsaved_rollouts': len(lengths) - len(audited),
        'json_status_by_budget': groups,
        'at_cap_invalid_json_without_eos': sum(row['at_cap'] and row['json_status'] == 'invalid_json'
                                              and not row['decoded_eos'] for row in audited),
        'hashes': {path.relative_to(run).as_posix(): files.digest(path)
                   for path in [run / 'training_summary.json', run / 'train.log', *paths]},
        'method': 'Native response validation on saved rollouts; decoded Qwen terminal tokens stripped before parsing',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gradient-accumulation-steps', type=int, required=True)
    args = parser.parse_args()
    if args.gradient_accumulation_steps < 1:
        parser.error('gradient accumulation must be positive')
    audit(args.run, args.output, args.gradient_accumulation_steps)
