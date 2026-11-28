"""Start the pinned vLLM service, or verify it before running LongTutor inference."""

import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_files(directory, hashes):
    for name, expected in hashes.items():
        if digest(directory / name) != expected:
            raise ValueError(f'File differs from the pinned model: {directory / name}')


def serve(args, manifest):
    if not 1 <= args.port <= 65535:
        raise ValueError('Choose a TCP port between 1 and 65535')
    installed = version('vllm')
    if installed != manifest['vllm_version']:
        raise ValueError(f"Expected vLLM {manifest['vllm_version']}, found {installed}")
    pin = read_json(BUNDLE / 'base_model.json')
    if pin['revision'] != manifest['base_revision']:
        raise ValueError('Model revision differs from the experiment manifest')
    model = args.model.resolve(strict=True)
    hashes = dict(pin['config_and_tokenizer_sha256'])
    hashes.update({name: data['sha256'] for name, data in pin['weights'].items()})
    hashes.update({'generation_config.json': pin['generation_config_sha256'],
                   'model.safetensors.index.json': pin['weight_index_sha256']})
    verify_files(model, hashes)
    if any((model / name).exists() for name in ('chat_template.jinja', 'chat_template.json', 'chat_templates')):
        raise ValueError('Use the embedded chat template in tokenizer_config.json')

    settings = manifest['inference']
    base = manifest['base_model']
    executable = Path(sys.executable).with_name('vllm')
    if not executable.is_file():
        raise FileNotFoundError(f'vLLM executable is missing: {executable}')
    command = [str(executable), 'serve', str(model), '--served-model-name', base,
               '--host', '127.0.0.1', '--port', str(args.port), '--dtype', settings['dtype'],
               '--max-model-len', str(settings['max_model_len']), '--max-num-seqs', '4',
               '--gpu-memory-utilization', '0.91', '--max-log-len', '48', '--generation-config', 'auto']
    roots = {base: str(model)}
    if args.adapter is not None:
        adapter = args.adapter.resolve(strict=True)
        verify_files(adapter, manifest['adapter_sha256'])
        roots['memory_first'] = str(adapter)
        config = json.dumps({'name': 'memory_first', 'path': str(adapter), 'base_model_name': base})
        command += ['--enable-lora', '--max-lora-rank', '8', '--max-cpu-loras', '1', '--lora-modules', config]
    receipt = {'manifest_sha256': digest(BUNDLE / 'manifest.json'), 'roots': roots,
               'base_revision': pin['revision'], 'port': args.port, 'pid': os.getpid(), 'vllm_version': installed,
               'dtype': settings['dtype'], 'max_model_len': settings['max_model_len']}
    print('Verified weights and service settings', flush=True)
    (BUNDLE / f'service.{args.port}.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    os.execve(executable, command, dict(os.environ, HF_HUB_OFFLINE='1', VLLM_NO_USAGE_STATS='1',
                                       VLLM_ALLOW_RUNTIME_LORA_UPDATING='0'))


def verify(args, manifest):
    receipt = read_json(args.receipt)
    settings = manifest['inference']
    expected = {'manifest_sha256': digest(BUNDLE / 'manifest.json'), 'base_revision': manifest['base_revision'],
                'port': args.port, 'vllm_version': manifest['vllm_version'],
                'dtype': settings['dtype'], 'max_model_len': settings['max_model_len']}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Service receipt does not match this experiment')
    if read_json(args.version)['version'] != manifest['vllm_version']:
        raise ValueError('Active vLLM version differs from the pinned version')
    if sys.platform.startswith('linux'):
        cmdline = (Path('/proc') / str(int(receipt['pid'])) / 'cmdline').read_bytes().split(b'\x00')
        options = [(b'--port', str(args.port).encode()), (b'--dtype', settings['dtype'].encode()),
                   (b'--max-model-len', str(settings['max_model_len']).encode())]
        if b'serve' not in cmdline or not set(options).issubset(pairwise(cmdline)):
            raise ValueError('The pinned service process or its launch settings changed')

    registered = {model['id']: model for model in read_json(args.models)['data']}
    base = manifest['base_model']
    if args.model not in registered or args.model not in receipt['roots']:
        raise ValueError(f'Model {args.model} is unavailable. Registered models: {sorted(registered)}')
    if registered[base]['max_model_len'] != settings['max_model_len']:
        raise ValueError('Active context length differs from the pinned context length')
    for alias in {base, args.model}:
        if registered[alias]['root'] != receipt['roots'][alias]:
            raise ValueError(f'Active checkpoint root differs from the pinned service receipt: {alias}')
    if args.model != base and registered[args.model]['parent'] != base:
        raise ValueError('Active adapter parent differs from the pinned base')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='action', required=True)
    start = subparsers.add_parser('serve')
    start.add_argument('--model', type=Path, required=True)
    start.add_argument('--adapter', type=Path)
    start.add_argument('--port', type=int, default=8000)
    check = subparsers.add_parser('verify')
    check.add_argument('--port', type=int, required=True)
    check.add_argument('--model', required=True)
    for option in ('receipt', 'models', 'version'):
        check.add_argument(f'--{option}', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(BUNDLE / 'manifest.json')
    if args.action == 'serve':
        serve(args, manifest)
    else:
        verify(args, manifest)


if __name__ == '__main__':
    main()
