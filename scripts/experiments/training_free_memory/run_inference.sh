#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: bash run_inference.sh {smoke|repeat|full} {Qwen2.5-7B-Instruct|memory_first} NEW_OUTPUT_DIR" >&2
    exit 2
fi

subset=$1
model=$2
output=$3
case "$subset" in smoke|repeat|full) ;; *) echo "Invalid subset: $subset" >&2; exit 2 ;; esac
case "$model" in Qwen2.5-7B-Instruct|memory_first) ;; *) echo "Invalid model: $model" >&2; exit 2 ;; esac
port=${VLLM_PORT:-8000}
python=${PYTHON:-python}
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
    echo "VLLM_PORT must be a TCP port" >&2
    exit 2
fi

bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
(cd -- "$bundle" && sha256sum --check --quiet SHA256SUMS)
settings=$("$python" - "$bundle/manifest.json" "$subset" <<'PY'
import json
import re
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
subset = sys.argv[2]
arms = manifest.get('subset_arms', {}).get(subset, []) if subset == 'repeat' else manifest['arm_order']
workers = manifest.get('workers', 4)
if (not arms or len(set(arms)) != len(arms) or workers not in (1, 4)
        or any(not re.fullmatch(r'[a-z][a-z0-9_]*', arm) for arm in arms)):
    raise SystemExit('Invalid subset, arm names, or concurrency in the manifest')
print(workers, *arms)
PY
)
read -r -a settings <<< "$settings"
workers=${settings[0]}
arms=("${settings[@]:1}")
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"

export PYTHONHASHSEED=0
export OPENAI_BASE_URL="http://127.0.0.1:$port/v1"
export OPENAI_API_KEY=EMPTY
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
{
    printf 'subset\t%s\nmodel\t%s\nbase_url\t%s\n' "$subset" "$model" "$OPENAI_BASE_URL"
    printf 'temperature\t0\nmax_tokens\t3000\nworkers\t%s\npython_hash_seed\t0\n' "$workers"
    printf 'manifest_sha256\t'
    sha256sum "$bundle/manifest.json" | cut -d ' ' -f 1
} > "$output/run.tsv"

curl --fail --silent --show-error --max-time 5 "$OPENAI_BASE_URL/models" > "$output/models.json"
curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:$port/version" > "$output/version.json"
cp -- "$bundle/service.$port.json" "$output/service.json"
"$python" "$bundle/service.py" verify --port "$port" --model "$model" \
    --receipt "$output/service.json" --models "$output/models.json" --version "$output/version.json"

for arm in "${arms[@]}"; do
    input="$bundle/$subset.$arm.jsonl"
    predictions="$output/$arm.jsonl"
    "$python" -u "$bundle/upstream/eval_ai_tutor.py" \
        --input "$input" --tests "$bundle/queries_only.jsonl" \
        --output "$predictions" --model "$model" --temperature 0 \
        --max_tokens 3000 --workers "$workers" 2>&1 | tee "$output/$arm.log"
    "$python" - "$input" "$predictions" <<'PY' | tee -a "$output/$arm.log"
import json
import sys
from pathlib import Path

def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

expected = {row['_key'] for row in rows(sys.argv[1])}
predictions = rows(sys.argv[2])
actual = {row['_key'] for row in predictions}
if len(predictions) != len(actual) or actual != expected:
    raise SystemExit('Official predictions have missing, extra or duplicate sample keys')

# Keep model-side format failures in the official output and stop on service or input failures.
failures = []
for row in predictions:
    reason = row['reason']
    official_format_failure = (
        reason.startswith('ERROR: RuntimeError: invalid_output_after_retries:')
        or (reason.startswith('ERROR: UnboundLocalError:') and "'parsed'" in reason)
    )
    if not row['diagnosis'] and reason.startswith('ERROR:') and not official_format_failure:
        failures.append(row['_key'])
if failures:
    raise SystemExit(f'{len(failures)} non-format failures. See the unmodified official predictions.')
print(f'Archived {len(predictions)} official predictions.')
PY
done
