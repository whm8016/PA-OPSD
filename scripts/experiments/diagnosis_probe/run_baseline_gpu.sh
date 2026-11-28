#!/usr/bin/env bash
set -euo pipefail

bundle=$(cd "${1:?extracted combined bundle}" && pwd)
model=$(cd "${2:?verified Qwen2.5-7B-Instruct directory}" && pwd)
train_python=${3:?absolute path to training Python}
eval_python=${4:?absolute path to vLLM Python}
output=${5:?new absolute result directory}
port=${BASELINE_PORT:-18039}
mkdir -p "$output"
output=$(cd "$output" && pwd)
cd "$bundle"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

for phase in oracle weighted; do
    adapter=()
    if [[ "$phase" == weighted ]]; then
        for size in smoke full; do
            extra=()
            [[ "$size" == smoke ]] && extra+=(--smoke)
            destination="$output/train_$size"
            [[ -f "$destination/run_receipt.json" ]] && continue
            if [[ -e "$destination" ]]; then
                printf '%s\n' "Incomplete training at $destination. Preserve it and restart this arm in a fresh result directory." >&2
                exit 2
            fi
            "$train_python" -m scripts.experiments.gold_evidence_first.run --bundle inputs train memory_first \
                --model "$model" --pin base_model.json --diagnosis-token-multiplier 8 \
                --output "$destination" "${extra[@]}"
        done
        adapter=(--adapter "$output/train_full")
    fi
    for size in smoke full; do
        extra=()
        [[ "$size" == smoke ]] && extra+=(--smoke)
        "$train_python" -m scripts.experiments.diagnosis_probe.baseline infer --bundle "$bundle" --model "$model" \
            --variant "$phase" --output "$output/${phase}_$size" "${adapter[@]}" "${extra[@]}" \
            >> "$output/${phase}_$size.log" 2>&1
    done
done

"$eval_python" - <<'PY'
from importlib.metadata import version
from pathlib import Path
from scripts.experiments.diagnosis_probe import run

directory = Path('official')
manifest = run.read_json(directory / 'manifest.json')
protocol = run.read_json(Path('protocol.json'))
if (manifest['upstream_commit'] != 'ba40af3b9e976f5960c42a15a645c0e9c3a7d718'
        or version('vllm') != manifest['vllm_version'] or set(manifest['ids']) != set(protocol['ids'])):
    raise ValueError('Official protocol or vLLM version differs from the earlier SFT')
run.verify_model_files(directory, {name: value for name, value in manifest['files'].items()
    if name.startswith('upstream/') or name in ('inputs.jsonl', 'gold.jsonl')})
PY

cd "$bundle/official"
api="http://127.0.0.1:$port/v1"
adapter=$("$eval_python" -c 'import json,sys; print(json.dumps(dict(name="diagnosis_sft", path=sys.argv[1], base_model_name="Qwen2.5-7B-Instruct")))' "$output/train_full")
"$(dirname "$eval_python")/vllm" serve "$model" --served-model-name Qwen2.5-7B-Instruct --host 127.0.0.1 \
    --port "$port" --dtype bfloat16 --max-model-len 32768 --max-num-seqs 4 --gpu-memory-utilization 0.91 \
    --enable-lora --max-lora-rank 8 --max-cpu-loras 1 --max-log-len 48 --generation-config auto \
    --lora-modules "$adapter" > "$output/vllm.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
for _ in $(seq 1 150); do
    kill -0 "$server"
    if curl -fsS --max-time 3 "$api/models" > "$output/models.json" 2>/dev/null; then break; fi
    sleep 2
done
"$eval_python" - "$output" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
models = {model['id']: model for model in json.loads((output / 'models.json').read_text())['data']}
adapter = models['diagnosis_sft']
if adapter['parent'] != 'Qwen2.5-7B-Instruct' or Path(adapter['root']).resolve() != output / 'train_full':
    raise ValueError('vLLM is serving an unexpected base or checkpoint')
PY

for count in 2 140; do
    preds="$output/official_$count.jsonl"
    OPENAI_API_KEY=EMPTY OPENAI_BASE_URL="$api" \
        "$eval_python" -u upstream/eval_ai_tutor.py --input inputs.jsonl --tests gold.jsonl \
            --output "$preds" --model diagnosis_sft --temperature 0 --max_tokens 3000 --workers 4 --limit "$count" \
            >> "$output/official_$count.log" 2>&1
    "$eval_python" - "$preds" "$count" <<'PY'
import json
import sys
from pathlib import Path

path, count = Path(sys.argv[1]), int(sys.argv[2])
expected = set(json.loads(Path('manifest.json').read_text())['ids'][:count])
rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
if len(rows) != count or {row['_key'] for row in rows} != expected:
    raise ValueError('Official generation has unexpected coverage')
for row in rows:
    reason = row['reason']
    if not row['diagnosis'] and not (reason.startswith('ERROR: RuntimeError: invalid_output_after_retries:')
                                    or "cannot access local variable 'parsed'" in reason):
        raise ValueError('Check the official log and model service for a generation fault')
PY
done
sha256sum "$output/official_140.jsonl" > "$output/official_140.sha256"
