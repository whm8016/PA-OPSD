#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode=${1:-smoke}
case "$mode" in
    setup|preflight|smoke|run|final|base|eval-sft|sft-cont|pa-opsd|merge) ;;
    *) printf 'Usage: bash run.sh {setup|preflight|smoke|run|final RUN_DIR|base|eval-sft SFT|sft-cont SFT|pa-opsd SFT|merge BASE SFT SFT_CONT PA_OPSD}\n' >&2; exit 2 ;;
esac
sha256sum --check --quiet manifest.sha256
py=${PA_OPSD_PYTHON:-.venv/bin/python}
if [[ ! -x "$py" && -z ${PA_OPSD_PYTHON:-} && "$mode" != setup ]]; then
    py=python
fi
model=${PA_OPSD_MODEL_DIR:-models/Qwen2.5-7B-Instruct}
pin=base_model.json
manifest=split_manifest.json
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false

if [[ "$mode" == setup ]]; then
    if [[ -z ${PA_OPSD_PYTHON:-} && ! -x "$py" ]]; then
        python -m venv --system-site-packages .venv
    fi
    "$py" -c 'import torch; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda); assert torch.cuda.is_available(), "Use an AutoDL image with a GPU-enabled PyTorch"'
    packages=(transformers==4.57.1 trl==0.26.0 accelerate==1.11.0 datasets==3.6.0
        peft==0.17.1 bitsandbytes==0.48.2 'openai>=1,<3')
    if ! "$py" -m pip install "${packages[@]}"; then
        "$py" -m pip install --index-url https://pypi.org/simple "${packages[@]}"
    fi
    if [[ -z ${PA_OPSD_MODEL_DIR:-} ]]; then
        "$py" - "$pin" "$model" <<'PY'
import json
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

pin = json.loads(Path(sys.argv[1]).read_text())
snapshot_download(pin['model'], revision=pin['revision'], local_dir=sys.argv[2],
                  allow_patterns=['*.json', '*.safetensors', 'merges.txt'])
PY
    fi
    exit 0
fi

driver=scripts/run_pa_opsd_gold.py
max_length=32768
eval_tokens=1536
if [[ "$mode" == merge ]]; then
    if [[ $# -ne 5 ]]; then
        printf 'Usage: bash run.sh merge BASE SFT SFT_CONT PA_OPSD\n' >&2
        exit 2
    fi
    names=(base sft sft_cont pa_opsd)
    dirs=("$2" "$3" "$4" "$5")
    anchor_hash=$(<"$3/sft_anchor.sha256")
    if [[ ! "$anchor_hash" =~ ^[0-9a-f]{64}$ || "$anchor_hash" != "$(<"$4/sft_anchor.sha256")" \
          || "$anchor_hash" != "$(<"$5/sft_anchor.sha256")" ]]; then
        printf 'The tuned arms must use the same SFT checkpoint.\n' >&2
        exit 2
    fi
    mkdir -p runs
    out=$(mktemp -d 'runs/merged_XXXXXX')
    for i in "${!names[@]}"; do
        [[ -f "${dirs[i]}/arm.complete" ]]
        cp -- "${dirs[i]}/${names[i]}.jsonl" "$out/"
        cp -- "${dirs[i]}/${names[i]}_generation.json" "$out/"
    done
    "$py" - "$manifest" "$out" "$max_length" "$eval_tokens" "${names[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifest, out, context, completion, *arms = sys.argv[1:]
split = json.loads(Path(manifest).read_text(encoding='utf-8'))
expected = {
    'data_sha256': split['files']['development/prompts.jsonl'],
    'selected_rows': split['sizes']['validation'],
    'split': 'validation',
    'max_length': int(context),
    'max_new_tokens': int(completion),
    'completion_no_repeat_ngram_size': 0,
    'do_sample': False,
    'load_in_4bit': True,
}
for arm in arms:
    summary = json.loads((Path(out) / f'{arm}_generation.json').read_text(encoding='utf-8'))
    if bool(summary['adapter']) != (arm != 'base') or any(summary[key] != value for key, value in expected.items()):
        raise SystemExit(f'Generation settings or frozen validation split differ for {arm}.')
PY
    printf '%s\n' "$anchor_hash" > "$out/sft_anchor.sha256"
    "$py" "$driver" score "$manifest" development "$out" > "$out/score.log"
    touch -- "$out/development.complete"
    printf 'Completed. Metrics in %s/metrics.json\n' "$out"
    exit 0
fi

model=$(realpath -e -- "$model")
train_data=development/train.jsonl
prompts=development/prompts.jsonl
scope=development
rollout_tokens=512
common=(--seed-manifest "$manifest" --max-length "$max_length" --load-in-4bit --seed 42)
generation=(--model "$model" --local-files-only --max-new-tokens "$eval_tokens" --split validation)
limit=()

if [[ "$mode" == final ]]; then
    run=${2:?Usage: bash run.sh final RUN_DIR}
    [[ -f "$run/development.complete" && -s "$run/metrics.json" && -d sealed ]]
    "$py" "$driver" check-final "$manifest" "$model" "$run" "$pin"
    out="$run/final"
    mkdir -- "$out"
    prompts=sealed/prompts.jsonl
    scope=sealed
else
    mkdir -p runs
    out=$(mktemp -d "runs/${mode}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    run=$out
fi
printf 'Run directory %s\n' "$out"
trap 'printf "Failed at line %s. Inspect %s\n" "$LINENO" "$out" >&2' ERR

if [[ "$mode" != final ]]; then
    "$py" "$driver" preflight "$manifest" "$model" "$out" "$pin" | tee "$out/preflight.log"
    if [[ "$mode" == preflight ]]; then
        exit 0
    fi
fi

nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv | tee "$out/gpu.txt"
"$py" - <<'PY' | tee "$out/cuda.txt"
import bitsandbytes
import torch
from opsd_trainer import OPSDTrainer

assert torch.cuda.device_count() == 1, 'Select one visible CUDA GPU'
assert torch.cuda.is_bf16_supported(), 'This run uses CUDA BF16'
free, total = torch.cuda.mem_get_info()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), free, total)
assert free >= 20 * 1024**3, 'Free at least 20 GiB on the selected CUDA GPU'
PY

train_sft() {
    local phase=$1 data=$2 steps=$3 lr=$4
    shift 4
    printf 'Training %s (%s optimizer steps)\n' "$phase" "$steps"
    "$py" -u "$driver" sft train --data "$data" --model "$model" "${common[@]}" \
        --output-dir "$out/$phase" --local-files-only --max-steps "$steps" \
        --batch-size 1 --gradient-accumulation-steps "$accumulation" --learning-rate "$lr" \
        --lora-r 8 --gradient-checkpointing --assistant-logits-only --offload-saved-tensors \
        "$@" > "$out/$phase.log" 2>&1
}

train_continuation() {
    local anchor=$1
    train_sft sft_cont "$opsd_data" "$continuation_steps" 0.000001 --init-adapter "$anchor"
    "$py" "$driver" check-update "$out/sft_cont" --initial "$anchor"
}

train_opsd() {
    local anchor=$1
    printf 'Training pa_opsd (%s optimizer steps)\n' "$continuation_steps"
    "$py" -u "$driver" opsd --train-jsonl "$opsd_data" "${common[@]}" \
        --sft-checkpoint "$anchor" --teacher-mode local-same-model-privileged \
        --output-dir "$out/pa_opsd" --batch-size 1 --gradient-accumulation-steps "$accumulation" \
        --max-steps "$continuation_steps" --learning-rate 0.000001 \
        --max-completion-length "$rollout_tokens" --gradient-checkpointing \
        > "$out/pa_opsd.log" 2>&1
    "$py" "$driver" check-update "$out/pa_opsd" --initial "$anchor"
}

generate() {
    local phase=$1 status=0
    shift
    printf 'Evaluating %s on %s\n' "$phase" "$scope"
    "$py" -u "$driver" sft generate --data "$prompts" "${common[@]}" "${generation[@]}" \
        "${limit[@]}" "$@" --generation-summary "$out/${phase}_generation.json" \
        > "$out/$phase.jsonl" 2> "$out/${phase}_generation.log" || status=$?
    if [[ "$status" -gt 1 || ! -s "$out/${phase}_generation.json" ]]; then
        tail -n 25 "$out/${phase}_generation.log" >&2
        return 2
    fi
}

accumulation=4
continuation_steps=35
opsd_data="$out/continuation.jsonl"

case "$mode" in
    base)
        generate base
        touch -- "$out/arm.complete"
        exit 0
        ;;
    eval-sft|sft-cont|pa-opsd)
        anchor=$(realpath -e -- "${2:?Supply the completed Gold SFT adapter directory}")
        "$py" - "$anchor" "$model" "$manifest" <<'PY'
import json
import sys
from pathlib import Path

anchor, model, manifest = map(Path, sys.argv[1:])
config = json.loads((anchor / 'adapter_config.json').read_text(encoding='utf-8'))
summary = json.loads((anchor / 'training_summary.json').read_text(encoding='utf-8'))
split = json.loads(manifest.read_text(encoding='utf-8'))
if Path(config['base_model_name_or_path']).resolve() != model:
    raise SystemExit('Mount the verified model at the same absolute path as the original Gold SFT run.')
if (summary['data_sha256'] != split['files']['development/train.jsonl']
        or summary['rows'] != split['sizes']['train'] or summary['optimizer_restarted']
        or summary['seed'] != 42 or summary['lora_r'] != 8 or not summary['load_in_4bit']
        or summary['metrics']['epoch'] < 1.0):
    raise SystemExit('Expected the full 560-row Gold SFT checkpoint.')
PY
        "$py" "$driver" check-update "$anchor"
        (
            cd -- "$anchor"
            shopt -s nullglob
            sha256sum -- adapter_model.safetensors adapter_config.json tokenizer.json tokenizer_config.json \
                special_tokens_map.json* added_tokens.json* merges.txt* vocab.json*
        ) | sha256sum | cut -d ' ' -f 1 > "$out/sft_anchor.sha256"
        case "$mode" in
            eval-sft) generate sft --adapter "$anchor" ;;
            sft-cont)
                train_continuation "$anchor"
                generate sft_cont --adapter "$out/sft_cont"
                ;;
            pa-opsd)
                train_opsd "$anchor"
                generate pa_opsd --adapter "$out/pa_opsd"
                ;;
        esac
        touch -- "$out/arm.complete"
        exit 0
        ;;
esac

if [[ "$mode" != final ]]; then
    sft_steps=140
    if [[ "$mode" == smoke ]]; then
        accumulation=1
        sft_steps=1
        continuation_steps=1
        train_data="$out/probe_sft.jsonl"
        opsd_data="$out/probe_pa_opsd.jsonl"
        limit=(--limit 4)
    fi
    train_sft sft "$train_data" "$sft_steps" 0.00002
    "$py" "$driver" check-update "$out/sft"
    train_continuation "$out/sft"
    train_opsd "$out/sft"
fi

generate base
for arm in sft sft_cont pa_opsd; do
    generate "$arm" --adapter "$run/$arm"
done
"$py" "$driver" score "$manifest" "$scope" "$out" "${limit[@]}" > "$out/score.log"
if [[ "$mode" == run ]]; then
    touch -- "$out/development.complete"
fi
printf 'Completed. Metrics in %s/metrics.json\n' "$out"
