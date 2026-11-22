#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode=${1:-run}
case "$mode" in
    preflight|run) ;;
    *) printf 'Usage: bash run.sh [preflight|run]\n' >&2; exit 2 ;;
esac
py=${PA_OPSD_PYTHON:-.venv/bin/python}
trainer=scripts/train_pa_opsd_sft.py
checks=scripts/build_pa_opsd_silver_baseline.py
model=models/Qwen2.5-7B-Instruct
validation=pilot/validation12.jsonl
audit=pilot/audit.json

sha256sum --check --quiet manifest.sha256
if [[ -n ${PA_OPSD_MODEL_DIR:-} ]]; then
    target=$(realpath -e -- "$PA_OPSD_MODEL_DIR")
    mkdir -p models
    if [[ ! -e "$model" && ! -L "$model" ]]; then
        ln -s -- "$target" "$model"
    elif [[ "$(realpath -e -- "$model")" != "$target" ]]; then
        printf 'Configured base model does not match %s\n' "$model" >&2
        exit 2
    fi
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p runs
out=$(mktemp -d "runs/silver_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
printf 'Run directory %s\n' "$out"
trap 'printf "Failed at line %s. Inspect %s\n" "$LINENO" "$out" >&2' ERR

"$py" "$checks" model "$model" pilot/base_model.json "$validation" "$audit" | tee "$out/model_check.log"
"$py" "$trainer" audit --data pilot/train.jsonl --model "$model" --local-files-only \
    --max-length 32768 | tee "$out/train_input_check.log"
if [[ "$mode" == preflight ]]; then
    exit 0
fi

nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv | tee "$out/gpu.txt"
"$py" -c 'import bitsandbytes, peft, torch; assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(); assert torch.cuda.mem_get_info()[1] >= 22 * 1024**3'

train() {
    local phase=$1 data=$2 steps=$3
    printf 'Training %s for %s step(s)\n' "$phase" "$steps"
    "$py" -u "$trainer" train --data "$data" --model "$model" --local-files-only \
        --max-length 32768 --max-steps "$steps" --batch-size 1 --gradient-accumulation-steps 1 \
        --learning-rate 0.00002 --lora-r 8 --seed 42 --load-in-4bit --gradient-checkpointing \
        --assistant-logits-only --offload-saved-tensors --attn-implementation sdpa \
        --output-dir "$out/$phase" > "$out/$phase.log" 2>&1
    "$py" "$checks" checkpoint "$out/$phase" "$data" "$trainer" | tee "$out/${phase}_check.log"
}

generate() {
    local phase=$1 status=0
    shift
    printf 'Generating %s\n' "$phase"
    "$py" -u "$trainer" generate --data "$validation" --model "$model" --split validation \
        --local-files-only --load-in-4bit --attn-implementation sdpa --max-length 32768 \
        --max-new-tokens 1536 --generation-summary "$out/${phase}_summary.json" "$@" \
        > "$out/$phase.jsonl" 2> "$out/${phase}_stderr.log" || status=$?
    printf '%s\n' "$status" > "$out/${phase}_exit.txt"
    # The generator returns 1 when one or more outputs fail native JSON validation.
    if [[ "$status" -gt 1 || ! -s "$out/${phase}_summary.json" ]]; then
        tail -n 25 "$out/${phase}_stderr.log" >&2
        return 2
    fi
}

train probe pilot/probe.jsonl 1
train silver pilot/train.jsonl "$(wc -l < pilot/train.jsonl)"
generate base
generate silver --adapter "$out/silver"
"$py" "$checks" pair "$out" "$audit" "$validation" "$trainer" "$model"
printf 'Completed. Results in %s\n' "$out"
