#!/usr/bin/env bash
set -Eeuo pipefail

py=${1:?pass the Python executable from the existing AutoDL training environment}
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRL_EXPERIMENTAL_SILENCE=1

sha256sum --check <<'EOF'
677d7714d1d0710fca8070b7253a395bafb786b3afedf258f78b524cc0c836fd  continuation.jsonl
e48ee4d7846964b03876437acfbf5ea965578ba851451f672c856651749d5657  questions.jsonl
40be07306e9070de6b47d9dfdd372ae7a6183b81aad1d728ae27141f4f37313b  anchor/adapter_model.safetensors
f826486770549a04126d964ac37f9b16464bf0fd01790540ccea4b43aeb2e9f8  smoke_input.jsonl
EOF

mkdir smoke formal
flags=(--seed-manifest split_manifest.json --sft-checkpoint anchor
       --teacher-mode local-same-model-privileged --standard-solution-bank questions.jsonl
       --load-in-4bit --max-length 32768 --max-completion-length 1024
       --batch-size 1 --learning-rate 0.000001 --seed 42 --gradient-checkpointing)

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 3 > gpu_monitor.csv &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT

printf 'Start 1024-token smoke on the longest teacher input\n'
"$py" -u scripts/run_pa_opsd_gold.py opsd "${flags[@]}" \
    --train-jsonl smoke_input.jsonl --output-dir smoke --gradient-accumulation-steps 1 --max-steps 1 > smoke/train.log 2>&1
"$py" scripts/run_pa_opsd_gold.py check-update smoke --initial anchor

printf 'Smoke passed; start 35 formal optimizer steps\n'
"$py" -u scripts/run_pa_opsd_gold.py opsd "${flags[@]}" \
    --train-jsonl continuation.jsonl --output-dir formal --gradient-accumulation-steps 4 \
    --max-steps 35 --checkpoint-steps 3 > formal/train.log 2>&1
"$py" scripts/run_pa_opsd_gold.py check-update formal --initial anchor
printf 'Formal training completed\n'
