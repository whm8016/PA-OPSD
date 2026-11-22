#!/usr/bin/env bash
# Start Qwen2.5-7B-Instruct in the same conda environment used by B1 evaluation.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/autodl/local.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/autodl/local.env"
fi
# shellcheck source=autodl/conda_env.sh
source "$ROOT/autodl/conda_env.sh"

if ! conda_env_exists; then
  cat >&2 <<EOF
ERROR: conda environment '$CONDA_ENV' does not exist.
Run this first: bash autodl/setup_longtutor_b1.sh
EOF
  exit 2
fi

activate_longtutor_env

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
if [[ "$TARGET_MODEL" == "Qwen/Qwen2.5-7B-Instruct" ]]; then
  AUTO_MODEL="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"
  if [[ -f "$AUTO_MODEL/config.json" ]]; then
    TARGET_MODEL="$AUTO_MODEL"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
  fi
fi
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

echo "[vLLM] env: $CONDA_ENV"
echo "[vLLM] model: $TARGET_MODEL"
echo "[vLLM] endpoint: http://127.0.0.1:$PORT/v1"
echo "[vLLM] max_model_len: $MAX_MODEL_LEN"

exec vllm serve "$TARGET_MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --dtype bfloat16
