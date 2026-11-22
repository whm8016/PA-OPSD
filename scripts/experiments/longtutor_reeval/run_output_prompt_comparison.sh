#!/usr/bin/env bash
set -Eeuo pipefail

source_root=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
run=$(pwd -P)
previous=$1
bundle=$2
runtime=$3
prompts=$4
score_py=$5
arm=${6:-sft}
port=8022
py="$runtime/.venv/bin/python"
entry="$source_root/scripts/experiments/longtutor_reeval/compare_output_prompts.py"
server=

cleanup() {
  code=$?
  if [[ -n "$server" ]]; then
    kill "$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
  fi
  printf '%s\n' "$code" > "$run/job.exit.part"
  mv "$run/job.exit.part" "$run/job.exit"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 PYTHONUTF8=1 PYTHONUNBUFFERED=1

"$py" "$entry" --bundle "$bundle" --prompts "$prompts" --check
"$score_py" "$entry" --bundle "$bundle" --prompts "$prompts" --check --score
wait_for_stage() {
  local stage=$1
  while [[ ! -s "$previous/$stage.exit" ]]; do
    if ! kill -0 "$(cat "$previous/$stage.pid")" 2>/dev/null; then
      if [[ -s "$previous/$stage.exit" ]]; then break; fi
      printf 'Previous %s process exited without a status file\n' "$stage" >&2
      exit 1
    fi
    sleep 15
  done
}

printf '%s waiting for previous evaluation to release the GPU\n' "$(date -Iseconds)"
generation_args=(--arm "$arm")
score_args=(--arm "$arm")
if [[ "$arm" == base ]]; then
  wait_for_stage job
  generation_args+=(--strict-only --base-root "$runtime/qwen_bf16")
  score_args+=(--strict-only)
  if [[ "$(cat "$previous/job.exit")" == 0 ]]; then
    score_args+=(--match-sft "$previous/evaluation/strict")
  fi
else
  wait_for_stage evaluate
  previous_code=$(cat "$previous/evaluate.exit")
  printf '%s previous evaluator exited with %s\n' "$(date -Iseconds)" "$previous_code"
  if [[ "$previous_code" != 0 ]]; then wait_for_stage queue; fi
fi
for _ in $(seq 1 15); do
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -z "$gpu_pids" ]]; then break; fi
  sleep 2
done
if [[ -n "$gpu_pids" ]]; then printf 'GPU is still in use\n' >&2; exit 1; fi

"$py" "$bundle/run.py" serve --bundle "$bundle" --model "$runtime/qwen_bf16" --port "$port" > service.log 2>&1 &
server=$!
for _ in $(seq 1 150); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" > registered_models.json; then break; fi
  kill -0 "$server"
  sleep 2
done
"$py" "$entry" --bundle "$bundle" --prompts "$prompts" --port "$port" --output smoke --smoke "${generation_args[@]}"
"$py" "$entry" --bundle "$bundle" --prompts "$prompts" --port "$port" --output evaluation "${generation_args[@]}"
"$score_py" "$entry" --bundle "$bundle" --prompts "$prompts" --output evaluation --score "${score_args[@]}"
printf '%s finished Gold development evaluation of %s\n' "$(date -Iseconds)" "$arm"
