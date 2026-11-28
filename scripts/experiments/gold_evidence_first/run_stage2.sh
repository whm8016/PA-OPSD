#!/usr/bin/env bash
set -euo pipefail
umask 077

arm=${1:?Usage: run_stage2.sh ARM OUTPUT_DIRECTORY}
out=${2:?Usage: run_stage2.sh ARM OUTPUT_DIRECTORY}
case "$arm" in
  sft_cont|opsd|opsd_compact|gkd_frozen) ;;
  *) printf 'Unknown arm %s\n' "$arm" >&2; exit 2 ;;
esac

python_bin=${OPD_PYTHON:-python}
model=${OPD_STUDENT_MODEL:-/root/autodl-tmp/longtutor_gold_ba40af3/qwen_bf16}
lr=${OPD_LEARNING_RATE:-5e-6}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

mkdir -p "$(dirname "$out")"
mkdir "$out"
trap 'code=$?; printf "%s\n" "$code" > "$out/complete.exit"' EXIT
exec > >(tee "$out/driver.log") 2>&1

gold() {
  "$python_bin" -u -m scripts.experiments.gold_evidence_first.run --bundle inputs "$@"
}
common=(--model "$model" --pin base_model.json)
train=(--anchor anchor --learning-rate "$lr")
teacher=()
if [[ "$arm" == gkd_frozen ]]; then
  teacher=(--teacher-model "${OPD_FROZEN_TEACHER:?Set OPD_FROZEN_TEACHER to the pinned 14B cache}"
           --teacher-pin teacher_14b.json)
fi

printf '%s 起始组 %s\n' "$(date -Iseconds)" "$arm"
if [[ "$arm" != sft_cont ]]; then
  probe=(--checkpoint anchor --teacher-pair --teacher-arm "$arm" --student-eval anchor_eval)
  if [[ "$arm" == opsd_compact ]]; then
    gold evaluate "${common[@]}" "${probe[@]}" --probe-manifest compact_probe_ids.json --output "$out/probe24"
  fi
  gold evaluate "${common[@]}" "${probe[@]}" "${teacher[@]}" --output "$out/teacher_probe"
  train+=(--teacher-report "$out/teacher_probe/report.json")
fi

gold train "$arm" "${common[@]}" "${train[@]}" "${teacher[@]}" --smoke --output "$out/smoke"
gold train "$arm" "${common[@]}" "${train[@]}" "${teacher[@]}" --output "$out/train"
gold evaluate "${common[@]}" --checkpoint "$out/train" --output "$out/eval"
printf '%s 训练和评测完成\n' "$(date -Iseconds)"
