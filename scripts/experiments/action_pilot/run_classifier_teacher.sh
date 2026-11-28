#!/usr/bin/env bash
set -euo pipefail
umask 077

# Run from the unpacked bundle. Arguments are absolute paths.
python=${1:?Pass the vLLM environment Python}
model=${2:?Pass the pinned base model}
adapter=${3:?Pass the pinned Gold SFT adapter}
port=8808
export PATH="$(dirname -- "$python"):$PATH" PYTHONIOENCODING=utf-8
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
exec 9>/root/autodl-tmp/.action_teacher_evidence_gpu.lock
flock -n 9
mkdir -- runs
touch service.log gpu_before.txt
exec >> queue.log 2>&1
date -Iseconds > started_at.txt
service_pid=
probe_pid=

cleanup() {
    local status=$?
    trap - EXIT
    trap '' TERM INT
    if [[ -n "$probe_pid" ]]; then
        kill -TERM "$probe_pid" 2>/dev/null || true
        wait "$probe_pid" 2>/dev/null || true
    fi
    if [[ -n "$service_pid" ]]; then
        kill -TERM -- "-$service_pid" 2>/dev/null || true
        for attempt in {1..12}; do
            if ! kill -0 -- "-$service_pid" 2>/dev/null; then break; fi
            sleep 1
        done
        kill -KILL -- "-$service_pid" 2>/dev/null || true
        wait "$service_pid" 2>/dev/null || true
    fi
    nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > gpu_after.txt || true
    date -Iseconds > finished_at.txt
    printf '%s\n' "$status" > queue.exit
    local files=(runs service.log queue.log queue.exit started_at.txt finished_at.txt gpu_before.txt gpu_after.txt manifest.json SHA256SUMS)
    if [[ -f "service.$port.json" ]]; then files+=("service.$port.json"); fi
    tar -czf gpu_results.tmp.tar.gz "${files[@]}"
    mv -- gpu_results.tmp.tar.gz gpu_results.tar.gz
    sha256sum gpu_results.tar.gz > gpu_results.sha256
    mv -- queue.exit queue.done
    exit "$status"
}

run_phase() {
    local phase=$1 status=0
    "$python" -u -m scripts.experiments.action_pilot.classifier_teacher probe \
        --bundle . --phase "$phase" --output "runs/$phase.jsonl" --port "$port" &
    probe_pid=$!
    wait "$probe_pid" || status=$?
    probe_pid=
    return "$status"
}

trap cleanup EXIT
trap 'exit 124' TERM
trap 'exit 130' INT
sha256sum --check --quiet SHA256SUMS
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > gpu_before.txt
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$active" ]]; then
    echo 'Another compute process is active' >&2
    exit 1
fi
"$python" - "$port" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PY
setsid "$python" -u service.py serve --model "$model" --adapter "$adapter" --port "$port" > service.log 2>&1 &
service_pid=$!
ready=false
for attempt in {1..125}; do
    kill -0 "$service_pid"
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$port/v1/models" > /dev/null; then
        ready=true
        break
    fi
    sleep 2
done
"$ready"
run_phase smoke
if ! run_phase full; then
    if grep -Eq '^openai\.(APIConnectionError|APITimeoutError):' queue.log; then
        run_phase full
    else
        exit 1
    fi
fi
