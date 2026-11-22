#!/usr/bin/env bash
# One-time setup for the LongTutor Gold B1 baseline on AutoDL.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LT_REPO="${LT_REPO:-$ROOT/data/longtutor/repo}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"
LT_SEED_ZIP="${LT_SEED_ZIP:-}"

# shellcheck source=autodl/conda_env.sh
source "$ROOT/autodl/conda_env.sh"

if ! conda_env_exists; then
  echo "[Setup] Creating conda environment: $CONDA_ENV"
  "$CONDA_BIN" create -y -n "$CONDA_ENV" python=3.10
else
  echo "[Setup] Reusing conda environment: $CONDA_ENV"
fi

activate_longtutor_env
echo "[Setup] Python: $(command -v python)"
python --version

mkdir -p "$(dirname "$LT_REPO")"

find_longtutor_seed() {
  if [[ -n "$LT_SEED_ZIP" ]]; then
    printf '%s\n' "$LT_SEED_ZIP"
    return 0
  fi
  local candidate
  for candidate in \
    "$ROOT"/*.zip \
    "$ROOT"/autodl/*.zip \
    "$ROOT"/offline/*.zip \
    "$ROOT"/data/*.zip; do
    case "$(basename "$candidate")" in
      LongTutor_official_seed*.zip)
        printf '%s\n' "$candidate"
        return 0
        ;;
    esac
  done
  return 1
}

extract_longtutor_seed() {
  local seed_zip="$1"
  local extract_tmp="$ROOT/.longtutor_seed_extract"
  printf '[Setup] Using offline LongTutor seed: %s\n' "$seed_zip"
  rm -rf "$extract_tmp"
  mkdir -p "$extract_tmp"
  python - "$seed_zip" "$extract_tmp" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
dest = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise SystemExit(f"corrupt zip member: {bad}")
    archive.extractall(dest)
PY

  if [[ -d "$extract_tmp/LongTutor" ]]; then
    rm -rf "$LT_REPO"
    mkdir -p "$(dirname "$LT_REPO")"
    mv "$extract_tmp/LongTutor" "$LT_REPO"
    rm -rf "$extract_tmp"
  else
    echo "ERROR: seed zip must contain a top-level LongTutor/ directory." >&2
    rm -rf "$extract_tmp"
    exit 2
  fi
}

clone_longtutor() {
  local clone_tmp="$ROOT/.longtutor_git_clone"
  rm -rf "$clone_tmp"

  if [[ -f /etc/network_turbo ]]; then
    echo '[Setup] Trying AutoDL academic acceleration (/etc/network_turbo)...'
    if bash -c 'source /etc/network_turbo && git clone --depth 1 https://github.com/liano3/LongTutor.git "$1"' _ "$clone_tmp"; then
      rm -rf "$LT_REPO"
      mkdir -p "$(dirname "$LT_REPO")"
      mv "$clone_tmp" "$LT_REPO"
      return 0
    fi
    rm -rf "$clone_tmp"
  fi

  echo '[Setup] Trying direct GitHub clone...'
  if git clone --depth 1 https://github.com/liano3/LongTutor.git "$clone_tmp"; then
    rm -rf "$LT_REPO"
    mkdir -p "$(dirname "$LT_REPO")"
    mv "$clone_tmp" "$LT_REPO"
    return 0
  fi
  rm -rf "$clone_tmp"
  return 1
}

if [[ ! -d "$LT_REPO/.git" && ! -f "$LT_REPO/.offline-commit" ]]; then
  if [[ -e "$LT_REPO" ]]; then
    if [[ "$LT_REPO" == "$ROOT/data/longtutor/repo" && ! -f "$LT_REPO/requirements.txt" ]]; then
      echo '[Setup] Removing incomplete default LongTutor clone.'
      rm -rf "$LT_REPO"
    else
      echo "ERROR: $LT_REPO exists but is not a git repository." >&2
      exit 2
    fi
  fi

  if ! clone_longtutor; then
    if seed_zip="$(find_longtutor_seed)"; then
      extract_longtutor_seed "$seed_zip"
    else
      cat >&2 <<'EOF'
ERROR: GitHub clone failed and no offline seed zip was found.

Do one of these:
1) Run AutoDL acceleration and retry:
   source /etc/network_turbo
   bash autodl/setup_longtutor_b1.sh

2) Upload LongTutor_official_seed_*.zip beside this project's autodl/ directory,
   then rerun:
   bash autodl/setup_longtutor_b1.sh
EOF
      exit 2
    fi
  fi
fi

pip_install_with_fallback() {
  local specs=("$@")
  local index

  export PIP_DISABLE_PIP_VERSION_CHECK=1
  echo "[Setup] pip install: ${specs[*]}"
  if python -m pip install "${specs[@]}"; then
    return 0
  fi

  for index in \
    https://mirrors.aliyun.com/pypi/simple/ \
    https://pypi.tuna.tsinghua.edu.cn/simple \
    https://pypi.org/simple; do
    echo "[Setup] Retrying with index: $index"
    if python -m pip install --index-url "$index" "${specs[@]}"; then
      return 0
    fi
  done

  if [[ -f /etc/network_turbo ]]; then
    echo "[Setup] Retrying with AutoDL academic acceleration and official PyPI."
    if bash -c 'source /etc/network_turbo && python -m pip install --index-url https://pypi.org/simple "$@"' _ "${specs[@]}"; then
      return 0
    fi
  fi

  echo "ERROR: all configured Python package indexes failed." >&2
  return 1
}

pip_install_with_fallback --upgrade pip setuptools wheel
pip_install_with_fallback -r "$LT_REPO/requirements.txt"

if [[ "$INSTALL_VLLM" == "1" ]]; then
  echo "[Setup] Installing vLLM: $VLLM_SPEC"
  pip_install_with_fallback "$VLLM_SPEC"
else
  echo "[Setup] INSTALL_VLLM=0; skip vLLM installation."
fi

python -m pip check
python "$ROOT/scripts/build_longtutor_xes_history.py"
python "$ROOT/scripts/verify_longtutor_gold_alignment.py"
python -m pip freeze > "$ROOT/autodl/longtutor_b1_freeze.txt"

cat <<EOF

[Setup complete]
LongTutor repo: $LT_REPO

Next:
1) Configure once: bash autodl/configure_longtutor_b1.sh
2) Start vLLM: bash autodl/start_vllm_qwen25_7b.sh
3) Smoke test: bash autodl/run_longtutor_b1.sh smoke
4) Full run: bash autodl/run_longtutor_b1.sh full
EOF
