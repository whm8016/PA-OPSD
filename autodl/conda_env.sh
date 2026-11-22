#!/usr/bin/env bash
# Shared conda environment configuration for LongTutor B1 scripts.

CONDA_ENV="${CONDA_ENV:-longtutor-b1}"

if [[ -z "${CONDA_BIN:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  else
    for candidate in \
      /root/miniconda3/bin/conda \
      /opt/conda/bin/conda \
      /usr/local/miniconda3/bin/conda \
      /usr/local/anaconda3/bin/conda; do
      if [[ -x "$candidate" ]]; then
        CONDA_BIN="$candidate"
        break
      fi
    done
  fi
fi

if [[ -z "${CONDA_BIN:-}" || ! -x "$CONDA_BIN" ]]; then
  cat >&2 <<'EOF'
ERROR: conda executable was not found.
AutoDL images normally provide it at /root/miniconda3/bin/conda.
Open a new AutoDL terminal, then rerun this script.
EOF
  exit 2
fi
export CONDA_BIN

conda_env_exists() {
  "$CONDA_BIN" env list | awk -v env="$CONDA_ENV" '$1 == env {found=1} END {exit !found}'
}

activate_longtutor_env() {
  local conda_root
  conda_root="$(cd "$(dirname "$CONDA_BIN")/.." && pwd)"
  # shellcheck source=/dev/null
  source "$conda_root/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}
