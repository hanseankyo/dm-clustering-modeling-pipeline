#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_modeling.sh <pretrain|train|shap>" >&2
  exit 1
fi

TASK="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${ROOT_DIR}/src/Modeling/main.py" "${TASK}"
