#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/run_clustering.sh <ANAS|KoGES|SNUH> <script.py>" >&2
  echo "Example: scripts/run_clustering.sh ANAS PCA2_Kclustering.py" >&2
  exit 1
fi

COHORT="$1"
SCRIPT="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${ROOT_DIR}/src/Clustering/${COHORT}/${SCRIPT}"

if [[ ! -f "${TARGET}" ]]; then
  echo "Script not found: ${TARGET}" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
export KMP_AFFINITY="${KMP_AFFINITY:-disabled}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"

python "${TARGET}"
