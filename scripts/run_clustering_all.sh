#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_clustering_all.sh <ANAS|KoGES|SNUH>" >&2
  exit 1
fi

COHORT="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COHORT_DIR="${ROOT_DIR}/src/Clustering/${COHORT}"

if [[ ! -d "${COHORT_DIR}" ]]; then
  echo "Cohort directory not found: ${COHORT_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

mapfile -t SCRIPTS < <(find "${COHORT_DIR}" -maxdepth 1 -type f -name '*.py' -print | sort)

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
  echo "No clustering scripts found in ${COHORT_DIR}" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
export KMP_AFFINITY="${KMP_AFFINITY:-disabled}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"

# Default to sequential execution for stability on shared/HPC environments.
JOBS="${CLUSTER_JOBS:-1}"
echo "Running ${#SCRIPTS[@]} scripts for cohort ${COHORT} (jobs=${JOBS})"

failed=0
if [[ "${JOBS}" -le 1 ]]; then
  for script in "${SCRIPTS[@]}"; do
    echo "[START] ${script}"
    if ! python "${script}"; then
      failed=1
    fi
  done
else
  pids=()
  for script in "${SCRIPTS[@]}"; do
    echo "[START] ${script}"
    python "${script}" &
    pids+=("$!")
    while [[ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]]; do
      sleep 0.2
    done
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
fi

if [[ ${failed} -ne 0 ]]; then
  echo "One or more clustering scripts failed." >&2
  exit 1
fi

echo "All clustering scripts completed successfully."
