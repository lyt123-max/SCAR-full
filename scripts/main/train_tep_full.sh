#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${1:-full}"
EXP_BASE_NAME="${2:-tep_full_mechanism}"
EXP_NAME_OVERRIDE="${3:-}"
RESUME="${RESUME:-0}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-./artifacts}"
DATA_ROOT="${4:-${DATA_ROOT:-./Multi-mode-Fault-Diagnosis-Datasets-with-TE-process}}"
TEP_MECHANISM_POSTPROCESS="${TEP_MECHANISM_POSTPROCESS:-1}"
TEP_ALLOW_PLOTS="${TEP_ALLOW_PLOTS:-0}"
FORMAL_REBUTTAL="${FORMAL_REBUTTAL:-1}"

if [[ "${FORMAL_REBUTTAL}" == "1" && "${TEP_ALLOW_PLOTS}" == "1" ]]; then
  echo "[TEP Full] formal rebuttal mode forbids plot generation." >&2
  exit 2
fi
if [[ "${FORMAL_REBUTTAL}" == "1" && "${MAX_TEST_SEQUENCES:-0}" != "0" ]]; then
  echo "[TEP Full] formal rebuttal mode requires all test sequences." >&2
  exit 2
fi

if [[ -n "${EXP_NAME_OVERRIDE}" ]]; then
  EXP_NAME="${EXP_NAME_OVERRIDE}"
elif [[ "${RESUME}" == "1" ]]; then
  EXP_NAME="${EXP_NAME:-${EXP_BASE_NAME}}"
else
  TIMESTAMP="${EXP_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
  EXP_NAME="${EXP_NAME:-${EXP_BASE_NAME}_${TIMESTAMP}}"
fi

export EXP_NAME
export RESUME
export ARTIFACT_ROOT
export DATA_ROOT
export TEP_PROTOCOL="full"

EXP_DIR="${ARTIFACT_ROOT}/${EXP_NAME}"
AUDIT_DIR="${EXP_DIR}/tep_data_audit"
mkdir -p "${AUDIT_DIR}"

python scripts/tep/audit_tep_dataset.py \
  --data-root "${DATA_ROOT}" \
  --protocol full \
  --seq-len "${SEQ_LEN:-128}" \
  --output-dir "${AUDIT_DIR}"

bash scripts/main/train_tep.sh "${STAGE}" "${EXP_BASE_NAME}"

case "${STAGE}" in
  full|test)
    if [[ "${TEP_MECHANISM_POSTPROCESS}" == "1" ]]; then
      EXPORT_ARGS=(--experiment_dir "${EXP_DIR}")
      if [[ "${RESUME}" == "1" ]]; then
        EXPORT_ARGS+=(--resume)
      fi
      python scripts/tep/export_mechanism_logs.py "${EXPORT_ARGS[@]}"
      python scripts/tep/compute_mechanism_metrics.py \
        --experiment_dirs "${EXP_DIR}"
      if [[ "${TEP_ALLOW_PLOTS}" == "1" ]]; then
        python scripts/tep/plot_mechanism_figures.py \
          --experiment_dirs "${EXP_DIR}"
      fi
      TABLE_ARGS=(
        --full-experiment "${EXP_DIR}"
        --output-dir "${EXP_DIR}/tep_rebuttal_tables"
      )
      if [[ -n "${TEP_SELECTED_EXPERIMENT_DIR:-}" ]]; then
        TABLE_ARGS+=(--selected-experiment "${TEP_SELECTED_EXPERIMENT_DIR}")
      fi
      python scripts/tep/generate_rebuttal_tables.py "${TABLE_ARGS[@]}"
    fi
    ;;
esac

echo "[TEP Full] completed stage=${STAGE} experiment=${EXP_DIR}"
