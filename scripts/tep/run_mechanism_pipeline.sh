#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_DIR="${1:?Usage: run_mechanism_pipeline.sh <experiment_dir> [ablation_key]}"
ABLATION_KEY="${2:-}"

PYTHON_EXE="${PYTHON_EXE:-python}"
OUTPUT_DIR="${TEP_MECHANISM_OUTPUT_DIR:-${EXPERIMENT_DIR}/tep_mechanism}"
WINDOW_AGGREGATION="${TEP_MECHANISM_WINDOW_AGGREGATION:-p95}"
EMBEDDING_METHOD="${TEP_MECHANISM_EMBEDDING_METHOD:-umap}"
TOP_K="${TEP_MECHANISM_TOP_K:-0}"
SMC_K="${TEP_MECHANISM_SMC_K:-10}"
SMR_K="${TEP_MECHANISM_SMR_K:-10}"
FPR_QUANTILE="${TEP_MECHANISM_FPR_QUANTILE:-0.95}"
RUN_EXPORT="${TEP_MECHANISM_RUN_EXPORT:-1}"
RUN_METRICS="${TEP_MECHANISM_RUN_METRICS:-1}"
RUN_PLOTS="${TEP_MECHANISM_RUN_PLOTS:-1}"
RUN_A4_COMPARE="${TEP_MECHANISM_RUN_A4_COMPARE:-1}"
RUN_A4_COMPARE_STRICT="${TEP_A4_COMPARE_STRICT:-1}"
A4_COMPARE_STATUS_PATH="${OUTPUT_DIR}/a4_compare_status.json"

mkdir -p "${OUTPUT_DIR}"

echo "[TEP Mechanism] experiment_dir=${EXPERIMENT_DIR}"
echo "[TEP Mechanism] output_dir=${OUTPUT_DIR} ablation_key=${ABLATION_KEY:-<none>}"

write_a4_compare_status() {
  local status="$1"
  local reason="$2"
  local short_dir="${3:-}"
  local long_dir="${4:-}"
  local multi_dir="${5:-}"
  "${PYTHON_EXE}" -c "import json, sys; json.dump({'status': sys.argv[1], 'reason': sys.argv[2], 'ablation_key': sys.argv[3], 'short_dir': sys.argv[4], 'long_dir': sys.argv[5], 'multi_dir': sys.argv[6]}, open(sys.argv[7], 'w', encoding='utf-8'), indent=2)" \
    "${status}" \
    "${reason}" \
    "${ABLATION_KEY}" \
    "${short_dir}" \
    "${long_dir}" \
    "${multi_dir}" \
    "${A4_COMPARE_STATUS_PATH}"
}

if [[ "${RUN_EXPORT}" == "1" ]]; then
  "${PYTHON_EXE}" scripts/tep/export_mechanism_logs.py \
    --experiment_dir "${EXPERIMENT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --window_aggregation "${WINDOW_AGGREGATION}" \
    --top_k "${TOP_K}"
fi

if [[ "${RUN_METRICS}" == "1" ]]; then
  "${PYTHON_EXE}" scripts/tep/compute_mechanism_metrics.py \
    --experiment_dirs "${EXPERIMENT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --smc_k "${SMC_K}" \
    --smr_k "${SMR_K}" \
    --fpr_quantile "${FPR_QUANTILE}"
fi

if [[ "${RUN_PLOTS}" == "1" ]]; then
  "${PYTHON_EXE}" scripts/tep/plot_mechanism_figures.py \
    --experiment_dirs "${EXPERIMENT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --embedding_method "${EMBEDDING_METHOD}" \
    --smr_k "${SMR_K}"
fi

if [[ "${RUN_A4_COMPARE}" != "1" ]]; then
  write_a4_compare_status "disabled" "TEP_MECHANISM_RUN_A4_COMPARE=0"
  exit 0
fi

if [[ "${ABLATION_KEY}" != "multi_scale" ]]; then
  write_a4_compare_status "not_applicable" "ablation_key_is_not_multi_scale"
  exit 0
fi

EXP_DIR_PATH="$(cd "${EXPERIMENT_DIR}" && pwd)"
EXP_PARENT="$(dirname "${EXP_DIR_PATH}")"
EXP_BASENAME="$(basename "${EXP_DIR_PATH}")"

SHORT_EXP_DIR="${TEP_A4_SHORT_EXP_DIR:-}"
LONG_EXP_DIR="${TEP_A4_LONG_EXP_DIR:-}"
MULTI_EXP_DIR="${TEP_A4_MULTI_EXP_DIR:-${EXP_DIR_PATH}}"

if [[ -z "${SHORT_EXP_DIR}" ]]; then
  CANDIDATE_SHORT="${EXP_BASENAME/multi_scale/single_scale_short}"
  if [[ "${CANDIDATE_SHORT}" != "${EXP_BASENAME}" && -d "${EXP_PARENT}/${CANDIDATE_SHORT}" ]]; then
    SHORT_EXP_DIR="${EXP_PARENT}/${CANDIDATE_SHORT}"
  fi
fi

if [[ -z "${LONG_EXP_DIR}" ]]; then
  CANDIDATE_LONG="${EXP_BASENAME/multi_scale/single_scale_long}"
  if [[ "${CANDIDATE_LONG}" != "${EXP_BASENAME}" && -d "${EXP_PARENT}/${CANDIDATE_LONG}" ]]; then
    LONG_EXP_DIR="${EXP_PARENT}/${CANDIDATE_LONG}"
  fi
fi

if [[ -z "${SHORT_EXP_DIR}" || -z "${LONG_EXP_DIR}" || ! -d "${MULTI_EXP_DIR}" ]]; then
  write_a4_compare_status "missing_inputs" "short_long_multi_experiment_dirs_are_not_all_available" "${SHORT_EXP_DIR}" "${LONG_EXP_DIR}" "${MULTI_EXP_DIR}"
  echo "[TEP Mechanism] A4 compare inputs are incomplete."
  echo "[TEP Mechanism] status file: ${A4_COMPARE_STATUS_PATH}"
  echo "[TEP Mechanism] hint: set TEP_A4_SHORT_EXP_DIR / TEP_A4_LONG_EXP_DIR / TEP_A4_MULTI_EXP_DIR if names are custom."
  if [[ "${RUN_A4_COMPARE_STRICT}" == "1" ]]; then
    echo "[TEP Mechanism] strict mode is enabled; aborting."
    exit 1
  fi
  echo "[TEP Mechanism] strict mode is disabled; skip A4 compare."
  exit 0
fi

echo "[TEP Mechanism] A4 compare short=${SHORT_EXP_DIR} long=${LONG_EXP_DIR} multi=${MULTI_EXP_DIR}"
write_a4_compare_status "running" "inputs_resolved" "${SHORT_EXP_DIR}" "${LONG_EXP_DIR}" "${MULTI_EXP_DIR}"

if [[ "${RUN_METRICS}" == "1" ]]; then
  "${PYTHON_EXE}" scripts/tep/compute_mechanism_metrics.py \
    --short_dir "${SHORT_EXP_DIR}" \
    --long_dir "${LONG_EXP_DIR}" \
    --multi_dir "${MULTI_EXP_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --smc_k "${SMC_K}" \
    --smr_k "${SMR_K}" \
    --fpr_quantile "${FPR_QUANTILE}"
fi

if [[ "${RUN_PLOTS}" == "1" ]]; then
  "${PYTHON_EXE}" scripts/tep/plot_mechanism_figures.py \
    --short_dir "${SHORT_EXP_DIR}" \
    --long_dir "${LONG_EXP_DIR}" \
    --multi_dir "${MULTI_EXP_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --embedding_method "${EMBEDDING_METHOD}" \
    --smr_k "${SMR_K}"
fi

write_a4_compare_status "completed" "a4_compare_finished" "${SHORT_EXP_DIR}" "${LONG_EXP_DIR}" "${MULTI_EXP_DIR}"
