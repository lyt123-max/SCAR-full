#!/usr/bin/env bash
set -euo pipefail

# Title: A3 ablation matrix (PIT fusion, fusion variants)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${1:-auto}"
DATASETS_STR="${DATASETS:-MSL SMAP PSM SWAT SMD TEP}"
ABLATIONS_STR="${ABLATIONS:-pit_fusion raw_max zscore_mean}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
MATRIX_RESUME="${RESUME:-0}"
PYTHON_EXE="${PYTHON_EXE:-python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-./artifacts}"
COLLECT_RESULTS="${COLLECT_RESULTS:-1}"

if [[ "${MATRIX_RESUME}" == "1" ]]; then
  DEFAULT_EXPERIMENT_TEMPLATE="{dataset_lower}_ablation_{ablation}"
  DEFAULT_SUMMARY_OUTPUT_PREFIX="${ARTIFACT_ROOT}/a3_ablation_summary"
else
  DEFAULT_EXPERIMENT_TEMPLATE="{dataset_lower}_ablation_{ablation}_${RUN_TIMESTAMP}"
  DEFAULT_SUMMARY_OUTPUT_PREFIX="${ARTIFACT_ROOT}/a3_ablation_summary_${RUN_TIMESTAMP}"
fi

SUMMARY_OUTPUT_PREFIX="${SUMMARY_OUTPUT_PREFIX:-${DEFAULT_SUMMARY_OUTPUT_PREFIX}}"
SUMMARY_MARKDOWN_SCORE_KEY="${SUMMARY_MARKDOWN_SCORE_KEY:-cdf_mean}"
SUMMARY_METRIC_KEYS_STR="${SUMMARY_METRIC_KEYS:-}"
SUMMARY_MARKDOWN_METRIC_KEYS_STR="${SUMMARY_MARKDOWN_METRIC_KEYS:-}"

read -r -a DATASETS <<< "${DATASETS_STR}"
read -r -a ABLATIONS <<< "${ABLATIONS_STR}"
read -r -a SUMMARY_METRIC_KEYS <<< "${SUMMARY_METRIC_KEYS_STR}"
read -r -a SUMMARY_MARKDOWN_METRIC_KEYS <<< "${SUMMARY_MARKDOWN_METRIC_KEYS_STR}"

for dataset in "${DATASETS[@]}"; do
  for ablation in "${ABLATIONS[@]}"; do
    echo "[A3Matrix] dataset=${dataset} ablation=${ablation} stage=${STAGE}"
    EXP_TIMESTAMP="${RUN_TIMESTAMP}" RESUME="${MATRIX_RESUME}" bash "${SCRIPT_DIR}/run_dataset_ablation.sh" "${dataset}" "${ablation}" "${STAGE}"
  done
done

if [[ "${COLLECT_RESULTS}" == "1" ]]; then
  SUMMARY_CMD=(
    "${PYTHON_EXE}"
    "${SCRIPT_DIR}/collect_results.py"
    --artifact_root "${ARTIFACT_ROOT}"
    --datasets "${DATASETS[@]}"
    --ablations "${ABLATIONS[@]}"
    --experiment_name_template "${DEFAULT_EXPERIMENT_TEMPLATE}"
    --output_prefix "${SUMMARY_OUTPUT_PREFIX}"
    --markdown_score_key "${SUMMARY_MARKDOWN_SCORE_KEY}"
  )

  if [[ ${#SUMMARY_METRIC_KEYS[@]} -gt 0 ]]; then
    SUMMARY_CMD+=(--metric_keys "${SUMMARY_METRIC_KEYS[@]}")
  fi
  if [[ ${#SUMMARY_MARKDOWN_METRIC_KEYS[@]} -gt 0 ]]; then
    SUMMARY_CMD+=(--markdown_metric_keys "${SUMMARY_MARKDOWN_METRIC_KEYS[@]}")
  fi

  echo "[A3Matrix] collecting summary results"
  "${SUMMARY_CMD[@]}"
fi
