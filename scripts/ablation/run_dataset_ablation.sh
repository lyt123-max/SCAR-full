#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:?Usage: run_dataset_ablation.sh <MSL|SMAP|PSM|SWAT|SMD|TEP> <ablation> [stage] [exp_base_name]}"
ABLATION="${2:?Usage: run_dataset_ablation.sh <MSL|SMAP|PSM|SWAT|SMD|TEP> <ablation> [stage] [exp_base_name]}"
REQUESTED_STAGE="${3:-auto}"
EXP_BASE_NAME="${4:-}"

run_target() {
  local script_name="$1"
  if [[ -n "${EXP_BASE_NAME}" ]]; then
    exec bash "${SCRIPT_DIR}/${script_name}" "${DATASET}" "${REQUESTED_STAGE}" "${EXP_BASE_NAME}"
  fi
  exec bash "${SCRIPT_DIR}/${script_name}" "${DATASET}" "${REQUESTED_STAGE}"
}

case "${ABLATION}" in
  full)
    run_target "run_full_ablation.sh"
    ;;
  wo_decomposition)
    run_target "run_a1_wo_decomposition.sh"
    ;;
  wo_state_aware_representation)
    run_target "run_a1_wo_state_aware_representation.sh"
    ;;
  global_retrieval)
    run_target "run_a2_global_retrieval.sh"
    ;;
  state_only_retrieval)
    run_target "run_a2_state_only_retrieval.sh"
    ;;
  context_only_retrieval)
    run_target "run_a2_context_only_retrieval.sh"
    ;;
  dual_condition_retrieval)
    run_target "run_a2_dual_condition_retrieval.sh"
    ;;
  pit_fusion)
    run_target "run_a3_pit_fusion.sh"
    ;;
  raw_max)
    run_target "run_a3_raw_max.sh"
    ;;
  zscore_mean)
    run_target "run_a3_zscore_mean.sh"
    ;;
  single_scale_short)
    run_target "run_a4_single_scale_short.sh"
    ;;
  single_scale_long)
    run_target "run_a4_single_scale_long.sh"
    ;;
  multi_scale)
    run_target "run_a4_multi_scale.sh"
    ;;
  *)
    echo "[Ablation] unsupported ablation=${ABLATION}"
    exit 1
    ;;
esac
