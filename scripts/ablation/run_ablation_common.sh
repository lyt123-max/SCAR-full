#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_EXE="${PYTHON_EXE:-python}"

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ -z "${ABLATION_KEY:-}" ]]; then
  echo "[Ablation] ABLATION_KEY is required."
  exit 1
fi

DATASET="${1:?Usage: <script> <MSL|SMAP|PSM|SWAT|SMD|TEP> [stage] [exp_base_name]}"
REQUESTED_STAGE="${2:-auto}"
DEFAULT_EXP_BASE_NAME="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')_ablation_${ABLATION_KEY}"
EXP_BASE_NAME="${3:-${DEFAULT_EXP_BASE_NAME}}"
DATASET_LOWER="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"

case "${DATASET}" in
  MSL|SMAP|PSM|SWAT|SMD|TEP)
    ;;
  *)
    echo "[Ablation] unsupported dataset=${DATASET}. Use MSL, SMAP, PSM, SWAT, SMD, or TEP."
    exit 1
    ;;
esac

TIMESTAMP="${EXP_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
RESUME="${RESUME:-0}"
if [[ "${RESUME}" == "1" ]]; then
  EXP_NAME="${EXP_NAME:-${EXP_BASE_NAME}}"
else
  EXP_NAME="${EXP_NAME:-${EXP_BASE_NAME}_${TIMESTAMP}}"
fi
RUN_RESUME="${RUN_RESUME:-${RESUME}}"

DEVICE="${DEVICE:-cuda}"
DEFAULT_DATA_ROOT="./dataset/anomaly_detect"
if [[ "${DATASET}" == "TEP" ]]; then
  DEFAULT_DATA_ROOT="./TEP-DATA/TEP_Selected_Data"
fi
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-./artifacts}"

SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${BATCH_SIZE}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
MEMORY_BATCH_SIZE="${MEMORY_BATCH_SIZE:-${BATCH_SIZE}}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-${BATCH_SIZE}}"
TRAIN_STRIDE="${TRAIN_STRIDE:-1}"
TEST_STRIDE="${TEST_STRIDE:-1}"
DEFAULT_MEMORY_BUILD_STRIDE="1"

VAL_RATIO="${VAL_RATIO:-0.15}"
VAL_GAP="${VAL_GAP:-1}"
DEFAULT_VAL_MIN_TRAIN_WINDOWS="16"
DEFAULT_VAL_SPLIT_MODE="tail"

STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-100}"
DEFAULT_EARLY_STOP_PATIENCE="15"
DEFAULT_LR="2e-3"
DEFAULT_COMPLETION_DROPOUT="0.1"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
SCHEDULER_ETA_MIN_RATIO="${SCHEDULER_ETA_MIN_RATIO:-0.01}"
MASK_RATIO="${MASK_RATIO:-0.25}"
N_MASK_GROUPS="${N_MASK_GROUPS:-4}"
LAMBDA_PRED="${LAMBDA_PRED:-0.5}"
LAMBDA_SMOOTH="${LAMBDA_SMOOTH:-0.01}"

PATCH_SIZES="${PATCH_SIZES:-8 32}"
D_Z="${D_Z:-128}"
TOP_M="${TOP_M:-50}"
TOP_K="${TOP_K:-20}"
KNN_K="${KNN_K:-5}"
CLEAN_RATIO="${CLEAN_RATIO:-0.02}"

USE_FAISS="${USE_FAISS:-1}"
FAISS_USE_GPU="${FAISS_USE_GPU:-1}"
FAISS_EXACT_THRESHOLD="${FAISS_EXACT_THRESHOLD:-100000}"
FAISS_IVF_NPROBE="${FAISS_IVF_NPROBE:-16}"

CORESET_KEEP_RATIO="${CORESET_KEEP_RATIO:-1.0}"
CORESET_MAX_PATCHES_PER_SCALE="${CORESET_MAX_PATCHES_PER_SCALE:-200000}"
CORESET_FPS_THRESHOLD="${CORESET_FPS_THRESHOLD:-100000}"

NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-0}"
MAX_TEST_WINDOWS="${MAX_TEST_WINDOWS:-0}"
SEED="${SEED:-42}"
SEQUENCE_SCORE_AGGREGATION="${SEQUENCE_SCORE_AGGREGATION:-p95}"

if [[ "${DATASET}" == "TEP" && "${MAX_TEST_WINDOWS}" != "0" ]]; then
  echo "[Ablation] dataset=TEP overriding MAX_TEST_WINDOWS=${MAX_TEST_WINDOWS} -> 0 to preserve full sequence coverage."
  MAX_TEST_WINDOWS="0"
fi

USE_STSD_DECOMPOSITION="${USE_STSD_DECOMPOSITION:-1}"
USE_CHANNEL_MODULATION="${USE_CHANNEL_MODULATION:-1}"
USE_TWO_LEVEL_RETRIEVAL="${USE_TWO_LEVEL_RETRIEVAL:-1}"
USE_CONTEXT_KEY_RETRIEVAL="${USE_CONTEXT_KEY_RETRIEVAL:-1}"
USE_COMPLETION_HEAD="${USE_COMPLETION_HEAD:-1}"
USE_COMPLETION_SELF_CLEANING="${USE_COMPLETION_SELF_CLEANING:-1}"
USE_COMPLETION_SCORE_FUSION="${USE_COMPLETION_SCORE_FUSION:-1}"
USE_PROTOTYPE_SUPPORT="${USE_PROTOTYPE_SUPPORT:-0}"
EVALUATION_SCORE_KEY="${EVALUATION_SCORE_KEY:-cdf_mean}"
ABLATION_TITLE="${ABLATION_TITLE:-${ABLATION_KEY}}"
RECOMMENDED_PIPELINE="${RECOMMENDED_PIPELINE:-full}"
BOOTSTRAP_STAGE_A="${BOOTSTRAP_STAGE_A:-0}"
BOOTSTRAP_STAGE_B="${BOOTSTRAP_STAGE_B:-0}"
DEFAULT_PARENT_EXPERIMENT="${DATASET_LOWER}_ablation_full"
if [[ "${RESUME}" != "1" ]]; then
  DEFAULT_PARENT_EXPERIMENT="${DEFAULT_PARENT_EXPERIMENT}_${TIMESTAMP}"
fi
PARENT_EXPERIMENT="${PARENT_EXPERIMENT:-${DEFAULT_PARENT_EXPERIMENT}}"
PARENT_ARTIFACT_ROOT="${PARENT_ARTIFACT_ROOT:-${ARTIFACT_ROOT}}"

if [[ "${PARENT_ARTIFACT_ROOT}" == "${ARTIFACT_ROOT}" ]]; then
  case "${ABLATION_KEY}" in
    global_retrieval|state_only_retrieval|context_only_retrieval|dual_condition_retrieval|pit_fusion|raw_max|zscore_mean)
      artifact_leaf="$(basename "${ARTIFACT_ROOT}")"
      artifact_parent="$(dirname "${ARTIFACT_ROOT}")"
      if [[ "${artifact_leaf}" == "A2" || "${artifact_leaf}" == "A3" ]]; then
        inferred_parent_artifact_root="${artifact_parent}/A1"
        if [[ -d "${inferred_parent_artifact_root}" ]]; then
          PARENT_ARTIFACT_ROOT="${inferred_parent_artifact_root}"
        fi
      fi
      ;;
  esac
fi

if [[ "${BOOTSTRAP_STAGE_A}" == "1" || "${BOOTSTRAP_STAGE_B}" == "1" ]]; then
  if [[ "${RUN_RESUME}" != "1" ]]; then
    echo "[Ablation] enabling RUN_RESUME=1 because bootstrapping is active."
    RUN_RESUME="1"
  fi
fi

case "${DATASET}" in
  PSM)
    DEFAULT_VAL_SPLIT_MODE="interleaved"
    DEFAULT_LR="5e-4"
    ;;
  SWAT)
    DEFAULT_MEMORY_BUILD_STRIDE="2"
    DEFAULT_VAL_MIN_TRAIN_WINDOWS="50"
    DEFAULT_EARLY_STOP_PATIENCE="10"
    DEFAULT_LR="1e-3"
    DEFAULT_COMPLETION_DROPOUT="0.2"
    ;;
  SMD)
    DEFAULT_MEMORY_BUILD_STRIDE="4"
    DEFAULT_VAL_SPLIT_MODE="interleaved"
    DEFAULT_EARLY_STOP_PATIENCE="6"
    DEFAULT_LR="1e-3"
    DEFAULT_VAL_MIN_TRAIN_WINDOWS="50"
    ;;
  TEP)
    DEFAULT_LR="1e-3"
    ;;
esac

MEMORY_BUILD_STRIDE="${MEMORY_BUILD_STRIDE:-${DEFAULT_MEMORY_BUILD_STRIDE}}"
VAL_MIN_TRAIN_WINDOWS="${VAL_MIN_TRAIN_WINDOWS:-${DEFAULT_VAL_MIN_TRAIN_WINDOWS}}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-${DEFAULT_VAL_SPLIT_MODE}}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-${DEFAULT_EARLY_STOP_PATIENCE}}"
LR="${LR:-${DEFAULT_LR}}"
COMPLETION_DROPOUT="${COMPLETION_DROPOUT:-${DEFAULT_COMPLETION_DROPOUT}}"

case "${REQUESTED_STAGE}" in
  auto)
    EFFECTIVE_STAGE="${RECOMMENDED_PIPELINE}"
    ;;
  full|stage_a|stage_b|test|stage_b_test)
    EFFECTIVE_STAGE="${REQUESTED_STAGE}"
    ;;
  *)
    echo "[Ablation] unsupported stage=${REQUESTED_STAGE}"
    exit 1
    ;;
esac

mkdir -p "${ARTIFACT_ROOT}/${EXP_NAME}"
LOG_DIR="${ARTIFACT_ROOT}/${EXP_NAME}/logs"
mkdir -p "${LOG_DIR}"
RUN_LOG_DIR="${LOG_DIR}/${EXP_NAME}"
mkdir -p "${RUN_LOG_DIR}"

CHANNELS="$(DATA_ROOT="${DATA_ROOT}" DATASET="${DATASET}" "${PYTHON_EXE}" - <<'PY'
import os
from coremad.data import load_raw_dataset_bundle

bundle = load_raw_dataset_bundle(os.environ["DATASET"], os.environ["DATA_ROOT"])
print(bundle.train.shape[1])
PY
)"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --data_root "${DATA_ROOT}"
  --artifact_root "${ARTIFACT_ROOT}"
  --experiment_name "${EXP_NAME}"
  --resume "${RUN_RESUME}"
  --seq_len "${SEQ_LEN}"
  --batch_size "${BATCH_SIZE}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --val_batch_size "${VAL_BATCH_SIZE}"
  --memory_batch_size "${MEMORY_BATCH_SIZE}"
  --test_batch_size "${TEST_BATCH_SIZE}"
  --train_stride "${TRAIN_STRIDE}"
  --test_stride "${TEST_STRIDE}"
  --memory_build_stride "${MEMORY_BUILD_STRIDE}"
  --val_ratio "${VAL_RATIO}"
  --val_gap "${VAL_GAP}"
  --val_min_train_windows "${VAL_MIN_TRAIN_WINDOWS}"
  --val_split_mode "${VAL_SPLIT_MODE}"
  --stage_a_epochs "${STAGE_A_EPOCHS}"
  --early_stop_patience "${EARLY_STOP_PATIENCE}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip_norm "${GRAD_CLIP_NORM}"
  --scheduler_eta_min_ratio "${SCHEDULER_ETA_MIN_RATIO}"
  --mask_ratio "${MASK_RATIO}"
  --n_mask_groups "${N_MASK_GROUPS}"
  --completion_dropout "${COMPLETION_DROPOUT}"
  --lambda_pred "${LAMBDA_PRED}"
  --lambda_smooth "${LAMBDA_SMOOTH}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --patch_sizes ${PATCH_SIZES}
  --d_z "${D_Z}"
  --top_M "${TOP_M}"
  --top_K "${TOP_K}"
  --knn_k "${KNN_K}"
  --clean_ratio "${CLEAN_RATIO}"
  --use_faiss "${USE_FAISS}"
  --faiss_use_gpu "${FAISS_USE_GPU}"
  --faiss_exact_threshold "${FAISS_EXACT_THRESHOLD}"
  --faiss_ivf_nprobe "${FAISS_IVF_NPROBE}"
  --coreset_keep_ratio "${CORESET_KEEP_RATIO}"
  --coreset_max_patches_per_scale "${CORESET_MAX_PATCHES_PER_SCALE}"
  --coreset_fps_threshold "${CORESET_FPS_THRESHOLD}"
  --num_workers "${NUM_WORKERS}"
  --max_train_windows "${MAX_TRAIN_WINDOWS}"
  --max_test_windows "${MAX_TEST_WINDOWS}"
  --sequence_score_aggregation "${SEQUENCE_SCORE_AGGREGATION}"
  --use_stsd_decomposition "${USE_STSD_DECOMPOSITION}"
  --use_channel_modulation "${USE_CHANNEL_MODULATION}"
  --use_two_level_retrieval "${USE_TWO_LEVEL_RETRIEVAL}"
  --use_context_key_retrieval "${USE_CONTEXT_KEY_RETRIEVAL}"
  --use_completion_head "${USE_COMPLETION_HEAD}"
  --use_completion_self_cleaning "${USE_COMPLETION_SELF_CLEANING}"
  --use_completion_score_fusion "${USE_COMPLETION_SCORE_FUSION}"
  --use_prototype_support "${USE_PROTOTYPE_SUPPORT}"
  --evaluation_score_key "${EVALUATION_SCORE_KEY}"
)

bootstrap_stage_a_if_needed() {
  if [[ "${BOOTSTRAP_STAGE_A}" != "1" && "${BOOTSTRAP_STAGE_B}" != "1" ]]; then
    return
  fi
  if [[ -f "${ARTIFACT_ROOT}/${EXP_NAME}/stage_a.pt" ]]; then
    return
  fi
  if [[ "${PARENT_EXPERIMENT}" == "${EXP_NAME}" ]]; then
    echo "[Ablation] skip Stage-A bootstrap: parent_experiment equals exp_name=${EXP_NAME}"
    return
  fi
  "${PYTHON_EXE}" "${SCRIPT_DIR}/bootstrap_stage_a.py" \
    --artifact_root "${ARTIFACT_ROOT}" \
    --source_artifact_root "${PARENT_ARTIFACT_ROOT}" \
    --source_experiment "${PARENT_EXPERIMENT}" \
    --target_experiment "${EXP_NAME}"
}

bootstrap_stage_b_if_needed() {
  if [[ "${BOOTSTRAP_STAGE_B}" != "1" ]]; then
    return
  fi
  if [[ -f "${ARTIFACT_ROOT}/${EXP_NAME}/memory.pt" \
        && -f "${ARTIFACT_ROOT}/${EXP_NAME}/memory_meta.json" \
        && -f "${ARTIFACT_ROOT}/${EXP_NAME}/cdf_fusion.npz" \
        && -f "${ARTIFACT_ROOT}/${EXP_NAME}/cdf_fusion.json" \
        && -f "${ARTIFACT_ROOT}/${EXP_NAME}/zscore_fusion.json" ]]; then
    return
  fi
  if [[ "${PARENT_EXPERIMENT}" == "${EXP_NAME}" ]]; then
    echo "[Ablation] skip Stage-B bootstrap: parent_experiment equals exp_name=${EXP_NAME}"
    return
  fi
  bootstrap_stage_a_if_needed
  "${PYTHON_EXE}" "${SCRIPT_DIR}/bootstrap_stage_b.py" \
    --artifact_root "${ARTIFACT_ROOT}" \
    --source_artifact_root "${PARENT_ARTIFACT_ROOT}" \
    --source_experiment "${PARENT_EXPERIMENT}" \
    --target_experiment "${EXP_NAME}"
}

prepare_bootstrap_artifacts() {
  case "${EFFECTIVE_STAGE}" in
    full)
      bootstrap_stage_a_if_needed
      bootstrap_stage_b_if_needed
      ;;
    stage_a)
      bootstrap_stage_a_if_needed
      ;;
    stage_b|stage_b_test)
      bootstrap_stage_a_if_needed
      bootstrap_stage_b_if_needed
      ;;
    test)
      bootstrap_stage_a_if_needed
      bootstrap_stage_b_if_needed
      ;;
  esac
}

print_header() {
  local stage_name="$1"
  local log_file="$2"
  echo "[Ablation] dataset=${DATASET} ablation=${ABLATION_KEY} title=${ABLATION_TITLE}"
  echo "[Ablation] stage=${stage_name} effective_stage=${EFFECTIVE_STAGE} exp_name=${EXP_NAME}"
  echo "[Ablation] data_root=${DATA_ROOT} artifact_root=${ARTIFACT_ROOT}"
  echo "[Ablation] resume=${RESUME} run_resume=${RUN_RESUME} log_file=${log_file}"
  echo "[Ablation] conda_prefix=${CONDA_PREFIX:-<unset>}"
  echo "[Ablation] python_exe=${PYTHON_EXE}"
  echo "[Ablation] parent_experiment=${PARENT_EXPERIMENT}"
  echo "[Ablation] parent_artifact_root=${PARENT_ARTIFACT_ROOT}"
  echo "[Ablation] bootstrap_stage_a=${BOOTSTRAP_STAGE_A} bootstrap_stage_b=${BOOTSTRAP_STAGE_B}"
  echo "[Ablation] d_z=${D_Z} patch_sizes=${PATCH_SIZES} lambda_pred=${LAMBDA_PRED} lambda_smooth=${LAMBDA_SMOOTH}"
  echo "[Ablation] completion_dropout=${COMPLETION_DROPOUT}"
  echo "[Ablation] use_stsd=${USE_STSD_DECOMPOSITION} use_channel_mod=${USE_CHANNEL_MODULATION} use_two_level_retrieval=${USE_TWO_LEVEL_RETRIEVAL} use_context_key_retrieval=${USE_CONTEXT_KEY_RETRIEVAL}"
  echo "[Ablation] use_completion_head=${USE_COMPLETION_HEAD} use_completion_self_cleaning=${USE_COMPLETION_SELF_CLEANING} use_completion_score_fusion=${USE_COMPLETION_SCORE_FUSION}"
  echo "[Ablation] use_prototype_support=${USE_PROTOTYPE_SUPPORT}"
  echo "[Ablation] max_train_windows=${MAX_TRAIN_WINDOWS} max_test_windows=${MAX_TEST_WINDOWS}"
  echo "[Ablation] evaluation_score_key=${EVALUATION_SCORE_KEY}"
  echo "[Ablation] sequence_score_aggregation=${SEQUENCE_SCORE_AGGREGATION}"
  echo "[Ablation] detected_n_channels=${CHANNELS}"
}

run_one_stage() {
  local stage_name="$1"
  local log_file="${RUN_LOG_DIR}/${EXP_NAME}_${stage_name}.log"
  {
    print_header "${stage_name}" "${log_file}"
    "${PYTHON_EXE}" run.py --stage "${stage_name}" "${COMMON_ARGS[@]}"
  } 2>&1 | tee -a "${log_file}"
}

maybe_run_tep_mechanism_postprocess() {
  if [[ "${DATASET}" != "TEP" ]]; then
    return
  fi
  if [[ "${TEP_MECHANISM_POSTPROCESS:-1}" != "1" ]]; then
    echo "[Ablation] skip TEP mechanism postprocess: TEP_MECHANISM_POSTPROCESS=${TEP_MECHANISM_POSTPROCESS:-1}"
    return
  fi

  local post_log="${RUN_LOG_DIR}/${EXP_NAME}_tep_mechanism.log"
  {
    echo "[Ablation] starting TEP mechanism postprocess for exp_name=${EXP_NAME}"
    PYTHON_EXE="${PYTHON_EXE}" bash "${REPO_ROOT}/scripts/tep/run_mechanism_pipeline.sh" "${ARTIFACT_ROOT}/${EXP_NAME}" "${ABLATION_KEY}"
  } 2>&1 | tee -a "${post_log}"
}

prepare_bootstrap_artifacts

case "${EFFECTIVE_STAGE}" in
  full)
    run_one_stage "stage_a"
    run_one_stage "stage_b"
    run_one_stage "test"
    maybe_run_tep_mechanism_postprocess
    ;;
  stage_a|stage_b|test)
    run_one_stage "${EFFECTIVE_STAGE}"
    if [[ "${EFFECTIVE_STAGE}" == "test" ]]; then
      maybe_run_tep_mechanism_postprocess
    fi
    ;;
  stage_b_test)
    run_one_stage "stage_b"
    run_one_stage "test"
    maybe_run_tep_mechanism_postprocess
    ;;
esac
