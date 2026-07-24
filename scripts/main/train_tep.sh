#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

STAGE="${1:-full}"
EXP_BASE_NAME="${2:-tep_baseline}"
TIMESTAMP="${EXP_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
RESUME="${RESUME:-0}"
if [[ "${RESUME}" == "1" ]]; then
  EXP_NAME="${EXP_NAME:-${EXP_BASE_NAME}}"
else
  EXP_NAME="${EXP_NAME:-${EXP_BASE_NAME}_${TIMESTAMP}}"
fi
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-./TEP-DATA/TEP_Selected_Data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-./artifacts}"
TEP_PROTOCOL="${TEP_PROTOCOL:-selected}"

SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${BATCH_SIZE}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
MEMORY_BATCH_SIZE="${MEMORY_BATCH_SIZE:-${BATCH_SIZE}}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-${BATCH_SIZE}}"
TRAIN_STRIDE="${TRAIN_STRIDE:-1}"
TEST_STRIDE="${TEST_STRIDE:-1}"
MEMORY_BUILD_STRIDE="${MEMORY_BUILD_STRIDE:-1}"

VAL_RATIO="${VAL_RATIO:-0.15}"
VAL_GAP="${VAL_GAP:-1}"
VAL_MIN_TRAIN_WINDOWS="${VAL_MIN_TRAIN_WINDOWS:-16}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-tail}"

STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"
LR="${LR:-1e-3}"
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

NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-0}"
MAX_TEST_WINDOWS="${MAX_TEST_WINDOWS:-0}"
MAX_TEST_SEQUENCES="${MAX_TEST_SEQUENCES:-0}"
EVALUATION_SCORE_KEY="${EVALUATION_SCORE_KEY:-cdf_mean}"
SEED="${SEED:-42}"
MEMORY_AUDIT_MODE="${MEMORY_AUDIT_MODE:-full}"
RESOURCE_MONITOR="${RESOURCE_MONITOR:-1}"
SEQUENCE_SCORE_AGGREGATION="${SEQUENCE_SCORE_AGGREGATION:-p95}"

if [[ "${MAX_TEST_WINDOWS}" != "0" ]]; then
  echo "[TEP Script] overriding MAX_TEST_WINDOWS=${MAX_TEST_WINDOWS} -> 0 to preserve full sequence coverage."
  MAX_TEST_WINDOWS="0"
fi

mkdir -p "${ARTIFACT_ROOT}/${EXP_NAME}"
LOG_DIR="${ARTIFACT_ROOT}/${EXP_NAME}/logs"
mkdir -p "${LOG_DIR}"
RUN_LOG_DIR="${LOG_DIR}/${EXP_NAME}"
mkdir -p "${RUN_LOG_DIR}"
CHANNELS="$(DATA_ROOT="${DATA_ROOT}" TEP_PROTOCOL="${TEP_PROTOCOL}" python - <<'PY'
import os
from coremad.config import CoReMADConfig
from coremad.data import load_raw_dataset_bundle

config = CoReMADConfig(
    dataset="TEP",
    data_root=os.environ["DATA_ROOT"],
    tep_protocol=os.environ["TEP_PROTOCOL"],
)
bundle = load_raw_dataset_bundle("TEP", os.environ["DATA_ROOT"], config=config)
print(bundle.train.shape[1])
PY
)"

COMMON_ARGS=(
  --dataset TEP
  --data_root "${DATA_ROOT}"
  --tep_protocol "${TEP_PROTOCOL}"
  --artifact_root "${ARTIFACT_ROOT}"
  --experiment_name "${EXP_NAME}"
  --resume "${RESUME}"
  --memory_audit_mode "${MEMORY_AUDIT_MODE}"
  --resource_monitor "${RESOURCE_MONITOR}"
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
  --max_test_sequences "${MAX_TEST_SEQUENCES}"
  --evaluation_score_key "${EVALUATION_SCORE_KEY}"
  --sequence_score_aggregation "${SEQUENCE_SCORE_AGGREGATION}"
)

print_header() {
  local stage_name="$1"
  local log_file="$2"
  echo "[TEP Script] repo_root=${REPO_ROOT}"
  echo "[TEP Script] stage=${stage_name} exp_name=${EXP_NAME} device=${DEVICE}"
  echo "[TEP Script] data_root=${DATA_ROOT} artifact_root=${ARTIFACT_ROOT}"
  echo "[TEP Script] tep_protocol=${TEP_PROTOCOL}"
  echo "[TEP Script] resume=${RESUME} log_file=${log_file}"
  echo "[TEP Script] conda_prefix=${CONDA_PREFIX:-<unset>}"
  echo "[TEP Script] d_z=${D_Z} patch_sizes=${PATCH_SIZES} seq_len=${SEQ_LEN}"
  echo "[TEP Script] val_ratio=${VAL_RATIO} val_gap=${VAL_GAP} val_split_mode=${VAL_SPLIT_MODE}"
  echo "[TEP Script] max_train_windows=${MAX_TRAIN_WINDOWS} max_test_windows=${MAX_TEST_WINDOWS}"
  echo "[TEP Script] train_batch_size=${TRAIN_BATCH_SIZE} val_batch_size=${VAL_BATCH_SIZE}"
  echo "[TEP Script] memory_batch_size=${MEMORY_BATCH_SIZE} test_batch_size=${TEST_BATCH_SIZE}"
  echo "[TEP Script] evaluation_score_key=${EVALUATION_SCORE_KEY}"
  echo "[TEP Script] sequence_score_aggregation=${SEQUENCE_SCORE_AGGREGATION}"
  echo "[TEP Script] detected_n_channels=${CHANNELS}"
}

run_one_stage() {
  local stage_name="$1"
  local log_file="${RUN_LOG_DIR}/${EXP_NAME}_${stage_name}.log"
  {
    print_header "${stage_name}" "${log_file}"
    python run.py --stage "${stage_name}" "${COMMON_ARGS[@]}"
  } 2>&1 | tee -a "${log_file}"
}

if [[ "${STAGE}" == "full" ]]; then
  run_one_stage "stage_a"
  run_one_stage "stage_b"
  run_one_stage "test"
else
  run_one_stage "${STAGE}"
fi
