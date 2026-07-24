#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SCAR_PYTHON="${SCAR_PYTHON:?set SCAR_PYTHON to the scar-paano environment Python}"
DEVICE="${DEVICE:-cuda:0}"
SMOKE_ROOT="${SMOKE_ROOT:-./artifacts/smoke}"
ANOMALY_DATA_ROOT="${ANOMALY_DATA_ROOT:-./dataset/anomaly_detect}"
TSB_M_DATA_ROOT="${TSB_M_DATA_ROOT:-./dataset/TSB-AD-M}"
TSB_U_DATA_ROOT="${TSB_U_DATA_ROOT:-./dataset/TSB-AD-U}"
TEP_DATA_ROOT="${TEP_DATA_ROOT:-./Multi-mode-Fault-Diagnosis-Datasets-with-TE-process}"

"${SCAR_PYTHON}" scripts/experiments/freeze_conda_env.py \
  --output-dir "${SMOKE_ROOT}/environment_locks/SCAR"

"${SCAR_PYTHON}" scripts/experiments/preflight.py \
  --artifact-root "${SMOKE_ROOT}" \
  --output "${SMOKE_ROOT}/preflight.json" \
  --strict-runtime

for dataset in MSL PSM SMAP SMD SWAT; do
  lower="$(printf '%s' "${dataset}" | tr '[:upper:]' '[:lower:]')"
  "${SCAR_PYTHON}" run.py \
    --stage full --dataset "${dataset}" \
    --data_root "${ANOMALY_DATA_ROOT}" \
    --artifact_root "${SMOKE_ROOT}" \
    --experiment_name "smoke_scar_${lower}" \
    --seed 42 --stage_a_epochs 1 --early_stop_patience 1 \
    --max_train_windows 256 --max_test_windows 128 \
    --memory_audit_mode full --resource_monitor 1 \
    --device "${DEVICE}"
done

"${SCAR_PYTHON}" scripts/tsb_ad/run_benchmark.py \
  --edition M --split eval --limit 1 --seed 42 \
  --data-root "${TSB_M_DATA_ROOT}" \
  --stage_a_epochs 1 --max_train_windows 256 --max_test_windows 128 \
  --artifact-root "${SMOKE_ROOT}/tsb_ad" --device "${DEVICE}"
"${SCAR_PYTHON}" scripts/tsb_ad/run_benchmark.py \
  --edition U --split eval --limit 1 --seed 42 \
  --data-root "${TSB_U_DATA_ROOT}" \
  --stage_a_epochs 1 --max_train_windows 256 --max_test_windows 128 \
  --artifact-root "${SMOKE_ROOT}/tsb_ad" --device "${DEVICE}"

STAGE_A_EPOCHS=1 EARLY_STOP_PATIENCE=1 MAX_TRAIN_WINDOWS=256 \
MAX_TEST_SEQUENCES=2 \
FORMAL_REBUTTAL=0 TEP_MECHANISM_POSTPROCESS=0 ARTIFACT_ROOT="${SMOKE_ROOT}" \
EXP_NAME=smoke_tep DEVICE="${DEVICE}" \
bash scripts/main/train_tep_full.sh full smoke_tep smoke_tep "${TEP_DATA_ROOT}"

for method in PaAno MEMTO PUAD PGRF-Net; do
  variable="$(printf '%s' "${method}" | tr '[:lower:]-' '[:upper:]_')_PYTHON"
  baseline_python="${!variable:?set ${variable} to the method environment Python}"
  "${baseline_python}" scripts/experiments/freeze_conda_env.py \
    --output-dir "${SMOKE_ROOT}/environment_locks/${method}"
  "${baseline_python}" scripts/rebuttal/baselines/run_baseline.py \
    --method "${method}" --dataset MSL --seed 42 --device "${DEVICE}" \
    --scar-python "${SCAR_PYTHON}" --baseline-python "${baseline_python}" \
    --source-experiment "${SMOKE_ROOT}/smoke_scar_msl" \
    --output-dir "${SMOKE_ROOT}/baseline_${method}"
done

"${SCAR_PYTHON}" -m unittest discover -s tests -p 'test_*.py'
echo "[remote_smoke] all smoke gates passed"
