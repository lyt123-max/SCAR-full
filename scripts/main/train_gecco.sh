#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${VAL_SPLIT_MODE:=interleaved}"
: "${LR:=5e-4}"
: "${EARLY_STOP_PATIENCE:=6}"
: "${COMPLETION_DROPOUT:=0.25}"
: "${EVALUATION_SCORE_KEY:=cdf_mean}"
exec bash "${SCRIPT_DIR}/_train_detect_dataset.sh" "GECCO" "gecco_baseline" "$@"
