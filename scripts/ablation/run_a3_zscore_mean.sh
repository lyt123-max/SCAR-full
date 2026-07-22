#!/usr/bin/env bash
set -euo pipefail

# Title: A3 ablation - zscore-mean fusion

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="zscore_mean"
ABLATION_TITLE="A3: zscore-mean fusion"
EVALUATION_SCORE_KEY="zscore_mean"
BOOTSTRAP_STAGE_A="${BOOTSTRAP_STAGE_A:-1}"
BOOTSTRAP_STAGE_B="${BOOTSTRAP_STAGE_B:-1}"

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
