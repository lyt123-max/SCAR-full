#!/usr/bin/env bash
set -euo pipefail

# Title: A3 ablation - raw-max fusion

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="raw_max"
ABLATION_TITLE="A3: raw-max fusion"
EVALUATION_SCORE_KEY="raw_max"
BOOTSTRAP_STAGE_A="${BOOTSTRAP_STAGE_A:-1}"
BOOTSTRAP_STAGE_B="${BOOTSTRAP_STAGE_B:-1}"

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
