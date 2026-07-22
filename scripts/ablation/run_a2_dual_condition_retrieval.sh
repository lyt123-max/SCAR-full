#!/usr/bin/env bash
set -euo pipefail

# Title: A2 ablation - dual-condition retrieval

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="dual_condition_retrieval"
ABLATION_TITLE="A2: dual-condition retrieval"
USE_TWO_LEVEL_RETRIEVAL=1
USE_CONTEXT_KEY_RETRIEVAL=1
BOOTSTRAP_STAGE_A="${BOOTSTRAP_STAGE_A:-1}"

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
