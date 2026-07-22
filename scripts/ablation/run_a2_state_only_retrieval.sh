#!/usr/bin/env bash
set -euo pipefail

# Title: A2 ablation - state-only retrieval

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="state_only_retrieval"
ABLATION_TITLE="A2: state-only retrieval"
USE_TWO_LEVEL_RETRIEVAL=1
USE_CONTEXT_KEY_RETRIEVAL=0
BOOTSTRAP_STAGE_A="${BOOTSTRAP_STAGE_A:-1}"

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
