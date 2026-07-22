#!/usr/bin/env bash
set -euo pipefail

# Title: A2 ablation - global retrieval

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="global_retrieval"
ABLATION_TITLE="A2: global retrieval"
USE_TWO_LEVEL_RETRIEVAL=0
USE_CONTEXT_KEY_RETRIEVAL=0
BOOTSTRAP_STAGE_A="${BOOTSTRAP_STAGE_A:-1}"

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
