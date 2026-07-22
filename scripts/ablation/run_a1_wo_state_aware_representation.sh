#!/usr/bin/env bash
set -euo pipefail

# Title: A1 ablation - w/o state-guided modulation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="wo_state_aware_representation"
ABLATION_TITLE="A1: w/o state-guided modulation"
USE_CHANNEL_MODULATION=0

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
