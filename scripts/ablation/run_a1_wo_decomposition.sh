#!/usr/bin/env bash
set -euo pipefail

# Title: A1 ablation - w/o decomposition

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="wo_decomposition"
ABLATION_TITLE="A1: w/o decomposition"
USE_STSD_DECOMPOSITION=0
LAMBDA_SMOOTH=0.0

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
