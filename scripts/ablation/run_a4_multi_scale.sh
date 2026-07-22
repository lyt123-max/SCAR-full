#!/usr/bin/env bash
set -euo pipefail

# Title: A4 ablation - multi-scale

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="multi_scale"
ABLATION_TITLE="A4: multi-scale"
PATCH_SIZES="8 32"
EVALUATION_SCORE_KEY="cdf_mean"

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
