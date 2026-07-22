#!/usr/bin/env bash
set -euo pipefail

# Title: A4 ablation - single-scale (short)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="single_scale_short"
ABLATION_TITLE="A4: single-scale (short)"
PATCH_SIZES="8"
EVALUATION_SCORE_KEY="cdf_mean"

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
