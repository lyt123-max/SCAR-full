#!/usr/bin/env bash
set -euo pipefail

# Title: A4 ablation - single-scale (long)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="single_scale_long"
ABLATION_TITLE="A4: single-scale (long)"
PATCH_SIZES="32"
EVALUATION_SCORE_KEY="cdf_mean"

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
