#!/usr/bin/env bash
set -euo pipefail

# Title: A3 ablation - CDF/PIT mean fusion

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="pit_fusion"
ABLATION_TITLE="A3: CDF/PIT mean fusion"
EVALUATION_SCORE_KEY="cdf_mean"
BOOTSTRAP_STAGE_A="${BOOTSTRAP_STAGE_A:-1}"
BOOTSTRAP_STAGE_B="${BOOTSTRAP_STAGE_B:-1}"

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
