#!/usr/bin/env bash
set -euo pipefail

# Title: Full baseline for ablation comparisons

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_KEY="full"
ABLATION_TITLE="Full baseline"

source "${SCRIPT_DIR}/run_ablation_common.sh" "$@"
