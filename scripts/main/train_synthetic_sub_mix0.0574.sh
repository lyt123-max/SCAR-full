#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_train_detect_dataset.sh" "synthetic_sub_mix0.0574" "synthetic_sub_mix0.0574_baseline" "$@"
