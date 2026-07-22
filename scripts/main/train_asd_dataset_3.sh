#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_train_detect_dataset.sh" "ASD_dataset_3" "asd_dataset_3_baseline" "$@"
