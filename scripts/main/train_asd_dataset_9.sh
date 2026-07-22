#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_train_detect_dataset.sh" "ASD_dataset_9" "asd_dataset_9_baseline" "$@"
