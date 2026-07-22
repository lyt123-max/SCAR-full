#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_train_detect_dataset.sh" "synthetic_tre0.0482" "synthetic_tre0.0482_baseline" "$@"
