#!/usr/bin/env bash
# Stage 1 of the sharpness-vs-batch-size experiment: bracket the SGD and Adafactor learning rates (see bracket.py).
#   sharpness/scripts/run_bg.sh sharp-bracket bash sharpness/scripts/run_bracket.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$REPO"
python -m sharpness.scripts.bracket "$@"
