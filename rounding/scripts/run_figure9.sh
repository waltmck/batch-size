#!/usr/bin/env bash
# The rounding experiment (the paper's Figure 9 with the corrected Adafactor): 12 runs = B in 16 256 4096 1 (the sharpness
# bracket's Adafactor hyperparameters, sharpness/runs/bracket/best.json) x weight precision fp32 / bf16 (nearest) / bf16sr
# (stochastic), at the paper's full budget. See figure9.py. Output: rounding/runs/figure9/. Restartable.
#   rounding/scripts/run_bg.sh rounding bash rounding/scripts/run_figure9.sh
set -euo pipefail
# JAX memory: one preallocated pool of 90% of the GPU (devenv.nix sets XLA_PYTHON_CLIENT_PREALLOCATE=false, see sharpness/scripts/run_full.sh).
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$REPO"
python -m rounding.scripts.figure9 "$@"
