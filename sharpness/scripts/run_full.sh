#!/usr/bin/env bash
# Stage 2 of the sharpness-vs-batch-size experiment: the 12 full-budget runs (vanilla SGD, Adafactor, Adam at B in
# 4096 256 16 1) at the hyperparameters in sharpness/runs/bracket/best.json (SGD / Adafactor: the bracketed
# peak_lr [+ t2]; Adam: the paper's tuned table via --hparams lm11m_fwedu_adam). train.py does the rest: TF32, the
# paper's budget cut to whole updates, snapshots + validation loss + the sharpness lower bounds at the 36 instants and the
# upper bounds at the four checkpoints (sharpness/sharpness.ipynb), logged to W&B group 'sharpness'.
# Output: sharpness/runs/<opt>_b<B>. Restartable (finished runs skipped, interrupted runs resume).
#   sharpness/scripts/run_bg.sh sharp-full bash sharpness/scripts/run_full.sh
set -euo pipefail
# JAX memory for this job: one preallocated pool of 90% of the GPU. devenv.nix sets XLA_PYTHON_CLIENT_PREALLOCATE=false, under which
# the pool grows in separate regions and every buffer must fit inside one of them; on 2026-10-17 the sgd B=1 run died at its first
# measurement because the ~10.5 GiB Hessian-vector-product workspace could not get a region under the default 75% limit.
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$REPO"
OUT=${OUT:-$REPO/sharpness/runs}
BEST=${BEST:-$OUT/bracket/best.json}
OPTS=${OPTS:-"sgd adafactor adamw"}; BATCH_SIZES=${BATCH_SIZES:-"4096 256 16 1"}
EXTRA=${EXTRA:-}
for opt in $OPTS; do for B in $BATCH_SIZES; do
  run=$OUT/${opt}_b$B
  hp=$(python -c "import json; d=json.load(open('$BEST'))['$opt']['$B']; print(' '.join(f'--{k.replace(\"_\", \"-\")} {v}' for k, v in d.items() if v is not None))")
  echo "[$(date -u +%FT%TZ)] $opt B=$B: $hp"
  python -m sharpness.scripts.train --optimizer "$opt" --batch-size "$B" $hp --out "$run" $EXTRA
done; done
echo "[$(date -u +%FT%TZ)] all runs done"
