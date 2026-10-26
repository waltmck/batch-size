#!/usr/bin/env bash
# Register the devenv Python (.devenv/state/venv) as a Jupyter kernel named "batch-size".
# After this, the kernel shows up in the JupyterLab server already running on this box and in VS Code.
#
# Kernels started by JupyterLab do not go through `devenv shell`, so the env vars devenv.nix sets are
# baked into the kernel spec here. Re-run after recreating the venv or changing those variables.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
devenv shell -- bash -c '
  python -m ipykernel install --user --name batch-size --display-name "Python 3.12 (batch-size devenv)" \
    --env XLA_PYTHON_CLIENT_PREALLOCATE "$XLA_PYTHON_CLIENT_PREALLOCATE" \
    --env TQDM_MININTERVAL "$TQDM_MININTERVAL"
'
cat "$HOME/.local/share/jupyter/kernels/batch-size/kernel.json"
