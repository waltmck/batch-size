#!/usr/bin/env bash
# Run a command as a detached background job that keeps running after you disconnect from SSH.
#
# Uses a transient systemd *user* unit (lingering is enabled for this user, so units outlive logins).
# The command runs inside `devenv shell`, so it gets the project's Python env without you being in it.
#
# Usage:   rounding/scripts/run_bg.sh <job-name> <command> [args...]
# Log:     rounding/logs/<job-name>.log            (tail -f it; tqdm output is throttled by TQDM_MININTERVAL)
# Status:  systemctl --user status <job-name>  |  list: systemctl --user list-units 'run-*' --all
# Stop:    systemctl --user stop <job-name>
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <job-name> <command> [args...]" >&2; exit 2; }
name=$1; shift
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log="$REPO/rounding/logs/$name.log"
mkdir -p "$REPO/rounding/logs"
devenv_bin=$(command -v devenv || echo "$HOME/.nix-profile/bin/devenv")
echo "[$(date -Is)] starting job '$name' in $PWD: $*" >> "$log"
systemd-run --user --unit="$name" --collect --same-dir \
  -p StandardOutput=append:"$log" -p StandardError=append:"$log" \
  -E PATH="$PATH" -E HOME="$HOME" -E PYTHONUNBUFFERED=1 \
  "$devenv_bin" shell -- "$@"
echo "log: $log"
