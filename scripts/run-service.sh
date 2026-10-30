#!/usr/bin/env bash
# Run a kalshi-trader command inside the project's Nix dev shell, which provides Python and
# the runtime libraries the prebuilt PyPI wheels need on NixOS. `exec` keeps the Python
# process in the foreground so systemd's SIGTERM reaches it and it shuts down gracefully.
#
#   scripts/run-service.sh run              # engine + dashboard
#   scripts/run-service.sh run --record-only
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [ ! -x "$PROJECT_DIR/.venv/bin/kalshi-trader" ]; then
  nix develop "$PROJECT_DIR" --command uv sync --frozen
fi
exec nix develop "$PROJECT_DIR" --command "$PROJECT_DIR/.venv/bin/kalshi-trader" "$@"
