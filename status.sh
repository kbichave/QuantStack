#!/usr/bin/env bash
# QuantStack — status dashboard (Textual TUI).
# Usage:
#   ./status.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec uv run python -m quantstack.tui
