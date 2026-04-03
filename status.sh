#!/usr/bin/env bash
# QuantStack — status dashboard.
# Usage:
#   ./status.sh                         # print once and exit
#   ./status.sh --watch                 # live refresh every 10s
#   ./status.sh --watch --interval 30   # custom refresh interval

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec python3 scripts/dashboard.py "$@"
