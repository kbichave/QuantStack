#!/usr/bin/env bash
# QuantStack — health dashboard.
# Usage:
#   ./status.sh             # print once and exit
#   ./status.sh --watch     # live refresh every 10s (Ctrl+C to quit)
#   ./status.sh --watch --interval 30   # custom refresh interval

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

exec python3 scripts/dashboard.py "$@"
