#!/usr/bin/env bash
# Start the Trading Loop in a tmux session.
#
# Position monitoring, entry scanning, execution.
# Does NOT optimize — executes and records outcomes.
# The research loop reads those outcomes to improve.
#
# Usage:
#   ./scripts/start_trading_loop.sh
#
# To stop:   tmux kill-session -t quantstack-trading
# To attach: tmux attach -t quantstack-trading

set -euo pipefail

if [[ "${FORCE_LOOPS:-0}" != "1" ]]; then
    echo "ERROR: Loops are PAUSED."
    echo "Reason: P&L attribution system not yet built."
    echo "To force-start: FORCE_LOOPS=1 $0"
    exit 1
fi

SESSION="quantstack-trading"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLAUDE_CMD="${QUANTPOD_CLAUDE_CMD:-claude}"
SLEEP=300

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION" -n trading -c "$REPO_DIR"
tmux send-keys -t "$SESSION:trading" \
    "while :; do echo \"[\$(date)] Trading iteration starting...\"; cat prompts/trading_loop.md | $CLAUDE_CMD --continue 2>&1 | tail -20; echo \"[\$(date)] Sleeping ${SLEEP}s...\"; sleep $SLEEP; done" C-m

echo "Trading loop started in tmux session '$SESSION'"
echo "  Attach: tmux attach -t $SESSION"
echo "  Stop:   tmux kill-session -t $SESSION"

tmux attach -t "$SESSION"
