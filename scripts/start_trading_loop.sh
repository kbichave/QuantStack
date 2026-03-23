#!/usr/bin/env bash
# Start the Autonomous Trading Loop in a tmux session.
#
# Claude runs the full trading lifecycle: position monitoring, entry scanning,
# instrument selection (equity/options), and execution. All reasoning is LLM-based.
# MCP tools provide data; Claude provides all decisions.
#
# The loop runs every ~5 minutes during market hours (9:30-16:00 ET).
# Positions are held as long as the thesis supports (intraday to weeks).
#
# Usage:
#   ./scripts/start_trading_loop.sh
#
# To stop:   tmux kill-session -t quantstack-trading
# To attach: tmux attach -t quantstack-trading

set -euo pipefail

SESSION="quantstack-trading"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLAUDE_CMD="${QUANTPOD_CLAUDE_CMD:-claude}"
SLEEP="${TRADING_LOOP_INTERVAL:-300}"  # 5 min default

# Load environment if .env exists
if [[ -f "$REPO_DIR/.env" ]]; then
    set -a
    source "$REPO_DIR/.env"
    set +a
fi

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION" -n trading -c "$REPO_DIR"
tmux send-keys -t "$SESSION:trading" \
    "while :; do echo \"[\$(date)] Trading iteration starting...\"; cat prompts/trading_loop.md | $CLAUDE_CMD --continue 2>&1 | tail -20; echo \"[\$(date)] Sleeping ${SLEEP}s...\"; sleep $SLEEP; done" C-m

echo "Trading loop started in tmux session '$SESSION'"
echo "  Interval: ${SLEEP}s between iterations"
echo "  Attach:   tmux attach -t $SESSION"
echo "  Stop:     tmux kill-session -t $SESSION"

tmux attach -t "$SESSION"
