#!/usr/bin/env bash
# Start the Research Loop in a tmux session.
#
# Strategy discovery, ML training, parameter optimization.
# Reads optimization feedback (reflexion_episodes, judge_verdicts, prompt_critiques)
# to bias research direction. Spawns desk agents as needed.
#
# Usage:
#   ./scripts/start_research_loop.sh
#
# To stop:   tmux kill-session -t quantstack-research
# To attach: tmux attach -t quantstack-research

set -euo pipefail

if [[ "${FORCE_LOOPS:-0}" != "1" ]]; then
    echo "ERROR: Loops are PAUSED."
    echo "Reason: P&L attribution system not yet built."
    echo "To force-start: FORCE_LOOPS=1 $0"
    exit 1
fi

SESSION="quantstack-research"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLAUDE_CMD="${QUANTPOD_CLAUDE_CMD:-claude}"
SLEEP=120

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION" -n research -c "$REPO_DIR"
tmux send-keys -t "$SESSION:research" \
    "while :; do echo \"[\$(date)] Research iteration starting...\"; cat prompts/research_loop.md | $CLAUDE_CMD --continue 2>&1 | tail -20; echo \"[\$(date)] Sleeping ${SLEEP}s...\"; sleep $SLEEP; done" C-m

echo "Research loop started in tmux session '$SESSION'"
echo "  Attach: tmux attach -t $SESSION"
echo "  Stop:   tmux kill-session -t $SESSION"

tmux attach -t "$SESSION"
