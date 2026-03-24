#!/usr/bin/env bash
# Start the Research Loop in a tmux session.
#
# Strategy discovery, ML training, parameter optimization.
# Reads optimization feedback (reflexion_episodes, judge_verdicts, prompt_critiques)
# to bias research direction. Spawns desk agents as needed.
#
# Usage:
#   ./scripts/start_research_loop.sh [equity|options|both]
#   Or: RESEARCH_MODE=equity ./scripts/start_research_loop.sh
#
# Modes:
#   equity  — equity investment + swing/position strategies only
#   options — options strategies only
#   both    — full portfolio (default)
#
# Run all three independently:
#   ./scripts/start_research_loop.sh equity   # tmux: quantstack-research-equity
#   ./scripts/start_research_loop.sh options  # tmux: quantstack-research-options
#
# To stop:   tmux kill-session -t quantstack-research-<mode>
# To attach: tmux attach -t quantstack-research-<mode>

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# --- Parse mode argument ---
RESEARCH_MODE="${1:-${RESEARCH_MODE:-both}}"
case "$RESEARCH_MODE" in
    equity|options|both) ;;
    *)
        echo "Usage: $0 [equity|options|both]"
        echo "  Or set RESEARCH_MODE env var. Default: both"
        exit 1
        ;;
esac

# Load env vars (API keys, risk limits, etc.)
if [[ -f "$REPO_DIR/.env" ]]; then
    set -a
    source "$REPO_DIR/.env"
    set +a
fi

# Export mode so the prompt can read it
export RESEARCH_MODE

SESSION="quantstack-research-${RESEARCH_MODE}"
CLAUDE_CMD="${QUANTPOD_CLAUDE_CMD:-claude}"
SLEEP=120

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION" -n research -c "$REPO_DIR"
tmux send-keys -t "$SESSION:research" \
    "set -a; source .env; set +a; export RESEARCH_MODE=$RESEARCH_MODE; while :; do echo \"[\$(date)] Research iteration starting (mode=$RESEARCH_MODE)...\"; cat prompts/research_loop.md | $CLAUDE_CMD --continue 2>&1 | tail -20; echo \"[\$(date)] Sleeping ${SLEEP}s...\"; sleep $SLEEP; done" C-m

echo "Research loop started in tmux session '$SESSION' (mode: $RESEARCH_MODE)"
echo "  Attach: tmux attach -t $SESSION"
echo "  Stop:   tmux kill-session -t $SESSION"

tmux attach -t "$SESSION"
