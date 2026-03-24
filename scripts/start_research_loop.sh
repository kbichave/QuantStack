#!/usr/bin/env bash
# Start the Research Loop in a tmux session.
#
# Strategy discovery, ML training, parameter optimization.
# Reads optimization feedback (reflexion_episodes, judge_verdicts, prompt_critiques)
# to bias research direction. Spawns desk agents as needed.
#
# Usage:
#   ./scripts/start_research_loop.sh [investment|swing|options|all]
#   Or: RESEARCH_MODE=investment ./scripts/start_research_loop.sh
#
# Modes:
#   investment — equity investment (fundamental, long-hold) only
#   swing      — equity swing/position trading only
#   options    — options strategies only
#   all        — orchestrator runs all three domains (default)
#
# Run independently in parallel:
#   ./scripts/start_research_loop.sh investment  # tmux: qs-research-investment
#   ./scripts/start_research_loop.sh swing       # tmux: qs-research-swing
#   ./scripts/start_research_loop.sh options     # tmux: qs-research-options
#
# Run all via orchestrator:
#   ./scripts/start_research_loop.sh             # tmux: qs-research-all
#
# To stop:   tmux kill-session -t qs-research-<mode>
# To attach: tmux attach -t qs-research-<mode>

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# --- Parse mode argument ---
RESEARCH_MODE="${1:-${RESEARCH_MODE:-all}}"

# Map prompt file per mode
case "$RESEARCH_MODE" in
    investment)
        PROMPT_FILE="prompts/research_equity_investment.md"
        ;;
    swing)
        PROMPT_FILE="prompts/research_equity_swing.md"
        ;;
    options)
        PROMPT_FILE="prompts/research_options.md"
        ;;
    all)
        PROMPT_FILE="prompts/research_loop.md"
        ;;
    *)
        echo "Usage: $0 [investment|swing|options|all]"
        echo ""
        echo "Modes:"
        echo "  investment — equity investment (fundamental, long-hold)"
        echo "  swing      — equity swing/position trading"
        echo "  options    — options strategies"
        echo "  all        — orchestrator runs all domains (default)"
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

SESSION="qs-research-${RESEARCH_MODE}"
CLAUDE_CMD="${QUANTPOD_CLAUDE_CMD:-claude}"
SLEEP=120

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION" -n research -c "$REPO_DIR"
tmux send-keys -t "$SESSION:research" \
    "set -a; source .env; set +a; export RESEARCH_MODE=$RESEARCH_MODE; while :; do echo \"[\$(date)] Research iteration starting (mode=$RESEARCH_MODE, prompt=$PROMPT_FILE)...\"; cat $PROMPT_FILE | $CLAUDE_CMD --continue 2>&1 | tail -20; echo \"[\$(date)] Sleeping ${SLEEP}s...\"; sleep $SLEEP; done" C-m

echo "Research loop started in tmux session '$SESSION' (mode: $RESEARCH_MODE)"
echo "  Prompt: $PROMPT_FILE"
echo "  Attach: tmux attach -t $SESSION"
echo "  Stop:   tmux kill-session -t $SESSION"

tmux attach -t "$SESSION"
