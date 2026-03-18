#!/usr/bin/env bash
# Start QuantPod autonomous loops in a tmux session.
#
# Three loops (inspired by Karpathy's autoresearch):
#   Pane 1 — Strategy Factory (rule-based strategy discovery, every ~60s)
#   Pane 2 — Live Trader (position monitoring + execution, every ~300s)
#   Pane 3 — ML Research (autonomous model training + feature engineering, every ~120s)
#
# Usage:
#   ./scripts/start_loops.sh              # All three loops
#   ./scripts/start_loops.sh factory      # Strategy Factory only
#   ./scripts/start_loops.sh trader       # Live Trader only
#   ./scripts/start_loops.sh ml           # ML Research only
#   ./scripts/start_loops.sh trading      # Factory + Trader (no ML)
#
# To stop: tmux kill-session -t quantpod-loops
# To attach: tmux attach -t quantpod-loops

set -euo pipefail

SESSION="quantpod-loops"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Kill existing session if running
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

MODE="${1:-all}"

case "$MODE" in
    factory)
        tmux new-session -d -s "$SESSION" -n factory -c "$REPO_DIR"
        tmux send-keys -t "$SESSION:factory" \
            "while :; do cat prompts/strategy_factory.md | claude --continue; echo '--- sleeping 60s ---'; sleep 60; done" C-m
        ;;
    trader)
        tmux new-session -d -s "$SESSION" -n trader -c "$REPO_DIR"
        tmux send-keys -t "$SESSION:trader" \
            "while :; do cat prompts/live_trader.md | claude --continue; echo '--- sleeping 300s ---'; sleep 300; done" C-m
        ;;
    ml)
        tmux new-session -d -s "$SESSION" -n ml -c "$REPO_DIR"
        tmux send-keys -t "$SESSION:ml" \
            "while :; do cat prompts/ml_research.md | claude --continue; echo '--- sleeping 120s ---'; sleep 120; done" C-m
        ;;
    trading)
        # Factory + Trader only (no ML research)
        tmux new-session -d -s "$SESSION" -n loops -c "$REPO_DIR"
        tmux send-keys -t "$SESSION:loops" \
            "while :; do cat prompts/strategy_factory.md | claude --continue; echo '--- sleeping 60s ---'; sleep 60; done" C-m
        tmux split-window -h -t "$SESSION:loops" -c "$REPO_DIR"
        tmux send-keys -t "$SESSION:loops.1" \
            "while :; do cat prompts/live_trader.md | claude --continue; echo '--- sleeping 300s ---'; sleep 300; done" C-m
        ;;
    all)
        # All three loops in a tiled layout
        tmux new-session -d -s "$SESSION" -n loops -c "$REPO_DIR"

        # Pane 0: Strategy Factory (top-left)
        tmux send-keys -t "$SESSION:loops" \
            "while :; do cat prompts/strategy_factory.md | claude --continue; echo '--- sleeping 60s ---'; sleep 60; done" C-m

        # Pane 1: Live Trader (top-right)
        tmux split-window -h -t "$SESSION:loops" -c "$REPO_DIR"
        tmux send-keys -t "$SESSION:loops.1" \
            "while :; do cat prompts/live_trader.md | claude --continue; echo '--- sleeping 300s ---'; sleep 300; done" C-m

        # Pane 2: ML Research (bottom, full width)
        tmux split-window -v -t "$SESSION:loops.0" -c "$REPO_DIR"
        tmux send-keys -t "$SESSION:loops.2" \
            "while :; do cat prompts/ml_research.md | claude --continue; echo '--- sleeping 120s ---'; sleep 120; done" C-m

        # Even out the layout
        tmux select-layout -t "$SESSION:loops" tiled
        ;;
    *)
        echo "Usage: $0 [all|factory|trader|ml|trading]"
        exit 1
        ;;
esac

echo "QuantPod loops started in tmux session '$SESSION' (mode: $MODE)"
echo "  Attach: tmux attach -t $SESSION"
echo "  Stop:   tmux kill-session -t $SESSION"

tmux attach -t "$SESSION"
