#!/usr/bin/env bash
# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Supervised Ralph Wiggum loops — with health monitoring and git auto-commit.
#
# Panes:
#   0: Strategy Factory (every 60s)
#   1: Live Trader (every 300s)
#   2: ML Research (every 120s)
#   3: Loop Supervisor (checks health every 60s)
#   4: Git auto-commit (every 300s)
#
# Usage:
#   ./scripts/start_supervised_loops.sh [all|factory|trader|ml|supervisor]
#   tmux kill-session -t quantpod-loops   # to stop everything

set -euo pipefail

SESSION="quantpod-loops"
WORKDIR="${QUANTPOD_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CLAUDE_CMD="${QUANTPOD_CLAUDE_CMD:-claude}"
SLEEP_FACTORY=60
SLEEP_TRADER=300
SLEEP_ML=120
SUPERVISOR_INTERVAL=60
GIT_COMMIT_INTERVAL=300

MODE="${1:-all}"

cd "$WORKDIR"

# Kill existing session if running
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1

# Create session with first pane (Strategy Factory)
tmux new-session -d -s "$SESSION" -n loops -x 220 -y 50

# ── Pane 0: Strategy Factory ────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "factory" ]]; then
  tmux send-keys -t "$SESSION:loops.0" "cd $WORKDIR && echo '=== Strategy Factory ===' && while :; do echo \"[\$(date)] Factory iteration starting...\"; cat prompts/strategy_factory.md | $CLAUDE_CMD --continue 2>&1 | tail -20; echo \"[\$(date)] Sleeping ${SLEEP_FACTORY}s...\"; sleep $SLEEP_FACTORY; done" Enter
fi

# ── Pane 1: Live Trader ─────────────────────────────────────────────────────
tmux split-window -t "$SESSION:loops" -h
if [[ "$MODE" == "all" || "$MODE" == "trader" ]]; then
  tmux send-keys -t "$SESSION:loops.1" "cd $WORKDIR && echo '=== Live Trader ===' && while :; do echo \"[\$(date)] Trader iteration starting...\"; cat prompts/live_trader.md | $CLAUDE_CMD --continue 2>&1 | tail -20; echo \"[\$(date)] Sleeping ${SLEEP_TRADER}s...\"; sleep $SLEEP_TRADER; done" Enter
fi

# ── Pane 2: ML Research ─────────────────────────────────────────────────────
tmux split-window -t "$SESSION:loops" -v
if [[ "$MODE" == "all" || "$MODE" == "ml" ]]; then
  tmux send-keys -t "$SESSION:loops.2" "cd $WORKDIR && echo '=== ML Research ===' && while :; do echo \"[\$(date)] ML iteration starting...\"; cat prompts/ml_research.md | $CLAUDE_CMD --continue 2>&1 | tail -20; echo \"[\$(date)] Sleeping ${SLEEP_ML}s...\"; sleep $SLEEP_ML; done" Enter
fi

# ── Pane 3: Loop Supervisor ──────────────────────────────────────────────────
tmux split-window -t "$SESSION:loops.0" -v
if [[ "$MODE" == "all" || "$MODE" == "supervisor" ]]; then
  tmux send-keys -t "$SESSION:loops.3" "cd $WORKDIR && echo '=== Loop Supervisor ===' && python -m quant_pod.coordination.supervisor 2>&1" Enter
fi

# ── Pane 4: Git Auto-Commit ─────────────────────────────────────────────────
tmux split-window -t "$SESSION:loops.3" -v
tmux send-keys -t "$SESSION:loops.4" "cd $WORKDIR && echo '=== Git Auto-Commit ===' && while :; do git add .claude/memory/ 2>/dev/null && git diff --cached --quiet 2>/dev/null || git commit -m \"memory: auto-commit \$(date +%Y-%m-%dT%H:%M)\" 2>/dev/null; sleep $GIT_COMMIT_INTERVAL; done" Enter

# Tile the panes
tmux select-layout -t "$SESSION:loops" tiled

echo "Supervised loops started in tmux session '$SESSION'"
echo "  Attach: tmux attach -t $SESSION"
echo "  Stop:   tmux kill-session -t $SESSION"
