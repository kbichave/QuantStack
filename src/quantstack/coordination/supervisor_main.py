# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
supervisor_main — standalone entry point for the loop supervisor.

Runs inside the `quantstack-loops:supervisor` tmux window. Monitors trading_loop
and research_loop heartbeats and restarts dead loops via tmux send-keys.

Usage:
    python -m quantstack.coordination.supervisor_main
    quantstack-supervisor  (console script)
"""

import logging
import sys

from quantstack.coordination.supervisor import LoopConfig, LoopSupervisor
from quantstack.db import open_db

LOOP_CONFIGS = [
    LoopConfig(
        name="trading_loop",
        expected_interval_seconds=300,
        tmux_target="quantstack-loops:trading",
        restart_command="cat prompts/trading_loop.md | claude",
    ),
    LoopConfig(
        name="research_loop",
        expected_interval_seconds=120,
        tmux_target="quantstack-loops:research",
        restart_command="cat prompts/research_loop.md | claude",
    ),
]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    try:
        conn = open_db()
    except Exception as exc:
        logging.error(f"[Supervisor] Cannot connect to PostgreSQL: {exc}")
        sys.exit(1)

    supervisor = LoopSupervisor(conn, LOOP_CONFIGS)
    supervisor.run_forever(check_interval=60)


if __name__ == "__main__":
    main()
