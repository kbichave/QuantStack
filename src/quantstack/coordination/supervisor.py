# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Loop supervisor — monitors Ralph loop health via heartbeat events.

Each Ralph loop (Factory, Trader, ML) publishes LOOP_HEARTBEAT events at the
start and end of each iteration.  The supervisor polls these heartbeats and
detects when a loop is stale (no heartbeat for 3x expected interval) or dead
(10x expected interval).

On stall detection, the supervisor checks if the tmux pane is alive and
restarts the loop with exponential backoff (max 5 attempts).

Run as:
    python -m quantstack.coordination.supervisor

Or in the supervisor tmux pane via start_supervised_loops.sh.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from quantstack.db import PgConnection

# Path to the autoresclaw_runner script (relative to repo root).
_RUNNER_SCRIPT = Path(__file__).parents[3] / "scripts" / "autoresclaw_runner.py"


@dataclass
class LoopConfig:
    """Configuration for a monitored loop."""

    name: str
    expected_interval_seconds: int
    tmux_target: str = ""  # e.g., "quantstack-loops:loops.0"
    restart_command: str = ""
    max_restart_attempts: int = 5
    backoff_base_seconds: int = 30


@dataclass
class LoopHealth:
    """Health status of a single loop."""

    loop_name: str
    last_heartbeat: datetime | None = None
    last_iteration: int = 0
    staleness_seconds: float = 0.0
    status: str = "unknown"  # "healthy", "stale", "dead", "unknown"
    consecutive_errors: int = 0
    restart_count: int = 0


DEFAULT_LOOP_CONFIGS = [
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


class LoopSupervisor:
    """
    Monitors Ralph loop health via heartbeat events in PostgreSQL.

    Args:
        conn: PostgreSQL connection (supervisor reads heartbeat events).
        configs: Loop configurations. Defaults to the three Ralph loops.
    """

    def __init__(
        self,
        conn: PgConnection,
        configs: list[LoopConfig] | None = None,
    ) -> None:
        self._conn = conn
        self._configs = {c.name: c for c in (configs or DEFAULT_LOOP_CONFIGS)}
        self._restart_counts: dict[str, int] = {}
        self._last_restart: dict[str, datetime] = {}

    def check_health(self) -> list[LoopHealth]:
        """Check all monitored loops and return their health status."""
        results: list[LoopHealth] = []
        # Use naive local time for comparison
        now = datetime.now()

        for name, config in self._configs.items():
            health = self._check_one(name, config, now)
            results.append(health)

        return results

    def _check_one(self, name: str, config: LoopConfig, now: datetime) -> LoopHealth:
        """Check health of a single loop."""
        health = LoopHealth(loop_name=name)

        try:
            row = self._conn.execute(
                """
                SELECT iteration, started_at, finished_at, errors, status
                FROM loop_heartbeats
                WHERE loop_name = ?
                  AND status != 'orphaned'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                [name],
            ).fetchone()
        except Exception:
            return health

        if not row:
            health.status = "unknown"
            return health

        iteration, started_at, finished_at, errors, hb_status = row
        health.last_iteration = iteration
        health.consecutive_errors = errors or 0

        # Use finished_at if available, else started_at
        last_ts = finished_at or started_at
        if last_ts:
            if not isinstance(last_ts, datetime):
                health.status = "unknown"
                return health
            # Normalize to naive local time for comparison
            if last_ts.tzinfo is not None:
                last_ts = last_ts.replace(tzinfo=None)
            health.last_heartbeat = last_ts
            health.staleness_seconds = (now - last_ts).total_seconds()

            stale_threshold = config.expected_interval_seconds * 3
            dead_threshold = config.expected_interval_seconds * 10

            if health.staleness_seconds < stale_threshold:
                health.status = "healthy"
            elif health.staleness_seconds < dead_threshold:
                health.status = "stale"
            else:
                health.status = "dead"
        else:
            health.status = "unknown"

        health.restart_count = self._restart_counts.get(name, 0)
        return health

    def restart_loop(self, loop_name: str) -> bool:
        """
        Attempt to restart a stale/dead loop in its tmux pane.

        Returns True if restart command was sent successfully.
        """
        config = self._configs.get(loop_name)
        if not config or not config.tmux_target:
            logger.warning(f"[Supervisor] No tmux target for {loop_name}")
            return False

        count = self._restart_counts.get(loop_name, 0)
        if count >= config.max_restart_attempts:
            logger.error(
                f"[Supervisor] {loop_name} has exceeded max restart attempts "
                f"({config.max_restart_attempts}). Manual intervention required."
            )
            return False

        # Exponential backoff
        last = self._last_restart.get(loop_name)
        if last:
            backoff = config.backoff_base_seconds * (2**count)
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
            if elapsed < backoff:
                logger.debug(
                    f"[Supervisor] {loop_name} in backoff ({elapsed:.0f}s < {backoff}s)"
                )
                return False

        # Check if tmux pane exists
        try:
            result = subprocess.run(
                ["tmux", "list-panes", "-t", config.tmux_target.split(".")[0]],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                logger.warning(f"[Supervisor] tmux session not found for {loop_name}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

        # Send restart command to the pane
        if config.restart_command:
            try:
                subprocess.run(
                    [
                        "tmux",
                        "send-keys",
                        "-t",
                        config.tmux_target,
                        config.restart_command,
                        "Enter",
                    ],
                    timeout=5,
                )
                self._restart_counts[loop_name] = count + 1
                self._last_restart[loop_name] = datetime.now(timezone.utc)
                logger.info(f"[Supervisor] Restarted {loop_name} (attempt {count + 1})")
                return True
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                logger.error(f"[Supervisor] Failed to restart {loop_name}: {exc}")
                return False

        return False

    def run_forever(self, check_interval: int = 60) -> None:
        """
        Main supervisor loop.  Checks health every ``check_interval`` seconds.

        Also starts a background thread that watches research_queue for pending
        bug_fix tasks and dispatches AutoResearchClaw immediately — no waiting
        for the Sunday scheduled run.

        Blocks forever — run in a dedicated tmux pane or as a background process.
        """
        logger.info(f"[Supervisor] Starting (health_interval={check_interval}s)")

        # Start the bug-fix watcher in a daemon thread so it dies with the supervisor.
        watcher = threading.Thread(
            target=self._bug_fix_watcher,
            kwargs={"poll_interval": 60},
            daemon=True,
            name="bug-fix-watcher",
        )
        watcher.start()
        logger.info("[Supervisor] Bug-fix watcher thread started (poll=60s)")

        while True:
            try:
                results = self.check_health()
                for health in results:
                    if health.status == "dead":
                        logger.warning(
                            f"[Supervisor] {health.loop_name} is DEAD "
                            f"(last heartbeat {health.staleness_seconds:.0f}s ago)"
                        )
                        self.restart_loop(health.loop_name)
                    elif health.status == "stale":
                        logger.info(
                            f"[Supervisor] {health.loop_name} is STALE "
                            f"(last heartbeat {health.staleness_seconds:.0f}s ago)"
                        )
                    elif health.status == "healthy":
                        logger.debug(
                            f"[Supervisor] {health.loop_name} healthy "
                            f"(iter={health.last_iteration})"
                        )
            except Exception as exc:
                logger.error(f"[Supervisor] Health check failed: {exc}")

            time.sleep(check_interval)

    def _bug_fix_watcher(self, poll_interval: int = 60) -> None:
        """
        Background thread: polls the bugs table every poll_interval seconds for
        open bugs that have a linked research_queue task pending, and dispatches
        AutoResearchClaw immediately — highest priority first, no weekly wait.

        Uses the bugs table (not research_queue) as the source of truth so we
        get proper deduplication and lifecycle tracking. The research_queue entry
        is the dispatch handle; bugs.arc_task_id links them.

        Runs one task at a time to avoid parallel ARC sessions clobbering each other.
        """
        logger.info("[BugFixWatcher] Started — polling bugs table every %ds", poll_interval)
        dispatched: set[str] = set()

        while True:
            try:
                # Find the highest-priority open bug that has a pending dispatch task
                # and that we haven't already started this session.
                row = self._conn.execute(
                    """
                    SELECT b.bug_id, b.arc_task_id, b.tool_name, b.priority
                    FROM bugs b
                    JOIN research_queue rq ON rq.task_id = b.arc_task_id
                    WHERE b.status = 'open'
                      AND rq.status = 'pending'
                    ORDER BY b.priority DESC, b.consecutive_errors DESC, b.created_at ASC
                    LIMIT 1
                    """,
                ).fetchone()

                if row:
                    bug_id, task_id, tool_name, priority = row
                    if task_id not in dispatched:
                        dispatched.add(task_id)
                        logger.warning(
                            f"[BugFixWatcher] Bug {bug_id} ({tool_name}, priority={priority}) "
                            f"→ dispatching ARC task {task_id}"
                        )
                        # Mark bug in_progress before we fire ARC
                        try:
                            self._conn.execute(
                                "UPDATE bugs SET status='in_progress' WHERE bug_id=%s",
                                [bug_id],
                            )
                            self._conn.commit()
                        except Exception:
                            pass
                        self._run_arc_task(task_id)

            except Exception as exc:
                logger.error(f"[BugFixWatcher] Poll error: {exc}")

            time.sleep(poll_interval)

    def _run_arc_task(self, task_id: str) -> None:
        """Fire autoresclaw_runner.py for a specific task_id (blocking)."""
        if not _RUNNER_SCRIPT.exists():
            logger.error(f"[BugFixWatcher] Runner script not found: {_RUNNER_SCRIPT}")
            return

        cmd = [sys.executable, str(_RUNNER_SCRIPT), "--task-id", task_id]
        logger.info(f"[BugFixWatcher] Running: {' '.join(cmd)}")
        try:
            # Timeout: 90 min max per task (generous but bounded).
            result = subprocess.run(cmd, timeout=5400)
            if result.returncode == 0:
                logger.info(f"[BugFixWatcher] Task {task_id} completed")
            else:
                logger.error(
                    f"[BugFixWatcher] Task {task_id} exited with code {result.returncode}"
                )
        except subprocess.TimeoutExpired:
            logger.error(f"[BugFixWatcher] Task {task_id} timed out after 90 min")
        except Exception as exc:
            logger.error(f"[BugFixWatcher] Task {task_id} failed to start: {exc}")
