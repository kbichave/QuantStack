# Copyright 2024 QuantPod Contributors
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
    python -m quant_pod.coordination.supervisor

Or in the supervisor tmux pane via start_supervised_loops.sh.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import duckdb
from loguru import logger


@dataclass
class LoopConfig:
    """Configuration for a monitored loop."""

    name: str
    expected_interval_seconds: int
    tmux_target: str = ""  # e.g., "quantpod-loops:loops.0"
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
        name="strategy_factory",
        expected_interval_seconds=60,
        tmux_target="quantpod-loops:loops.0",
    ),
    LoopConfig(
        name="live_trader",
        expected_interval_seconds=300,
        tmux_target="quantpod-loops:loops.1",
    ),
    LoopConfig(
        name="ml_research",
        expected_interval_seconds=120,
        tmux_target="quantpod-loops:loops.2",
    ),
]


class LoopSupervisor:
    """
    Monitors Ralph loop health via heartbeat events in DuckDB.

    Args:
        conn: DuckDB read-only connection (supervisor only reads).
        configs: Loop configurations. Defaults to the three Ralph loops.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        configs: list[LoopConfig] | None = None,
    ) -> None:
        self._conn = conn
        self._configs = {c.name: c for c in (configs or DEFAULT_LOOP_CONFIGS)}
        self._restart_counts: dict[str, int] = {}
        self._last_restart: dict[str, datetime] = {}

    def check_health(self) -> list[LoopHealth]:
        """Check all monitored loops and return their health status."""
        results: list[LoopHealth] = []
        # Use naive local time — DuckDB stores timestamps without timezone
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
            # DuckDB returns naive timestamps in local time
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

        Blocks forever — run in a dedicated tmux pane or as a background process.
        """
        logger.info(f"[Supervisor] Starting (interval={check_interval}s)")

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
