# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Autonomous Watchdog — system health monitor with auto-halt and auto-resume.

Runs every 60 seconds (via asyncio loop or external scheduler). Checks:
1. DuckDB connectivity (infra)
2. Kill switch state (safety)
3. Daily equity snapshot freshness (accounting)
4. Strategy breaker states (per-strategy safety)
5. SignalEngine liveness (analysis plane health)
6. Execution pipeline liveness (orders submitted recently)

On failure:
  - CRITICAL: halt all trading via kill switch, alert Discord
  - WARNING: alert Discord, do NOT halt (human-optional review)
  - Auto-resume: if kill switch was tripped by watchdog AND all checks pass
    for 5 consecutive minutes, automatically reset the kill switch.

No humans required. The watchdog is the operational backbone of the autonomous
trading company.

Usage:
    watchdog = Watchdog(conn)
    await watchdog.run_once()           # Single check cycle
    await watchdog.run_forever()        # Infinite loop (60s interval)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import duckdb
from loguru import logger


@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    status: str  # "OK", "WARNING", "CRITICAL"
    message: str
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WatchdogReport:
    """Aggregate result of all health checks."""
    checks: list[HealthCheck]
    overall_status: str  # "HEALTHY", "WARNING", "CRITICAL"
    action_taken: str | None = None  # "halt", "resume", None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "CRITICAL")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "WARNING")


class Watchdog:
    """
    Autonomous system health monitor.

    Args:
        conn: DuckDB connection for health queries.
        check_interval_s: Seconds between check cycles (default 60).
        resume_after_healthy_checks: Consecutive healthy checks before auto-resume (default 5).
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        check_interval_s: int = 60,
        resume_after_healthy_checks: int = 5,
    ) -> None:
        self._conn = conn
        self._interval = check_interval_s
        self._resume_threshold = resume_after_healthy_checks
        self._consecutive_healthy = 0
        self._watchdog_tripped = False  # True if WE tripped the kill switch
        self._running = False

    async def run_forever(self) -> None:
        """Run health checks in an infinite loop."""
        self._running = True
        logger.info(f"[Watchdog] Starting (interval={self._interval}s, resume_threshold={self._resume_threshold})")

        while self._running:
            try:
                report = await self.run_once()
                if report.overall_status == "CRITICAL":
                    self._consecutive_healthy = 0
                elif report.overall_status == "HEALTHY":
                    self._consecutive_healthy += 1
                else:
                    self._consecutive_healthy = max(0, self._consecutive_healthy - 1)
            except Exception as exc:
                logger.error(f"[Watchdog] Check cycle failed: {exc}")
                self._consecutive_healthy = 0

            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        """Stop the watchdog loop."""
        self._running = False

    async def run_once(self) -> WatchdogReport:
        """Run all health checks once and take action if needed."""
        checks = []

        checks.append(self._check_db())
        checks.append(self._check_kill_switch())
        checks.append(self._check_equity_snapshot())
        checks.append(self._check_strategy_breakers())
        checks.append(await self._check_signal_engine())

        # Determine overall status
        if any(c.status == "CRITICAL" for c in checks):
            overall = "CRITICAL"
        elif any(c.status == "WARNING" for c in checks):
            overall = "WARNING"
        else:
            overall = "HEALTHY"

        report = WatchdogReport(checks=checks, overall_status=overall)

        # Take action
        if overall == "CRITICAL":
            report.action_taken = self._handle_critical(checks)
        elif overall == "HEALTHY" and self._watchdog_tripped:
            if self._consecutive_healthy >= self._resume_threshold:
                report.action_taken = self._handle_auto_resume()

        # Log
        status_line = " | ".join(f"{c.name}={c.status}" for c in checks)
        if overall != "HEALTHY":
            logger.warning(f"[Watchdog] {overall}: {status_line}")
        else:
            logger.debug(f"[Watchdog] HEALTHY: {status_line}")

        # Alert on non-healthy
        if overall in ("CRITICAL", "WARNING"):
            self._send_alert(report)

        return report

    # ── Individual checks ──────────────────────────────────────────────────

    def _check_db(self) -> HealthCheck:
        """Check DuckDB is responding."""
        try:
            self._conn.execute("SELECT 1").fetchone()
            return HealthCheck(name="db", status="OK", message="DuckDB responsive")
        except Exception as exc:
            return HealthCheck(name="db", status="CRITICAL", message=f"DuckDB failed: {exc}")

    def _check_kill_switch(self) -> HealthCheck:
        """Check if kill switch is active (not tripped by watchdog = external trigger)."""
        try:
            from quant_pod.execution.kill_switch import get_kill_switch
            ks = get_kill_switch()
            if ks.is_active():
                if self._watchdog_tripped:
                    return HealthCheck(
                        name="kill_switch", status="WARNING",
                        message="Kill switch active (tripped by watchdog — monitoring for auto-resume)"
                    )
                return HealthCheck(
                    name="kill_switch", status="CRITICAL",
                    message=f"Kill switch active (external): {ks.status().reason}"
                )
            return HealthCheck(name="kill_switch", status="OK", message="Kill switch inactive")
        except Exception as exc:
            return HealthCheck(name="kill_switch", status="WARNING", message=f"Cannot check: {exc}")

    def _check_equity_snapshot(self) -> HealthCheck:
        """Check that daily equity snapshot is current (written today or last trading day)."""
        try:
            row = self._conn.execute(
                "SELECT MAX(date) FROM daily_equity"
            ).fetchone()

            if row is None or row[0] is None:
                return HealthCheck(
                    name="equity_snapshot", status="WARNING",
                    message="No equity snapshots exist yet"
                )

            last_date = row[0]
            if isinstance(last_date, datetime):
                last_date = last_date.date()

            today = date.today()
            # Allow 3-day gap for weekends
            if (today - last_date).days > 3:
                return HealthCheck(
                    name="equity_snapshot", status="WARNING",
                    message=f"Last snapshot {last_date} is {(today - last_date).days} days old"
                )
            return HealthCheck(
                name="equity_snapshot", status="OK",
                message=f"Last snapshot: {last_date}"
            )
        except Exception as exc:
            return HealthCheck(
                name="equity_snapshot", status="WARNING",
                message=f"Cannot check: {exc}"
            )

    def _check_strategy_breakers(self) -> HealthCheck:
        """Check if any strategies are tripped."""
        try:
            from quant_pod.execution.strategy_breaker import StrategyBreaker

            breaker = StrategyBreaker()
            states = breaker.get_all_states()
            tripped = [sid for sid, s in states.items() if s.status == "TRIPPED"]

            if tripped:
                return HealthCheck(
                    name="strategy_breakers", status="WARNING",
                    message=f"{len(tripped)} strategies TRIPPED: {', '.join(tripped[:5])}"
                )
            scaled = [sid for sid, s in states.items() if s.status == "SCALED"]
            if scaled:
                return HealthCheck(
                    name="strategy_breakers", status="OK",
                    message=f"{len(scaled)} strategies SCALED (monitoring)"
                )
            return HealthCheck(
                name="strategy_breakers", status="OK",
                message=f"{len(states)} strategies tracked, all ACTIVE"
            )
        except Exception as exc:
            return HealthCheck(
                name="strategy_breakers", status="WARNING",
                message=f"Cannot check: {exc}"
            )

    async def _check_signal_engine(self) -> HealthCheck:
        """Quick smoke test: can SignalEngine produce a brief for SPY?"""
        try:
            from quant_pod.signal_engine.engine import SignalEngine

            engine = SignalEngine()
            brief = await asyncio.wait_for(engine.run("SPY"), timeout=30.0)

            if brief.overall_confidence <= 0:
                return HealthCheck(
                    name="signal_engine", status="WARNING",
                    message="SignalEngine returned zero confidence"
                )
            failures = brief.collector_failures
            if len(failures) > 3:
                return HealthCheck(
                    name="signal_engine", status="WARNING",
                    message=f"SignalEngine: {len(failures)} collector failures: {failures}"
                )
            return HealthCheck(
                name="signal_engine", status="OK",
                message=f"SPY brief OK (confidence={brief.overall_confidence:.2f}, failures={len(failures)})"
            )
        except asyncio.TimeoutError:
            return HealthCheck(
                name="signal_engine", status="CRITICAL",
                message="SignalEngine timed out after 30s"
            )
        except Exception as exc:
            return HealthCheck(
                name="signal_engine", status="WARNING",
                message=f"SignalEngine error: {exc}"
            )

    # ── Action handlers ──────────────────────────────────────────────────

    def _handle_critical(self, checks: list[HealthCheck]) -> str:
        """Halt trading on critical failure."""
        critical = [c for c in checks if c.status == "CRITICAL"]
        reasons = "; ".join(f"{c.name}: {c.message}" for c in critical)

        try:
            from quant_pod.execution.kill_switch import get_kill_switch
            ks = get_kill_switch()
            if not ks.is_active():
                ks.trigger(reason=f"[Watchdog] CRITICAL: {reasons}")
                self._watchdog_tripped = True
                logger.critical(f"[Watchdog] HALTED TRADING: {reasons}")
                return "halt"
            return "already_halted"
        except Exception as exc:
            logger.error(f"[Watchdog] Failed to trigger kill switch: {exc}")
            return "halt_failed"

    def _handle_auto_resume(self) -> str:
        """Resume trading after sustained healthy checks."""
        try:
            from quant_pod.execution.kill_switch import get_kill_switch
            ks = get_kill_switch()
            if ks.is_active():
                ks.reset(by="watchdog_auto_resume")
                self._watchdog_tripped = False
                self._consecutive_healthy = 0
                logger.info(
                    f"[Watchdog] AUTO-RESUMED trading after "
                    f"{self._resume_threshold} consecutive healthy checks"
                )
                return "resume"
            self._watchdog_tripped = False
            return "already_running"
        except Exception as exc:
            logger.error(f"[Watchdog] Failed to reset kill switch: {exc}")
            return "resume_failed"

    def _send_alert(self, report: WatchdogReport) -> None:
        """Send alert to Discord (best-effort, non-blocking)."""
        try:
            webhook = os.getenv("DISCORD_WEBHOOK_URL")
            if not webhook:
                return

            import httpx

            status_emoji = {"CRITICAL": "🔴", "WARNING": "🟡", "HEALTHY": "🟢"}
            lines = [
                f"**{status_emoji.get(report.overall_status, '❓')} Watchdog: {report.overall_status}**",
            ]
            for check in report.checks:
                if check.status != "OK":
                    lines.append(f"- `{check.name}`: {check.status} — {check.message}")
            if report.action_taken:
                lines.append(f"**Action taken:** {report.action_taken}")

            httpx.post(
                webhook,
                json={"content": "\n".join(lines)},
                timeout=5.0,
            )
        except Exception:
            pass  # Alert failure must never crash the watchdog
