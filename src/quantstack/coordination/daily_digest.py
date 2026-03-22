# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Daily digest — aggregated summary of all autonomous loop activity.

Collects data from DuckDB tables (events, heartbeats, trades, strategies)
and produces a structured report that can be sent to Discord or written to
a memory file.

Runs at 17:00 ET via the supervisor, or on demand via MCP tool.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import duckdb
from loguru import logger


@dataclass
class DigestReport:
    """Aggregated daily activity report."""

    report_date: date = field(default_factory=date.today)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Portfolio
    open_positions: int = 0
    trades_today: int = 0
    total_realized_pnl: float = 0.0

    # Strategy lifecycle
    strategies_promoted: list[str] = field(default_factory=list)
    strategies_retired: list[str] = field(default_factory=list)
    strategies_demoted: list[str] = field(default_factory=list)
    total_live: int = 0
    total_forward_testing: int = 0
    total_draft: int = 0

    # Loop health
    factory_iterations: int = 0
    trader_iterations: int = 0
    ml_iterations: int = 0
    total_loop_errors: int = 0

    # Risk
    kill_switch_triggered: bool = False
    daily_halt_triggered: bool = False
    breakers_tripped: list[str] = field(default_factory=list)

    # ML
    models_trained: list[str] = field(default_factory=list)

    # Screener
    universe_size: int = 0
    watchlist_size: int = 0


class DailyDigest:
    """
    Generates aggregated daily reports from DuckDB state.

    Args:
        conn: DuckDB connection (read-only is sufficient).
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def generate(self, target_date: date | None = None) -> DigestReport:
        """Generate a digest for the given date (defaults to today)."""
        target = target_date or date.today()
        report = DigestReport(report_date=target)

        # Time window for "today"
        day_start = datetime.combine(target, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )
        day_end = day_start + timedelta(days=1)

        self._fill_portfolio(report)
        self._fill_trades(report, day_start, day_end)
        self._fill_strategies(report, day_start, day_end)
        self._fill_loop_health(report, day_start, day_end)
        self._fill_events(report, day_start, day_end)
        self._fill_screener(report)

        return report

    def format_markdown(self, report: DigestReport) -> str:
        """Format the digest as a markdown string."""
        lines = [
            f"# Daily Digest — {report.report_date.isoformat()}",
            "",
            "## Portfolio",
            f"- Open positions: {report.open_positions}",
            f"- Trades today: {report.trades_today}",
            f"- Realized P&L: ${report.total_realized_pnl:,.2f}",
            "",
            "## Strategy Lifecycle",
            f"- Live: {report.total_live} | Forward testing: {report.total_forward_testing} | Draft: {report.total_draft}",
        ]
        if report.strategies_promoted:
            lines.append(f"- Promoted: {', '.join(report.strategies_promoted)}")
        if report.strategies_retired:
            lines.append(f"- Retired: {', '.join(report.strategies_retired)}")
        if report.strategies_demoted:
            lines.append(f"- Demoted: {', '.join(report.strategies_demoted)}")

        lines.extend(
            [
                "",
                "## Loop Health",
                f"- Factory iterations: {report.factory_iterations}",
                f"- Trader iterations: {report.trader_iterations}",
                f"- ML iterations: {report.ml_iterations}",
                f"- Total errors: {report.total_loop_errors}",
            ]
        )

        if report.kill_switch_triggered or report.daily_halt_triggered:
            lines.extend(
                [
                    "",
                    "## Risk Events",
                ]
            )
            if report.kill_switch_triggered:
                lines.append("- KILL SWITCH TRIGGERED")
            if report.daily_halt_triggered:
                lines.append("- DAILY HALT TRIGGERED")
        if report.breakers_tripped:
            lines.append(f"- Breakers tripped: {', '.join(report.breakers_tripped)}")

        if report.models_trained:
            lines.extend(
                [
                    "",
                    "## ML Models",
                    f"- Trained: {', '.join(report.models_trained)}",
                ]
            )

        lines.extend(
            [
                "",
                "## Universe",
                f"- Universe size: {report.universe_size}",
                f"- Watchlist size: {report.watchlist_size}",
            ]
        )

        return "\n".join(lines)

    def format_discord(self, report: DigestReport) -> dict[str, Any]:
        """Format as a Discord webhook embed payload."""
        color = 0x00FF00  # Green
        if report.kill_switch_triggered or report.daily_halt_triggered:
            color = 0xFF0000  # Red
        elif report.total_loop_errors > 5:
            color = 0xFFA500  # Orange

        fields = [
            {"name": "Positions", "value": str(report.open_positions), "inline": True},
            {"name": "Trades Today", "value": str(report.trades_today), "inline": True},
            {
                "name": "Realized P&L",
                "value": f"${report.total_realized_pnl:,.2f}",
                "inline": True,
            },
            {
                "name": "Strategies",
                "value": f"L:{report.total_live} FT:{report.total_forward_testing} D:{report.total_draft}",
                "inline": True,
            },
            {
                "name": "Loop Iterations",
                "value": f"F:{report.factory_iterations} T:{report.trader_iterations} ML:{report.ml_iterations}",
                "inline": True,
            },
            {"name": "Errors", "value": str(report.total_loop_errors), "inline": True},
        ]

        if report.strategies_promoted:
            fields.append(
                {
                    "name": "Promoted",
                    "value": ", ".join(report.strategies_promoted),
                    "inline": False,
                }
            )
        if report.breakers_tripped:
            fields.append(
                {
                    "name": "Breakers Tripped",
                    "value": ", ".join(report.breakers_tripped),
                    "inline": False,
                }
            )

        return {
            "embeds": [
                {
                    "title": f"QuantPod Daily Digest — {report.report_date.isoformat()}",
                    "color": color,
                    "fields": fields,
                    "footer": {
                        "text": f"Generated at {report.generated_at.strftime('%H:%M UTC')}"
                    },
                }
            ]
        }

    def send_slack(self, report: DigestReport) -> bool:
        """Send the digest to Slack #system channel. Returns True on success."""
        try:
            from quantstack.coordination.slack_client import SlackClient

            client = SlackClient()
            if not client.is_configured:
                return False

            md = self.format_markdown(report)
            # Convert markdown to Slack mrkdwn (close enough)
            ts = client.post_system(md)
            if ts:
                logger.info("[DailyDigest] Sent to Slack #system")
                return True
            return False
        except Exception as exc:
            logger.debug(f"[DailyDigest] Slack send failed: {exc}")
            return False

    def send_discord(self, report: DigestReport) -> bool:
        """Send the digest to Discord via webhook. Returns True on success."""
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            logger.debug("[DailyDigest] No DISCORD_WEBHOOK_URL set — skipping")
            return False

        try:
            import urllib.request

            payload = json.dumps(self.format_discord(report)).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            logger.info("[DailyDigest] Sent to Discord")
            return True
        except Exception as exc:
            logger.warning(f"[DailyDigest] Discord send failed: {exc}")
            return False

    # ── Private data gathering ───────────────────────────────────────────────

    def _fill_portfolio(self, report: DigestReport) -> None:
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM positions").fetchone()
            report.open_positions = row[0] if row else 0
        except Exception:
            pass

    def _fill_trades(
        self, report: DigestReport, start: datetime, end: datetime
    ) -> None:
        try:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(realized_pnl), 0) FROM closed_trades WHERE closed_at >= ? AND closed_at < ?",
                [start, end],
            ).fetchone()
            if row:
                report.trades_today = row[0]
                report.total_realized_pnl = row[1]
        except Exception:
            pass

    def _fill_strategies(
        self, report: DigestReport, start: datetime, end: datetime
    ) -> None:
        try:
            for status in ("live", "forward_testing", "draft"):
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM strategies WHERE status = ?", [status]
                ).fetchone()
                count = row[0] if row else 0
                if status == "live":
                    report.total_live = count
                elif status == "forward_testing":
                    report.total_forward_testing = count
                else:
                    report.total_draft = count
        except Exception:
            pass

    def _fill_loop_health(
        self, report: DigestReport, start: datetime, end: datetime
    ) -> None:
        try:
            for loop_name in ("strategy_factory", "live_trader", "ml_research"):
                row = self._conn.execute(
                    """
                    SELECT COUNT(*), COALESCE(SUM(errors), 0)
                    FROM loop_heartbeats
                    WHERE loop_name = ? AND started_at >= ? AND started_at < ?
                    """,
                    [loop_name, start, end],
                ).fetchone()
                if row:
                    iterations, errors = row
                    if loop_name == "strategy_factory":
                        report.factory_iterations = iterations
                    elif loop_name == "live_trader":
                        report.trader_iterations = iterations
                    elif loop_name == "ml_research":
                        report.ml_iterations = iterations
                    report.total_loop_errors += errors
        except Exception:
            pass

    def _fill_events(
        self, report: DigestReport, start: datetime, end: datetime
    ) -> None:
        try:
            rows = self._conn.execute(
                """
                SELECT event_type, payload FROM loop_events
                WHERE created_at >= ? AND created_at < ?
                ORDER BY created_at
                """,
                [start, end],
            ).fetchall()

            for etype, payload_raw in rows:
                payload = json.loads(payload_raw) if payload_raw else {}
                sid = payload.get("strategy_id", "?")

                if etype == "strategy_promoted":
                    report.strategies_promoted.append(sid)
                elif etype == "strategy_retired":
                    report.strategies_retired.append(sid)
                elif etype == "strategy_demoted":
                    report.strategies_demoted.append(sid)
                elif etype == "model_trained":
                    symbol = payload.get("symbol", "?")
                    report.models_trained.append(symbol)
                elif etype == "degradation_detected":
                    if payload.get("severity") == "critical":
                        report.breakers_tripped.append(sid)
        except Exception:
            pass

    def _fill_screener(self, report: DigestReport) -> None:
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM universe WHERE is_active = TRUE"
            ).fetchone()
            report.universe_size = row[0] if row else 0

            row = self._conn.execute(
                """
                SELECT COUNT(*) FROM screener_results
                WHERE screened_at = (SELECT MAX(screened_at) FROM screener_results)
                """
            ).fetchone()
            report.watchlist_size = row[0] if row else 0
        except Exception:
            pass
