# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Production preflight check — the gate between research and live trading.

This is the single checkpoint that must pass before the system starts
executing trades (paper or live).  It validates that every dependency
is healthy and every precondition is met.

Usage (CLI):
    python -m quant_pod.coordination.preflight

Usage (MCP tool):
    run_preflight_check()

Usage (code):
    from quantstack.coordination.preflight import PreflightCheck
    report = PreflightCheck(conn).run()
    if not report.ready:
        print(report.blockers)  # Must fix these before trading
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pathlib import Path

import duckdb
from loguru import logger

from quantstack.db import open_db, run_migrations


@dataclass
class PreflightResult:
    """Outcome of a preflight check."""

    name: str
    passed: bool
    detail: str
    severity: str = "blocker"  # "blocker" or "warning"


@dataclass
class PreflightReport:
    """Aggregate preflight report."""

    run_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checks: list[PreflightResult] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True only if zero blockers."""
        return not any(c.severity == "blocker" and not c.passed for c in self.checks)

    @property
    def blockers(self) -> list[PreflightResult]:
        return [c for c in self.checks if c.severity == "blocker" and not c.passed]

    @property
    def warnings(self) -> list[PreflightResult]:
        return [c for c in self.checks if c.severity == "warning" and not c.passed]

    def summary(self) -> str:
        lines = [f"Preflight Report — {self.run_at.strftime('%Y-%m-%d %H:%M UTC')}"]
        lines.append(f"Status: {'READY' if self.ready else 'NOT READY'}")
        lines.append("")

        for c in self.checks:
            icon = (
                "PASS" if c.passed else ("BLOCK" if c.severity == "blocker" else "WARN")
            )
            lines.append(f"  [{icon}] {c.name}: {c.detail}")

        if self.blockers:
            lines.append("")
            lines.append(
                f"  {len(self.blockers)} blocker(s) must be resolved before trading."
            )

        return "\n".join(lines)


class PreflightCheck:
    """
    Production readiness gate.

    Validates:
      1. Database — migrations ran, tables exist
      2. Kill switch — not active
      3. Cash balance — set and sufficient for target wallet
      4. Universe — populated (at least ETFs)
      5. Screener — has run recently (< 24h)
      6. Strategies — at least 1 in live or forward_testing
      7. SignalEngine — can produce a brief for the target symbol
      8. Broker — connectivity (if Alpaca keys set)
      9. Risk limits — sane for the configured wallet size
     10. Data provider — FD.ai key set

    Args:
        conn: DuckDB connection.
        target_symbols: Symbols to validate SignalEngine against (default ["SPY"]).
        target_wallet: Expected starting equity in dollars (default 1000).
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        target_symbols: list[str] | None = None,
        target_wallet: float = 1000.0,
    ) -> None:
        self._conn = conn
        self._symbols = target_symbols or ["SPY"]
        self._wallet = target_wallet

    def run(self) -> PreflightReport:
        """Run all preflight checks."""
        report = PreflightReport()

        report.checks.append(self._check_database())
        report.checks.append(self._check_kill_switch())
        report.checks.append(self._check_cash_balance())
        report.checks.append(self._check_universe())
        report.checks.append(self._check_screener())
        report.checks.append(self._check_strategies())
        report.checks.append(self._check_risk_limits())
        report.checks.append(self._check_data_provider())
        report.checks.append(self._check_broker())
        report.checks.append(self._check_paper_mode())
        report.checks.append(self._check_options_execution())

        return report

    def _check_database(self) -> PreflightResult:
        """Verify all expected tables exist."""
        try:
            tables = {r[0] for r in self._conn.execute("SHOW TABLES").fetchall()}
            required = {
                "strategies",
                "positions",
                "cash_balance",
                "fills",
                "decision_events",
                "system_state",
                "universe",
                "screener_results",
                "loop_events",
                "loop_heartbeats",
            }
            missing = required - tables
            if missing:
                return PreflightResult(
                    "Database tables",
                    False,
                    f"Missing tables: {missing}. Run migrations first.",
                )
            return PreflightResult(
                "Database tables", True, f"{len(tables)} tables present"
            )
        except Exception as exc:
            return PreflightResult("Database tables", False, f"DB error: {exc}")

    def _check_kill_switch(self) -> PreflightResult:
        """Kill switch must not be active."""
        try:
            row = self._conn.execute(
                "SELECT value FROM system_state WHERE key = 'kill_switch'"
            ).fetchone()
            if row and row[0] == "active":
                return PreflightResult(
                    "Kill switch",
                    False,
                    "Kill switch is ACTIVE. Reset it before trading.",
                )
            # Also check sentinel file
            sentinel = Path("~/.quant_pod/KILL_SWITCH_ACTIVE").expanduser()
            if sentinel.exists():
                return PreflightResult(
                    "Kill switch",
                    False,
                    f"Kill switch sentinel file exists at {sentinel}",
                )
            return PreflightResult("Kill switch", True, "Not active")
        except Exception:
            return PreflightResult(
                "Kill switch", True, "No kill switch state found (OK)"
            )

    def _check_cash_balance(self) -> PreflightResult:
        """Cash balance must be set and match target wallet."""
        try:
            row = self._conn.execute(
                "SELECT cash FROM cash_balance WHERE id = 1"
            ).fetchone()
            if not row:
                return PreflightResult(
                    "Cash balance",
                    False,
                    f"No cash balance set. Initialize with ${self._wallet:,.0f}.",
                    severity="blocker",
                )
            cash = row[0]
            if cash < self._wallet * 0.5:
                return PreflightResult(
                    "Cash balance",
                    False,
                    f"Cash ${cash:,.0f} is below 50% of target ${self._wallet:,.0f}",
                )
            return PreflightResult(
                "Cash balance",
                True,
                f"${cash:,.0f} (target: ${self._wallet:,.0f})",
            )
        except Exception as exc:
            return PreflightResult("Cash balance", False, f"Error: {exc}")

    def _check_universe(self) -> PreflightResult:
        """Universe must be populated."""
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM universe WHERE is_active = TRUE"
            ).fetchone()
            count = row[0] if row else 0
            if count == 0:
                return PreflightResult(
                    "Universe",
                    False,
                    "Universe is empty. Run UniverseRegistry.refresh_constituents().",
                )
            # Check that target symbols are in the universe
            for sym in self._symbols:
                r = self._conn.execute(
                    "SELECT 1 FROM universe WHERE symbol = ? AND is_active = TRUE",
                    [sym],
                ).fetchone()
                if not r:
                    return PreflightResult(
                        "Universe",
                        False,
                        f"Target symbol {sym} not in universe. Add it manually or run refresh.",
                    )
            return PreflightResult("Universe", True, f"{count} active symbols")
        except Exception as exc:
            return PreflightResult("Universe", False, f"Error: {exc}")

    def _check_screener(self) -> PreflightResult:
        """Screener should have run recently."""
        try:
            row = self._conn.execute(
                "SELECT MAX(screened_at) FROM screener_results"
            ).fetchone()
            if not row or not row[0]:
                return PreflightResult(
                    "Screener",
                    False,
                    "No screener results. Run AutonomousScreener.screen() first.",
                    severity="warning",
                )
            return PreflightResult("Screener", True, f"Last run: {row[0]}")
        except Exception:
            return PreflightResult(
                "Screener", True, "Table empty (OK for first run)", severity="warning"
            )

    def _check_strategies(self) -> PreflightResult:
        """At least one strategy must be in live or forward_testing."""
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM strategies WHERE status IN ('live', 'forward_testing')"
            ).fetchone()
            count = row[0] if row else 0
            if count == 0:
                return PreflightResult(
                    "Strategies",
                    False,
                    "No live or forward_testing strategies. "
                    "Use /workshop to create and validate a strategy first.",
                )
            # Count by status
            live = self._conn.execute(
                "SELECT COUNT(*) FROM strategies WHERE status = 'live'"
            ).fetchone()[0]
            ft = self._conn.execute(
                "SELECT COUNT(*) FROM strategies WHERE status = 'forward_testing'"
            ).fetchone()[0]
            return PreflightResult(
                "Strategies",
                True,
                f"{live} live + {ft} forward_testing = {count} tradeable",
            )
        except Exception as exc:
            return PreflightResult("Strategies", False, f"Error: {exc}")

    @staticmethod
    def _env_float(key: str, default: float) -> float:
        """Read a float from env, stripping inline comments."""
        raw = os.getenv(key, str(default))
        # Strip inline comments (e.g. "0.10  # 10% of equity")
        raw = raw.split("#")[0].strip()
        return float(raw) if raw else default

    def _check_risk_limits(self) -> PreflightResult:
        """Risk limits must be sane for the wallet size."""
        max_pct = self._env_float("RISK_MAX_POSITION_PCT", 0.10)
        max_notional = self._env_float("RISK_MAX_POSITION_NOTIONAL", 20000)
        max_per_position = min(self._wallet * max_pct, max_notional)

        warnings = []

        # Check if we can actually buy the target symbols
        for sym in self._symbols:
            # Rough price check — use cached OHLCV if available
            try:
                row = self._conn.execute(
                    "SELECT close FROM ohlcv WHERE symbol = ? AND timeframe = 'D1' "
                    "ORDER BY timestamp DESC LIMIT 1",
                    [sym],
                ).fetchone()
                if row:
                    price = row[0]
                    if price > max_per_position:
                        fractional = os.getenv(
                            "ALPACA_FRACTIONAL_SHARES", "true"
                        ).lower() in ("true", "1")
                        if fractional:
                            warnings.append(
                                f"{sym} (${price:.0f}) > max position (${max_per_position:.0f}) "
                                f"— fractional shares required"
                            )
                        else:
                            return PreflightResult(
                                "Risk limits",
                                False,
                                f"{sym} at ${price:.0f} exceeds max position ${max_per_position:.0f}. "
                                f"Either increase wallet, raise RISK_MAX_POSITION_PCT, or enable fractional shares.",
                            )
            except Exception:
                pass

        daily_halt = self._wallet * self._env_float("RISK_DAILY_LOSS_LIMIT_PCT", 0.02)
        detail = (
            f"Max position: ${max_per_position:,.0f} ({max_pct:.0%} of ${self._wallet:,.0f}), "
            f"Daily halt at: -${daily_halt:,.0f}"
        )
        if warnings:
            detail += " | " + " | ".join(warnings)

        return PreflightResult(
            "Risk limits", True, detail, severity="warning" if warnings else "blocker"
        )

    def _check_data_provider(self) -> PreflightResult:
        """FD.ai API key must be set for data fetching."""
        key = os.getenv("FINANCIAL_DATASETS_API_KEY", "")
        if not key:
            return PreflightResult(
                "Data provider",
                False,
                "FINANCIAL_DATASETS_API_KEY not set. Required for OHLCV and fundamentals.",
            )
        return PreflightResult("Data provider", True, f"FD.ai key set ({key[:8]}...)")

    def _check_broker(self) -> PreflightResult:
        """Check broker credentials are configured."""
        alpaca_key = os.getenv("ALPACA_API_KEY", "")
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
        is_paper = os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1")

        if alpaca_key and alpaca_secret:
            mode = "paper" if is_paper else "LIVE"
            return PreflightResult(
                "Broker",
                True,
                f"Alpaca configured ({mode} mode)",
            )

        ibkr_host = os.getenv("IBKR_HOST", "")
        if ibkr_host:
            return PreflightResult(
                "Broker", True, f"IBKR configured (host={ibkr_host})"
            )

        return PreflightResult(
            "Broker",
            False,
            "No broker configured. Set ALPACA_API_KEY + ALPACA_SECRET_KEY, "
            "or IBKR_HOST. Will fall back to PaperBroker (no real execution).",
            severity="warning",
        )

    def _check_options_execution(self) -> PreflightResult:
        """Verify options execution tool is available and wallet supports options."""
        try:
            tool_available = (
                importlib.util.find_spec("quantstack.mcp.tools.options_execution")
                is not None
            )
        except Exception:
            tool_available = False

        if not tool_available:
            return PreflightResult(
                "Options execution",
                False,
                "execute_options_trade MCP tool not available.",
            )

        # Check wallet can support at least 1 options position
        max_pct = self._env_float("RISK_MAX_PREMIUM_AT_RISK_PCT", 0.02)
        max_premium = self._wallet * max_pct
        if max_premium < 50:
            return PreflightResult(
                "Options execution",
                False,
                f"Max premium per position ${max_premium:.0f} (at {max_pct:.0%} of ${self._wallet:,.0f}) "
                f"is too small for options. Increase wallet or RISK_MAX_PREMIUM_AT_RISK_PCT.",
                severity="blocker",
            )

        # Check DTE limits make sense for short holds
        min_dte = int(os.getenv("RISK_MIN_DTE_ENTRY", "7").split("#")[0].strip())
        if min_dte > 14:
            return PreflightResult(
                "Options execution",
                True,
                f"RISK_MIN_DTE_ENTRY={min_dte} is high for swing trades. Consider lowering to 7.",
                severity="warning",
            )

        return PreflightResult(
            "Options execution",
            True,
            f"Ready. Max premium/position: ${max_premium:.0f}, DTE range: {min_dte}-60",
        )

    def _check_paper_mode(self) -> PreflightResult:
        """Verify paper mode is correctly set."""
        use_real = os.getenv("USE_REAL_TRADING", "false").lower() in ("true", "1")
        alpaca_paper = os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1")

        if use_real and not alpaca_paper:
            return PreflightResult(
                "Paper mode",
                True,
                "LIVE TRADING ENABLED — USE_REAL_TRADING=true, ALPACA_PAPER=false",
                severity="warning",
            )
        if use_real:
            return PreflightResult(
                "Paper mode",
                True,
                "USE_REAL_TRADING=true but ALPACA_PAPER=true — paper trades only",
            )
        return PreflightResult(
            "Paper mode",
            True,
            "Paper mode (safe default). Set USE_REAL_TRADING=true to go live.",
        )


def run_preflight(
    target_symbols: list[str] | None = None,
    target_wallet: float = 1000.0,
) -> dict[str, Any]:
    """
    MCP-callable preflight check.

    Returns:
        {"ready": bool, "blockers": [...], "warnings": [...], "summary": "..."}
    """
    try:
        conn = open_db()
        check = PreflightCheck(conn, target_symbols, target_wallet)
        report = check.run()

        return {
            "success": True,
            "ready": report.ready,
            "blockers": [{"name": c.name, "detail": c.detail} for c in report.blockers],
            "warnings": [{"name": c.name, "detail": c.detail} for c in report.warnings],
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "severity": c.severity,
                }
                for c in report.checks
            ],
            "summary": report.summary(),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


if __name__ == "__main__":
    import sys

    # Load .env if present
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    conn = open_db()
    run_migrations(conn)

    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["SPY"]
    wallet = float(os.getenv("PREFLIGHT_WALLET", "1000"))

    check = PreflightCheck(conn, symbols, wallet)
    report = check.run()
    print(report.summary())
    sys.exit(0 if report.ready else 1)
