# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Intraday Monitor Flow — lightweight hourly regime + P&L check.

Addresses GAP-12 in the gap analysis:
  "Positions change intraday; risk exposure is updated only at day start.
   Market regime can shift within hours (VIX spike, news event).
   Decision latency of daily bars means reacting to yesterday's regime."

Deliberately NOT a full TradingDayFlow:
  - No LLM calls — deterministic regime check + metrics only
  - No CrewAI — runs in seconds, not minutes
  - No new orders — monitoring and alerting only
  - Safe to run every 30–60 minutes via cron or scheduled API call

What it does each run:
  1. Update position mark-to-market prices (from DataProvider)
  2. Re-run RegimeDetectorAgent on all held symbols
  3. Detect regime reversals vs. entry regime (flips that invalidate the thesis)
  4. Compute intraday P&L and compare vs. daily loss limit
  5. Run AlphaMonitor + DegradationDetector checks
  6. Post Discord alert if any action items detected

Usage:
    flow = IntradayMonitorFlow()
    report = flow.run()
    # report.action_items → list of strings for ops review
    # report.regime_reversals → symbols where regime has flipped
    # report.discord_alert_sent → bool

Cron example (every 30 minutes during trading hours):
    # Add to system crontab or use CronCreate tool:
    # */30 9-16 * * 1-5 python -m quant_pod.flows.intraday_monitor_flow
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

import requests
from loguru import logger

from quantstack.agents.regime_detector import RegimeDetectorAgent
from quantstack.audit.decision_log import get_decision_log
from quantstack.audit.models import AuditQuery
from quantstack.execution.portfolio_state import get_portfolio_state
from quantstack.execution.risk_gate import get_risk_gate
from quantstack.monitoring.alpha_monitor import get_alpha_monitor
from quantstack.monitoring.degradation_detector import get_degradation_detector

# =============================================================================
# REPORT MODEL
# =============================================================================


@dataclass
class RegimeChange:
    """A detected regime reversal on a held position."""

    symbol: str
    entry_regime: str  # Regime when position was opened (from audit log)
    current_regime: str  # Current detected regime
    entry_confidence: float
    current_confidence: float
    position_side: str  # "long" or "short"
    action_hint: str  # Human-readable recommended action


@dataclass
class IntradayReport:
    """Full intraday monitoring run result."""

    run_at: datetime
    duration_seconds: float

    # Portfolio metrics
    open_positions: list[dict]
    intraday_pnl: float
    intraday_pnl_pct: float
    daily_loss_pct: float  # As fraction of equity (negative = loss)
    daily_loss_limit_pct: float  # From RiskGate config

    # Regime status
    regime_checks: dict[str, dict]  # symbol → current regime
    regime_reversals: list[RegimeChange]

    # Degradation status (from AlphaMonitor + DegradationDetector)
    alpha_status: str  # "clean" | "warning" | "critical"
    degradation_status: str  # "clean" | "warning" | "critical" | "insufficient_data"

    # Findings and recommended actions
    action_items: list[str] = field(default_factory=list)
    discord_alert_sent: bool = False

    @property
    def requires_attention(self) -> bool:
        return bool(self.action_items) or bool(self.regime_reversals)

    @property
    def overall_status(self) -> str:
        if self.degradation_status == "critical" or self.alpha_status == "critical":
            return "critical"
        if (
            self.regime_reversals
            or self.degradation_status == "warning"
            or self.alpha_status == "warning"
        ):
            return "warning"
        return "clean"


# =============================================================================
# FLOW
# =============================================================================


class IntradayMonitorFlow:
    """
    Lightweight intraday monitoring — runs in seconds, no LLM.

    Can be called:
    - Directly: ``IntradayMonitorFlow().run()``
    - Via API: ``GET /monitor/intraday``
    - Via cron: ``python -m quant_pod.flows.intraday_monitor_flow``

    Design constraints:
    - Must complete in < 30 seconds (no LLM, no full crew)
    - Must not submit orders — read-only relative to broker state
    - Must not crash if DataProvider is unavailable (graceful degradation)
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        min_regime_confidence: float = 0.6,
    ) -> None:
        """
        Args:
            webhook_url: Discord webhook URL. Falls back to DISCORD_WEBHOOK_URL env var.
            min_regime_confidence: Only flag regime reversals if current detection
                                   confidence exceeds this threshold (avoid noise on
                                   low-confidence regime reads).
        """
        self._webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self._min_confidence = min_regime_confidence

    def run(self) -> IntradayReport:
        """Execute the full intraday monitoring cycle."""
        start = datetime.now()
        logger.info("[INTRADAY] Starting intraday monitor cycle")

        portfolio = self._get_portfolio()
        positions = portfolio.get("positions", [])
        snapshot = portfolio.get("snapshot", {})

        # 1. Price update + P&L computation
        intraday_pnl, intraday_pnl_pct = self._compute_intraday_pnl(snapshot)
        daily_loss_pct = snapshot.get("daily_pnl", 0.0) / max(
            snapshot.get("total_equity", 1.0), 1.0
        )
        daily_loss_limit = self._get_daily_loss_limit()

        # 2. Regime check on all held symbols
        held_symbols = [p["symbol"] for p in positions]
        regime_checks = self._check_regimes(held_symbols)

        # 3. Regime reversal detection
        reversals = self._detect_reversals(positions, regime_checks)

        # 4. Alpha / degradation status
        alpha_status = self._run_alpha_check()
        degradation_status = self._run_degradation_check()

        # 5. Compile action items
        action_items = self._compile_action_items(
            daily_loss_pct=daily_loss_pct,
            daily_loss_limit=daily_loss_limit,
            reversals=reversals,
            alpha_status=alpha_status,
            degradation_status=degradation_status,
        )

        duration = (datetime.now() - start).total_seconds()

        report = IntradayReport(
            run_at=start,
            duration_seconds=round(duration, 2),
            open_positions=positions,
            intraday_pnl=round(intraday_pnl, 2),
            intraday_pnl_pct=round(intraday_pnl_pct, 4),
            daily_loss_pct=round(daily_loss_pct, 4),
            daily_loss_limit_pct=daily_loss_limit,
            regime_checks=regime_checks,
            regime_reversals=reversals,
            alpha_status=alpha_status,
            degradation_status=degradation_status,
            action_items=action_items,
        )

        # 6. Discord alert if anything requires attention
        if report.requires_attention:
            self._send_discord_alert(report)
            report.discord_alert_sent = self._webhook_url is not None

        log_level = "warning" if report.requires_attention else "info"
        getattr(logger, log_level)(
            f"[INTRADAY] Cycle complete in {duration:.1f}s | "
            f"status={report.overall_status} | "
            f"{len(positions)} positions | "
            f"P&L={intraday_pnl_pct:+.2%} | "
            f"{len(reversals)} regime reversals | "
            f"{len(action_items)} action items"
        )

        return report

    # -------------------------------------------------------------------------
    # Step implementations
    # -------------------------------------------------------------------------

    def _get_portfolio(self) -> dict:
        """Load current portfolio state."""
        try:
            ps = get_portfolio_state()
            snapshot = ps.get_snapshot()
            positions = [p.model_dump() for p in ps.get_positions()]
            return {"snapshot": snapshot.model_dump(), "positions": positions}
        except Exception as e:
            logger.warning(f"[INTRADAY] Portfolio load failed: {e}")
            return {"snapshot": {}, "positions": []}

    def _compute_intraday_pnl(self, snapshot: dict) -> tuple[float, float]:
        """Compute intraday P&L from portfolio snapshot."""
        daily_pnl = snapshot.get("daily_pnl", 0.0)
        equity = max(snapshot.get("total_equity", 1.0), 1.0)
        return daily_pnl, daily_pnl / equity

    def _get_daily_loss_limit(self) -> float:
        """Read daily loss limit from RiskGate config."""
        try:
            limits = get_risk_gate().limits
            return getattr(limits, "daily_loss_limit_pct", 0.02)
        except Exception:
            return float(os.getenv("RISK_DAILY_LOSS_LIMIT_PCT", "0.02"))

    def _check_regimes(self, symbols: list[str]) -> dict[str, dict]:
        """Run RegimeDetectorAgent on all held symbols. Never throws."""
        detector = RegimeDetectorAgent()
        regimes = {}
        for symbol in symbols:
            try:
                result = detector.detect_regime(symbol)
                regimes[symbol] = result
            except Exception as e:
                logger.warning(f"[INTRADAY] Regime check failed for {symbol}: {e}")
                regimes[symbol] = {"success": False, "error": str(e)}
        return regimes

    def _detect_reversals(
        self,
        positions: list[dict],
        current_regimes: dict[str, dict],
    ) -> list[RegimeChange]:
        """
        Compare current regime to regime at time of position entry.

        Entry regime is read from the most recent audit log event for each
        symbol with event_type="super_trader_decision". This is where
        TradingDayFlow stores the regime context at analysis time.

        A reversal is detected when:
          - Entry was trending_up AND current is trending_down/ranging (for longs)
          - Entry was trending_down AND current is trending_up/ranging (for shorts)
        """
        reversals = []
        entry_regimes = self._load_entry_regimes([p["symbol"] for p in positions])

        for pos in positions:
            symbol = pos["symbol"]
            current = current_regimes.get(symbol, {})
            if not current.get("success"):
                continue

            current_trend = current.get("trend_regime", "unknown")
            current_conf = current.get("confidence", 0.0)
            if current_conf < self._min_confidence:
                continue  # Too uncertain to act on

            entry = entry_regimes.get(symbol, {})
            entry_trend = entry.get("trend", "unknown")
            side = pos.get("side", "long")

            reversal = self._is_reversal(entry_trend, current_trend, side)
            if reversal:
                action_hint = self._reversal_action_hint(
                    side, entry_trend, current_trend
                )
                reversals.append(
                    RegimeChange(
                        symbol=symbol,
                        entry_regime=entry_trend,
                        current_regime=current_trend,
                        entry_confidence=entry.get("confidence", 0.0),
                        current_confidence=current_conf,
                        position_side=side,
                        action_hint=action_hint,
                    )
                )
                logger.warning(
                    f"[INTRADAY] REGIME REVERSAL {symbol}: "
                    f"{entry_trend} → {current_trend} ({side} position)"
                )

        return reversals

    @staticmethod
    def _is_reversal(entry_trend: str, current_trend: str, side: str) -> bool:
        """Return True if current regime invalidates the original thesis."""
        if entry_trend == "unknown" or current_trend == "unknown":
            return False
        if side == "long":
            return entry_trend == "trending_up" and current_trend in (
                "trending_down",
                "ranging",
            )
        if side == "short":
            return entry_trend == "trending_down" and current_trend in (
                "trending_up",
                "ranging",
            )
        return False

    @staticmethod
    def _reversal_action_hint(side: str, entry_trend: str, current_trend: str) -> str:
        if side == "long":
            return (
                f"Long position was entered in {entry_trend} regime. "
                f"Regime is now {current_trend}. "
                "Consider reducing or closing position — trend thesis no longer valid."
            )
        return (
            f"Short position was entered in {entry_trend} regime. "
            f"Regime is now {current_trend}. "
            "Consider covering — counter-trend risk increasing."
        )

    def _load_entry_regimes(self, symbols: list[str]) -> dict[str, dict]:
        """
        Load the regime context recorded at position-entry time from audit log.

        Falls back to empty dict if audit log is unavailable — reversal detection
        is disabled for that symbol (avoids false positives on missing data).
        """
        entry_regimes: dict[str, dict] = {}
        try:
            log = get_decision_log()
            for symbol in symbols:
                events = log.query(
                    AuditQuery(
                        symbol=symbol,
                        event_type="super_trader_decision",
                        limit=1,
                    )
                )
                if events:
                    # The regime at entry is stored in output_structured
                    output = events[0].output_structured or {}
                    entry_regimes[symbol] = output.get("regime", {})
        except Exception as e:
            logger.debug(f"[INTRADAY] Could not load entry regimes from audit: {e}")
        return entry_regimes

    def _run_alpha_check(self) -> str:
        """Run AlphaMonitor and return overall status string."""
        try:
            monitor = get_alpha_monitor()
            # Suppress Discord from here — intraday_monitor handles notification
            monitor._webhook_url = None
            report = monitor.check_all_agents()
            return report.overall_status
        except Exception as e:
            logger.debug(f"[INTRADAY] Alpha check failed: {e}")
            return "unknown"

    def _run_degradation_check(self) -> str:
        """Run DegradationDetector and return overall status string."""
        try:
            detector = get_degradation_detector()
            reports = detector.check_all()
            if not reports:
                return "insufficient_data"
            # Return worst status across all strategies
            severity_order = {
                "critical": 0,
                "warning": 1,
                "clean": 2,
                "insufficient_data": 3,
            }
            worst = min(reports, key=lambda r: severity_order.get(r.status.value, 99))
            return worst.status.value
        except Exception as e:
            logger.debug(f"[INTRADAY] Degradation check failed: {e}")
            return "unknown"

    def _compile_action_items(
        self,
        daily_loss_pct: float,
        daily_loss_limit: float,
        reversals: list[RegimeChange],
        alpha_status: str,
        degradation_status: str,
    ) -> list[str]:
        """Build a prioritised list of action items for the ops dashboard."""
        items = []

        # P&L-based alerts
        if daily_loss_pct < -(daily_loss_limit * 0.80):
            items.append(
                f"DAILY LOSS ALERT: Current loss {daily_loss_pct:.1%} is approaching "
                f"the {daily_loss_limit:.1%} daily limit. "
                f"Consider reducing exposure before limit is breached."
            )

        if daily_loss_pct < -(daily_loss_limit * 0.50):
            items.append(
                f"WARNING: Daily loss {daily_loss_pct:.1%} is 50%+ of limit. "
                "Review all open positions."
            )

        # Regime reversals
        for rev in reversals:
            items.append(f"REGIME REVERSAL: {rev.symbol} — {rev.action_hint}")

        # Alpha / degradation
        if alpha_status == "critical":
            items.append(
                "ALPHA CRITICAL: One or more agents show negative rolling IC. "
                "Check /skills/degradation for details."
            )
        elif alpha_status == "warning":
            items.append(
                "ALPHA WARNING: IC decay detected. Review /skills/degradation."
            )

        if degradation_status == "critical":
            items.append(
                "STRATEGY CRITICAL: Live performance severely below backtest expectations. "
                "Reduce position sizes immediately. Check /monitor/degradation."
            )
        elif degradation_status == "warning":
            items.append(
                "STRATEGY WARNING: IS/OOS performance gap widening. "
                "Consider 50% size reduction. Check /monitor/degradation."
            )

        return items

    # -------------------------------------------------------------------------
    # Discord notification
    # -------------------------------------------------------------------------

    def _send_discord_alert(self, report: IntradayReport) -> None:
        """Post intraday monitoring summary to Discord webhook."""
        if not self._webhook_url:
            return

        color_map = {"critical": 0xFF0000, "warning": 0xFFAA00, "clean": 0x2ECC71}
        color = color_map.get(report.overall_status, 0x95A5A6)

        fields = [
            {
                "name": "Open Positions",
                "value": str(len(report.open_positions)),
                "inline": True,
            },
            {
                "name": "Intraday P&L",
                "value": f"{report.intraday_pnl_pct:+.2%}",
                "inline": True,
            },
            {
                "name": "Daily Loss",
                "value": f"{report.daily_loss_pct:+.2%} / {report.daily_loss_limit_pct:.0%} limit",
                "inline": True,
            },
            {
                "name": "Regime Reversals",
                "value": str(len(report.regime_reversals)),
                "inline": True,
            },
            {
                "name": "Alpha Status",
                "value": report.alpha_status.upper(),
                "inline": True,
            },
            {
                "name": "Degradation",
                "value": report.degradation_status.upper(),
                "inline": True,
            },
        ]

        description = (
            "\n".join(f"• {item}" for item in report.action_items[:5])
            or "No action items."
        )

        embed = {
            "title": f"QuantPod Intraday Monitor — {report.overall_status.upper()}",
            "description": description,
            "color": color,
            "fields": fields,
            "timestamp": report.run_at.isoformat(),
            "footer": {"text": f"Completed in {report.duration_seconds:.1f}s"},
        }

        try:
            resp = requests.post(
                self._webhook_url,
                json={"username": "QuantPod Monitor", "embeds": [embed]},
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"[INTRADAY] Discord webhook failed: {e}")


# =============================================================================
# CLI entry point (for cron scheduling)
# =============================================================================


def main() -> None:
    """
    Standalone entry point for cron execution.

    Usage:
        python -m quant_pod.flows.intraday_monitor_flow
    """
    flow = IntradayMonitorFlow()
    report = flow.run()
    print(
        json.dumps(
            {
                "status": report.overall_status,
                "run_at": report.run_at.isoformat(),
                "action_items": report.action_items,
                "regime_reversals": [
                    {"symbol": r.symbol, "hint": r.action_hint}
                    for r in report.regime_reversals
                ],
                "intraday_pnl_pct": report.intraday_pnl_pct,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
