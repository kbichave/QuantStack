# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Emergency kill switch — halts all agents and closes all positions.

One call stops everything. Designed to be callable from a CLI,
a monitoring alert, or a keyboard interrupt.

Usage:
    switch = get_kill_switch()

    # Trigger emergency halt
    switch.trigger(reason="Daily loss limit breached — manual halt")

    # Check state before submitting any order
    if switch.is_active():
        raise RuntimeError("Kill switch is active — no orders allowed")

    # Check from a monitoring script
    status = switch.status()
    print(status.triggered_at, status.reason)
"""

from __future__ import annotations

import os
import signal
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

from loguru import logger
from pydantic import BaseModel

# =============================================================================
# DATA MODELS
# =============================================================================


class KillSwitchStatus(BaseModel):
    """Current state of the kill switch."""

    active: bool = False
    triggered_at: datetime | None = None
    reason: str | None = None
    reset_at: datetime | None = None
    reset_by: str | None = None


# =============================================================================
# KILL SWITCH
# =============================================================================


class KillSwitch:
    """
    Emergency halt mechanism.

    State is written to a sentinel file so it survives process restarts
    and can be checked by monitoring scripts independently.

    On trigger():
      1. Write sentinel file with reason + timestamp
      2. Log CRITICAL to stderr
      3. Mark all open orders as cancelled in PaperBroker (best-effort)
      4. Optionally call broker.close_all_positions() if a closer is registered

    On is_active():
      1. Check in-memory flag (fast path)
      2. Fall back to sentinel file check (survives restart)
    """

    SENTINEL_FILE = Path(
        os.getenv("KILL_SWITCH_SENTINEL", "~/.quant_pod/KILL_SWITCH_ACTIVE")
    ).expanduser()

    _lock = Lock()

    def __init__(self):
        self._status = KillSwitchStatus()
        self._position_closer = None  # Optional[Callable[[], None]]
        # Check if a previous kill was never reset
        if self.SENTINEL_FILE.exists():
            self._load_from_file()
            if self._status.active:
                logger.critical(
                    "[KILL SWITCH] Active sentinel found from previous session — "
                    "trading is HALTED. Call reset() to resume."
                )

    # -------------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------------

    def trigger(self, reason: str = "Manual trigger") -> None:
        """
        Activate the kill switch. Halts all trading immediately.

        This is irreversible until reset() is explicitly called.
        """
        with self._lock:
            self._status = KillSwitchStatus(
                active=True,
                triggered_at=datetime.now(),
                reason=reason,
            )
            self._write_sentinel()

        logger.critical(
            f"\n{'=' * 70}\n"
            f"  *** KILL SWITCH ACTIVATED ***\n"
            f"  Reason: {reason}\n"
            f"  Time:   {self._status.triggered_at}\n"
            f"  All trading is HALTED. Call reset() to resume.\n"
            f"{'=' * 70}"
        )

        # Best-effort: close all positions via registered closer
        if self._position_closer is not None:
            try:
                logger.info("[KILL SWITCH] Closing all positions...")
                self._position_closer()
                logger.info("[KILL SWITCH] All positions closed")
            except Exception as e:
                logger.error(f"[KILL SWITCH] Position closer failed: {e}")

    def reset(self, reset_by: str = "manual") -> None:
        """
        Deactivate the kill switch. Allows trading to resume.

        Should only be called after the root cause is investigated.
        """
        with self._lock:
            self._status.active = False
            self._status.reset_at = datetime.now()
            self._status.reset_by = reset_by
            # Atomic delete: no existence check needed — unlink raises FileNotFoundError
            # if already gone, which we catch. This avoids a TOCTOU race where another
            # process re-creates the file between our exists() check and unlink().
            try:
                self.SENTINEL_FILE.unlink()
            except FileNotFoundError:
                pass  # Already absent — that's fine

        logger.info(
            f"[KILL SWITCH] Reset by '{reset_by}' at {self._status.reset_at}. Trading may resume."
        )

    def is_active(self) -> bool:
        """
        True if kill switch is active.

        Checks in-memory state first, then falls back to sentinel file
        (handles case where switch was triggered in another process).
        """
        if self._status.active:
            return True
        # Check file (cross-process safety)
        if self.SENTINEL_FILE.exists():
            self._load_from_file()
            return self._status.active
        return False

    def guard(self) -> None:
        """
        Raise RuntimeError if kill switch is active.

        Call this at the top of any function that submits orders.
        """
        if self.is_active():
            raise RuntimeError(
                f"Kill switch is ACTIVE — trading halted. "
                f"Reason: {self._status.reason}. "
                f"Triggered at: {self._status.triggered_at}"
            )

    def status(self) -> KillSwitchStatus:
        """Return current kill switch status."""
        return self._status.model_copy()

    # -------------------------------------------------------------------------
    # Position closer registration
    # -------------------------------------------------------------------------

    def register_position_closer(self, closer) -> None:
        """
        Register a callable that closes all open positions.

        Called automatically on trigger().

        Example:
            switch.register_position_closer(lambda: broker.close_all())
        """
        self._position_closer = closer

    # -------------------------------------------------------------------------
    # SIGTERM / SIGINT handler
    # -------------------------------------------------------------------------

    def install_signal_handlers(self) -> None:
        """
        Install SIGTERM and SIGINT handlers that activate the kill switch.

        Call this once at process startup.
        """

        def handler(signum, frame):
            sig_name = signal.Signals(signum).name
            self.trigger(reason=f"Process received {sig_name}")
            sys.exit(1)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        logger.info("[KILL SWITCH] Signal handlers installed (SIGTERM, SIGINT)")

    # -------------------------------------------------------------------------
    # Auto-trigger monitoring (Phase 4.2)
    # -------------------------------------------------------------------------

    def create_auto_trigger(self) -> AutoTriggerMonitor:
        """
        Create an AutoTriggerMonitor bound to this kill switch.

        The monitor tracks failure events and triggers the kill switch
        automatically when thresholds are breached.
        """
        return AutoTriggerMonitor(self)

    # -------------------------------------------------------------------------
    # Sentinel file
    # -------------------------------------------------------------------------

    def _write_sentinel(self) -> None:
        self.SENTINEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.SENTINEL_FILE, "w") as f:
            f.write(
                f"triggered_at={self._status.triggered_at}\nreason={self._status.reason}\n"
            )

    def _load_from_file(self) -> None:
        try:
            lines = self.SENTINEL_FILE.read_text().strip().splitlines()
            data = {}
            for line in lines:
                if "=" in line:
                    k, v = line.split("=", 1)
                    data[k.strip()] = v.strip()
            self._status = KillSwitchStatus(
                active=True,
                triggered_at=datetime.fromisoformat(
                    data.get("triggered_at", str(datetime.now()))
                ),
                reason=data.get("reason", "Unknown — loaded from sentinel file"),
            )
        except Exception as e:
            logger.warning(f"[KILL SWITCH] Could not parse sentinel file: {e}")
            self._status = KillSwitchStatus(
                active=True,
                triggered_at=datetime.now(),
                reason="Unknown — sentinel file present but unreadable",
            )


# Singleton
_kill_switch: KillSwitch | None = None


def get_kill_switch() -> KillSwitch:
    """Get the singleton KillSwitch instance."""
    global _kill_switch
    if _kill_switch is None:
        _kill_switch = KillSwitch()
    return _kill_switch


# =============================================================================
# AUTO TRIGGER MONITOR (Phase 4.2)
# =============================================================================


class AutoTriggerMonitor:
    """
    Automatic kill switch triggers based on operational health signals.

    Tracks events and triggers the kill switch when thresholds are breached.
    Each trigger is independent — any one is sufficient to halt trading.

    Triggers:
      1. Consecutive broker API failures (default: 3)
      2. Market-wide circuit breaker (SPY halted)
      3. Rolling 3-day drawdown > 3x daily loss limit
      4. Model drift on >50% of active strategies

    Usage:
        monitor = get_kill_switch().create_auto_trigger()

        # After each broker call:
        monitor.record_broker_result(success=False, error="timeout")

        # Periodically (every 60s):
        monitor.check_rolling_drawdown(daily_pnls=[-0.5, -0.8, -1.2])
        monitor.check_market_circuit_breaker(spy_halted=True)
        monitor.check_model_drift(drifted_count=5, total_active=8)
    """

    _lock = Lock()

    # Configurable thresholds (override via env)
    MAX_CONSECUTIVE_BROKER_FAILURES = int(os.getenv("KILL_MAX_BROKER_FAILURES", "3"))
    ROLLING_DRAWDOWN_DAYS = 3
    ROLLING_DRAWDOWN_MULTIPLIER = 3.0  # 3x the daily loss limit
    MODEL_DRIFT_THRESHOLD_PCT = 0.50  # >50% of active strategies drifted

    def __init__(self, kill_switch: KillSwitch) -> None:
        self._ks = kill_switch
        self._consecutive_broker_failures = 0
        # Track daily P&L for rolling drawdown (most recent at right)
        self._daily_pnls: deque[float] = deque(maxlen=self.ROLLING_DRAWDOWN_DAYS)

    # -------------------------------------------------------------------------
    # Trigger 1: Consecutive broker API failures
    # -------------------------------------------------------------------------

    def record_broker_result(self, success: bool, error: str = "") -> None:
        """
        Record the outcome of a broker API call.

        Triggers kill switch after MAX_CONSECUTIVE_BROKER_FAILURES consecutive
        failures. A single success resets the counter.
        """
        with self._lock:
            if success:
                if self._consecutive_broker_failures > 0:
                    logger.debug(
                        f"[AUTO TRIGGER] Broker success — resetting failure counter "
                        f"(was {self._consecutive_broker_failures})"
                    )
                self._consecutive_broker_failures = 0
                return

            self._consecutive_broker_failures += 1
            logger.warning(
                f"[AUTO TRIGGER] Broker failure #{self._consecutive_broker_failures}: {error}"
            )

            if (
                self._consecutive_broker_failures
                >= self.MAX_CONSECUTIVE_BROKER_FAILURES
            ):
                self._ks.trigger(
                    reason=(
                        f"Auto-trigger: {self._consecutive_broker_failures} consecutive "
                        f"broker API failures (last: {error})"
                    )
                )

    # -------------------------------------------------------------------------
    # Trigger 2: Market-wide circuit breaker
    # -------------------------------------------------------------------------

    def check_market_circuit_breaker(self, spy_halted: bool) -> None:
        """
        Trigger kill switch if a market-wide circuit breaker is detected.

        SPY being halted is a proxy for Level 1/2/3 circuit breakers.
        """
        if spy_halted and not self._ks.is_active():
            self._ks.trigger(
                reason="Auto-trigger: market-wide circuit breaker detected (SPY halted)"
            )

    # -------------------------------------------------------------------------
    # Trigger 3: Rolling 3-day drawdown
    # -------------------------------------------------------------------------

    def record_daily_pnl(self, pnl_pct: float) -> None:
        """Record today's P&L as a percentage of equity for rolling tracking."""
        with self._lock:
            self._daily_pnls.append(pnl_pct)

    def check_rolling_drawdown(
        self,
        daily_loss_limit_pct: float = 0.02,
        daily_pnls: list[float] | None = None,
    ) -> None:
        """
        Trigger if the rolling 3-day loss exceeds 3x the daily loss limit.

        A daily loss limit of 2% means a 3-day rolling limit of 6%.
        This catches sustained bleeding that individual daily limits miss.

        Args:
            daily_loss_limit_pct: The per-day loss limit (default 2%).
            daily_pnls: Optional explicit P&L list (overrides internal deque).
        """
        with self._lock:
            pnls = (
                list(daily_pnls) if daily_pnls is not None else list(self._daily_pnls)
            )

        if not pnls:
            return

        rolling_loss = abs(min(0, sum(pnls)))
        threshold = daily_loss_limit_pct * self.ROLLING_DRAWDOWN_MULTIPLIER

        if rolling_loss >= threshold and not self._ks.is_active():
            self._ks.trigger(
                reason=(
                    f"Auto-trigger: {self.ROLLING_DRAWDOWN_DAYS}-day rolling drawdown "
                    f"{rolling_loss:.2%} >= threshold {threshold:.2%} "
                    f"({self.ROLLING_DRAWDOWN_MULTIPLIER}x daily limit)"
                )
            )

    # -------------------------------------------------------------------------
    # Trigger 4: Model drift on majority of active strategies
    # -------------------------------------------------------------------------

    def check_model_drift(self, drifted_count: int, total_active: int) -> None:
        """
        Trigger if >50% of active strategies show model drift.

        Drift is detected by DriftDetector or concept_drift checks upstream.
        The caller passes aggregate counts; this method only checks the threshold.
        """
        if total_active <= 0:
            return

        drift_pct = drifted_count / total_active
        if drift_pct > self.MODEL_DRIFT_THRESHOLD_PCT and not self._ks.is_active():
            self._ks.trigger(
                reason=(
                    f"Auto-trigger: model drift detected on {drifted_count}/{total_active} "
                    f"active strategies ({drift_pct:.0%} > {self.MODEL_DRIFT_THRESHOLD_PCT:.0%} threshold)"
                )
            )

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def status(self) -> dict:
        """Return current auto-trigger state for debugging/monitoring."""
        with self._lock:
            return {
                "consecutive_broker_failures": self._consecutive_broker_failures,
                "max_broker_failures": self.MAX_CONSECUTIVE_BROKER_FAILURES,
                "rolling_pnls": list(self._daily_pnls),
                "kill_switch_active": self._ks.is_active(),
            }
