"""Kill switch auto-recovery and escalation managers.

AutoRecoveryManager: Automatic reset for broker failures only. Other conditions
(drawdown, drift, SPY halt) carry signal about model/market state and must
remain manual.

KillSwitchEscalationManager: Tiered notification escalation (Discord/email)
for all kill switch types.

SizingRampBack: Gradual return to full position sizing after auto-reset.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from quantstack.db import db_conn
from quantstack.execution.kill_switch import KillSwitch

logger = logging.getLogger(__name__)

# Broker failure trigger reasons start with this prefix (set by AutoTriggerMonitor)
_BROKER_FAILURE_PREFIX = "Auto-trigger:"
_BROKER_FAILURE_PATTERN = "consecutive broker"


def _is_broker_failure(reason: str | None) -> bool:
    """Return True if the kill switch reason indicates a broker failure."""
    if not reason:
        return False
    lower = reason.lower()
    return _BROKER_FAILURE_PATTERN in lower


# =============================================================================
# AUTO RECOVERY MANAGER
# =============================================================================


class AutoRecoveryManager:
    """Manages tiered recovery for broker failure kill switch triggers.

    Only broker failures qualify for auto-recovery. All other kill switch
    conditions (drawdown, drift, SPY halt) require manual reset because
    they carry signal about model or market state.

    Recovery timeline:
      0 min  - Kill switch triggers, Discord alert fires
      5 min  - Auto-investigate: call broker health check endpoint
      15 min - If broker responsive AND trigger was broker failure: auto-reset
               with sizing_scalar=0.5
      15 min - If broker still down: do NOT reset, continue escalation

    Safety: MAX_AUTO_RESETS_PER_DAY = 2. Third trigger stays halted.
    """

    MAX_AUTO_RESETS_PER_DAY = int(os.getenv("MAX_AUTO_RESETS_PER_DAY", "2"))
    INVESTIGATION_DELAY = timedelta(minutes=5)
    RESET_DELAY = timedelta(minutes=15)
    BACKOFF_WINDOW = timedelta(minutes=5)

    def __init__(self, kill_switch: KillSwitch) -> None:
        self._ks = kill_switch
        self._last_reset_at: datetime | None = None

    def check(self) -> None:
        """Run one recovery check cycle (called by supervisor every 5 min)."""
        if not self._ks.is_active():
            self._last_reset_at = None
            return

        status = self._ks.status()
        if not _is_broker_failure(status.reason):
            return

        elapsed = datetime.now() - status.triggered_at
        if elapsed < self.INVESTIGATION_DELAY:
            return

        # Investigation phase (5-15 min)
        if elapsed < self.RESET_DELAY:
            logger.info(
                "[AutoRecovery] Investigating broker failure (elapsed: %s). "
                "Running broker health check.",
                elapsed,
            )
            responsive = self._check_broker_health()
            logger.info("[AutoRecovery] Broker health check: %s", "responsive" if responsive else "down")
            return

        # Reset phase (>= 15 min)
        # Back-off: don't reset if we just reset recently
        if self._last_reset_at and (datetime.now() - self._last_reset_at) < self.BACKOFF_WINDOW:
            logger.warning(
                "[AutoRecovery] Skipping reset — last reset was %s ago (back-off window: %s)",
                datetime.now() - self._last_reset_at, self.BACKOFF_WINDOW,
            )
            return

        responsive = self._check_broker_health()
        if not responsive:
            logger.info("[AutoRecovery] Broker still down at %s. Not resetting.", elapsed)
            return

        # Check daily reset cap
        if self._get_daily_reset_count() >= self.MAX_AUTO_RESETS_PER_DAY:
            logger.warning(
                "[AutoRecovery] Daily reset cap reached (%d). Manual intervention required.",
                self.MAX_AUTO_RESETS_PER_DAY,
            )
            return

        # Perform auto-reset
        logger.info("[AutoRecovery] Broker responsive. Initiating auto-reset.")
        self._ks.reset(
            reset_by="auto_recovery",
            reason=f"Broker failure auto-recovery: broker responsive after {elapsed}",
        )
        self._last_reset_at = datetime.now()
        self._increment_daily_reset_count()

        # Set sizing ramp-back
        SizingRampBack().initialize_after_reset()

    def _check_broker_health(self) -> bool:
        """Check if the broker API is responsive. Returns True if healthy."""
        try:
            from quantstack.execution.broker import get_broker
            broker = get_broker()
            broker.get_account()
            return True
        except Exception as exc:
            logger.debug("[AutoRecovery] Broker health check failed: %s", exc)
            return False

    def _get_daily_reset_count(self) -> int:
        """Read today's auto-reset count from system_state."""
        key = f"auto_resets_{datetime.now().strftime('%Y-%m-%d')}"
        try:
            with db_conn() as conn:
                row = conn.execute(
                    "SELECT value FROM system_state WHERE key = %s",
                    (key,),
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception as exc:
            logger.warning("[AutoRecovery] Failed to read daily reset count: %s", exc)
            return 0

    def _increment_daily_reset_count(self) -> None:
        """Increment today's auto-reset count in system_state."""
        key = f"auto_resets_{datetime.now().strftime('%Y-%m-%d')}"
        try:
            with db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO system_state (key, value, updated_at)
                    VALUES (%s, '1', NOW())
                    ON CONFLICT (key) DO UPDATE
                    SET value = (COALESCE(system_state.value::int, 0) + 1)::text,
                        updated_at = NOW()
                    """,
                    (key,),
                )
        except Exception as exc:
            logger.warning("[AutoRecovery] Failed to increment daily reset count: %s", exc)


# =============================================================================
# SIZING RAMP-BACK
# =============================================================================


class SizingRampBack:
    """Gradual return to full position sizing after auto-reset.

    Ramp schedule:
      Auto-reset: sizing_scalar = 0.5, successful_cycles = 0
      After 1 successful cycle: 0.75
      After 2 successful cycles: 1.0
      After 3 successful cycles: keys removed (fully recovered)

    If kill switch re-triggers during ramp, reset to 0.5.
    """

    RAMP_SCHEDULE = {0: 0.5, 1: 0.75, 2: 1.0}

    def initialize_after_reset(self) -> None:
        """Set initial ramp-back state after an auto-reset."""
        try:
            with db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO system_state (key, value, updated_at) VALUES
                    ('kill_switch_sizing_scalar', '0.5', NOW()),
                    ('kill_switch_successful_cycles', '0', NOW())
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                    """,
                )
        except Exception as exc:
            logger.warning("[SizingRampBack] Failed to initialize: %s", exc)

    def reset_on_retrigger(self) -> None:
        """Reset ramp state when kill switch re-triggers during ramp."""
        self.initialize_after_reset()

    def record_successful_cycle(self) -> None:
        """Increment successful cycle count and adjust sizing scalar."""
        try:
            with db_conn() as conn:
                row = conn.execute(
                    "SELECT value FROM system_state WHERE key = 'kill_switch_successful_cycles'"
                ).fetchone()

            if row is None:
                return  # No ramp in progress

            cycles = int(row[0]) + 1

            if cycles >= 3:
                # Fully recovered — remove ramp state
                with db_conn() as conn:
                    conn.execute(
                        "DELETE FROM system_state WHERE key IN "
                        "('kill_switch_sizing_scalar', 'kill_switch_successful_cycles')"
                    )
                logger.info("[SizingRampBack] Fully recovered — sizing scalar removed.")
                return

            scalar = self.RAMP_SCHEDULE.get(cycles, 1.0)
            with db_conn() as conn:
                conn.execute(
                    """
                    UPDATE system_state SET value = %s, updated_at = NOW()
                    WHERE key = 'kill_switch_successful_cycles'
                    """,
                    (str(cycles),),
                )
                conn.execute(
                    """
                    UPDATE system_state SET value = %s, updated_at = NOW()
                    WHERE key = 'kill_switch_sizing_scalar'
                    """,
                    (str(scalar),),
                )
            logger.info(
                "[SizingRampBack] Cycle %d complete — sizing scalar now %.2f",
                cycles, scalar,
            )
        except Exception as exc:
            logger.warning("[SizingRampBack] Failed to record cycle: %s", exc)

    @staticmethod
    def get_current_scalar() -> float:
        """Read current sizing scalar from system_state. Returns 1.0 if not set."""
        try:
            with db_conn() as conn:
                row = conn.execute(
                    "SELECT value FROM system_state WHERE key = 'kill_switch_sizing_scalar'"
                ).fetchone()
            return float(row[0]) if row else 1.0
        except Exception:
            return 1.0


# =============================================================================
# ESCALATION MANAGER
# =============================================================================


class KillSwitchEscalationManager:
    """Sends progressively urgent notifications while the kill switch is active.

    Escalation schedule:
      0 min  -> Discord: trigger reason + open positions + unrealized P&L
      4 hours -> Enhanced Discord (or email if configured)
      24 hours -> Emergency Discord: all-caps header, full position summary

    Stores last_escalation_tier in system_state to avoid duplicate sends.
    Resets when kill switch is cleared.
    """

    TIER_2_DELAY = timedelta(hours=4)
    TIER_3_DELAY = timedelta(hours=24)

    def __init__(self, kill_switch: KillSwitch) -> None:
        self._ks = kill_switch

    def check(self) -> None:
        """Run one escalation check cycle (called by supervisor every 5 min)."""
        if not self._ks.is_active():
            self._clear_tier()
            return

        status = self._ks.status()
        elapsed = datetime.now() - status.triggered_at
        current_tier = self._get_current_tier()

        if elapsed >= self.TIER_3_DELAY and current_tier < 3:
            self._send_discord(
                f"**🚨 EMERGENCY: KILL SWITCH ACTIVE FOR {elapsed} 🚨**\n\n"
                f"Reason: {status.reason}\n"
                f"Triggered: {status.triggered_at}\n"
                f"IMMEDIATE INVESTIGATION REQUIRED.",
                tier=3,
            )
        elif elapsed >= self.TIER_2_DELAY and current_tier < 2:
            self._send_discord(
                f"**Kill Switch Escalation (4h+)**\n\n"
                f"Reason: {status.reason}\n"
                f"Triggered: {status.triggered_at}\n"
                f"Elapsed: {elapsed}\n\n"
                f"Action required: Log in and investigate. "
                f"Run `switch.reset(reset_by='manual', reason='...')` "
                f"after confirming root cause.",
                tier=2,
            )
        elif current_tier < 1:
            self._send_discord(
                f"**Kill Switch Activated**\n\n"
                f"Reason: {status.reason}\n"
                f"Triggered: {status.triggered_at}\n"
                f"Auto-recovery eligible: {_is_broker_failure(status.reason)}",
                tier=1,
            )

    def _send_discord(self, message: str, tier: int) -> None:
        """Send a Discord notification and update the escalation tier."""
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            logger.warning("[Escalation] DISCORD_WEBHOOK_URL not set — cannot send tier %d alert", tier)
            self._set_tier(tier)
            return

        try:
            import httpx
            httpx.post(webhook_url, json={"content": message}, timeout=10)
            logger.info("[Escalation] Tier %d notification sent to Discord.", tier)
        except Exception as exc:
            logger.warning("[Escalation] Discord send failed (tier %d): %s", tier, exc)
            return  # Don't advance tier if send failed

        self._set_tier(tier)

    def _get_current_tier(self) -> int:
        """Read the current escalation tier from system_state."""
        try:
            with db_conn() as conn:
                row = conn.execute(
                    "SELECT value FROM system_state WHERE key = 'kill_switch_last_escalation_tier'"
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def _set_tier(self, tier: int) -> None:
        """Write the escalation tier to system_state."""
        try:
            with db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO system_state (key, value, updated_at)
                    VALUES ('kill_switch_last_escalation_tier', %s, NOW())
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                    """,
                    (str(tier),),
                )
        except Exception as exc:
            logger.warning("[Escalation] Failed to set tier: %s", exc)

    def _clear_tier(self) -> None:
        """Remove escalation tier tracking when kill switch is cleared."""
        try:
            with db_conn() as conn:
                conn.execute(
                    "DELETE FROM system_state WHERE key = 'kill_switch_last_escalation_tier'"
                )
        except Exception:
            pass  # Non-critical
