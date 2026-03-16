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
from datetime import datetime
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
    # Sentinel file
    # -------------------------------------------------------------------------

    def _write_sentinel(self) -> None:
        self.SENTINEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.SENTINEL_FILE, "w") as f:
            f.write(f"triggered_at={self._status.triggered_at}\nreason={self._status.reason}\n")

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
                triggered_at=datetime.fromisoformat(data.get("triggered_at", str(datetime.now()))),
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
