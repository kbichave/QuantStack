"""
KillSwitch — file-sentinel based emergency halt for the execution engine.

Design
------
The kill switch monitors for the presence of a sentinel file (default:
``/tmp/KILL_TRADING``).  When the file exists the switch is ACTIVE and all
order submission is blocked.  When the file is absent it is INACTIVE and
orders flow normally.

Why a file, not a flag in memory?
  A flag in memory requires the operator to have a Python handle to the
  running process.  A file can be created by any shell one-liner, a cron job,
  an ops runbook, or a monitoring script — even when the process is wedged
  in a tight loop.  On process restart the file persists, so the engine stays
  halted until an operator deliberately removes it.

Activation:
    ``touch /tmp/KILL_TRADING``  →  orders blocked immediately on next check

Deactivation:
    ``rm /tmp/KILL_TRADING``     →  orders resume on next check

Programmatic use:
    ``KillSwitch.activate()``    →  writes the sentinel file
    ``KillSwitch.deactivate()``  →  removes the sentinel file
    ``KillSwitch.is_active()``   →  True when halted
    ``KillSwitch.check()``       →  raises KillSwitchError if active

Thread safety: os.path.exists() is atomic enough for a read-only check;
writes and removes use os.makedirs / os.remove which are also atomic on Linux.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger


class KillSwitchError(RuntimeError):
    """Raised when an order submission attempt is made while the kill switch is active."""


class KillSwitch:
    """File-sentinel kill switch.  All methods are class-level (no instance needed).

    Args:
        sentinel_path: Path to the sentinel file.  Override in tests to avoid
                       leaving state in /tmp.
    """

    _DEFAULT_PATH = Path("/tmp/KILL_TRADING")

    def __init__(self, sentinel_path: str | Path | None = None) -> None:
        self._path = Path(sentinel_path) if sentinel_path else self._DEFAULT_PATH

    # ── State ─────────────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        """Return True when the kill switch is engaged (sentinel file exists)."""
        return self._path.exists()

    def check(self) -> None:
        """Raise KillSwitchError if the kill switch is currently active.

        Call this at every order submission point.
        """
        if self.is_active():
            raise KillSwitchError(
                f"Kill switch is ACTIVE — order submission blocked. "
                f"Remove {self._path} to resume trading."
            )

    # ── Control ───────────────────────────────────────────────────────────────

    def activate(self, reason: str = "") -> None:
        """Engage the kill switch by writing the sentinel file.

        Args:
            reason: Optional human-readable reason logged alongside activation.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(reason or "kill switch activated\n")
            logger.critical(
                f"[KillSwitch] ACTIVATED — all order submission halted. "
                f"Sentinel: {self._path}" + (f" | reason: {reason}" if reason else "")
            )
        except OSError as exc:
            logger.error(f"[KillSwitch] Failed to write sentinel file: {exc}")
            raise

    def deactivate(self) -> None:
        """Disengage the kill switch by removing the sentinel file."""
        try:
            if self._path.exists():
                self._path.unlink()
                logger.info(
                    f"[KillSwitch] DEACTIVATED — trading resumed. Sentinel removed: {self._path}"
                )
            else:
                logger.debug(
                    "[KillSwitch] deactivate() called but sentinel was not present"
                )
        except OSError as exc:
            logger.error(f"[KillSwitch] Failed to remove sentinel file: {exc}")
            raise

    def status(self) -> dict:
        """Return a status dict suitable for MCP tool responses."""
        active = self.is_active()
        result: dict = {"active": active, "sentinel_path": str(self._path)}
        if active:
            try:
                result["reason"] = self._path.read_text().strip()
            except OSError:
                result["reason"] = ""
        return result
