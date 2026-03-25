# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
In-memory signal cache — the boundary between the LLM analysis plane and
the tick executor.

Architecture contract:
  - The MINUTE ANALYST (TradingDayFlow + TradingCrew) WRITES signals here
    after each analysis cycle, with a TTL that expires if analysis stalls.
  - The TICK EXECUTOR reads signals from here in the hot path — no DB,
    no LLM, no I/O.  If a signal is absent or stale, the executor skips.
  - The `signal_state` table is used only for persistence / crash recovery.

Invariants:
  - Expired signals are NEVER executed (staleness is a hard safety property).
  - The cache is eventually consistent with the DB; the DB is authoritative
    on restart.
  - Thread-safe: reads and writes use an RLock.

Usage:
    cache = SignalCache()

    # Analyst writes after crew analysis
    cache.update(TradeSignal(
        symbol="SPY",
        action="BUY",
        confidence=0.78,
        position_size_pct=0.05,
        stop_loss=448.0,
        take_profit=460.0,
        expires_in_seconds=900,   # 15-minute TTL
        session_id=session_id,
    ))

    # Tick executor reads — returns None if absent or stale
    signal = cache.get("SPY")
    if signal is None:
        continue  # No valid signal — hold

    # Check staleness explicitly
    if cache.is_stale("SPY"):
        continue
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Literal

from loguru import logger

from quantstack.db import PgConnection

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TradeSignal:
    """
    A trading signal produced by the LLM analysis plane.

    The tick executor acts on this signal; it must never call back into
    the analysis plane to refresh it.
    """

    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float  # 0.0–1.0
    position_size_pct: float  # fraction of equity (0.0–1.0)
    stop_loss: float | None = None  # absolute price
    take_profit: float | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    session_id: str = ""

    @classmethod
    def create(
        cls,
        symbol: str,
        action: Literal["BUY", "SELL", "HOLD"],
        confidence: float,
        position_size_pct: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        expires_in_seconds: int = 900,  # 15 min default TTL
        session_id: str = "",
    ) -> TradeSignal:
        """Convenience constructor with TTL expressed in seconds."""
        now = datetime.now(UTC)
        return cls(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size_pct=position_size_pct,
            stop_loss=stop_loss,
            take_profit=take_profit,
            generated_at=now,
            expires_at=now + timedelta(seconds=expires_in_seconds),
            session_id=session_id,
        )

    @property
    def is_expired(self) -> bool:
        """True if this signal has passed its expiry time."""
        return datetime.now(UTC) >= self.expires_at

    @property
    def age_seconds(self) -> float:
        """Seconds since this signal was generated."""
        return (datetime.now(UTC) - self.generated_at).total_seconds()


# ---------------------------------------------------------------------------
# Signal cache
# ---------------------------------------------------------------------------


class SignalCache:
    """
    Thread-safe in-memory store of the latest signal per symbol.

    Only the most recent signal per symbol is retained — older signals
    are overwritten on update.  If the analysis plane produces no new
    signal before the TTL expires, the executor sees None.
    """

    def __init__(
        self,
        conn: PgConnection | None = None,
        default_ttl_seconds: int = 900,
    ):
        self._lock = RLock()
        self._signals: dict[str, TradeSignal] = {}
        self._default_ttl = default_ttl_seconds
        self._conn = conn
        # Recover signals from DB on startup (in case of crash restart)
        if conn is not None:
            self._load_from_db()

    # -----------------------------------------------------------------------
    # Write path (analysis plane → cache)
    # -----------------------------------------------------------------------

    def update(self, signal: TradeSignal) -> None:
        """
        Store or replace the signal for a symbol.

        Also persists to the database for crash recovery.
        """
        with self._lock:
            self._signals[signal.symbol.upper()] = signal
            if self._conn is not None:
                self._persist(signal)

        ttl = max(0.0, (signal.expires_at - datetime.now(UTC)).total_seconds())
        logger.debug(
            f"[SignalCache] {signal.symbol} updated: {signal.action} "
            f"conf={signal.confidence:.0%} expires_in={ttl:.0f}s"
        )

    def update_batch(self, signals: list[TradeSignal]) -> None:
        """Store multiple signals atomically (single lock acquisition)."""
        with self._lock:
            for signal in signals:
                self._signals[signal.symbol.upper()] = signal
            if self._conn is not None:
                for signal in signals:
                    self._persist(signal)

    # -----------------------------------------------------------------------
    # Read path (tick executor ← cache) — must be nanosecond-fast
    # -----------------------------------------------------------------------

    def get(self, symbol: str) -> TradeSignal | None:
        """
        Return the signal for a symbol, or None if absent or expired.

        This is the hot-path read for the tick executor.  No I/O.
        """
        with self._lock:
            signal = self._signals.get(symbol.upper())
        if signal is None:
            return None
        if signal.is_expired:
            return None
        return signal

    def get_all_valid(self) -> dict[str, TradeSignal]:
        """Return all non-expired signals, keyed by symbol."""
        with self._lock:
            return {
                sym: sig for sym, sig in self._signals.items() if not sig.is_expired
            }

    def is_stale(self, symbol: str) -> bool:
        """True if no valid (non-expired) signal exists for this symbol."""
        return self.get(symbol) is None

    def staleness_seconds(self, symbol: str) -> float | None:
        """
        Seconds since the signal was generated, or None if no signal exists.

        Used for the `signal_staleness_seconds` Prometheus metric.
        """
        with self._lock:
            signal = self._signals.get(symbol.upper())
        return signal.age_seconds if signal else None

    # -----------------------------------------------------------------------
    # Maintenance
    # -----------------------------------------------------------------------

    def invalidate(self, symbol: str) -> None:
        """Remove the signal for a symbol (forces hold until next analysis)."""
        with self._lock:
            self._signals.pop(symbol.upper(), None)
            if self._conn is not None:
                try:
                    self._conn.execute(
                        "DELETE FROM signal_state WHERE symbol = ?", [symbol.upper()]
                    )
                except Exception as e:
                    logger.warning(
                        f"[SignalCache] Failed to delete {symbol} from DB: {e}"
                    )

    def invalidate_all(self) -> None:
        """Clear all signals (e.g., on session end or kill switch trigger)."""
        with self._lock:
            self._signals.clear()
            if self._conn is not None:
                try:
                    self._conn.execute("DELETE FROM signal_state")
                except Exception as e:
                    logger.warning(f"[SignalCache] Failed to clear signal_state: {e}")

    # -----------------------------------------------------------------------
    # Persistence helpers
    # -----------------------------------------------------------------------

    def _persist(self, signal: TradeSignal) -> None:
        """
        Upsert signal to the database for crash recovery.

        Called with self._lock held — don't acquire it here.
        """
        if self._conn is None:
            return
        try:
            self._conn.execute(
                """
                INSERT INTO signal_state
                    (symbol, action, confidence, position_size_pct,
                     stop_loss, take_profit, generated_at, expires_at, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol) DO UPDATE SET
                    action = excluded.action,
                    confidence = excluded.confidence,
                    position_size_pct = excluded.position_size_pct,
                    stop_loss = excluded.stop_loss,
                    take_profit = excluded.take_profit,
                    generated_at = excluded.generated_at,
                    expires_at = excluded.expires_at,
                    session_id = excluded.session_id
                """,
                [
                    signal.symbol.upper(),
                    signal.action,
                    signal.confidence,
                    signal.position_size_pct,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.generated_at,
                    signal.expires_at,
                    signal.session_id,
                ],
            )
        except Exception as e:
            # Persistence failure must NOT block execution
            logger.warning(
                f"[SignalCache] Failed to persist {signal.symbol} to DB: {e}"
            )

    def _load_from_db(self) -> None:
        """
        Recover non-expired signals from the database on startup.

        Only loads signals that haven't expired — stale signals from before
        a crash are not re-loaded.
        """
        if self._conn is None:
            return
        try:
            rows = self._conn.execute(
                """
                SELECT symbol, action, confidence, position_size_pct,
                       stop_loss, take_profit, generated_at, expires_at, session_id
                FROM signal_state
                WHERE expires_at > CURRENT_TIMESTAMP
                """
            ).fetchall()
            for row in rows:
                sym, action, conf, pos_pct, sl, tp, gen_at, exp_at, sid = row
                signal = TradeSignal(
                    symbol=sym,
                    action=action,
                    confidence=conf,
                    position_size_pct=pos_pct,
                    stop_loss=sl,
                    take_profit=tp,
                    generated_at=_ensure_tz(gen_at),
                    expires_at=_ensure_tz(exp_at),
                    session_id=sid or "",
                )
                self._signals[sym] = signal
            if rows:
                logger.info(f"[SignalCache] Recovered {len(rows)} signals from DB")
        except Exception as e:
            logger.warning(
                f"[SignalCache] Could not load signals from DB on startup: {e}"
            )


def _ensure_tz(dt: datetime) -> datetime:
    """Attach UTC timezone if the datetime is naive (may be stripped by the database driver)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt
