# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
ReflectionManager — automatic post-trade analysis with SQL-based memory.

Fires after every trade close and at daily market close. No LLM required.

Architecture:
  1. On trade close: record outcome + market context snapshot
  2. SQL-query similar past situations by regime, symbol, strategy
  3. Surface relevant lessons from similar past failures
  4. Persist learnings to DuckDB + memory files
  5. On daily close: aggregate, identify patterns, update workshop_lessons.md
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Trade reflection record
# ---------------------------------------------------------------------------


@dataclass
class TradeReflection:
    """A single trade outcome with market context."""

    symbol: str
    strategy_id: str
    action: str  # "buy" or "sell"
    entry_price: float
    exit_price: float
    realized_pnl_pct: float
    holding_days: int
    regime_at_entry: str
    regime_at_exit: str = "unknown"
    conviction_at_entry: float = 0.0
    signals_at_entry: str = ""  # serialized key signals
    signals_at_exit: str = ""
    lesson: str = ""  # auto-generated or empty
    timestamp: str = ""


# ---------------------------------------------------------------------------
# ReflectionManager
# ---------------------------------------------------------------------------

REFLECTIONS_TABLE = "trade_reflections"
MEMORY_DIR = Path(".claude/memory")
WORKSHOP_FILE = MEMORY_DIR / "workshop_lessons.md"
JOURNAL_FILE = MEMORY_DIR / "trade_journal.md"

_REFLECTION_COLUMNS = (
    "symbol, strategy_id, action, entry_price, exit_price, "
    "realized_pnl_pct, holding_days, regime_at_entry, regime_at_exit, "
    "conviction, signals_entry, signals_exit, lesson"
)


class ReflectionManager:
    """Automatic post-trade analysis with SQL-based memory retrieval.

    Usage:
        rm = ReflectionManager(conn)
        # After trade close:
        rm.record_outcome(symbol, strategy_id, ...)
        # At market close:
        rm.daily_reflection(snapshot_date, ...)
    """

    def __init__(self, conn: Any):
        self._conn = conn
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the reflections table if it doesn't exist."""
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {REFLECTIONS_TABLE} (
                id              INTEGER PRIMARY KEY,
                symbol          VARCHAR NOT NULL,
                strategy_id     VARCHAR,
                action          VARCHAR,
                entry_price     DOUBLE,
                exit_price      DOUBLE,
                realized_pnl_pct DOUBLE,
                holding_days    INTEGER,
                regime_at_entry VARCHAR,
                regime_at_exit  VARCHAR,
                conviction      DOUBLE,
                signals_entry   VARCHAR,
                signals_exit    VARCHAR,
                lesson          VARCHAR,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    @staticmethod
    def _row_to_reflection(row: tuple) -> TradeReflection:
        """Convert a DB row to a TradeReflection."""
        return TradeReflection(
            symbol=row[0] or "",
            strategy_id=row[1] or "",
            action=row[2] or "",
            entry_price=row[3] or 0.0,
            exit_price=row[4] or 0.0,
            realized_pnl_pct=row[5] or 0.0,
            holding_days=row[6] or 0,
            regime_at_entry=row[7] or "",
            regime_at_exit=row[8] or "",
            conviction_at_entry=row[9] or 0.0,
            signals_at_entry=row[10] or "",
            signals_at_exit=row[11] or "",
            lesson=row[12] or "",
        )

    # ------------------------------------------------------------------
    # Record trade outcome
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        symbol: str,
        strategy_id: str,
        action: str,
        entry_price: float,
        exit_price: float,
        realized_pnl_pct: float,
        holding_days: int = 0,
        regime_at_entry: str = "unknown",
        regime_at_exit: str = "unknown",
        conviction: float = 0.0,
        signals_entry: str = "",
        signals_exit: str = "",
    ) -> TradeReflection:
        """Record a trade outcome and generate automatic reflection.

        Called after every trade close. Generates a lesson for losing trades
        by querying similar past situations from the DB.
        """
        # Auto-generate lesson for losses
        lesson = ""
        if realized_pnl_pct < -1.0:
            lesson = self._generate_loss_lesson(
                symbol,
                strategy_id,
                regime_at_entry,
                regime_at_exit,
                realized_pnl_pct,
                signals_entry,
            )

        ref = TradeReflection(
            symbol=symbol,
            strategy_id=strategy_id,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            realized_pnl_pct=realized_pnl_pct,
            holding_days=holding_days,
            regime_at_entry=regime_at_entry,
            regime_at_exit=regime_at_exit,
            conviction_at_entry=conviction,
            signals_at_entry=signals_entry,
            signals_at_exit=signals_exit,
            lesson=lesson,
            timestamp=datetime.now().isoformat(),
        )

        # Persist to DB
        try:
            self._conn.execute(
                f"INSERT INTO {REFLECTIONS_TABLE} "
                f"(symbol, strategy_id, action, entry_price, exit_price, "
                f"realized_pnl_pct, holding_days, regime_at_entry, regime_at_exit, "
                f"conviction, signals_entry, signals_exit, lesson) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    symbol,
                    strategy_id,
                    action,
                    entry_price,
                    exit_price,
                    realized_pnl_pct,
                    holding_days,
                    regime_at_entry,
                    regime_at_exit,
                    conviction,
                    signals_entry,
                    signals_exit,
                    lesson,
                ],
            )
        except Exception as exc:
            logger.warning(f"[Reflection] Failed to persist: {exc}")

        # Append to trade journal
        self._append_journal(ref)

        if lesson:
            logger.info(f"[Reflection] {symbol} loss lesson: {lesson[:100]}")

        return ref

    # ------------------------------------------------------------------
    # Find similar past situations
    # ------------------------------------------------------------------

    def find_similar(
        self,
        symbol: str,
        regime: str,
        signals: str = "",
        top_k: int = 3,
    ) -> list[TradeReflection]:
        """Find past trades in the same regime, preferring same symbol.

        Uses SQL filtering on structured columns instead of lexical matching.
        """
        try:
            rows = self._conn.execute(
                f"""SELECT {_REFLECTION_COLUMNS}
                    FROM {REFLECTIONS_TABLE}
                    WHERE regime_at_entry = ?
                    ORDER BY
                        (symbol = ?) DESC,
                        abs(realized_pnl_pct) DESC
                    LIMIT ?""",
                [regime, symbol, top_k],
            ).fetchall()
        except Exception as exc:
            logger.debug(f"[Reflection] find_similar query failed: {exc}")
            rows = []
        return [self._row_to_reflection(row) for row in rows]

    # ------------------------------------------------------------------
    # Loss lesson generation (deterministic, no LLM)
    # ------------------------------------------------------------------

    def _generate_loss_lesson(
        self,
        symbol: str,
        strategy_id: str,
        regime_entry: str,
        regime_exit: str,
        pnl_pct: float,
        signals: str,
    ) -> str:
        """Generate a structured lesson from a losing trade.

        Deterministic analysis — no LLM needed.
        """
        parts = []

        # Regime shift during hold?
        if regime_entry != regime_exit and regime_exit != "unknown":
            parts.append(
                f"Regime shifted {regime_entry}\u2192{regime_exit} during hold. "
                f"Strategy was designed for {regime_entry}."
            )

        # Severity classification
        if pnl_pct < -5.0:
            parts.append(
                f"Severe loss ({pnl_pct:.1f}%). Stop-loss may have been too wide or missing."
            )
        elif pnl_pct < -2.0:
            parts.append(
                f"Moderate loss ({pnl_pct:.1f}%). Review entry timing and signal strength."
            )
        else:
            parts.append(f"Minor loss ({pnl_pct:.1f}%). Within normal variance.")

        # Check for repeat losses via SQL
        try:
            repeat_count = self._conn.execute(
                f"""SELECT COUNT(*) FROM {REFLECTIONS_TABLE}
                    WHERE symbol = ? AND regime_at_entry = ? AND strategy_id = ?
                      AND realized_pnl_pct < -1.0""",
                [symbol, regime_entry, strategy_id],
            ).fetchone()[0]
        except Exception:
            repeat_count = 0

        if repeat_count >= 2:
            parts.append(
                f"REPEAT PATTERN: {repeat_count} similar past losses found. "
                f"Consider retiring or modifying strategy '{strategy_id}' for {regime_entry}."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Daily reflection (market close)
    # ------------------------------------------------------------------

    def daily_reflection(
        self,
        snapshot_date: date,
        daily_pnl: float,
        daily_return_pct: float,
        closed_trades: list[dict] | None = None,
    ) -> str:
        """Generate daily reflection summary. Called at market close.

        Returns the summary text (also appended to workshop_lessons.md).
        """
        summary_parts = [f"## Daily Reflection \u2014 {snapshot_date}"]
        summary_parts.append(f"P&L: ${daily_pnl:,.2f} ({daily_return_pct:+.2f}%)")

        if closed_trades:
            wins = [t for t in closed_trades if t.get("realized_pnl", 0) > 0]
            losses = [t for t in closed_trades if t.get("realized_pnl", 0) < 0]
            summary_parts.append(
                f"Trades: {len(closed_trades)} ({len(wins)}W / {len(losses)}L)"
            )

            # Worst loss analysis
            if losses:
                worst = min(losses, key=lambda t: t.get("realized_pnl", 0))
                summary_parts.append(
                    f"Worst: {worst.get('symbol')} "
                    f"${worst.get('realized_pnl', 0):,.2f} "
                    f"(strategy: {worst.get('strategy_id', 'unknown')})"
                )

            # Pattern detection: same strategy losing on multiple symbols
            loss_strategies = Counter(
                t.get("strategy_id", "") for t in losses if t.get("strategy_id")
            )
            repeated = {s: c for s, c in loss_strategies.items() if c >= 2}
            if repeated:
                for strat, count in repeated.items():
                    summary_parts.append(
                        f"WARNING: Strategy '{strat}' lost on {count} symbols today. "
                        f"May be misaligned with current regime."
                    )
        else:
            summary_parts.append("No closed trades today.")

        summary = "\n".join(summary_parts)

        # Append to workshop lessons
        self._append_workshop(summary)

        return summary

    # ------------------------------------------------------------------
    # File persistence
    # ------------------------------------------------------------------

    def _append_journal(self, ref: TradeReflection) -> None:
        """Append trade outcome to trade_journal.md."""
        try:
            JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
            entry = (
                f"\n### {ref.symbol} \u2014 {ref.timestamp[:10]}\n"
                f"- Strategy: {ref.strategy_id}\n"
                f"- Entry: ${ref.entry_price:.2f} \u2192 Exit: ${ref.exit_price:.2f}\n"
                f"- P&L: {ref.realized_pnl_pct:+.2f}% ({ref.holding_days}d hold)\n"
                f"- Regime: {ref.regime_at_entry} \u2192 {ref.regime_at_exit}\n"
            )
            if ref.lesson:
                entry += f"- Lesson: {ref.lesson}\n"
            with open(JOURNAL_FILE, "a") as f:
                f.write(entry)
        except Exception as exc:
            logger.debug(f"[Reflection] Failed to write journal: {exc}")

    def _append_workshop(self, summary: str) -> None:
        """Append daily summary to workshop_lessons.md."""
        try:
            WORKSHOP_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(WORKSHOP_FILE, "a") as f:
                f.write(f"\n{summary}\n")
        except Exception as exc:
            logger.debug(f"[Reflection] Failed to write workshop lessons: {exc}")
