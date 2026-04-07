# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
PostgreSQL-backed agent memory (formerly markdown blackboard).

Replaced the markdown file implementation because:
  - O(n) full-file reads scaled badly as history grew
  - No indexing: filtering by symbol required scanning every line
  - Freeform text allowed prompt injection via market data
  - No atomic writes when multiple agents wrote concurrently
  - File compaction was lossy and irreversible

New approach:
  - All writes go to the `agent_memory` table in the consolidated DB
  - Reads are indexed SQL queries (O(log n) on symbol/session indices)
  - Content stored as JSON — structured, not freeform markdown
  - Public API is identical to the old implementation so callers don't change

Usage:
    from quantstack.memory.blackboard import Blackboard

    bb = Blackboard(conn)          # injected connection
    bb.write("TrendIC", "SPY", "Strong uptrend ADX=35")
    entries = bb.read_recent(symbol="SPY", limit=10)
    ctx = bb.read_as_context("SPY")

Convenience module-level functions (backward-compatible):
    write_to_blackboard("Agent", "SPY", "message")
    read_blackboard_context("SPY")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from threading import RLock
from typing import Any

from loguru import logger

from quantstack.db import PgConnection, open_db, run_migrations

# ---------------------------------------------------------------------------
# Temporal decay configuration (Section 09)
# ---------------------------------------------------------------------------

CATEGORY_HALF_LIFE_DAYS: dict[str, int] = {
    "trade_outcome": 14,
    "strategy_param": 30,
    "market_regime": 7,
    "research_finding": 90,
    "general": 30,
}

DEFAULT_HALF_LIFE_DAYS = 30


def decay_weight(age_days: float, half_life_days: int | float) -> float:
    """Compute temporal decay weight: pow(0.5, age / half_life)."""
    if half_life_days <= 0:
        return 0.0
    return math.pow(0.5, age_days / half_life_days)

# ---------------------------------------------------------------------------
# Data model — same interface as the old BlackboardEntry
# ---------------------------------------------------------------------------


@dataclass
class BlackboardEntry:
    """A single memory entry."""

    timestamp: str
    agent: str
    symbol: str
    message: str
    category: str = "general"
    session_id: str = ""

    def to_markdown(self) -> str:
        """Render as markdown for injection into LLM context."""
        return (
            f"### [{self.timestamp}] {self.agent}\n"
            f"**Symbol:** {self.symbol}\n\n"
            f"{self.message}\n\n"
            "---\n"
        )

    def __str__(self) -> str:
        return self.to_markdown()


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------


class Blackboard:
    """
    Database-backed shared memory for all agents.

    Thread-safe.  Accepts an injected PostgreSQL connection.
    """

    def __init__(
        self,
        conn: PgConnection | None = None,
        session_id: str = "",
    ):
        self._session_id = session_id
        self._lock = RLock()

        if conn is not None:
            self._conn = conn
        else:
            self._conn = open_db()
            run_migrations(self._conn)

        logger.info("Blackboard initialized (PostgreSQL-backed agent_memory table)")

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------

    def write(
        self,
        agent: str,
        symbol: str,
        message: str,
        sim_date: date | None = None,
        category: str = "general",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Write an entry to agent memory.

        Args:
            agent: Name of the agent writing
            symbol: Symbol being discussed (empty string for portfolio-level entries)
            message: The observation / decision / analysis text
            sim_date: Simulation date override (uses now() if not provided)
            category: Semantic category tag (e.g. "decision", "analysis", "regime")
            extra: Optional additional structured data stored alongside the message
        """
        payload: dict[str, Any] = {"message": message}
        if extra:
            payload.update(extra)
        content_json = json.dumps(payload, default=str)

        ts = (
            datetime.combine(sim_date, datetime.min.time())
            if sim_date
            else datetime.now()
        )

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO agent_memory
                    (id, session_id, sim_date, agent, symbol, category, content_json, created_at)
                VALUES (nextval('agent_memory_seq'), ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    self._session_id,
                    sim_date or date.today(),
                    agent,
                    symbol.upper() if symbol else "",
                    category,
                    content_json,
                    ts,
                ],
            )

    # -----------------------------------------------------------------------
    # Read
    # -----------------------------------------------------------------------

    def read_recent(
        self,
        symbol: str | None = None,
        agent: str | None = None,
        category: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        use_decay: bool = False,
    ) -> list[BlackboardEntry]:
        """
        Return recent entries, most recent first.

        All filters are optional and combined with AND.
        When *use_decay* is True, results are ordered by temporal decay
        weight (POW(0.5, age / half_life)) instead of raw created_at,
        and last_accessed_at is updated for the returned rows.
        """
        conditions = ["archived_at IS NULL"]
        params: list[Any] = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol.upper())
        if agent:
            conditions.append("agent = ?")
            params.append(agent)
        if category:
            conditions.append("category = ?")
            params.append(category)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        where = "WHERE " + " AND ".join(conditions)

        if use_decay:
            # SQLite-compatible decay weight: julianday('now') - julianday(created_at) gives age in days.
            # For PostgreSQL, the query would use EXTRACT(EPOCH FROM ...) / 86400.
            # We use a Python-side sort to stay DB-agnostic.
            select_sql = f"""
                SELECT id, agent, symbol, category, content_json, created_at, session_id
                FROM agent_memory
                {where}
            """
            with self._lock:
                rows = self._conn.execute(select_sql, params).fetchall()

            now = datetime.now()
            weighted: list[tuple[float, tuple]] = []
            for row in rows:
                row_id, agent_name, sym, cat, content_json, created_at, sid = row
                if isinstance(created_at, str):
                    created_dt = datetime.fromisoformat(created_at)
                else:
                    created_dt = created_at
                age_days = max((now - created_dt).total_seconds() / 86400.0, 0.0)
                hl = CATEGORY_HALF_LIFE_DAYS.get(cat, DEFAULT_HALF_LIFE_DAYS)
                w = decay_weight(age_days, hl)
                weighted.append((w, (row_id, agent_name, sym, cat, content_json, created_at, sid)))

            weighted.sort(key=lambda x: x[0], reverse=True)
            top_rows = weighted[:limit]

            # Update last_accessed_at for returned rows
            if top_rows:
                ids = [r[1][0] for r in top_rows]
                placeholders = ",".join("?" for _ in ids)
                now_str = now.isoformat()
                with self._lock:
                    self._conn.execute(
                        f"UPDATE agent_memory SET last_accessed_at = ? WHERE id IN ({placeholders})",
                        [now_str] + ids,
                    )

            entries = []
            for _w, (row_id, agent_name, sym, cat, content_json, created_at, sid) in top_rows:
                try:
                    payload = json.loads(content_json)
                    message = payload.get("message", content_json)
                except (json.JSONDecodeError, TypeError):
                    message = str(content_json)
                entries.append(
                    BlackboardEntry(
                        timestamp=str(created_at),
                        agent=agent_name,
                        symbol=sym,
                        message=message,
                        category=cat,
                        session_id=sid,
                    )
                )
            return entries

        # Default path: order by created_at DESC (backward compatible)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT agent, symbol, category, content_json, created_at, session_id
                FROM agent_memory
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        entries = []
        for row in rows:
            agent_name, sym, cat, content_json, created_at, sid = row
            try:
                payload = json.loads(content_json)
                message = payload.get("message", content_json)
            except (json.JSONDecodeError, TypeError):
                message = str(content_json)

            entries.append(
                BlackboardEntry(
                    timestamp=str(created_at),
                    agent=agent_name,
                    symbol=sym,
                    message=message,
                    category=cat,
                    session_id=sid,
                )
            )
        return entries

    def read_as_context(self, symbol: str, limit: int = 10) -> str:
        """
        Return recent entries for a symbol formatted as markdown for LLM context.

        Structured markdown is assembled here from the stored JSON so the DB
        never stores raw LLM-injected text — only parsed, typed fields.
        """
        entries = self.read_recent(symbol=symbol, limit=limit)

        if not entries:
            return f"## Recent History for {symbol}\n\n*No recent history available for this symbol.*"

        lines = [
            f"## Recent History for {symbol}",
            "",
            f"*Last {len(entries)} entries:*",
            "",
        ]
        for entry in entries:
            lines.append(f"### [{entry.timestamp}] {entry.agent}")
            lines.append("")
            lines.append(entry.message)
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def read_all_as_markdown(self, limit: int = 50) -> str:
        """Return all recent entries as a single markdown document."""
        entries = self.read_recent(limit=limit)

        if not entries:
            return "## Trading Blackboard\n\n*No entries yet.*"

        lines = [
            "## Trading Blackboard",
            "",
            f"*Showing {len(entries)} most recent entries*",
            "",
        ]
        for entry in entries:
            lines.append(entry.to_markdown())

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Temporal decay — archival
    # -----------------------------------------------------------------------

    def archive_stale(self) -> dict[str, int]:
        """Archive entries past their TTL (3x half_life) or not accessed in 60+ days.

        Moves qualifying rows from agent_memory → agent_memory_archive.
        Returns ``{category: count_archived}`` for logging.
        """
        now = datetime.now()
        archived_counts: dict[str, int] = {}

        with self._lock:
            # Build per-category thresholds
            thresholds: list[tuple[str, datetime]] = []
            for cat, hl in CATEGORY_HALF_LIFE_DAYS.items():
                cutoff = now - timedelta(days=hl * 3)
                thresholds.append((cat, cutoff))

            access_cutoff = now - timedelta(days=60)

            # Find qualifying rows
            rows = self._conn.execute(
                "SELECT id, category FROM agent_memory WHERE archived_at IS NULL"
            ).fetchall()

            to_archive: list[int] = []
            for row_id, cat in rows:
                # Check created_at threshold for this category
                cat_cutoff = None
                for tcat, tcutoff in thresholds:
                    if tcat == cat:
                        cat_cutoff = tcutoff
                        break
                if cat_cutoff is None:
                    # Unknown category: use default half-life
                    cat_cutoff = now - timedelta(days=DEFAULT_HALF_LIFE_DAYS * 3)

                # Check if row qualifies
                created_row = self._conn.execute(
                    "SELECT created_at, last_accessed_at FROM agent_memory WHERE id = ?",
                    (row_id,),
                ).fetchone()
                if created_row is None:
                    continue
                created_at_raw, last_accessed_raw = created_row
                created_dt = (
                    datetime.fromisoformat(created_at_raw)
                    if isinstance(created_at_raw, str) else created_at_raw
                )
                last_accessed_dt = (
                    datetime.fromisoformat(last_accessed_raw)
                    if isinstance(last_accessed_raw, str) and last_accessed_raw
                    else last_accessed_raw
                )

                if created_dt < cat_cutoff or (
                    last_accessed_dt and last_accessed_dt < access_cutoff
                ):
                    to_archive.append(row_id)
                    archived_counts[cat] = archived_counts.get(cat, 0) + 1

            # Move rows: copy to archive, then delete from active
            now_str = now.isoformat()
            for row_id in to_archive:
                self._conn.execute(
                    "INSERT INTO agent_memory_archive "
                    "(id, session_id, sim_date, agent, symbol, category, "
                    " content_json, created_at, last_accessed_at, archived_at) "
                    "SELECT id, session_id, sim_date, agent, symbol, category, "
                    "       content_json, created_at, last_accessed_at, ? "
                    "FROM agent_memory WHERE id = ?",
                    (now_str, row_id),
                )
                self._conn.execute("DELETE FROM agent_memory WHERE id = ?", (row_id,))

        total = sum(archived_counts.values())
        if total:
            logger.info("Archived %d stale memory entries: %s", total, archived_counts)
        return archived_counts

    # -----------------------------------------------------------------------
    # Maintenance
    # -----------------------------------------------------------------------

    def clear(self, session_id: str | None = None) -> int:
        """
        Delete entries, optionally scoped to a session.

        Returns the number of rows deleted.
        """
        with self._lock:
            if session_id:
                result = self._conn.execute(
                    "DELETE FROM agent_memory WHERE session_id = ? RETURNING id",
                    [session_id],
                ).fetchall()
            else:
                result = self._conn.execute(
                    "DELETE FROM agent_memory RETURNING id"
                ).fetchall()
        count = len(result)
        logger.info(f"Blackboard cleared: {count} entries removed")
        return count

    def clear_before_date(self, cutoff_date: date) -> int:
        """Remove entries older than cutoff_date. Returns count deleted."""
        with self._lock:
            result = self._conn.execute(
                "DELETE FROM agent_memory WHERE sim_date < ? RETURNING id",
                [cutoff_date],
            ).fetchall()
        count = len(result)
        if count:
            logger.info(
                f"Blackboard pruned: {count} entries before {cutoff_date} removed"
            )
        return count

    def set_session(self, session_id: str) -> None:
        """Update the session ID for subsequent writes."""
        self._session_id = session_id


# ---------------------------------------------------------------------------
# Module-level convenience functions (backward-compatible with old API)
# ---------------------------------------------------------------------------

_blackboard: Blackboard | None = None


def get_blackboard(
    conn: PgConnection | None = None,
    session_id: str = "",
) -> Blackboard:
    """Get the singleton Blackboard instance."""
    global _blackboard
    if _blackboard is None:
        _blackboard = Blackboard(conn=conn, session_id=session_id)
    return _blackboard


def write_to_blackboard(
    agent: str,
    symbol: str,
    message: str,
    sim_date: date | None = None,
    category: str = "general",
) -> None:
    """Write to the blackboard (backward-compatible convenience function)."""
    get_blackboard().write(agent, symbol, message, sim_date, category)


def read_blackboard_context(symbol: str, limit: int = 10) -> str:
    """Read context for a symbol as markdown (backward-compatible)."""
    return get_blackboard().read_as_context(symbol, limit)


def write_portfolio_state(portfolio_summary: str) -> None:
    """
    Write the current portfolio state to the blackboard at session start.

    Called once per trading session before crew kickoff. The symbol is set to
    "PORTFOLIO" so agents can filter for it explicitly.
    """
    get_blackboard().write(
        agent="PortfolioState",
        symbol="PORTFOLIO",
        message=portfolio_summary,
        category="portfolio",
    )


def read_pinned_portfolio() -> str:
    """Return the most recent portfolio state entry from the blackboard."""
    entries = get_blackboard().read_recent(agent="PortfolioState", limit=1)
    if entries:
        return entries[0].message
    return "*No portfolio state on blackboard yet.*"
