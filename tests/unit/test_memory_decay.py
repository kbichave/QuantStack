"""Tests for Section 09: Memory Temporal Decay.

Validates decay weighting, archival, last_accessed_at tracking, and SQL correctness.
Uses an in-memory SQLite stand-in that mirrors the agent_memory schema with
the new columns (last_accessed_at, archived_at) and the archive table.
"""

import json
import math
import sqlite3
from datetime import datetime, timedelta

import pytest

from quantstack.memory.blackboard import (
    Blackboard,
    CATEGORY_HALF_LIFE_DAYS,
    DEFAULT_HALF_LIFE_DAYS,
    decay_weight,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight in-memory DB that mirrors the PG schema
# ---------------------------------------------------------------------------

def _make_db():
    """Create an in-memory SQLite DB with agent_memory + archive tables."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE agent_memory ("
                 "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 "  session_id TEXT NOT NULL,"
                 "  sim_date DATE,"
                 "  agent TEXT NOT NULL,"
                 "  symbol TEXT DEFAULT '',"
                 "  category TEXT DEFAULT 'general',"
                 "  content_json TEXT NOT NULL,"
                 "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                 "  last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                 "  archived_at TIMESTAMP DEFAULT NULL"
                 ")")
    conn.execute("CREATE TABLE agent_memory_archive ("
                 "  id INTEGER PRIMARY KEY,"
                 "  session_id TEXT NOT NULL,"
                 "  sim_date DATE,"
                 "  agent TEXT NOT NULL,"
                 "  symbol TEXT DEFAULT '',"
                 "  category TEXT DEFAULT 'general',"
                 "  content_json TEXT NOT NULL,"
                 "  created_at TIMESTAMP,"
                 "  last_accessed_at TIMESTAMP,"
                 "  archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                 ")")
    return conn


def _insert_row(conn, *, agent="TestAgent", symbol="SPY", category="general",
                message="test", session_id="s1", days_ago=0,
                last_accessed_days_ago=None):
    """Insert a row with created_at = now - days_ago."""
    created = datetime.now() - timedelta(days=days_ago)
    last_accessed = (
        datetime.now() - timedelta(days=last_accessed_days_ago)
        if last_accessed_days_ago is not None
        else created
    )
    payload = json.dumps({"message": message})
    conn.execute(
        "INSERT INTO agent_memory "
        "(session_id, sim_date, agent, symbol, category, content_json, created_at, last_accessed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, created.date().isoformat(), agent, symbol, category,
         payload, created.isoformat(), last_accessed.isoformat()),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


# ---------------------------------------------------------------------------
# Import the constants and functions under test
# ---------------------------------------------------------------------------

class TestCategoryHalfLife:
    """CATEGORY_HALF_LIFE_DAYS constant is defined correctly."""

    def test_known_categories(self):
        from quantstack.memory.blackboard import CATEGORY_HALF_LIFE_DAYS
        assert CATEGORY_HALF_LIFE_DAYS["trade_outcome"] == 14
        assert CATEGORY_HALF_LIFE_DAYS["market_regime"] == 7
        assert CATEGORY_HALF_LIFE_DAYS["research_finding"] == 90
        assert CATEGORY_HALF_LIFE_DAYS["strategy_param"] == 30
        assert CATEGORY_HALF_LIFE_DAYS["general"] == 30

    def test_default_half_life(self):
        from quantstack.memory.blackboard import DEFAULT_HALF_LIFE_DAYS
        assert DEFAULT_HALF_LIFE_DAYS == 30


# ---------------------------------------------------------------------------
# Temporal decay weighting (pure Python formula)
# ---------------------------------------------------------------------------

class TestDecayWeight:
    """decay_weight() pure function computes POW(0.5, age_days / half_life)."""

    def test_today_weight_near_one(self):
        from quantstack.memory.blackboard import decay_weight
        w = decay_weight(age_days=0.0, half_life_days=14)
        assert w >= 0.95

    def test_one_half_life_weight_near_half(self):
        from quantstack.memory.blackboard import decay_weight
        w = decay_weight(age_days=14.0, half_life_days=14)
        assert 0.45 <= w <= 0.55

    def test_two_half_lives_weight_near_quarter(self):
        from quantstack.memory.blackboard import decay_weight
        w = decay_weight(age_days=28.0, half_life_days=14)
        assert 0.20 <= w <= 0.30

    def test_different_categories_different_weights(self):
        """Same age, different half-lives → different weights."""
        from quantstack.memory.blackboard import (
            decay_weight, CATEGORY_HALF_LIFE_DAYS,
        )
        age = 21.0
        w_regime = decay_weight(age, CATEGORY_HALF_LIFE_DAYS["market_regime"])   # hl=7
        w_research = decay_weight(age, CATEGORY_HALF_LIFE_DAYS["research_finding"])  # hl=90
        # market_regime decayed much more (3 half-lives)
        assert w_regime < w_research
        assert w_regime < 0.15  # 0.5^3 = 0.125
        assert w_research > 0.80

    def test_matches_python_pow(self):
        from quantstack.memory.blackboard import decay_weight
        age, hl = 10.0, 14.0
        expected = math.pow(0.5, age / hl)
        assert abs(decay_weight(age, hl) - expected) < 0.001


# ---------------------------------------------------------------------------
# Archival via archive_stale()
# ---------------------------------------------------------------------------

class TestArchiveStale:
    """archive_stale() moves expired entries to agent_memory_archive."""

    def test_old_entry_archived(self):
        """Entry older than 3x half_life is archived."""
        from quantstack.memory.blackboard import Blackboard
        conn = _make_db()
        _insert_row(conn, category="trade_outcome", days_ago=43)  # 3*14=42 threshold
        bb = Blackboard(conn=conn, session_id="s1")
        result = bb.archive_stale()
        assert result.get("trade_outcome", 0) >= 1
        # Verify moved to archive
        archived = conn.execute("SELECT COUNT(*) FROM agent_memory_archive").fetchone()[0]
        assert archived >= 1
        # Verify removed from active table
        active = conn.execute("SELECT COUNT(*) FROM agent_memory").fetchone()[0]
        assert active == 0

    def test_archived_entry_has_timestamp(self):
        conn = _make_db()
        row_id = _insert_row(conn, category="trade_outcome", days_ago=50)
        bb = Blackboard(conn=conn, session_id="s1")
        bb.archive_stale()
        archived_at = conn.execute(
            "SELECT archived_at FROM agent_memory_archive WHERE id = ?", (row_id,)
        ).fetchone()
        assert archived_at is not None
        assert archived_at[0] is not None

    def test_active_entry_not_archived(self):
        """Entry within TTL is not touched."""
        conn = _make_db()
        _insert_row(conn, category="trade_outcome", days_ago=10)  # well within 42 days
        bb = Blackboard(conn=conn, session_id="s1")
        result = bb.archive_stale()
        assert sum(result.values()) == 0
        active = conn.execute("SELECT COUNT(*) FROM agent_memory").fetchone()[0]
        assert active == 1

    def test_stale_access_triggers_archive(self):
        """Entry not accessed in 60+ days is archived even if created_at is recent."""
        conn = _make_db()
        _insert_row(conn, category="general", days_ago=20, last_accessed_days_ago=65)
        bb = Blackboard(conn=conn, session_id="s1")
        result = bb.archive_stale()
        assert sum(result.values()) >= 1

    def test_multiple_categories(self):
        """Archival respects per-category thresholds."""
        conn = _make_db()
        # market_regime half_life=7, threshold=21 days
        _insert_row(conn, category="market_regime", days_ago=25)
        # research_finding half_life=90, threshold=270 days
        _insert_row(conn, category="research_finding", days_ago=25)
        bb = Blackboard(conn=conn, session_id="s1")
        result = bb.archive_stale()
        assert result.get("market_regime", 0) >= 1
        assert result.get("research_finding", 0) == 0


# ---------------------------------------------------------------------------
# Decay-weighted reads
# ---------------------------------------------------------------------------

class TestDecayWeightedReads:
    """read_recent() orders by decay_weight when use_decay=True."""

    def test_decay_changes_ordering(self):
        """Old research_finding outranks recent market_regime by decay weight."""
        conn = _make_db()
        _insert_row(conn, category="research_finding", message="old-research", days_ago=20)
        _insert_row(conn, category="market_regime", message="recent-regime", days_ago=5)
        bb = Blackboard(conn=conn, session_id="s1")
        entries = bb.read_recent(limit=2, use_decay=True)
        # research_finding (hl=90, age=20 → w≈0.86) should rank above
        # market_regime (hl=7, age=5 → w≈0.61)
        assert entries[0].category == "research_finding"
        assert entries[1].category == "market_regime"

    def test_limit_applied_after_decay(self):
        """LIMIT returns top-N by decay_weight, not arbitrary N."""
        conn = _make_db()
        # Insert 10 old market_regime entries (low weight)
        for i in range(10):
            _insert_row(conn, category="market_regime", message=f"old-{i}", days_ago=15+i)
        # Insert 2 fresh research_finding entries (high weight)
        for i in range(2):
            _insert_row(conn, category="research_finding", message=f"fresh-{i}", days_ago=i)
        bb = Blackboard(conn=conn, session_id="s1")
        entries = bb.read_recent(limit=3, use_decay=True)
        # The 2 fresh research findings should be in top 3
        categories = [e.category for e in entries]
        assert categories.count("research_finding") == 2

    def test_default_read_still_works(self):
        """read_recent without use_decay maintains backward compat (created_at DESC)."""
        conn = _make_db()
        _insert_row(conn, message="older", days_ago=5)
        _insert_row(conn, message="newer", days_ago=1)
        bb = Blackboard(conn=conn, session_id="s1")
        entries = bb.read_recent(limit=2)
        assert entries[0].message == "newer"


# ---------------------------------------------------------------------------
# last_accessed_at tracking
# ---------------------------------------------------------------------------

class TestLastAccessedAt:
    """read_recent(use_decay=True) updates last_accessed_at."""

    def test_read_updates_last_accessed(self):
        conn = _make_db()
        old_time = datetime.now() - timedelta(days=10)
        row_id = _insert_row(conn, days_ago=10)
        # Force last_accessed_at to 10 days ago
        conn.execute(
            "UPDATE agent_memory SET last_accessed_at = ? WHERE id = ?",
            (old_time.isoformat(), row_id),
        )
        conn.commit()
        bb = Blackboard(conn=conn, session_id="s1")
        bb.read_recent(limit=5, use_decay=True)
        # Check last_accessed_at was updated
        new_accessed = conn.execute(
            "SELECT last_accessed_at FROM agent_memory WHERE id = ?", (row_id,)
        ).fetchone()[0]
        # Parse and verify it's recent (within last minute)
        accessed_dt = datetime.fromisoformat(new_accessed)
        assert (datetime.now() - accessed_dt).total_seconds() < 60


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

class TestSupervisorWiring:
    """Supervisor scheduled_tasks node calls archive_stale()."""

    def test_memory_pruning_in_scheduled_tasks(self):
        """Verify the supervisor nodes.py references memory_pruning task."""
        import inspect
        from quantstack.graphs.supervisor import nodes
        source = inspect.getsource(nodes.make_scheduled_tasks)
        assert "memory_pruning" in source
        assert "archive_stale" in source


class TestSchemaMigration:
    """_migrate_memory_pg adds the new columns and archive table."""

    def test_migration_adds_columns(self):
        """Verify the migration SQL includes last_accessed_at and archived_at."""
        import inspect
        from quantstack.db import _migrate_memory_pg
        source = inspect.getsource(_migrate_memory_pg)
        assert "last_accessed_at" in source
        assert "archived_at" in source

    def test_migration_creates_archive_table(self):
        """Verify the migration SQL creates agent_memory_archive."""
        import inspect
        from quantstack.db import _migrate_memory_pg
        source = inspect.getsource(_migrate_memory_pg)
        assert "agent_memory_archive" in source
