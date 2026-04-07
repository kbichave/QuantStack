"""Tests for Phase 4 DB migration and policy update."""

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from quantstack.db import _migrate_phase4_coordination_pg


# ---------------------------------------------------------------------------
# Policy test
# ---------------------------------------------------------------------------


def test_claude_md_risk_gate_rule_updated():
    """CLAUDE.md says 'Never weaken or bypass', not 'Never modify'."""
    claude_md = Path(__file__).resolve().parents[2] / "CLAUDE.md"
    text = claude_md.read_text()
    assert "Never weaken or bypass" in text
    assert "Never modify." not in text


def test_claude_md_permits_strengthening():
    """CLAUDE.md explicitly permits strengthening the risk gate."""
    claude_md = Path(__file__).resolve().parents[2] / "CLAUDE.md"
    text = claude_md.read_text()
    assert "Strengthening checks" in text


# ---------------------------------------------------------------------------
# Migration function tests (mock DB)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_conn():
    """PgConnection mock that records execute calls."""
    conn = MagicMock()
    conn.execute = MagicMock()
    return conn


def test_migration_creates_circuit_breaker_table(mock_conn):
    """circuit_breaker_state table DDL is executed."""
    _migrate_phase4_coordination_pg(mock_conn)
    sqls = [c.args[0] for c in mock_conn.execute.call_args_list]
    cb_sqls = [s for s in sqls if "circuit_breaker_state" in s]
    assert len(cb_sqls) >= 1
    cb_ddl = cb_sqls[0]
    assert "CREATE TABLE IF NOT EXISTS" in cb_ddl
    assert "breaker_key" in cb_ddl
    assert "failure_count" in cb_ddl
    assert "cooldown_seconds" in cb_ddl


def test_migration_creates_agent_dlq_table(mock_conn):
    """agent_dlq table DDL is executed."""
    _migrate_phase4_coordination_pg(mock_conn)
    sqls = [c.args[0] for c in mock_conn.execute.call_args_list]
    dlq_sqls = [s for s in sqls if "agent_dlq" in s and "CREATE" in s]
    assert len(dlq_sqls) >= 1
    dlq_ddl = " ".join(dlq_sqls)
    assert "agent_name" in dlq_ddl
    assert "graph_name" in dlq_ddl
    assert "raw_output" in dlq_ddl
    assert "error_type" in dlq_ddl
    assert "prompt_hash" in dlq_ddl


def test_migration_creates_dlq_sequence(mock_conn):
    """agent_dlq_seq sequence is created."""
    _migrate_phase4_coordination_pg(mock_conn)
    sqls = [c.args[0] for c in mock_conn.execute.call_args_list]
    seq_sqls = [s for s in sqls if "agent_dlq_seq" in s]
    assert len(seq_sqls) >= 1
    assert "CREATE SEQUENCE IF NOT EXISTS" in seq_sqls[0]


def test_migration_creates_dlq_index(mock_conn):
    """Index on (agent_name, created_at) for DLQ rate queries."""
    _migrate_phase4_coordination_pg(mock_conn)
    sqls = [c.args[0] for c in mock_conn.execute.call_args_list]
    idx_sqls = [s for s in sqls if "ix_agent_dlq_agent_created" in s]
    assert len(idx_sqls) >= 1
    assert "agent_name" in idx_sqls[0]
    assert "created_at" in idx_sqls[0]


def test_migration_is_idempotent(mock_conn):
    """Running migration twice does not error."""
    _migrate_phase4_coordination_pg(mock_conn)
    _migrate_phase4_coordination_pg(mock_conn)
    # No exception raised = idempotent (IF NOT EXISTS handles it)


def test_circuit_breaker_defaults(mock_conn):
    """circuit_breaker_state has correct defaults: state='closed', failure_count=0, cooldown=300."""
    _migrate_phase4_coordination_pg(mock_conn)
    sqls = [c.args[0] for c in mock_conn.execute.call_args_list]
    cb_ddl = [s for s in sqls if "circuit_breaker_state" in s][0]
    assert "DEFAULT 'closed'" in cb_ddl
    assert "DEFAULT 0" in cb_ddl
    assert "DEFAULT 300" in cb_ddl
