# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for research_wip table — distributed work locks for parallel research agents.
"""

import pytest
import psycopg2
from quantstack.db import db_conn, run_migrations


@pytest.fixture
def db():
    """Provide a test database connection with migrations run."""
    with db_conn() as conn:
        run_migrations(conn)
        # Clean up any existing test data
        conn.execute("DELETE FROM research_wip WHERE symbol LIKE 'TEST%'")
        yield conn


def test_lock_acquisition(db):
    """Test that two agents cannot acquire the same lock simultaneously."""
    # Agent 1 acquires lock
    db.execute(
        "INSERT INTO research_wip (symbol, domain, agent_id) VALUES (%s, %s, %s)",
        ("TEST_AAPL", "investment", "agent1")
    )

    # Agent 2 tries same lock → should fail
    with pytest.raises(psycopg2.IntegrityError):
        db.execute(
            "INSERT INTO research_wip (symbol, domain, agent_id) VALUES (%s, %s, %s)",
            ("TEST_AAPL", "investment", "agent2")
        )

    # Cleanup
    db.execute("DELETE FROM research_wip WHERE symbol = %s", ("TEST_AAPL",))


def test_different_domains_can_lock_same_symbol(db):
    """Test that different domains can research the same symbol in parallel."""
    # Investment agent locks AAPL for investment research
    db.execute(
        "INSERT INTO research_wip (symbol, domain, agent_id) VALUES (%s, %s, %s)",
        ("TEST_AAPL", "investment", "agent1")
    )

    # Swing agent locks AAPL for swing research — should succeed
    db.execute(
        "INSERT INTO research_wip (symbol, domain, agent_id) VALUES (%s, %s, %s)",
        ("TEST_AAPL", "swing", "agent2")
    )

    # Options agent locks AAPL for options research — should succeed
    db.execute(
        "INSERT INTO research_wip (symbol, domain, agent_id) VALUES (%s, %s, %s)",
        ("TEST_AAPL", "options", "agent3")
    )

    # Verify all three locks exist
    result = db.execute(
        "SELECT COUNT(*) FROM research_wip WHERE symbol = %s",
        ("TEST_AAPL",)
    ).fetchone()
    assert result[0] == 3

    # Cleanup
    db.execute("DELETE FROM research_wip WHERE symbol = %s", ("TEST_AAPL",))


def test_stale_lock_cleanup(db):
    """Test cleanup of stale locks (agents that crashed)."""
    # Create stale lock (31 minutes old)
    db.execute("""
        INSERT INTO research_wip (symbol, domain, agent_id, heartbeat_at)
        VALUES (%s, %s, %s, NOW() - INTERVAL '31 minutes')
    """, ("TEST_NVDA", "swing", "agent3"))

    # Create fresh lock
    db.execute("""
        INSERT INTO research_wip (symbol, domain, agent_id, heartbeat_at)
        VALUES (%s, %s, %s, NOW())
    """, ("TEST_TSLA", "investment", "agent4"))

    # Cleanup stale locks (older than 30 minutes)
    db.execute("DELETE FROM research_wip WHERE heartbeat_at < NOW() - INTERVAL '30 minutes'")

    # Verify stale lock removed
    result = db.execute(
        "SELECT COUNT(*) FROM research_wip WHERE symbol = %s",
        ("TEST_NVDA",)
    ).fetchone()
    assert result[0] == 0

    # Verify fresh lock still exists
    result = db.execute(
        "SELECT COUNT(*) FROM research_wip WHERE symbol = %s",
        ("TEST_TSLA",)
    ).fetchone()
    assert result[0] == 1

    # Cleanup
    db.execute("DELETE FROM research_wip WHERE symbol = %s", ("TEST_TSLA",))


def test_lock_release(db):
    """Test that locks can be released after work is complete."""
    # Acquire lock
    db.execute(
        "INSERT INTO research_wip (symbol, domain, agent_id) VALUES (%s, %s, %s)",
        ("TEST_MSFT", "options", "agent5")
    )

    # Verify lock exists
    result = db.execute(
        "SELECT COUNT(*) FROM research_wip WHERE symbol = %s AND domain = %s",
        ("TEST_MSFT", "options")
    ).fetchone()
    assert result[0] == 1

    # Release lock
    db.execute(
        "DELETE FROM research_wip WHERE symbol = %s AND domain = %s",
        ("TEST_MSFT", "options")
    )

    # Verify lock removed
    result = db.execute(
        "SELECT COUNT(*) FROM research_wip WHERE symbol = %s AND domain = %s",
        ("TEST_MSFT", "options")
    ).fetchone()
    assert result[0] == 0


def test_heartbeat_update(db):
    """Test that agents can update their heartbeat to keep locks alive."""
    import time

    # Acquire lock with explicit timestamp
    db.execute("""
        INSERT INTO research_wip (symbol, domain, agent_id, heartbeat_at)
        VALUES (%s, %s, %s, NOW() - INTERVAL '5 seconds')
    """, ("TEST_GOOG", "investment", "agent6"))

    # Get initial heartbeat
    result = db.execute(
        "SELECT heartbeat_at FROM research_wip WHERE symbol = %s AND domain = %s",
        ("TEST_GOOG", "investment")
    ).fetchone()
    initial_heartbeat = result[0]

    # Small delay to ensure timestamp difference
    time.sleep(0.01)

    # Update heartbeat
    db.execute("""
        UPDATE research_wip
        SET heartbeat_at = NOW()
        WHERE symbol = %s AND domain = %s
    """, ("TEST_GOOG", "investment"))

    # Verify heartbeat updated
    result = db.execute(
        "SELECT heartbeat_at FROM research_wip WHERE symbol = %s AND domain = %s",
        ("TEST_GOOG", "investment")
    ).fetchone()
    updated_heartbeat = result[0]

    assert updated_heartbeat > initial_heartbeat

    # Cleanup
    db.execute("DELETE FROM research_wip WHERE symbol = %s", ("TEST_GOOG",))


def test_domain_constraint(db):
    """Test that only valid domains are allowed."""
    # Valid domains should work
    for domain in ["investment", "swing", "options"]:
        db.execute(
            "INSERT INTO research_wip (symbol, domain, agent_id) VALUES (%s, %s, %s)",
            (f"TEST_{domain.upper()}", domain, "agent7")
        )

    # Invalid domain should fail
    with pytest.raises(psycopg2.IntegrityError):
        db.execute(
            "INSERT INTO research_wip (symbol, domain, agent_id) VALUES (%s, %s, %s)",
            ("TEST_INVALID", "futures", "agent8")
        )

    # Cleanup
    db.execute("DELETE FROM research_wip WHERE agent_id = %s", ("agent7",))
