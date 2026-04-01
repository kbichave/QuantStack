# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for BLITZ mode — parallel research agent orchestration.
"""

import pytest
import psycopg2
from quantstack.db import db_conn, run_migrations
from quantstack.research.agent_aggregator import AgentResult, ResearchAggregator


@pytest.fixture
def db():
    """Provide a test database connection with migrations run."""
    with db_conn() as conn:
        run_migrations(conn)
        # Clean up any existing test data
        conn.execute("DELETE FROM research_wip WHERE symbol LIKE 'TEST%'")
        conn.execute("DELETE FROM strategies WHERE name LIKE 'test_%'")
        conn.execute("DELETE FROM alpha_research_program WHERE investigation_id LIKE 'test_%'")
        yield conn
        # Cleanup after tests
        conn.execute("DELETE FROM research_wip WHERE symbol LIKE 'TEST%'")
        conn.execute("DELETE FROM strategies WHERE name LIKE 'test_%'")
        conn.execute("DELETE FROM alpha_research_program WHERE investigation_id LIKE 'test_%'")


def test_blitz_mode_small_scale(db):
    """Test BLITZ mode with 3 symbols × 3 domains = 9 agents (simulated)."""
    # Setup: create test symbols in OHLCV
    for sym in ["TEST_A", "TEST_B", "TEST_C"]:
        db.execute(
            "INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING",
            (sym, "1d", "2024-01-01", 100, 105, 99, 102, 1000000)
        )

    # Simulate agent results (in real BLITZ mode, agents return these)
    mock_results = []
    for symbol in ["TEST_A", "TEST_B", "TEST_C"]:
        for domain in ["investment", "swing", "options"]:
            mock_results.append(AgentResult(
                symbol=symbol,
                domain=domain,
                status="success",
                strategies_registered=[f"{symbol.lower()}_{domain}_strategy"],
                models_trained=[],
                hypotheses_tested=1,
                breakthrough_features=["test_feature"],
                thesis_status="intact",
                thesis_summary=f"{symbol} {domain} thesis",
                conflicts=[],
                elapsed_seconds=60.0
            ))

    # Aggregate results
    aggregator = ResearchAggregator()
    summary = aggregator.aggregate(mock_results)

    # Verify: all 3 symbols should have complete coverage
    assert len(summary["symbols_complete"]) == 3
    assert len(summary["symbols_partial"]) == 0
    assert summary["total_strategies"] == 9
    assert summary["agents_spawned"] == 9
    assert summary["agents_succeeded"] == 9

    # Verify: no work locks remain (all released)
    locks = db.execute("SELECT symbol, domain FROM research_wip WHERE symbol LIKE 'TEST%'").fetchall()
    assert len(locks) == 0


def test_work_lock_prevents_duplicates(db):
    """Test that work locks prevent duplicate research by parallel agents."""
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


def test_stale_lock_cleanup(db):
    """Test cleanup of stale locks before BLITZ mode starts."""
    # Create stale lock (31 minutes old)
    db.execute("""
        INSERT INTO research_wip (symbol, domain, agent_id, heartbeat_at)
        VALUES (%s, %s, %s, NOW() - INTERVAL '31 minutes')
    """, ("TEST_NVDA", "swing", "agent_stale"))

    # Create fresh lock
    db.execute("""
        INSERT INTO research_wip (symbol, domain, agent_id, heartbeat_at)
        VALUES (%s, %s, %s, NOW())
    """, ("TEST_TSLA", "investment", "agent_fresh"))

    # Cleanup stale locks (orchestrator does this before BLITZ)
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


def test_partial_coverage_handling(db):
    """Test aggregation when some agents fail."""
    mock_results = [
        AgentResult(
            symbol="TEST_PARTIAL",
            domain="investment",
            status="success",
            strategies_registered=["test_inv"],
            hypotheses_tested=1,
            elapsed_seconds=60.0
        ),
        AgentResult(
            symbol="TEST_PARTIAL",
            domain="swing",
            status="failure",
            strategies_registered=[],
            hypotheses_tested=1,
            elapsed_seconds=30.0
        ),
        AgentResult(
            symbol="TEST_PARTIAL",
            domain="options",
            status="needs_more_data",
            strategies_registered=[],
            hypotheses_tested=0,
            elapsed_seconds=10.0
        ),
    ]

    aggregator = ResearchAggregator()
    summary = aggregator.aggregate(mock_results)

    # Partial coverage (1 success out of 3 domains)
    assert "TEST_PARTIAL" in summary["symbols_partial"]
    assert "TEST_PARTIAL" not in summary["symbols_complete"]
    assert summary["total_strategies"] == 1
    assert summary["agents_succeeded"] == 1
    assert summary["agents_spawned"] == 3


def test_cross_domain_conflict_detection(db):
    """Test that aggregator detects conflicting theses across domains."""
    mock_results = [
        AgentResult(
            symbol="TEST_CONFLICT",
            domain="investment",
            status="success",
            thesis_status="intact",
            thesis_summary="Bullish fundamentals",
            hypotheses_tested=1,
            elapsed_seconds=60.0
        ),
        AgentResult(
            symbol="TEST_CONFLICT",
            domain="swing",
            status="success",
            thesis_status="broken",
            thesis_summary="Bearish technicals",
            hypotheses_tested=1,
            elapsed_seconds=60.0
        ),
        AgentResult(
            symbol="TEST_CONFLICT",
            domain="options",
            status="success",
            thesis_status="weakening",
            thesis_summary="Elevated vol",
            hypotheses_tested=1,
            elapsed_seconds=60.0
        ),
    ]

    aggregator = ResearchAggregator()
    summary = aggregator.aggregate(mock_results)

    # Should detect conflict between intact and broken
    assert len(summary["conflicts"]) > 0
    assert any("TEST_CONFLICT" in conflict for conflict in summary["conflicts"])


def test_blitz_mode_with_locked_symbols(db):
    """Test that BLITZ mode skips symbols with active locks."""
    # Setup: create test symbols
    for sym in ["TEST_FREE", "TEST_LOCKED"]:
        db.execute(
            "INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING",
            (sym, "1d", "2024-01-01", 100, 105, 99, 102, 1000000)
        )

    # Lock one symbol
    db.execute(
        "INSERT INTO research_wip (symbol, domain, agent_id) VALUES (%s, %s, %s)",
        ("TEST_LOCKED", "investment", "external_agent")
    )

    # Query for available symbols (what orchestrator would do)
    available_symbols = db.execute("""
        SELECT DISTINCT symbol
        FROM ohlcv
        WHERE symbol LIKE 'TEST%'
          AND symbol NOT IN (SELECT DISTINCT symbol FROM research_wip)
        ORDER BY symbol
        LIMIT 10
    """).fetchall()

    # Verify: only TEST_FREE is available
    available = [row[0] for row in available_symbols]
    assert "TEST_FREE" in available
    assert "TEST_LOCKED" not in available

    # Cleanup
    db.execute("DELETE FROM research_wip WHERE symbol = %s", ("TEST_LOCKED",))


def test_completion_pct_calculation(db):
    """Test portfolio completion percentage calculation."""
    # Setup: create test strategies
    test_strategies = [
        ("test_inv_1", "investment", "equity"),
        ("test_swing_1", "swing", "equity"),
        ("test_opt_1", "position", "options"),
    ]

    for name, time_horizon, instrument_type in test_strategies:
        # Delete if exists, then insert
        db.execute("DELETE FROM strategies WHERE strategy_id = %s", (name,))
        db.execute("""
            INSERT INTO strategies (strategy_id, name, time_horizon, instrument_type, status, parameters, entry_rules, exit_rules)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            name,
            name,
            time_horizon,
            instrument_type,
            "forward_testing",
            '{}',
            '[]',
            '[]'
        ))

    # Calculate completion (counts of strategies by domain)
    completion_criteria = {
        "equity_invest": db.execute(
            "SELECT COUNT(*) FROM strategies WHERE time_horizon='investment' AND status NOT IN ('retired', 'draft')"
        ).fetchone()[0],
        "equity_swing": db.execute(
            "SELECT COUNT(*) FROM strategies WHERE time_horizon IN ('swing','position') AND instrument_type='equity' AND status NOT IN ('retired', 'draft')"
        ).fetchone()[0],
        "options": db.execute(
            "SELECT COUNT(*) FROM strategies WHERE instrument_type='options' AND status NOT IN ('retired', 'draft')"
        ).fetchone()[0],
    }

    # Verify test strategies were inserted
    assert completion_criteria["equity_invest"] >= 1
    assert completion_criteria["equity_swing"] >= 1
    assert completion_criteria["options"] >= 1


def test_breakthrough_features_cross_domain(db):
    """Test identification of features that work across multiple domains."""
    mock_results = [
        AgentResult(
            symbol="TEST_FEAT",
            domain="investment",
            status="success",
            breakthrough_features=["volume_spike", "institutional_flow"],
            hypotheses_tested=1,
            elapsed_seconds=60.0
        ),
        AgentResult(
            symbol="TEST_FEAT",
            domain="swing",
            status="success",
            breakthrough_features=["volume_spike", "rsi_divergence"],
            hypotheses_tested=1,
            elapsed_seconds=60.0
        ),
        AgentResult(
            symbol="TEST_FEAT2",
            domain="options",
            status="success",
            breakthrough_features=["volume_spike"],
            hypotheses_tested=1,
            elapsed_seconds=60.0
        ),
    ]

    aggregator = ResearchAggregator()
    summary = aggregator.aggregate(mock_results)

    # volume_spike appears in 3 results (>= 2 threshold)
    assert "volume_spike" in summary["breakthrough_features"]
    # institutional_flow and rsi_divergence only appear once
    assert "institutional_flow" not in summary["breakthrough_features"]
    assert "rsi_divergence" not in summary["breakthrough_features"]
