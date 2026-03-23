# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the coordination layer:
  - EventBus (publish, poll, cursor, TTL)
  - StrategyStatusLock (CAS, valid transitions, event publishing)
  - UniverseRegistry (upsert, refresh, deactivation)
  - AutonomousScreener (scoring, tiers, hard filters)
  - WatchlistLoader v2 (tiered loading, fallbacks)
  - AutoPromoter (criteria evaluation, ramp schedule)
  - DegradationEnforcer (detector → breaker bridge)
  - PortfolioOrchestrator (correlation, sector, position cap gating)
  - DailyDigest (report generation)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from datetime import date, timedelta as td
from quantstack.autonomous.screener import AutonomousScreener
from quantstack.autonomous.watchlist import DEFAULT_SYMBOLS, WatchlistLoader
from quantstack.coordination.auto_promoter import AutoPromoter, PositionRamp, PromotionCriteria
from quantstack.coordination.daily_digest import DailyDigest
from quantstack.coordination.degradation_enforcer import DegradationEnforcer
from quantstack.coordination.event_bus import Event, EventBus, EventType
from quantstack.coordination.portfolio_orchestrator import PortfolioOrchestrator, ProposedTrade
from quantstack.coordination.strategy_lock import StrategyStatusLock
from quantstack.coordination.supervisor import LoopConfig, LoopSupervisor
from quantstack.coordination.universe_registry import UniverseRegistry
from quantstack.db import _migrate_coordination, _migrate_screener, _migrate_universe
from quantstack.execution.strategy_breaker import StrategyBreaker
import os


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def db():
    """In-memory DuckDB with all coordination tables."""
    conn = duckdb.connect(":memory:")

    # Run the coordination migrations manually
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS loop_events (
            event_id VARCHAR PRIMARY KEY, event_type VARCHAR NOT NULL,
            source_loop VARCHAR NOT NULL, payload JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS le_type ON loop_events (event_type, created_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS loop_cursors (
            consumer_id VARCHAR PRIMARY KEY, last_event_id VARCHAR,
            last_polled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS loop_heartbeats (
            loop_name VARCHAR NOT NULL, iteration INTEGER NOT NULL,
            started_at TIMESTAMP NOT NULL, finished_at TIMESTAMP,
            symbols_processed INTEGER DEFAULT 0, errors INTEGER DEFAULT 0,
            status VARCHAR DEFAULT 'running', PRIMARY KEY (loop_name, iteration)
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategies (
            strategy_id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL UNIQUE,
            description TEXT DEFAULT '', asset_class VARCHAR DEFAULT 'equities',
            regime_affinity JSON, parameters JSON NOT NULL,
            entry_rules JSON NOT NULL, exit_rules JSON NOT NULL,
            risk_params JSON, backtest_summary JSON, walkforward_summary JSON,
            status VARCHAR DEFAULT 'draft', source VARCHAR DEFAULT 'manual',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by VARCHAR DEFAULT 'test',
            instrument_type VARCHAR DEFAULT 'equity',
            time_horizon VARCHAR DEFAULT 'swing',
            holding_period_days INTEGER DEFAULT 5
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_outcomes (
            id INTEGER PRIMARY KEY, strategy_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL, regime_at_entry VARCHAR DEFAULT 'unknown',
            action VARCHAR NOT NULL, entry_price DOUBLE NOT NULL,
            exit_price DOUBLE, realized_pnl_pct DOUBLE, outcome VARCHAR,
            opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, closed_at TIMESTAMP,
            session_id VARCHAR DEFAULT ''
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS regime_strategy_matrix (
            regime VARCHAR NOT NULL, strategy_id VARCHAR NOT NULL,
            allocation_pct DOUBLE NOT NULL, confidence DOUBLE DEFAULT 0.5,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (regime, strategy_id)
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS universe (
            symbol VARCHAR PRIMARY KEY, name VARCHAR NOT NULL,
            sector VARCHAR DEFAULT 'Unknown', source VARCHAR NOT NULL,
            market_cap DOUBLE, avg_daily_volume DOUBLE,
            is_active BOOLEAN DEFAULT TRUE,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_refreshed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deactivated_reason VARCHAR
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS screener_results (
            symbol VARCHAR NOT NULL, screened_at TIMESTAMP NOT NULL,
            regime_used VARCHAR, tier INTEGER NOT NULL,
            composite_score DOUBLE NOT NULL,
            momentum_score DOUBLE, volatility_rank DOUBLE,
            volume_surge DOUBLE, regime_fit DOUBLE, catalyst_proximity DOUBLE,
            PRIMARY KEY (symbol, screened_at)
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol VARCHAR, timeframe VARCHAR, timestamp TIMESTAMP,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            symbol VARCHAR PRIMARY KEY, quantity INTEGER NOT NULL,
            avg_cost DOUBLE NOT NULL, side VARCHAR DEFAULT 'long',
            opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS closed_trades (
            id INTEGER PRIMARY KEY, symbol VARCHAR NOT NULL,
            side VARCHAR NOT NULL, quantity INTEGER NOT NULL,
            entry_price DOUBLE NOT NULL, exit_price DOUBLE NOT NULL,
            realized_pnl DOUBLE NOT NULL, opened_at TIMESTAMP,
            closed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            holding_days INTEGER DEFAULT 0, session_id VARCHAR DEFAULT ''
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS earnings_calendar (
            symbol VARCHAR, report_date DATE, fiscal_date_ending DATE,
            estimate DOUBLE, reported_eps DOUBLE, surprise DOUBLE, surprise_pct DOUBLE
        )
    """
    )

    yield conn
    conn.close()


# ── EventBus Tests ───────────────────────────────────────────────────────────


class TestEventBus:
    def test_publish_and_poll(self, db):
        bus = EventBus(db)

        # Publish
        eid = bus.publish(
            Event(
                event_type=EventType.STRATEGY_PROMOTED,
                source_loop="factory",
                payload={"strategy_id": "abc"},
            )
        )
        assert eid

        # Poll
        events = bus.poll("trader_loop")
        assert len(events) == 1
        assert events[0].event_type == EventType.STRATEGY_PROMOTED
        assert events[0].payload["strategy_id"] == "abc"

        # Second poll — cursor advanced, no new events
        events2 = bus.poll("trader_loop")
        assert len(events2) == 0

    def test_poll_with_filter(self, db):
        bus = EventBus(db)
        bus.publish(
            Event(event_type=EventType.STRATEGY_PROMOTED, source_loop="factory")
        )
        bus.publish(Event(event_type=EventType.MODEL_TRAINED, source_loop="ml"))

        # Filter for MODEL_TRAINED only
        events = bus.poll("consumer1", event_types=[EventType.MODEL_TRAINED])
        assert len(events) == 1
        assert events[0].event_type == EventType.MODEL_TRAINED

    def test_get_latest(self, db):
        bus = EventBus(db)
        bus.publish(
            Event(
                event_type=EventType.LOOP_HEARTBEAT,
                source_loop="factory",
                payload={"iter": 1},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.LOOP_HEARTBEAT,
                source_loop="factory",
                payload={"iter": 2},
            )
        )

        latest = bus.get_latest(EventType.LOOP_HEARTBEAT)
        assert latest is not None
        assert latest.payload["iter"] == 2

    def test_count_events(self, db):
        bus = EventBus(db)
        bus.publish(
            Event(event_type=EventType.STRATEGY_PROMOTED, source_loop="factory")
        )
        bus.publish(Event(event_type=EventType.STRATEGY_RETIRED, source_loop="factory"))
        bus.publish(
            Event(event_type=EventType.STRATEGY_PROMOTED, source_loop="factory")
        )

        assert bus.count_events(EventType.STRATEGY_PROMOTED) == 2
        assert bus.count_events() == 3

    def test_independent_cursors(self, db):
        bus = EventBus(db)
        bus.publish(Event(event_type=EventType.MODEL_TRAINED, source_loop="ml"))

        # Two consumers independently see the same event
        events_a = bus.poll("consumer_a")
        events_b = bus.poll("consumer_b")
        assert len(events_a) == 1
        assert len(events_b) == 1


# ── StrategyStatusLock Tests ─────────────────────────────────────────────────


class TestStrategyStatusLock:
    def _insert_strategy(self, db, strategy_id, name, status="draft"):
        db.execute(
            "INSERT INTO strategies (strategy_id, name, parameters, entry_rules, exit_rules, status) "
            "VALUES (?, ?, '{}', '{}', '{}', ?)",
            [strategy_id, name, status],
        )

    def test_valid_transition(self, db):
        self._insert_strategy(db, "s1", "test_strat", "draft")
        lock = StrategyStatusLock(db)
        ok = lock.transition("s1", "draft", "forward_testing", "test promotion")
        assert ok

        status = lock.get_status("s1")
        assert status == "forward_testing"

    def test_cas_failure(self, db):
        self._insert_strategy(db, "s1", "test_strat", "forward_testing")
        lock = StrategyStatusLock(db)

        # Try to transition from 'draft' but it's actually 'forward_testing'
        ok = lock.transition("s1", "draft", "forward_testing", "should fail")
        assert not ok

    def test_invalid_transition_raises(self, db):
        lock = StrategyStatusLock(db)
        with pytest.raises(ValueError, match="Invalid transition"):
            lock.transition("s1", "draft", "live", "skip forward_testing")

    def test_publishes_event(self, db):
        self._insert_strategy(db, "s1", "test_strat", "draft")
        bus = EventBus(db)
        lock = StrategyStatusLock(db, event_bus=bus)

        lock.transition("s1", "draft", "forward_testing", "test")

        events = bus.poll("test_consumer", [EventType.STRATEGY_PROMOTED])
        assert len(events) == 1
        assert events[0].payload["strategy_id"] == "s1"


# ── UniverseRegistry Tests ───────────────────────────────────────────────────


class TestUniverseRegistry:
    def test_refresh_etfs(self, db):
        registry = UniverseRegistry(db, client=None)
        report = registry.refresh_constituents()

        # Should have loaded ~50 ETFs
        assert report.total_active > 40
        assert report.symbols_added > 40

        # Check that SPY is in the universe
        symbols = registry.get_active_symbols()
        assert "SPY" in symbols
        assert "QQQ" in symbols

    def test_deactivate_symbol(self, db):
        registry = UniverseRegistry(db, client=None)
        registry.refresh_constituents()

        registry.deactivate_symbol("SPY", "testing")
        symbols = registry.get_active_symbols()
        assert "SPY" not in symbols

    def test_count(self, db):
        registry = UniverseRegistry(db, client=None)
        registry.refresh_constituents()

        count = registry.count()
        assert count > 40

    def test_refresh_age(self, db):
        registry = UniverseRegistry(db, client=None)

        # Before refresh — infinite age
        assert registry.get_refresh_age_hours() == float("inf")

        registry.refresh_constituents()
        age = registry.get_refresh_age_hours()
        assert age < 1.0  # Just refreshed


# ── PortfolioOrchestrator Tests ──────────────────────────────────────────────


class TestPortfolioOrchestrator:
    def test_no_doubling(self):
        orch = PortfolioOrchestrator(max_positions=10)
        trades = [
            ProposedTrade(symbol="AAPL", action="buy", confidence=0.8),
            ProposedTrade(symbol="MSFT", action="buy", confidence=0.7),
        ]
        current = {"AAPL": {"quantity": 100}}

        report = orch.gate_entries(trades, current)
        assert report.rejected_duplicate == 1
        assert report.approved == 1

    def test_position_cap(self):
        orch = PortfolioOrchestrator(max_positions=2)
        trades = [
            ProposedTrade(symbol="AAPL", action="buy", confidence=0.9),
            ProposedTrade(symbol="MSFT", action="buy", confidence=0.8),
        ]
        current = {"NVDA": {"quantity": 50}}

        report = orch.gate_entries(trades, current)
        assert report.approved == 1  # max_positions=2, already holding 1
        assert report.rejected_position_cap == 1

    def test_sector_concentration(self):
        orch = PortfolioOrchestrator(max_positions=10, max_sector_pct=0.30)
        trades = [
            ProposedTrade(symbol="AAPL", action="buy", confidence=0.9, sector="Tech"),
            ProposedTrade(symbol="MSFT", action="buy", confidence=0.8, sector="Tech"),
            ProposedTrade(symbol="GOOGL", action="buy", confidence=0.7, sector="Tech"),
        ]
        # Already holding 2 tech stocks out of 3 total
        current = {"NVDA": {}, "AMD": {}, "JPM": {}}
        sector_map = {"NVDA": "Tech", "AMD": "Tech", "JPM": "Financials"}

        report = orch.gate_entries(trades, current, sector_map)
        # At least one tech trade should be rejected for sector concentration
        assert report.rejected_sector >= 1

    def test_confidence_ranking(self):
        orch = PortfolioOrchestrator(max_positions=2)
        trades = [
            ProposedTrade(symbol="LOW_CONF", action="buy", confidence=0.3),
            ProposedTrade(symbol="HIGH_CONF", action="buy", confidence=0.9),
        ]

        report = orch.gate_entries(trades, {})
        # HIGH_CONF should be approved first
        approved = [r for r in report.results if r.approved]
        assert approved[0].trade.symbol == "HIGH_CONF"


# ── AutoPromoter Tests ───────────────────────────────────────────────────────


class TestAutoPromoter:
    def test_ramp_schedule(self):
        ramp = PositionRamp()
        assert ramp.get_scale(0) == 0.25
        assert ramp.get_scale(6) == 0.25
        assert ramp.get_scale(7) == 0.50
        assert ramp.get_scale(14) == 0.75
        assert ramp.get_scale(21) == 1.00
        assert ramp.get_scale(100) == 1.00

    def test_disabled_by_default(self, db):
        promoter = AutoPromoter(db)
        assert not promoter.is_enabled()
        decisions = promoter.evaluate_all()
        assert decisions == []

    @patch.dict("os.environ", {"AUTO_PROMOTE_ENABLED": "true"})
    def test_too_young_strategy(self, db):
        # Insert a fresh forward_testing strategy
        db.execute(
            "INSERT INTO strategies (strategy_id, name, parameters, entry_rules, exit_rules, status, updated_at) "
            "VALUES ('s1', 'young_strat', '{}', '{}', '{}', 'forward_testing', ?)",
            [datetime.now(timezone.utc)],
        )

        promoter = AutoPromoter(db)
        decisions = promoter.evaluate_all()
        assert len(decisions) == 1
        assert decisions[0].decision == "hold"
        assert "Too young" in decisions[0].reason


# ── DegradationEnforcer Tests ────────────────────────────────────────────────


class TestDegradationEnforcer:
    def test_critical_trips_breaker(self):
        # Mock detector returning CRITICAL
        detector = MagicMock()
        report = MagicMock()
        report.status = "CRITICAL"
        report.findings = ["Live Sharpe < 0"]
        report.recommended_size_multiplier = 0.0
        detector.check.return_value = report

        # Mock breaker
        breaker = MagicMock()
        breaker.force_trip = MagicMock()

        enforcer = DegradationEnforcer(detector, breaker)
        result = enforcer.enforce("strat_1")

        assert result.severity == "critical"
        assert result.action_taken == "tripped"
        breaker.force_trip.assert_called_once()

    def test_warning_scales_breaker(self):
        detector = MagicMock()
        report = MagicMock()
        report.status = "WARNING"
        report.findings = ["IS/OOS ratio > 2"]
        report.recommended_size_multiplier = 0.5
        detector.check.return_value = report

        breaker = MagicMock()
        breaker.force_scale = MagicMock()

        enforcer = DegradationEnforcer(detector, breaker)
        result = enforcer.enforce("strat_1")

        assert result.severity == "warning"
        assert result.action_taken == "scaled"
        assert result.size_multiplier == 0.5
        breaker.force_scale.assert_called_once()

    def test_clean_no_action(self):
        detector = MagicMock()
        report = MagicMock()
        report.status = "clean"
        report.findings = []
        report.recommended_size_multiplier = 1.0
        detector.check.return_value = report

        breaker = MagicMock()
        enforcer = DegradationEnforcer(detector, breaker)
        result = enforcer.enforce("strat_1")

        assert result.severity == "clean"
        assert result.action_taken == "none"


# ── DailyDigest Tests ────────────────────────────────────────────────────────


class TestDailyDigest:
    def test_generate_empty(self, db):
        digest = DailyDigest(db)
        report = digest.generate()

        assert report.open_positions == 0
        assert report.trades_today == 0
        assert report.total_live == 0

    def test_format_markdown(self, db):
        digest = DailyDigest(db)
        report = digest.generate()
        md = digest.format_markdown(report)

        assert "Daily Digest" in md
        assert "Portfolio" in md
        assert "Strategy Lifecycle" in md

    def test_format_discord(self, db):
        digest = DailyDigest(db)
        report = digest.generate()
        payload = digest.format_discord(report)

        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        assert "fields" in payload["embeds"][0]


# ── EventBus Edge Cases ──────────────────────────────────────────────────────


class TestEventBusTTL:
    def test_old_events_pruned_on_publish(self, db):
        bus = EventBus(db)

        # Insert an old event directly (8 days ago — beyond 7-day TTL)
        old_ts = datetime.now(timezone.utc) - timedelta(days=8)
        db.execute(
            "INSERT INTO loop_events (event_id, event_type, source_loop, payload, created_at) "
            "VALUES ('old_evt', 'loop_heartbeat', 'factory', '{}', ?)",
            [old_ts],
        )
        assert bus.count_events() == 1

        # Publishing a new event should prune the old one
        bus.publish(
            Event(event_type=EventType.STRATEGY_PROMOTED, source_loop="factory")
        )
        assert bus.count_events(EventType.LOOP_HEARTBEAT) == 0
        assert bus.count_events(EventType.STRATEGY_PROMOTED) == 1

    def test_poll_empty_returns_empty(self, db):
        bus = EventBus(db)
        events = bus.poll("new_consumer")
        assert events == []

    def test_get_latest_missing_returns_none(self, db):
        bus = EventBus(db)
        assert bus.get_latest(EventType.MODEL_TRAINED) is None


# ── StrategyStatusLock Edge Cases ────────────────────────────────────────────


class TestStrategyStatusLockEdgeCases:
    def _insert_strategy(self, db, strategy_id, name, status="draft"):
        db.execute(
            "INSERT INTO strategies (strategy_id, name, parameters, entry_rules, exit_rules, status) "
            "VALUES (?, ?, '{}', '{}', '{}', ?)",
            [strategy_id, name, status],
        )

    def test_nonexistent_strategy(self, db):
        lock = StrategyStatusLock(db)
        ok = lock.transition("nonexistent", "draft", "forward_testing", "test")
        assert not ok

    def test_get_status(self, db):
        self._insert_strategy(db, "s1", "test_strat", "live")
        lock = StrategyStatusLock(db)
        assert lock.get_status("s1") == "live"
        assert lock.get_status("missing") is None

    def test_demotion_transition(self, db):
        """live → forward_testing is valid (degradation demotion)."""
        self._insert_strategy(db, "s1", "test_strat", "live")
        lock = StrategyStatusLock(db)
        ok = lock.transition("s1", "live", "forward_testing", "degradation CRITICAL")
        assert ok
        assert lock.get_status("s1") == "forward_testing"

    def test_full_lifecycle(self, db):
        """draft → forward_testing → live → retired."""
        self._insert_strategy(db, "s1", "lifecycle_test", "draft")
        lock = StrategyStatusLock(db)

        assert lock.transition("s1", "draft", "forward_testing", "good backtest")
        assert lock.transition(
            "s1", "forward_testing", "live", "21d paper trading passed"
        )
        assert lock.transition("s1", "live", "retired", "alpha decayed")
        assert lock.get_status("s1") == "retired"


# ── StrategyBreaker Extensions ───────────────────────────────────────────────


class TestStrategyBreakerExtensions:
    def test_force_trip(self, tmp_path):
        state_file = tmp_path / "breakers.json"
        breaker = StrategyBreaker(state_path=str(state_file))

        result = breaker.force_trip("strat_1", reason="Degradation CRITICAL")
        assert result.status == "TRIPPED"
        assert result.scale_factor == 0.0
        assert result.tripped_at is not None

        # Verify persisted
        factor = breaker.get_scale_factor("strat_1")
        assert factor == 0.0

    def test_force_scale(self, tmp_path):
        state_file = tmp_path / "breakers.json"
        breaker = StrategyBreaker(state_path=str(state_file))

        result = breaker.force_scale(
            "strat_1", scale_factor=0.25, reason="IS/OOS ratio > 2"
        )
        assert result.status == "SCALED"
        assert result.scale_factor == 0.25

    def test_force_scale_does_not_escalate_past_tripped(self, tmp_path):
        state_file = tmp_path / "breakers.json"
        breaker = StrategyBreaker(state_path=str(state_file))

        breaker.force_trip("strat_1", reason="tripped first")
        result = breaker.force_scale("strat_1", scale_factor=0.5, reason="warning")
        # Should remain TRIPPED (force_scale is a no-op on tripped strategies)
        assert result.status == "TRIPPED"
        assert result.scale_factor == 0.0

    def test_force_scale_keeps_lower_factor(self, tmp_path):
        state_file = tmp_path / "breakers.json"
        breaker = StrategyBreaker(state_path=str(state_file))

        breaker.force_scale("strat_1", scale_factor=0.25, reason="severe warning")
        result = breaker.force_scale("strat_1", scale_factor=0.5, reason="mild warning")
        # 0.5 > 0.25, so the more restrictive factor (0.25) should stay
        assert result.scale_factor == 0.25

    def test_force_trip_survives_reload(self, tmp_path):
        state_file = tmp_path / "breakers.json"
        breaker1 = StrategyBreaker(state_path=str(state_file))
        breaker1.force_trip("strat_1", reason="test persistence")

        # Create a new breaker instance — should reload from JSON
        breaker2 = StrategyBreaker(state_path=str(state_file))
        assert breaker2.get_scale_factor("strat_1") == 0.0
        state = breaker2.check("strat_1")
        assert state.status == "TRIPPED"


# ── AutonomousScreener Tests ─────────────────────────────────────────────────


class TestAutonomousScreener:
    def _seed_universe(self, db, symbols):
        """Insert symbols into the universe table."""
        for sym in symbols:
            db.execute(
                "INSERT INTO universe (symbol, name, sector, source, avg_daily_volume, is_active) "
                "VALUES (?, ?, 'Technology', 'sp500', 1000000, TRUE)",
                [sym, f"{sym} Inc"],
            )

    def _seed_ohlcv(self, db, symbol, n_bars=200, base_price=100.0):
        """Insert synthetic daily OHLCV data with an upward trend."""
        today = date.today()
        for i in range(n_bars):
            day = today - td(days=n_bars - i)
            price = base_price + i * 0.1  # Gentle uptrend
            db.execute(
                "INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume) "
                "VALUES (?, 'D1', ?, ?, ?, ?, ?, ?)",
                [
                    symbol,
                    day,
                    price - 0.5,
                    price + 1.0,
                    price - 1.0,
                    price,
                    500000 + i * 100,
                ],
            )

    def test_screen_empty_universe(self, db):
        screener = AutonomousScreener(db)
        result = screener._screen_sync("trending_up")
        assert result.universe_size == 0
        assert result.total_watchlist == 0

    def test_screen_produces_tiers(self, db):
        # Seed 30 symbols with OHLCV data
        symbols = [f"SYM{i:02d}" for i in range(30)]
        self._seed_universe(db, symbols)
        for sym in symbols:
            self._seed_ohlcv(db, sym)

        screener = AutonomousScreener(db)
        result = screener._screen_sync("trending_up")

        assert result.universe_size == 30
        assert len(result.tier_1) == 15
        assert len(result.tier_2) == 15  # 30 - 15 = 15 (< max tier 2 of 20)
        assert result.total_watchlist == 30

        # Tier 1 should have higher composite scores than tier 2
        if result.tier_1 and result.tier_2:
            assert result.tier_1[-1].composite >= result.tier_2[0].composite

    def test_hard_filter_excludes_restricted(self, db):
        self._seed_universe(db, ["AAPL", "BANNED"])
        self._seed_ohlcv(db, "AAPL")
        self._seed_ohlcv(db, "BANNED")

        with patch.dict("os.environ", {"RISK_RESTRICTED_SYMBOLS": "BANNED"}):
            screener = AutonomousScreener(db)
            result = screener._screen_sync("unknown")

        assert result.filtered_out == 1
        all_symbols = [s.symbol for s in result.tier_1 + result.tier_2 + result.tier_3]
        assert "BANNED" not in all_symbols
        assert "AAPL" in all_symbols

    def test_hard_filter_excludes_low_adv(self, db):
        # Insert one with low ADV
        db.execute(
            "INSERT INTO universe (symbol, name, sector, source, avg_daily_volume, is_active) "
            "VALUES ('ILLIQUID', 'Illiquid Corp', 'Tech', 'sp500', 100000, TRUE)"
        )
        db.execute(
            "INSERT INTO universe (symbol, name, sector, source, avg_daily_volume, is_active) "
            "VALUES ('LIQUID', 'Liquid Corp', 'Tech', 'sp500', 1000000, TRUE)"
        )
        self._seed_ohlcv(db, "ILLIQUID")
        self._seed_ohlcv(db, "LIQUID")

        screener = AutonomousScreener(db)
        result = screener._screen_sync("unknown")

        assert result.filtered_out == 1
        all_symbols = [s.symbol for s in result.tier_1 + result.tier_2 + result.tier_3]
        assert "ILLIQUID" not in all_symbols

    def test_results_persisted(self, db):
        self._seed_universe(db, ["AAPL", "MSFT"])
        self._seed_ohlcv(db, "AAPL")
        self._seed_ohlcv(db, "MSFT")

        screener = AutonomousScreener(db)
        screener._screen_sync("unknown")

        rows = db.execute("SELECT COUNT(*) FROM screener_results").fetchone()
        assert rows[0] == 2


# ── WatchlistLoader v2 Tests ─────────────────────────────────────────────────


class TestWatchlistLoaderV2:
    def _seed_screener(self, db, symbols, tier_map=None):
        """Insert screener results."""
        now = datetime.now(timezone.utc)
        tier_map = tier_map or {}
        for i, sym in enumerate(symbols):
            tier = tier_map.get(sym, (i // 15) + 1)  # Auto-assign tiers
            db.execute(
                "INSERT INTO screener_results (symbol, screened_at, regime_used, tier, composite_score) "
                "VALUES (?, ?, 'unknown', ?, ?)",
                [sym, now, tier, 1.0 - i * 0.01],
            )

    @patch.dict("os.environ", {"USE_TIERED_WATCHLIST": "true"}, clear=False)
    def test_load_tiered_from_screener(self, db):
        # Patch open_db_readonly to return our test db
        with patch("quantstack.autonomous.watchlist.open_db", return_value=db):
            symbols = [f"T1_{i}" for i in range(15)] + [f"T2_{i}" for i in range(10)]
            tier_map = {s: 1 for s in symbols[:15]}
            tier_map.update({s: 2 for s in symbols[15:]})
            self._seed_screener(db, symbols, tier_map)

            loader = WatchlistLoader()
            tiered = loader.load_tiered()

            assert len(tiered[1]) == 15
            assert len(tiered[2]) == 10

    @patch.dict("os.environ", {"USE_TIERED_WATCHLIST": "true"}, clear=False)
    def test_load_returns_t1_plus_t2(self, db):
        with patch("quantstack.autonomous.watchlist.open_db", return_value=db), \
             patch("quantstack.autonomous.watchlist.open_db_readonly", return_value=db):
            symbols = [f"S{i}" for i in range(25)]
            tier_map = {s: 1 for s in symbols[:15]}
            tier_map.update({s: 2 for s in symbols[15:]})
            self._seed_screener(db, symbols, tier_map)

            loader = WatchlistLoader()
            result = loader.load()
            assert len(result) == 25  # T1 + T2

    @patch.dict("os.environ", {"AUTONOMOUS_WATCHLIST": "XOM,MSFT,SPY"}, clear=False)
    def test_env_override_takes_precedence(self, db):
        loader = WatchlistLoader()
        result = loader.load()
        assert result == ["XOM", "MSFT", "SPY"]

    def test_fallback_to_defaults(self, db):
        """When tiered mode is off and no strategies, fall back to DEFAULT_SYMBOLS."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove any env overrides
            os.environ.pop("AUTONOMOUS_WATCHLIST", None)
            os.environ.pop("USE_TIERED_WATCHLIST", None)

            # Patch strategy loading to return empty (no DB)
            with patch(
                "quantstack.autonomous.watchlist._load_from_strategies", return_value=[]
            ):
                loader = WatchlistLoader()
                result = loader.load()
                assert result == DEFAULT_SYMBOLS


# ── Supervisor Tests ─────────────────────────────────────────────────────────


class TestSupervisor:
    def test_healthy_loop(self, db):
        # Insert a recent heartbeat
        db.execute(
            "INSERT INTO loop_heartbeats (loop_name, iteration, started_at, finished_at, status) "
            "VALUES ('strategy_factory', 1, ?, ?, 'completed')",
            [datetime.now(), datetime.now()],
        )

        configs = [LoopConfig(name="strategy_factory", expected_interval_seconds=60)]
        supervisor = LoopSupervisor(db, configs)
        results = supervisor.check_health()

        assert len(results) == 1
        assert results[0].status == "healthy"
        assert results[0].last_iteration == 1

    def test_stale_loop(self, db):
        # Insert an old heartbeat (5 minutes ago, expected interval 60s → 3x = 180s)
        old_ts = datetime.now() - timedelta(minutes=5)
        db.execute(
            "INSERT INTO loop_heartbeats (loop_name, iteration, started_at, finished_at, status) "
            "VALUES ('strategy_factory', 1, ?, ?, 'completed')",
            [old_ts, old_ts],
        )

        configs = [LoopConfig(name="strategy_factory", expected_interval_seconds=60)]
        supervisor = LoopSupervisor(db, configs)
        results = supervisor.check_health()

        assert results[0].status == "stale"

    def test_dead_loop(self, db):
        # Insert a very old heartbeat (15 minutes ago, expected 60s → 10x = 600s)
        old_ts = datetime.now() - timedelta(minutes=15)
        db.execute(
            "INSERT INTO loop_heartbeats (loop_name, iteration, started_at, finished_at, status) "
            "VALUES ('strategy_factory', 1, ?, ?, 'completed')",
            [old_ts, old_ts],
        )

        configs = [LoopConfig(name="strategy_factory", expected_interval_seconds=60)]
        supervisor = LoopSupervisor(db, configs)
        results = supervisor.check_health()

        assert results[0].status == "dead"

    def test_unknown_if_no_heartbeat(self, db):
        configs = [LoopConfig(name="strategy_factory", expected_interval_seconds=60)]
        supervisor = LoopSupervisor(db, configs)
        results = supervisor.check_health()

        assert results[0].status == "unknown"


# ── DB Migration Tests ───────────────────────────────────────────────────────


class TestDBMigrations:
    def test_new_tables_created(self):
        """Verify run_migrations creates all coordination tables."""
        conn = duckdb.connect(":memory:")

        # We can't call run_migrations directly (it requires shared module),
        # but we can verify the SQL is valid by running the migration functions
        conn.execute("BEGIN")
        _migrate_universe(conn)
        _migrate_screener(conn)
        _migrate_coordination(conn)
        conn.execute("COMMIT")

        # Verify tables exist
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        assert "universe" in tables
        assert "screener_results" in tables
        assert "loop_events" in tables
        assert "loop_cursors" in tables
        assert "loop_heartbeats" in tables

        conn.close()

    def test_migrations_idempotent(self):
        """Running migrations twice should not fail."""
        conn = duckdb.connect(":memory:")
        conn.execute("BEGIN")
        _migrate_universe(conn)
        _migrate_screener(conn)
        _migrate_coordination(conn)
        conn.execute("COMMIT")

        # Second run should be a no-op
        conn.execute("BEGIN")
        _migrate_universe(conn)
        _migrate_screener(conn)
        _migrate_coordination(conn)
        conn.execute("COMMIT")

        conn.close()


# ── AutoPromoter Edge Cases ──────────────────────────────────────────────────


class TestAutoPromoterEdgeCases:
    @patch.dict("os.environ", {"AUTO_PROMOTE_ENABLED": "true"})
    def test_insufficient_trades(self, db):
        # Strategy old enough but not enough trades
        old_date = datetime.now(timezone.utc) - timedelta(days=30)
        db.execute(
            "INSERT INTO strategies (strategy_id, name, parameters, entry_rules, exit_rules, status, updated_at) "
            "VALUES ('s1', 'few_trades', '{}', '{}', '{}', 'forward_testing', ?)",
            [old_date],
        )
        # Only 5 trades (needs 15)
        for i in range(5):
            db.execute(
                "INSERT INTO strategy_outcomes (id, strategy_id, symbol, action, entry_price, exit_price, "
                "realized_pnl_pct, opened_at, closed_at) "
                "VALUES (?, 's1', 'SPY', 'buy', 100, 102, 0.02, ?, ?)",
                [i + 1, old_date + timedelta(days=i), old_date + timedelta(days=i + 1)],
            )

        promoter = AutoPromoter(db)
        decisions = promoter.evaluate_all()
        assert decisions[0].decision == "hold"
        assert "Insufficient trades" in decisions[0].reason

    @patch.dict("os.environ", {"AUTO_PROMOTE_ENABLED": "true"})
    def test_strategy_cap_blocks_promotion(self, db):
        # Fill up live strategy slots
        for i in range(8):
            db.execute(
                "INSERT INTO strategies (strategy_id, name, parameters, entry_rules, exit_rules, status) "
                "VALUES (?, ?, '{}', '{}', '{}', 'live')",
                [f"live_{i}", f"live_strat_{i}"],
            )

        # Add a forward_testing strategy with good metrics
        old_date = datetime.now(timezone.utc) - timedelta(days=30)
        db.execute(
            "INSERT INTO strategies (strategy_id, name, parameters, entry_rules, exit_rules, status, updated_at) "
            "VALUES ('candidate', 'good_strat', '{}', '{}', '{}', 'forward_testing', ?)",
            [old_date],
        )
        for i in range(20):
            db.execute(
                "INSERT INTO strategy_outcomes (id, strategy_id, symbol, action, entry_price, exit_price, "
                "realized_pnl_pct, opened_at, closed_at) "
                "VALUES (?, 'candidate', 'SPY', 'buy', 100, 105, 0.05, ?, ?)",
                [
                    i + 100,
                    old_date + timedelta(days=i),
                    old_date + timedelta(days=i + 1),
                ],
            )

        criteria = PromotionCriteria(max_concurrent_live=8)
        promoter = AutoPromoter(db, criteria=criteria)
        decisions = promoter.evaluate_all()

        # Should be held due to cap
        assert any(
            "strategy_cap" in str(d.criteria_results)
            for d in decisions
            if d.decision == "hold"
        )
