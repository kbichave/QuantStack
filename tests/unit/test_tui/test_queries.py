"""Tests for src/quantstack/tui/queries/ — dashboard query functions.

Each test verifies: correct dataclass type on success, graceful degradation on error.
"""
from datetime import date, datetime, time
from unittest.mock import MagicMock

import pytest

from quantstack.db import PgConnection


def _mock_conn(rows=None, scalar=None):
    """Build a mock PgConnection returning given rows or scalar."""
    conn = MagicMock(spec=PgConnection)
    conn.execute.return_value = conn
    conn.fetchall.return_value = rows or []
    conn.fetchone.return_value = (scalar,) if scalar is not None else None
    return conn


def _failing_conn():
    """Build a mock PgConnection whose execute() raises."""
    conn = MagicMock(spec=PgConnection)
    conn.execute.side_effect = Exception("connection lost")
    return conn


# ── System queries ──────────────────────────────────────────────────

class TestSystemQueries:

    def test_fetch_kill_switch_active(self):
        from quantstack.tui.queries.system import fetch_kill_switch
        conn = _mock_conn(scalar="active")
        assert fetch_kill_switch(conn) is True

    def test_fetch_kill_switch_inactive(self):
        from quantstack.tui.queries.system import fetch_kill_switch
        conn = _mock_conn()  # fetchone returns None
        assert fetch_kill_switch(conn) is False

    def test_fetch_kill_switch_error(self):
        from quantstack.tui.queries.system import fetch_kill_switch
        assert fetch_kill_switch(_failing_conn()) is False

    def test_fetch_av_calls_returns_int(self):
        from quantstack.tui.queries.system import fetch_av_calls
        conn = _mock_conn(scalar="150")
        assert fetch_av_calls(conn) == 150

    def test_fetch_av_calls_error(self):
        from quantstack.tui.queries.system import fetch_av_calls
        assert fetch_av_calls(_failing_conn()) == 0

    def test_fetch_regime_returns_dataclass(self):
        from quantstack.tui.queries.system import RegimeState, fetch_regime
        now = datetime.now()
        conn = _mock_conn(rows=[("SPY", "trending_up", "normal", 0.85)])
        # fetchone should return a row
        conn.fetchone.return_value = ("SPY", "trending_up", "normal", 0.85)
        result = fetch_regime(conn)
        assert isinstance(result, RegimeState)
        assert result.trend == "trending_up"
        assert result.confidence == 0.85

    def test_fetch_regime_error(self):
        from quantstack.tui.queries.system import fetch_regime
        assert fetch_regime(_failing_conn()) is None

    def test_fetch_graph_checkpoints_returns_list(self):
        from quantstack.tui.queries.system import GraphCheckpoint, fetch_graph_checkpoints
        now = datetime.now()
        conn = _mock_conn(rows=[("research", "node1", 5, now, 12.5)])
        result = fetch_graph_checkpoints(conn)
        assert len(result) == 1
        assert isinstance(result[0], GraphCheckpoint)
        assert result[0].graph_name == "research"

    def test_fetch_graph_checkpoints_error(self):
        from quantstack.tui.queries.system import fetch_graph_checkpoints
        assert fetch_graph_checkpoints(_failing_conn()) == []

    def test_fetch_heartbeats_returns_list(self):
        from quantstack.tui.queries.system import Heartbeat, fetch_heartbeats
        now = datetime.now()
        conn = _mock_conn(rows=[("trading_loop", now, "running")])
        result = fetch_heartbeats(conn)
        assert len(result) == 1
        assert isinstance(result[0], Heartbeat)

    def test_fetch_heartbeats_error(self):
        from quantstack.tui.queries.system import fetch_heartbeats
        assert fetch_heartbeats(_failing_conn()) == []

    def test_fetch_agent_events_returns_list(self):
        from quantstack.tui.queries.system import AgentEvent, fetch_agent_events
        now = datetime.now()
        conn = _mock_conn(rows=[("trading", "scan", "scanner", "tool_call", "scanning AAPL", now)])
        result = fetch_agent_events(conn)
        assert len(result) == 1
        assert isinstance(result[0], AgentEvent)

    def test_fetch_agent_events_error(self):
        from quantstack.tui.queries.system import fetch_agent_events
        assert fetch_agent_events(_failing_conn()) == []


# ── Portfolio queries ───────────────────────────────────────────────

class TestPortfolioQueries:

    def test_fetch_equity_summary(self):
        from quantstack.tui.queries.portfolio import EquitySummary, fetch_equity_summary
        conn = _mock_conn()
        conn.fetchone.return_value = (100000.0, 25000.0, 500.0, 0.5, 105000.0, -4.76)
        result = fetch_equity_summary(conn)
        assert isinstance(result, EquitySummary)
        assert result.total_equity == 100000.0
        assert result.high_water == 105000.0
        assert result.drawdown_pct == -4.76

    def test_fetch_equity_summary_error(self):
        from quantstack.tui.queries.portfolio import fetch_equity_summary
        assert fetch_equity_summary(_failing_conn()) is None

    def test_fetch_positions(self):
        from quantstack.tui.queries.portfolio import Position, fetch_positions
        conn = _mock_conn(rows=[
            ("AAPL", 100, 150.0, 160.0, 1000.0, 6.67, "strat_1", 5),
        ])
        result = fetch_positions(conn)
        assert len(result) == 1
        assert isinstance(result[0], Position)
        assert result[0].symbol == "AAPL"

    def test_fetch_positions_error(self):
        from quantstack.tui.queries.portfolio import fetch_positions
        assert fetch_positions(_failing_conn()) == []

    def test_fetch_closed_trades(self):
        from quantstack.tui.queries.portfolio import ClosedTrade, fetch_closed_trades
        now = datetime.now()
        conn = _mock_conn(rows=[
            ("AAPL", "sell", 250.0, 3, "strat_1", "stop_loss", now),
        ])
        result = fetch_closed_trades(conn)
        assert len(result) == 1
        assert isinstance(result[0], ClosedTrade)

    def test_fetch_closed_trades_error(self):
        from quantstack.tui.queries.portfolio import fetch_closed_trades
        assert fetch_closed_trades(_failing_conn()) == []

    def test_fetch_equity_curve(self):
        from quantstack.tui.queries.portfolio import EquityPoint, fetch_equity_curve
        conn = _mock_conn(rows=[
            (date(2026, 4, 2), 100000.0),
            (date(2026, 4, 1), 99500.0),
        ])
        result = fetch_equity_curve(conn)
        assert len(result) == 2
        assert isinstance(result[0], EquityPoint)
        # reversed ordering
        assert result[0].date == date(2026, 4, 1)

    def test_fetch_equity_curve_error(self):
        from quantstack.tui.queries.portfolio import fetch_equity_curve
        assert fetch_equity_curve(_failing_conn()) == []

    def test_fetch_benchmark(self):
        from quantstack.tui.queries.portfolio import BenchmarkPoint, fetch_benchmark
        conn = _mock_conn(rows=[
            (date(2026, 4, 2), "SPY", 520.0, 0.5),
        ])
        result = fetch_benchmark(conn)
        assert len(result) == 1
        assert isinstance(result[0], BenchmarkPoint)
        assert result[0].symbol == "SPY"

    def test_fetch_benchmark_error(self):
        from quantstack.tui.queries.portfolio import fetch_benchmark
        assert fetch_benchmark(_failing_conn()) == []

    def test_fetch_pnl_by_strategy(self):
        from quantstack.tui.queries.portfolio import StrategyPnl, fetch_pnl_by_strategy
        conn = _mock_conn(rows=[
            ("s1", "Momentum AAPL", 500.0, 200.0, 5, 2, 1.5),
        ])
        result = fetch_pnl_by_strategy(conn)
        assert len(result) == 1
        assert isinstance(result[0], StrategyPnl)

    def test_fetch_pnl_by_strategy_error(self):
        from quantstack.tui.queries.portfolio import fetch_pnl_by_strategy
        assert fetch_pnl_by_strategy(_failing_conn()) == []

    def test_fetch_pnl_by_symbol(self):
        from quantstack.tui.queries.portfolio import SymbolPnl, fetch_pnl_by_symbol
        conn = _mock_conn(rows=[("AAPL", 750.0)])
        result = fetch_pnl_by_symbol(conn)
        assert len(result) == 1
        assert isinstance(result[0], SymbolPnl)

    def test_fetch_pnl_by_symbol_error(self):
        from quantstack.tui.queries.portfolio import fetch_pnl_by_symbol
        assert fetch_pnl_by_symbol(_failing_conn()) == []


# ── Strategy queries ────────────────────────────────────────────────

class TestStrategyQueries:

    def test_fetch_strategy_pipeline(self):
        from quantstack.tui.queries.strategies import StrategyCard, fetch_strategy_pipeline
        conn = _mock_conn(rows=[
            ("s1", "Momentum AAPL", "live", "AAPL", "equity", "swing",
             1.5, -0.08, 0.65, 10, 500.0, 30, 30),
        ])
        result = fetch_strategy_pipeline(conn)
        assert len(result) == 1
        assert isinstance(result[0], StrategyCard)
        assert result[0].status == "live"

    def test_fetch_strategy_pipeline_error(self):
        from quantstack.tui.queries.strategies import fetch_strategy_pipeline
        assert fetch_strategy_pipeline(_failing_conn()) == []


# ── Data health queries ─────────────────────────────────────────────

class TestDataHealthQueries:

    def test_fetch_ohlcv_freshness(self):
        from quantstack.tui.queries.data_health import fetch_ohlcv_freshness
        now = datetime.now()
        conn = _mock_conn(rows=[("AAPL", now)])
        result = fetch_ohlcv_freshness(conn)
        assert result == {"AAPL": now}

    def test_fetch_ohlcv_freshness_error(self):
        from quantstack.tui.queries.data_health import fetch_ohlcv_freshness
        assert fetch_ohlcv_freshness(_failing_conn()) == {}

    def test_fetch_news_freshness_error(self):
        from quantstack.tui.queries.data_health import fetch_news_freshness
        assert fetch_news_freshness(_failing_conn()) == {}

    def test_fetch_sentiment_freshness_error(self):
        from quantstack.tui.queries.data_health import fetch_sentiment_freshness
        assert fetch_sentiment_freshness(_failing_conn()) == {}

    def test_fetch_options_freshness_error(self):
        from quantstack.tui.queries.data_health import fetch_options_freshness
        assert fetch_options_freshness(_failing_conn()) == {}

    def test_fetch_insider_freshness_error(self):
        from quantstack.tui.queries.data_health import fetch_insider_freshness
        assert fetch_insider_freshness(_failing_conn()) == {}

    def test_fetch_macro_freshness_error(self):
        from quantstack.tui.queries.data_health import fetch_macro_freshness
        assert fetch_macro_freshness(_failing_conn()) == {}



# ── Signal queries ──────────────────────────────────────────────────

class TestSignalQueries:

    def test_fetch_active_signals(self):
        from quantstack.tui.queries.signals import Signal, fetch_active_signals
        now = datetime.now()
        conn = _mock_conn(rows=[("AAPL", "BUY", 0.85, 5.0, now)])
        result = fetch_active_signals(conn)
        assert len(result) == 1
        assert isinstance(result[0], Signal)
        assert result[0].factors == {}

    def test_fetch_active_signals_error(self):
        from quantstack.tui.queries.signals import fetch_active_signals
        assert fetch_active_signals(_failing_conn()) == []

    def test_fetch_signal_brief(self):
        from quantstack.tui.queries.signals import SignalBrief, fetch_signal_brief
        now = datetime.now()
        conn = _mock_conn()
        conn.fetchone.return_value = ("AAPL", "BUY", 0.85, 5.0, now)
        result = fetch_signal_brief(conn, "AAPL")
        assert isinstance(result, SignalBrief)
        assert result.ml_score is None

    def test_fetch_signal_brief_not_found(self):
        from quantstack.tui.queries.signals import fetch_signal_brief
        conn = _mock_conn()  # fetchone returns None
        assert fetch_signal_brief(conn, "UNKNOWN") is None

    def test_fetch_signal_brief_error(self):
        from quantstack.tui.queries.signals import fetch_signal_brief
        assert fetch_signal_brief(_failing_conn(), "AAPL") is None


# ── Calendar queries ────────────────────────────────────────────────

class TestCalendarQueries:

    def test_fetch_earnings_calendar(self):
        from quantstack.tui.queries.calendar import EarningsEvent, fetch_earnings_calendar
        conn = _mock_conn(rows=[
            ("AAPL", date(2026, 7, 24), 1.50, None, None),
        ])
        result = fetch_earnings_calendar(conn)
        assert len(result) == 1
        assert isinstance(result[0], EarningsEvent)

    def test_fetch_earnings_calendar_error(self):
        from quantstack.tui.queries.calendar import fetch_earnings_calendar
        assert fetch_earnings_calendar(_failing_conn()) == []


# ── Agent queries ───────────────────────────────────────────────────

class TestAgentQueries:

    def test_fetch_graph_activity(self):
        from quantstack.tui.queries.agents import GraphActivity, fetch_graph_activity
        now = datetime.now()
        conn = _mock_conn(rows=[("trading", "scan", 10, now, 5)])
        result = fetch_graph_activity(conn)
        assert len(result) == 1
        assert isinstance(result[0], GraphActivity)

    def test_fetch_graph_activity_error(self):
        from quantstack.tui.queries.agents import fetch_graph_activity
        assert fetch_graph_activity(_failing_conn()) == []

    def test_fetch_cycle_history_error(self):
        from quantstack.tui.queries.agents import fetch_cycle_history
        assert fetch_cycle_history(_failing_conn()) == []

    def test_fetch_agent_skills_error(self):
        from quantstack.tui.queries.agents import fetch_agent_skills
        assert fetch_agent_skills(_failing_conn()) == []

    def test_fetch_calibration(self):
        from quantstack.tui.queries.agents import CalibrationRecord, fetch_calibration
        conn = _mock_conn(rows=[("scanner", 0.8, 0.6)])
        result = fetch_calibration(conn)
        assert len(result) == 1
        assert isinstance(result[0], CalibrationRecord)
        assert result[0].is_overconfident is True  # 0.8 > 0.6 + 0.1

    def test_fetch_calibration_error(self):
        from quantstack.tui.queries.agents import fetch_calibration
        assert fetch_calibration(_failing_conn()) == []

    def test_fetch_prompt_versions_error(self):
        from quantstack.tui.queries.agents import fetch_prompt_versions
        assert fetch_prompt_versions(_failing_conn()) == []


# ── Research queries ────────────────────────────────────────────────

class TestResearchQueries:

    def test_fetch_research_wip_error(self):
        from quantstack.tui.queries.research import fetch_research_wip
        assert fetch_research_wip(_failing_conn()) == []

    def test_fetch_research_queue_error(self):
        from quantstack.tui.queries.research import fetch_research_queue
        assert fetch_research_queue(_failing_conn()) == []

    def test_fetch_ml_experiments(self):
        from quantstack.tui.queries.research import MlExperiment, fetch_ml_experiments
        now = datetime.now()
        conn = _mock_conn(rows=[
            ("exp-001", now, "lgbm", "AAPL", 0.72, 42, "promoted"),
        ])
        result = fetch_ml_experiments(conn)
        assert len(result) == 1
        assert isinstance(result[0], MlExperiment)
        assert result[0].verdict == "promoted"

    def test_fetch_ml_experiments_error(self):
        from quantstack.tui.queries.research import fetch_ml_experiments
        assert fetch_ml_experiments(_failing_conn()) == []

    def test_fetch_alpha_programs_error(self):
        from quantstack.tui.queries.research import fetch_alpha_programs
        assert fetch_alpha_programs(_failing_conn()) == []

    def test_fetch_breakthroughs_error(self):
        from quantstack.tui.queries.research import fetch_breakthroughs
        assert fetch_breakthroughs(_failing_conn()) == []

    def test_fetch_reflections_error(self):
        from quantstack.tui.queries.research import fetch_reflections
        assert fetch_reflections(_failing_conn()) == []

    def test_fetch_bugs_error(self):
        from quantstack.tui.queries.research import fetch_bugs
        assert fetch_bugs(_failing_conn()) == []

    def test_fetch_concept_drift_error(self):
        from quantstack.tui.queries.research import fetch_concept_drift
        assert fetch_concept_drift(_failing_conn()) == []


# ── Risk queries ────────────────────────────────────────────────────

class TestRiskQueries:

    def test_fetch_risk_snapshot(self):
        from quantstack.tui.queries.risk import RiskSnapshot, fetch_risk_snapshot
        now = datetime.now()
        conn = _mock_conn()
        conn.fetchone.return_value = (0.5, 0.3, 0.2, 0.15, 0.1, 0.02, -0.08, now)
        result = fetch_risk_snapshot(conn)
        assert isinstance(result, RiskSnapshot)
        assert result.var_1d == 0.02

    def test_fetch_risk_snapshot_empty(self):
        from quantstack.tui.queries.risk import fetch_risk_snapshot
        conn = _mock_conn()  # fetchone returns None
        assert fetch_risk_snapshot(conn) is None

    def test_fetch_risk_snapshot_error(self):
        from quantstack.tui.queries.risk import fetch_risk_snapshot
        assert fetch_risk_snapshot(_failing_conn()) is None

    def test_fetch_risk_events_error(self):
        from quantstack.tui.queries.risk import fetch_risk_events
        assert fetch_risk_events(_failing_conn()) == []

    def test_fetch_equity_alerts_error(self):
        from quantstack.tui.queries.risk import fetch_equity_alerts
        assert fetch_equity_alerts(_failing_conn()) == []


# ── Cross-cutting: all query functions use PgConnection.execute ─────

class TestQueryInvariants:

    def test_all_query_modules_importable(self):
        """Smoke test: all query modules import without error."""
        from quantstack.tui.queries import (
            agents, calendar, data_health, portfolio, research, risk, signals,
            strategies, system,
        )
        # Just verify they're modules
        assert all(hasattr(m, "__name__") for m in [
            agents, calendar, data_health, portfolio, research, risk,
            signals, strategies, system,
        ])
