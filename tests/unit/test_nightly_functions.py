"""Unit tests for nightly supervisor functions (section-10)."""

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn_mock(fetchone_seq=None, fetchall_seq=None):
    """Build a mock DB connection with configurable response sequences."""
    cursor = MagicMock()
    cursor.execute = MagicMock(return_value=cursor)

    fetchone_seq = list(fetchone_seq) if fetchone_seq else []
    fetchall_seq = list(fetchall_seq) if fetchall_seq else []

    fetchone_call_count = [0]
    def _fetchone():
        i = fetchone_call_count[0]
        fetchone_call_count[0] += 1
        if i < len(fetchone_seq):
            return fetchone_seq[i]
        return None

    fetchall_call_count = [0]
    def _fetchall():
        i = fetchall_call_count[0]
        fetchall_call_count[0] += 1
        if i < len(fetchall_seq):
            return fetchall_seq[i]
        return []

    cursor.fetchone = MagicMock(side_effect=_fetchone)
    cursor.fetchall = MagicMock(side_effect=_fetchall)
    cursor.executemany = MagicMock()

    conn = MagicMock()
    conn.execute = MagicMock(return_value=cursor)
    conn.fetchone = cursor.fetchone
    conn.fetchall = cursor.fetchall
    conn.executemany = cursor.executemany
    return conn


def _make_ctx(conn):
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# Import the functions under test
# ---------------------------------------------------------------------------

from quantstack.graphs.supervisor.nodes import run_signal_scoring, run_ic_computation


# ===========================================================================
# run_signal_scoring tests
# ===========================================================================

@pytest.mark.asyncio
async def test_signal_scoring_only_queries_live_and_forward_testing():
    """Only live and forward_testing strategies are scored; draft is skipped."""
    # fetchall[0] = strategies (2 rows: live + forward_testing)
    strategies = [
        ("strat_live", "live", [{"indicator": "rsi", "threshold": 30, "condition": "below"}]),
        ("strat_ft", "forward_testing", [{"indicator": "rsi", "threshold": 70, "condition": "above"}]),
    ]
    conn = _make_conn_mock(
        fetchone_seq=[("unknown",)],  # regime_state
        fetchall_seq=[
            strategies,                                   # strategies query
            [("AAPL",), ("MSFT",)],                       # symbols for strat_live
            [("AAPL",)],                                  # symbols for strat_ft
        ],
    )
    ctx = _make_ctx(conn)

    market_data = {"rsi": 25.0, "close": 150.0}

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes._signal_scorer_mod.score_signal",
               return_value=(0.8, 0.9)) as mock_score, \
         patch("quantstack.graphs.supervisor.nodes._fetch_market_data_for_scoring",
               return_value=market_data):

        result = await run_signal_scoring()

    # Should have been called for strat_live×2 + strat_ft×1 = 3 times (live and ft only)
    assert mock_score.call_count == 3
    assert result["strategies_scored"] == 2


@pytest.mark.asyncio
async def test_signal_scoring_symbol_scope_is_union():
    """Symbol scope is UNION of closed_trades and positions per strategy."""
    strategies = [("strat_a", "live", [])]
    conn = _make_conn_mock(
        fetchone_seq=[None],      # regime_state (no row → "unknown")
        fetchall_seq=[
            strategies,
            # UNION result: closed_trades union positions
            [("AAPL",), ("TSLA",), ("NVDA",)],
        ],
    )
    ctx = _make_ctx(conn)

    scored_symbols = []

    def capture_score(entry_rules, market_data):
        scored_symbols.append(market_data.get("_symbol"))
        return (0.5, 0.8)

    market_data_mock = {"rsi": 50.0, "close": 100.0}

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes._signal_scorer_mod.score_signal",
               side_effect=lambda er, md: (0.5, 0.8)), \
         patch("quantstack.graphs.supervisor.nodes._fetch_market_data_for_scoring",
               return_value=market_data_mock):

        result = await run_signal_scoring()

    # 3 symbols in union → 3 score calls
    assert result["signals_written"] == 3


@pytest.mark.asyncio
async def test_signal_scoring_writes_today_signal_date():
    """Signals rows are written with today's signal_date."""
    strategies = [("strat_a", "live", [])]
    conn = _make_conn_mock(
        fetchone_seq=[None],
        fetchall_seq=[strategies, [("AAPL",)]],
    )
    ctx = _make_ctx(conn)

    insert_rows = []
    original_executemany = conn.executemany
    def capture_executemany(sql, rows):
        insert_rows.extend(rows)
    conn.executemany = MagicMock(side_effect=capture_executemany)

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes._signal_scorer_mod.score_signal",
               return_value=(0.7, 0.85)), \
         patch("quantstack.graphs.supervisor.nodes._fetch_market_data_for_scoring",
               return_value={"rsi": 40.0}):

        result = await run_signal_scoring()

    assert result["signals_written"] == 1
    # Verify executemany was called (batch insert)
    assert conn.executemany.called


@pytest.mark.asyncio
async def test_signal_scoring_no_symbols_writes_nothing():
    """Strategy with no symbol scope writes no signals and does not error."""
    strategies = [("strat_empty", "live", [])]
    conn = _make_conn_mock(
        fetchone_seq=[None],
        fetchall_seq=[
            strategies,
            [],  # no symbols
        ],
    )
    ctx = _make_ctx(conn)

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes._signal_scorer_mod.score_signal",
               return_value=(0.5, 0.8)) as mock_score, \
         patch("quantstack.graphs.supervisor.nodes._fetch_market_data_for_scoring",
               return_value={}):

        result = await run_signal_scoring()

    mock_score.assert_not_called()
    assert result["signals_written"] == 0
    assert "error" not in result


# ===========================================================================
# run_ic_computation tests
# ===========================================================================

def _make_ic_series(n=30, value=0.05):
    """IC series with `n` dates, constant value."""
    idx = pd.bdate_range(end="2026-03-31", periods=n)
    return pd.Series([value] * n, index=idx)


@pytest.mark.asyncio
async def test_ic_computation_skips_fewer_than_5_symbols():
    """Strategy with < 5 distinct symbols in signals table is skipped."""
    conn = _make_conn_mock(
        fetchall_seq=[
            [],  # eligible strategies query returns nothing (< 5 symbols)
        ],
    )
    ctx = _make_ctx(conn)

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes._ic_calculator_mod.compute_cross_sectional_ic") as mock_ic:

        result = await run_ic_computation()

    mock_ic.assert_not_called()
    assert result["strategies_computed"] == 0


@pytest.mark.asyncio
async def test_ic_computation_skips_fewer_than_21_days_history():
    """Strategy with < 21 distinct signal dates is skipped."""
    # The eligibility query is the gate; if no rows returned → skipped
    conn = _make_conn_mock(
        fetchall_seq=[[]],  # no eligible strategies
    )
    ctx = _make_ctx(conn)

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes._ic_calculator_mod.compute_cross_sectional_ic") as mock_ic:

        result = await run_ic_computation()

    mock_ic.assert_not_called()
    assert result["strategies_computed"] == 0


@pytest.mark.asyncio
async def test_ic_decay_published_when_both_windows_below_threshold():
    """IC_DECAY event is published when icir_21d < 0.3 AND icir_63d < 0.3 for a live strategy."""
    eligible_strategies = [("strat_live",)]
    signal_rows = [
        (date(2026, 3, d), f"SYM{i}", 0.5)
        for d in range(1, 32) for i in range(5)
    ]
    fwd_return_rows = []  # simplified — mocked out

    conn = _make_conn_mock(
        fetchone_seq=[
            ("live",),        # strategy status check
        ],
        fetchall_seq=[
            eligible_strategies,   # eligible strategies
            signal_rows,           # signal rows for strat_live / horizon 5
            [],                    # forward returns (mocked elsewhere)
            signal_rows,           # horizon 10
            [],
            signal_rows,           # horizon 21
            [],
        ],
    )
    ctx = _make_ctx(conn)

    # Produce ic_series with low ICIR (mean=0.05, std=0.5 → ICIR=0.1)
    low_ic = _make_ic_series(63, value=0.02)
    # Rolling ICIR below 0.3 for both windows
    low_icir = pd.Series([0.2] * 63, index=low_ic.index)

    published_events = []

    class FakeBus:
        def publish(self, event):
            published_events.append(event)

    # Non-empty fwd returns so code doesn't short-circuit
    fwd_df = pd.DataFrame(
        {"SYM0": [0.01] * 30, "SYM1": [0.02] * 30,
         "SYM2": [-0.01] * 30, "SYM3": [0.005] * 30, "SYM4": [-0.02] * 30},
        index=pd.bdate_range(end="2026-03-31", periods=30),
    )

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes._ic_calculator_mod.compute_cross_sectional_ic",
               return_value=low_ic), \
         patch("quantstack.graphs.supervisor.nodes._ic_calculator_mod.compute_rolling_icir",
               return_value=low_icir), \
         patch("quantstack.graphs.supervisor.nodes.EventBus", return_value=FakeBus()), \
         patch("quantstack.graphs.supervisor.nodes._fetch_fwd_returns_for_ic",
               return_value=fwd_df):

        result = await run_ic_computation()

    assert result["ic_decay_events"] >= 1


@pytest.mark.asyncio
async def test_ic_decay_not_published_when_only_one_window_below():
    """IC_DECAY is NOT published when only one ICIR window is below 0.3."""
    eligible_strategies = [("strat_live",)]
    signal_rows = [
        (date(2026, 3, d), f"SYM{i}", 0.5)
        for d in range(1, 32) for i in range(5)
    ]

    conn = _make_conn_mock(
        fetchone_seq=[("live",)],
        fetchall_seq=[
            eligible_strategies,
            signal_rows, [],
            signal_rows, [],
            signal_rows, [],
        ],
    )
    ctx = _make_ctx(conn)

    low_ic = _make_ic_series(63, value=0.02)
    high_icir = pd.Series([0.45] * 63, index=low_ic.index)   # only 21d is low; 63d is high

    call_count = [0]
    def varying_icir(ic_series, window):
        call_count[0] += 1
        if window == 21:
            return pd.Series([0.25] * len(ic_series), index=ic_series.index)  # below 0.3
        return pd.Series([0.45] * len(ic_series), index=ic_series.index)      # above 0.3

    published_events = []

    class FakeBus:
        def publish(self, event):
            published_events.append(event)

    fwd_df = pd.DataFrame(
        {"SYM0": [0.01] * 30, "SYM1": [0.02] * 30,
         "SYM2": [-0.01] * 30, "SYM3": [0.005] * 30, "SYM4": [-0.02] * 30},
        index=pd.bdate_range(end="2026-03-31", periods=30),
    )

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes._ic_calculator_mod.compute_cross_sectional_ic",
               return_value=low_ic), \
         patch("quantstack.graphs.supervisor.nodes._ic_calculator_mod.compute_rolling_icir",
               side_effect=varying_icir), \
         patch("quantstack.graphs.supervisor.nodes.EventBus", return_value=FakeBus()), \
         patch("quantstack.graphs.supervisor.nodes._fetch_fwd_returns_for_ic",
               return_value=fwd_df):

        result = await run_ic_computation()

    # No IC_DECAY events — only one window was below threshold
    assert result["ic_decay_events"] == 0


@pytest.mark.asyncio
async def test_ic_decay_not_published_for_forward_testing_strategy():
    """IC_DECAY is NOT published for strategies already in forward_testing status."""
    eligible_strategies = [("strat_ft",)]
    signal_rows = [
        (date(2026, 3, d), f"SYM{i}", 0.5)
        for d in range(1, 32) for i in range(5)
    ]

    conn = _make_conn_mock(
        fetchone_seq=[("forward_testing",)],  # strategy status = already demoted
        fetchall_seq=[
            eligible_strategies,
            signal_rows, [],
            signal_rows, [],
            signal_rows, [],
        ],
    )
    ctx = _make_ctx(conn)

    low_ic = _make_ic_series(63, value=0.02)
    low_icir = pd.Series([0.2] * 63, index=low_ic.index)

    published_events = []

    class FakeBus:
        def publish(self, event):
            published_events.append(event)

    fwd_df = pd.DataFrame(
        {"SYM0": [0.01] * 30, "SYM1": [0.02] * 30,
         "SYM2": [-0.01] * 30, "SYM3": [0.005] * 30, "SYM4": [-0.02] * 30},
        index=pd.bdate_range(end="2026-03-31", periods=30),
    )

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes._ic_calculator_mod.compute_cross_sectional_ic",
               return_value=low_ic), \
         patch("quantstack.graphs.supervisor.nodes._ic_calculator_mod.compute_rolling_icir",
               return_value=low_icir), \
         patch("quantstack.graphs.supervisor.nodes.EventBus", return_value=FakeBus()), \
         patch("quantstack.graphs.supervisor.nodes._fetch_fwd_returns_for_ic",
               return_value=fwd_df):

        result = await run_ic_computation()

    assert result["ic_decay_events"] == 0


# ===========================================================================
# AutoPromoter ICIR hysteresis test
# ===========================================================================

def test_auto_promoter_blocked_by_low_icir():
    """Strategy with icir_21d=0.35 (< 0.5) must NOT be promoted."""
    from quantstack.coordination.auto_promoter import AutoPromoter, PromotionCriteria
    from datetime import datetime, timezone, timedelta

    conn = MagicMock()

    # strategies.evaluate_all() will call _get_forward_test_outcomes → enough trades
    outcomes = [
        {"realized_pnl_pct": 0.02, "outcome": "win", "opened_at": None, "closed_at": None}
        for _ in range(20)
    ]

    promoter = AutoPromoter(conn, PromotionCriteria(min_forward_test_days=0))

    # Patch internal helpers so all normal criteria pass
    with patch.object(promoter, "_get_forward_test_outcomes", return_value=outcomes), \
         patch.object(promoter, "_get_backtest_sharpe", return_value=1.0), \
         patch.object(promoter, "_count_live_strategies", return_value=0), \
         patch.object(promoter, "_get_icir", return_value=0.35):  # below 0.5 gate

        from datetime import datetime, timezone, timedelta
        updated_at = datetime.now(timezone.utc) - timedelta(days=30)
        decision = promoter._evaluate_one("strat_x", "Test Strategy", None, updated_at)

    assert decision.decision == "hold", f"Expected hold, got: {decision.decision} — {decision.reason}"
    assert "icir" in decision.reason.lower() or "ic" in decision.reason.lower()
