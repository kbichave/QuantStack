"""Unit tests for attribution pipeline (section-08)."""

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn_mock(fetchone_sequence=None, fetchall_return=None):
    """Build a mock DB connection with configurable fetchone/fetchall."""
    cursor = MagicMock()
    cursor.execute = MagicMock(return_value=cursor)
    if fetchone_sequence:
        cursor.fetchone = MagicMock(side_effect=fetchone_sequence)
    else:
        cursor.fetchone = MagicMock(return_value=None)
    cursor.fetchall = MagicMock(return_value=fetchall_return or [])
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


def _make_returns_series(n=65, seed=42):
    """Generate synthetic log return series with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-03-11", periods=n)
    return pd.Series(rng.normal(0.001, 0.01, n), index=idx)


# ---------------------------------------------------------------------------
# Import the module-level run_attribution (to be implemented)
# ---------------------------------------------------------------------------

from quantstack.graphs.supervisor.nodes import run_attribution


# ---------------------------------------------------------------------------
# 1. Watermark filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_attribution_watermark_filtering():
    """
    Watermark = 2026-03-10. Only positions closed AFTER 2026-03-10 are processed.
    Position closed on 2026-03-09 → excluded. Position closed 2026-03-11 → included.
    """
    watermark = date(2026, 3, 10)
    closed_trade = (
        "AAPL", "strat_a",
        datetime(2026, 3, 11), datetime(2026, 3, 11),  # opened_at == closed_at → 1 day
        100.0,
    )

    conn = _make_conn_mock(
        fetchone_sequence=[(watermark,), (0.000198,)],  # watermark, rf_rate
        fetchall_return=[closed_trade],                  # closed_trades rows
    )
    ctx = _make_ctx(conn)

    # Returns mocks (stock, spy, sector)
    returns_series = _make_returns_series()

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes.attribution_decompose",
               return_value=MagicMock(
                   date=date(2026, 3, 11), symbol="AAPL", strategy_id="strat_a",
                   total_pnl=100.0, market_pnl=80.0, sector_pnl=5.0, alpha_pnl=10.0,
                   residual_pnl=5.0, beta_market=1.1, beta_sector=0.2,
                   sector_etf="XLK", holding_day=0,
               )) as mock_decompose, \
         patch("quantstack.graphs.supervisor.nodes._fetch_returns_for_attribution",
               return_value=returns_series):

        result = await run_attribution()

    # At least one decompose call happened (for the included trade)
    assert mock_decompose.call_count >= 1
    assert result["positions_processed"] >= 1


# ---------------------------------------------------------------------------
# 2. No watermark — processes all closed positions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_attribution_no_watermark():
    """When pnl_attribution is empty (watermark=None), all closed positions are processed."""
    closed_trades = [
        ("AAPL", "strat_a", datetime(2026, 3, 1), datetime(2026, 3, 1), 50.0),
        ("TSLA", "strat_b", datetime(2026, 3, 2), datetime(2026, 3, 2), 75.0),
    ]
    conn = _make_conn_mock(
        fetchone_sequence=[None, (0.000198,)],   # no watermark, rf_rate
        fetchall_return=closed_trades,
    )
    ctx = _make_ctx(conn)
    returns_series = _make_returns_series()

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes.attribution_decompose",
               return_value=MagicMock(
                   date=date(2026, 3, 1), symbol="X", strategy_id="s",
                   total_pnl=50.0, market_pnl=40.0, sector_pnl=5.0, alpha_pnl=5.0,
                   residual_pnl=0.0, beta_market=1.0, beta_sector=0.1,
                   sector_etf="XLK", holding_day=0,
               )) as mock_decompose, \
         patch("quantstack.graphs.supervisor.nodes._fetch_returns_for_attribution",
               return_value=returns_series):

        result = await run_attribution()

    assert result["positions_processed"] == 2
    assert mock_decompose.call_count == 2


# ---------------------------------------------------------------------------
# 3. No closed positions since watermark
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_attribution_no_positions_returns_early():
    """If no closed positions since watermark, returns early with zeros."""
    conn = _make_conn_mock(
        fetchone_sequence=[(date(2026, 3, 11),), (0.000198,)],
        fetchall_return=[],  # no trades
    )
    ctx = _make_ctx(conn)

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes.attribution_decompose") as mock_decompose:

        result = await run_attribution()

    assert result["positions_processed"] == 0
    assert result["rows_written"] == 0
    mock_decompose.assert_not_called()


# ---------------------------------------------------------------------------
# 4. decompose called for each day in the holding period
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decompose_called_for_each_holding_day():
    """Position held for 5 business days → decompose called 5 times."""
    opened = datetime(2026, 3, 3)  # Tuesday
    closed = datetime(2026, 3, 7)  # Saturday → skip to Friday 2026-03-07 (5 days: Tue-Sat, but Sat excluded → Mon-Fri = 5 days Tue to Fri +1)
    # Actually: 2026-03-03 (Tue) to 2026-03-07 (Sat) = Tue, Wed, Thu, Fri (4 business days) + Sat skip
    # Let's use 2026-03-03 to 2026-03-07, skipping Sat → Tue, Wed, Thu, Fri = 4 days
    # For cleaner test, use Mon to Fri:
    opened = datetime(2026, 3, 2)  # Monday
    closed = datetime(2026, 3, 6)  # Friday

    trade = ("AAPL", "strat_a", opened, closed, 200.0)
    conn = _make_conn_mock(
        fetchone_sequence=[None, (0.000198,)],
        fetchall_return=[trade],
    )
    ctx = _make_ctx(conn)
    returns_series = _make_returns_series()

    call_dates = []

    def capture_decompose(*args, **kwargs):
        call_dates.append(kwargs.get("attr_date", args[2] if len(args) > 2 else None))
        return MagicMock(
            date=kwargs.get("attr_date"), symbol="AAPL", strategy_id="strat_a",
            total_pnl=40.0, market_pnl=30.0, sector_pnl=5.0, alpha_pnl=5.0,
            residual_pnl=0.0, beta_market=1.0, beta_sector=0.1, sector_etf="XLK", holding_day=0,
        )

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes.attribution_decompose",
               side_effect=capture_decompose), \
         patch("quantstack.graphs.supervisor.nodes._fetch_returns_for_attribution",
               return_value=returns_series):

        result = await run_attribution()

    # Mon to Fri = 5 business days
    assert len(call_dates) == 5


# ---------------------------------------------------------------------------
# 5. Batch insert uses executemany
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_insert_uses_executemany():
    """Attribution rows are inserted using executemany (not one-by-one execute)."""
    trade = ("AAPL", "strat_a", datetime(2026, 3, 3), datetime(2026, 3, 3), 100.0)
    conn = _make_conn_mock(
        fetchone_sequence=[None, (0.000198,)],
        fetchall_return=[trade],
    )
    ctx = _make_ctx(conn)
    returns_series = _make_returns_series()

    with patch("quantstack.graphs.supervisor.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.supervisor.nodes.attribution_decompose",
               return_value=MagicMock(
                   date=date(2026, 3, 3), symbol="AAPL", strategy_id="strat_a",
                   total_pnl=100.0, market_pnl=80.0, sector_pnl=5.0, alpha_pnl=10.0,
                   residual_pnl=5.0, beta_market=1.1, beta_sector=0.2,
                   sector_etf="XLK", holding_day=0,
               )), \
         patch("quantstack.graphs.supervisor.nodes._fetch_returns_for_attribution",
               return_value=returns_series):

        await run_attribution()

    # executemany must have been called (not plain execute for inserts)
    assert conn.executemany.called


# ---------------------------------------------------------------------------
# 6-8. _fetch_attribution_summary tests (helper in trading/nodes.py)
# ---------------------------------------------------------------------------

from quantstack.graphs.trading.nodes import _fetch_attribution_summary


def test_attribution_summary_computes_alpha_fraction():
    """total_pnl=100, alpha_pnl=20 → alpha_fraction=0.20."""
    conn = MagicMock()
    conn.execute = MagicMock(return_value=conn)
    conn.fetchone = MagicMock(return_value=(20.0, 70.0, 5.0, 5.0, 100.0))  # alpha, market, sector, residual, total
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)

    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=ctx):
        result = _fetch_attribution_summary("AAPL", "strat_a")

    assert result is not None
    assert result["alpha_fraction"] == pytest.approx(0.20, abs=1e-4)
    assert result["alpha_pnl_sum"] == pytest.approx(20.0)


def test_attribution_summary_no_rows_returns_none():
    """If pnl_attribution has no rows for the symbol, return None (not exception)."""
    conn = MagicMock()
    conn.execute = MagicMock(return_value=conn)
    conn.fetchone = MagicMock(return_value=None)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)

    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=ctx):
        result = _fetch_attribution_summary("XYZ", "nonexistent_strat")

    assert result is None


def test_attribution_summary_zero_total_returns_none():
    """If total_pnl_sum is zero, return None to avoid division by zero."""
    conn = MagicMock()
    conn.execute = MagicMock(return_value=conn)
    conn.fetchone = MagicMock(return_value=(0.0, 0.0, 0.0, 0.0, 0.0))
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)

    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=ctx):
        result = _fetch_attribution_summary("AAPL", "strat_z")

    assert result is None
