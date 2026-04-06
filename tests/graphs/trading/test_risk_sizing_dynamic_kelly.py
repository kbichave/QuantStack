"""Tests for dynamic Kelly wiring in make_risk_sizing() (section-03)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from quantstack.core.kelly_sizing import (
    KELLY_HARD_CEILING,
    KELLY_HARD_FLOOR,
    TARGET_VOL,
    VOL_SCALAR_CAP,
    regime_kelly_fraction,
)


def _make_state(**overrides) -> dict:
    """Build a minimal TradingState dict for testing."""
    base = {
        "cycle_number": 1,
        "regime": "unknown",
        "portfolio_context": {},
        "entry_candidates": [],
        "vol_state": "normal",
        "decisions": [],
        "errors": [],
    }
    base.update(overrides)
    return base


def _mock_db_conn_factory(regime_row=None, ic_rows=None, ohlcv_rows=None):
    """
    Return a context-manager mock for db_conn().

    regime_row: tuple (regime, confidence) or None
    ic_rows: dict[strategy_id -> tuple|None]
    ohlcv_rows: dict[symbol -> list[tuple(close,)]]
    """
    ic_rows = ic_rows or {}
    ohlcv_rows = ohlcv_rows or {}

    class FakeCursor:
        def __init__(self, result):
            self._result = result

        def fetchone(self):
            return self._result

        def fetchall(self):
            return self._result if isinstance(self._result, list) else []

    class FakeConn:
        def __init__(self):
            self._call_count = 0

        def execute(self, sql, params=None):
            sql_lower = sql.strip().lower()

            if "regime_state" in sql_lower:
                return FakeCursor(regime_row)

            if "signal_ic" in sql_lower:
                sid = params[0] if params else None
                row = ic_rows.get(sid)
                return FakeCursor(row)

            if "ohlcv" in sql_lower:
                sym = params[0] if params else None
                rows = ohlcv_rows.get(sym, [])
                return FakeCursor(rows)

            return FakeCursor(None)

    fake_conn = FakeConn()

    class FakeCtx:
        def __enter__(self):
            return fake_conn

        def __exit__(self, *args):
            pass

    return FakeCtx


def _generate_closes(n: int, daily_vol: float = 0.01, start: float = 100.0) -> list[tuple]:
    """Generate n closing prices with known daily vol, return as list of (close,) tuples in DESC order."""
    np.random.seed(42)
    returns = np.random.normal(0, daily_vol, n - 1)
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * np.exp(r))
    # Return in DESC order (most recent first) as the SQL query does
    return [(p,) for p in reversed(prices)]


@pytest.mark.asyncio
async def test_trending_up_low_vol_caps_at_ceiling():
    """trending_up + confidence 1.0 + low vol → kelly capped at KELLY_HARD_CEILING (0.50)."""
    # With regime=trending_up, vol_state=normal, confidence=1.0:
    # regime_multiplier = 0.50
    # Low vol (8%) → vol_scalar = min(0.20/max(0.08, 0.10), 1.5) = min(2.0, 1.5) = 1.5
    # raw_kelly = 0.50 * 1.5 = 0.75 → capped at 0.50

    closes = _generate_closes(64, daily_vol=0.005)  # ~8% annualized

    mock_ctx = _mock_db_conn_factory(
        regime_row=("trending_up", 1.0),
        ic_rows={"strat_1": (0.05,)},
        ohlcv_rows={"AAPL": closes},
    )

    candidates = [
        {"symbol": "AAPL", "strategy_id": "strat_1", "signal_value": 0.8, "verdict": "ENTER"},
    ]

    with patch("quantstack.graphs.trading.nodes.db_conn", side_effect=lambda: mock_ctx()):
        from quantstack.graphs.trading.nodes import make_risk_sizing

        node = make_risk_sizing()
        state = _make_state(entry_candidates=candidates)
        result = await node(state)

    assert result["alpha_signal_candidates"], "Should have candidates"
    decision = result["decisions"][0]
    assert decision["kelly_fraction"] <= KELLY_HARD_CEILING
    assert decision["regime"] == "trending_up"


@pytest.mark.asyncio
async def test_unknown_regime_normal_vol_uses_low_kelly():
    """unknown regime + normal vol → kelly ≈ 0.15."""
    closes = _generate_closes(64, daily_vol=0.0126)  # ~20% annualized (≈ TARGET_VOL)

    mock_ctx = _mock_db_conn_factory(
        regime_row=("unknown", 0.5),
        ic_rows={"strat_1": (0.03,)},
        ohlcv_rows={"SPY": closes},
    )

    candidates = [
        {"symbol": "SPY", "strategy_id": "strat_1", "signal_value": 0.6, "verdict": "ENTER"},
    ]

    with patch("quantstack.graphs.trading.nodes.db_conn", side_effect=lambda: mock_ctx()):
        from quantstack.graphs.trading.nodes import make_risk_sizing

        node = make_risk_sizing()
        state = _make_state(entry_candidates=candidates)
        result = await node(state)

    decision = result["decisions"][0]
    # unknown + low confidence → kelly close to UNKNOWN_KELLY (0.15) × vol_scalar ~1.0
    assert decision["kelly_fraction"] <= 0.20


@pytest.mark.asyncio
async def test_cold_start_symbol_skipped():
    """Symbol with < 21 days of OHLCV → cold start, position deferred."""
    # Only 15 closes → 14 returns → below min_periods=21
    short_closes = [(100.0 + i,) for i in range(15)]

    mock_ctx = _mock_db_conn_factory(
        regime_row=("trending_up", 1.0),
        ic_rows={"strat_1": (0.05,)},
        ohlcv_rows={"NEWIPO": short_closes},
    )

    candidates = [
        {"symbol": "NEWIPO", "strategy_id": "strat_1", "signal_value": 0.9, "verdict": "ENTER"},
    ]

    with patch("quantstack.graphs.trading.nodes.db_conn", side_effect=lambda: mock_ctx()):
        from quantstack.graphs.trading.nodes import make_risk_sizing

        node = make_risk_sizing()
        state = _make_state(entry_candidates=candidates)
        result = await node(state)

    # All candidates cold-start → no alpha signals
    assert result["alpha_signals"] == []
    assert result["decisions"][0]["action"] == "all_cold_start"


@pytest.mark.asyncio
async def test_vol_state_persisted_in_result():
    """vol_state is returned in result for hysteresis across cycles."""
    closes = _generate_closes(64, daily_vol=0.0126)

    mock_ctx = _mock_db_conn_factory(
        regime_row=("ranging", 0.9),
        ic_rows={"strat_1": (0.04,)},
        ohlcv_rows={"MSFT": closes},
    )

    candidates = [
        {"symbol": "MSFT", "strategy_id": "strat_1", "signal_value": 0.7, "verdict": "ENTER"},
    ]

    with patch("quantstack.graphs.trading.nodes.db_conn", side_effect=lambda: mock_ctx()):
        from quantstack.graphs.trading.nodes import make_risk_sizing

        node = make_risk_sizing()
        state = _make_state(entry_candidates=candidates, vol_state="normal")
        result = await node(state)

    assert "vol_state" in result
    assert result["vol_state"] in ("normal", "high")


@pytest.mark.asyncio
async def test_kelly_floor_applied():
    """When computed kelly is very low, floor of KELLY_HARD_FLOOR applies."""
    # High vol → regime_multiplier shrinks, vol_scalar < 1
    closes = _generate_closes(64, daily_vol=0.04)  # ~63% annualized

    mock_ctx = _mock_db_conn_factory(
        regime_row=("unknown", 0.1),  # Low confidence → kelly close to 0.15
        ic_rows={"strat_1": (0.02,)},
        ohlcv_rows={"VOLATILE": closes},
    )

    candidates = [
        {"symbol": "VOLATILE", "strategy_id": "strat_1", "signal_value": 0.5, "verdict": "ENTER"},
    ]

    with patch("quantstack.graphs.trading.nodes.db_conn", side_effect=lambda: mock_ctx()):
        from quantstack.graphs.trading.nodes import make_risk_sizing

        node = make_risk_sizing()
        state = _make_state(entry_candidates=candidates)
        result = await node(state)

    decision = result["decisions"][0]
    assert decision["kelly_fraction"] >= KELLY_HARD_FLOOR
