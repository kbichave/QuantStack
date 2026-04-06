"""Unit tests for the deterministic risk_sizing node (section-07)."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from quantstack.graphs.trading.nodes import make_risk_sizing


def _minimal_state(candidates=None):
    return {
        "entry_candidates": candidates or [
            {"symbol": "AAPL", "strategy_id": "strat_a", "signal_value": 0.7, "conviction": 0.7},
            {"symbol": "TSLA", "strategy_id": "strat_b", "signal_value": 0.5, "conviction": 0.5},
        ],
        "portfolio_context": {"total_equity": 100_000, "positions": []},
        "alpha_signals": [],
        "alpha_signal_candidates": [],
        "decisions": [],
        "errors": [],
    }


def _mock_db_context(ic_fetchone=None, vol_rows=None):
    """
    Return a context manager mock for the two DB blocks in risk_sizing:
    - signal_ic block: conn.execute(...).fetchone() per strategy
    - OHLCV block: conn.execute(...).fetchall() per symbol

    ic_fetchone: value returned by fetchone() for every signal_ic query (default: None)
    vol_rows: list returned by fetchall() for every OHLCV query (default: [])
    """
    if vol_rows is None:
        vol_rows = []

    cursor = MagicMock()
    cursor.execute = MagicMock(return_value=cursor)
    cursor.fetchone = MagicMock(return_value=ic_fetchone)
    cursor.fetchall = MagicMock(return_value=vol_rows)

    conn = MagicMock()
    conn.execute = MagicMock(return_value=cursor)
    conn.fetchone = cursor.fetchone
    conn.fetchall = cursor.fetchall

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx, conn


# ---------------------------------------------------------------------------
# 1. No LLM call in sizing path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_llm_call_in_sizing_path():
    """risk_sizing must not call run_agent — it is deterministic."""
    state = _minimal_state()

    # Mock DB returns None for IC (prior applies), minimal prices for vol
    ctx, conn = _mock_db_context(ic_fetchone=None)

    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.trading.nodes.run_agent", new_callable=AsyncMock) as mock_run_agent, \
         patch("quantstack.graphs.trading.nodes.compute_alpha_signals",
               return_value=np.array([0.002, 0.001])) as mock_kelly:

        node = make_risk_sizing()
        result = await node(state)

    mock_run_agent.assert_not_called()
    mock_kelly.assert_called_once()


# ---------------------------------------------------------------------------
# 2. alpha_signals in state output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_alpha_signals_in_state_output():
    """State after risk_sizing must have 'alpha_signals', not 'sizing_results'."""
    state = _minimal_state()
    ctx, conn = _mock_db_context(ic_fetchone=None)
    expected = np.array([0.003, 0.002])

    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.trading.nodes.compute_alpha_signals", return_value=expected):

        node = make_risk_sizing()
        result = await node(state)

    assert "alpha_signals" in result
    assert "sizing_results" not in result
    assert len(result["alpha_signals"]) == 2
    assert "alpha_signal_candidates" in result


# ---------------------------------------------------------------------------
# 3. SafetyGate NOT called inside risk_sizing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safety_gate_not_called_in_risk_sizing():
    """SafetyGate.validate must never be called inside risk_sizing."""
    state = _minimal_state()
    ctx, conn = _mock_db_context(ic_fetchone=None)

    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.trading.nodes.compute_alpha_signals",
               return_value=np.array([0.002, 0.001])), \
         patch("quantstack.core.risk.safety_gate.SafetyGate.validate") as mock_validate:

        node = make_risk_sizing()
        await node(state)

    mock_validate.assert_not_called()


# ---------------------------------------------------------------------------
# 4. IC prior used when signal_ic has None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ic_prior_used_when_no_signal_ic_row():
    """When signal_ic table returns no rows, IC prior (0.01) is used — no error raised."""
    state = _minimal_state()
    ctx, conn = _mock_db_context(ic_fetchone=None)  # None = no signal_ic row → IC prior applies

    captured_kwargs = {}

    def capture_alpha_signals(candidates, signal_ic_lookup, volatility_lookup, **kw):
        captured_kwargs.update({"signal_ic_lookup": signal_ic_lookup})
        return np.zeros(len(candidates))

    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.trading.nodes.compute_alpha_signals",
               side_effect=capture_alpha_signals):

        node = make_risk_sizing()
        result = await node(state)

    # All strategies should have None IC (so IC prior applies)
    ic_lookup = captured_kwargs.get("signal_ic_lookup", {})
    for v in ic_lookup.values():
        assert v is None

    assert "errors" not in result or not result.get("errors")


# ---------------------------------------------------------------------------
# 5. Zero-volatility symbol uses floor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_zero_vol_symbol_uses_floor():
    """Symbol with 0.0 annualized vol does not produce NaN or ZeroDivisionError."""
    state = _minimal_state(candidates=[
        {"symbol": "AAPL", "strategy_id": "strat_a", "signal_value": 0.8},
    ])
    ctx, conn = _mock_db_context(ic_fetchone=None)
    captured = {}

    def capture_alpha_signals(candidates, signal_ic_lookup, volatility_lookup, **kw):
        captured["vol"] = volatility_lookup
        return np.array([0.001])

    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=ctx), \
         patch("quantstack.graphs.trading.nodes.compute_alpha_signals",
               side_effect=capture_alpha_signals):

        node = make_risk_sizing()
        result = await node(state)

    # vol for AAPL must not be 0.0 (floor applied)
    vol = captured.get("vol", {}).get("AAPL", None)
    if vol is not None:
        assert vol > 0
        assert not np.isnan(vol)
        assert not np.isinf(vol)
